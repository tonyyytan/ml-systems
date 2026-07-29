"""
Project 03, Step 3: the model runner. THE FLOOR.

Steps 1-2 proved the roofline predicts decode throughput. This is the first
piece of actual engine: load a real model, generate tokens, measure the rate.
It is deliberately dumb -- no offload policy, no quantization, no paging. Its
only job is to be *correct* and to expose the one interface everything else
(llama.cpp, vllm, my engine) also exposes, so bench.py can time them uniformly:

    generate(prompts: list[str], max_tokens: int) -> list[str]

Why a floor at all? Because "my engine is fast" is meaningless without a plain,
honest baseline running the same model on the same card. This is that baseline.
We do NOT write our own model loading -- HuggingFace already has a correct one,
and Step 3 only needs a floor, not a contribution (see README "borrowable").

Two modes, and the second is the one that matters:
  - batch 1        : one prompt at a time. simplest possible decode loop.
  - static batching : several prompts at once, padded to the longest, advancing
                      in lockstep until the slowest finishes. This exists so the
                      final chart can prove "MY batching is good" against a plain
                      batching baseline -- not merely "batching helps", which is
                      obvious and which nobody doubts.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- config -----------------------------------------------------------------
# TODO: fill on the machine. Use the SAME model family the roofline is built
#       around (llama-3-8b) so the numbers are comparable. Precision here is
#       whatever HF loads (fp16); quantization is Step 4, not this file.
MODEL_ID = None       # e.g. "meta-llama/Meta-Llama-3-8B"
DEVICE = "cuda"
DTYPE = torch.float16


@dataclass
class GenConfig:
    """Decode settings. Greedy only for now -- deterministic output makes
    correctness checkable (same prompt -> same tokens, every run) and takes
    sampling randomness out of the throughput measurement."""
    max_tokens: int = 128
    # greedy = argmax each step. no temperature/top-p yet; add later only if a
    # measurement needs it. quality is measured separately (perplexity, README).


class Runner:
    """Loads the model once, then serves generate() calls. Holding the model on
    the class (not reloading per call) is the whole point -- loading 16 GB of
    weights is slow and we amortize it across every benchmark run."""

    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE).eval()

    @torch.no_grad()
    def generate(self, prompts: list[str], max_tokens: int) -> list[str]:
        """
        The uniform interface. Given prompts, return the generated continuations.
        Routes to batch-1 or static-batch based on how many prompts came in.
        This signature is the contract every baseline in bench.py implements.
        """
        if len(prompts) == 1:
            return [self._generate_one(prompts[0], max_tokens)]
        return self._generate_static_batch(prompts, max_tokens)

    def _generate_one(self, prompt: str, max_tokens: int) -> str:
        """
        Batch-1 greedy decode with a KV cache. The loop that IS decode:

          1. tokenize the prompt -> input_ids
          2. PREFILL: run the whole prompt through once. This fills the KV cache
             and gives logits for the next token. (compute-bound, done once.)
          3. DECODE loop, max_tokens times: (memory-bound, done every token)
               - take logits for the last position, argmax -> next token
               - append it, feed ONLY that one token back in (the cache holds
                 the rest -- this is why we don't re-run the whole sequence)
               - stop early on the EOS token
          4. detokenize the generated ids back to text

        The KV cache is the reason step 3 is cheap: without it, generating token
        N would re-process all N-1 previous tokens. With it, each step reads the
        cache (the kv_bytes term in the roofline) and computes only the new token.
        """
        #tokenize
        inputs = self.tokenizer(prompt, return_tensors = "pt")
        input_ids = inputs["input_ids"].to(DEVICE)
        generated_ids = []
        
        #prefill
        outputs = self.model(input_ids = input_ids, use_cache = True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        #decode
        for _ in range(max_tokens):
            next_token_id = torch.argmax(next_token_logits, dim = -1, keepdim=True)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            generated_ids.append(next_token_id.item())

            outputs = self.model(input_ids=next_token_id, past_key_values= past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        #detokenize original + generated ones from decode
        full_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=DEVICE)], dim=-1)
        return self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

    def _generate_static_batch(self, prompts: list[str], max_tokens: int) -> list[str]:
        """
        Static batching: pad every prompt to the longest, decode all of them in
        lockstep, and wait for the slowest to hit max_tokens (or EOS).

        "Static" = the batch is fixed for the whole run. A sequence that finishes
        early keeps occupying its slot, wasting compute -- that inefficiency is
        deliberate. It's the baseline that Step 7's continuous batching (admit and
        retire per step) has to BEAT. Left-pad so every sequence's real last token
        sits at the same position, which keeps the decode step aligned.
        """

        # tokenize all prompts
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(DEVICE) # (B, T)

        attention_mask = inputs["attention_mask"].to(DEVICE)
        prefill_mask = attention_mask

        B = input_ids.shape[0]
        eos_token_id = self.tokenizer.eos_token_id

        # diff position id after padding
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        # prefill
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids, use_cache = True)
        # (B, T, V)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        generated_ids = [[] for _ in range(B)]

        #(B, true/false)
        finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        
        #decode
        for _ in range(max_tokens):
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            for i in range(B):

                if finished[i]:
                    continue

                if next_token_ids[i].item() == self.tokenizer.eos_token_id:
                    finished[i] = True
                else:
                    generated_ids[i].append(next_token_ids[i].item())

            if finished.all():
                break
            
            ones = torch.ones((B, 1), dtype=attention_mask.dtype, device=DEVICE)
            attention_mask = torch.cat([attention_mask, ones], dim=-1)
            #(B, 1)
            next_position_ids = attention_mask.sum(-1, keepdim=True) - 1

            outputs = self.model(input_ids = next_token_ids, attention_mask = attention_mask, position_ids = next_position_ids, past_key_values = past_key_values, use_cache = True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
        results = []
        
        for i in range(B):
            real_ids = input_ids[i][prefill_mask[i] == 1]
            new_ids = torch.tensor(generated_ids[i], dtype=real_ids.dtype, device=DEVICE)

            full_ids = torch.cat([real_ids, new_ids], dim=-1)

            results.append(self.tokenizer.decode(full_ids, skip_special_tokens=True))

        return results

if __name__ == "__main__":
    assert MODEL_ID is not None, "set MODEL_ID before running"

    runner = Runner()
    # smoke test: one prompt, a handful of tokens, just prove it emits text.
    one = runner.generate(["The capital of France is"], max_tokens=32)
    two = runner.generate(["The capital of France is", "The capital of France is"], max_tokens=32)
    print(out[0])

    assert one[0] == two[1]
