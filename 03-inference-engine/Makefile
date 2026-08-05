NVCC     := nvcc
CXXFLAGS := -O3 -std=c++17 -diag-suppress 177
ARCH     := -arch=sm_120   # Blackwell / RTX 5060 laptop (GB206)
LIBS     := -lcublas

TARGET   := bench_gemv
SRC      := engine/kernels/bench_gemv.cu

.PHONY: all ext run clean

all: $(TARGET)

$(TARGET): $(SRC) engine/kernels/gemv.cu
	$(NVCC) $(CXXFLAGS) $(ARCH) $(LIBS) -o $@ $<

# torch extension (gemv_cuda), separate from the standalone benchmark
ext:
	python3 setup.py build_ext --inplace

run: all
	./$(TARGET) > gemv_results.csv

clean:
	rm -f $(TARGET) gemv_results.csv
	rm -rf build gemv_cuda*.so
