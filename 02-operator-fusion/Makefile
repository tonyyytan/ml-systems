NVCC     := nvcc
CXXFLAGS := -O3 -std=c++17 -diag-suppress 177
ARCH     := -arch=sm_120   # Blackwell / RTX 5060 laptop (GB206)

TARGET   := kernels
SRC      := kernels.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(ARCH) -o $@ $<

run: all
	./$(TARGET) > results.csv
	python3 plot_fusion.py results.csv

clean:
	rm -f $(TARGET) results.csv
