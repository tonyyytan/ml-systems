NVCC     := nvcc
CXXFLAGS := -O3 -std=c++17 -diag-suppress 177
ARCH     := -arch=sm_120   # Blackwell / RTX 5060 laptop (GB206)
LIBS     := -lcublas

TARGET   := roofline
SRC      := roofline.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(ARCH) $(LIBS) -o $@ $<

clean:
	rm -f $(TARGET) roofline.csv
