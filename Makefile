CXX = g++
NVCC = nvcc
HAS_NVCC := $(shell command -v $(NVCC) 2> /dev/null)

CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp -fPIC
LDFLAGS = -lpthread

TARGET = puzzlesolver

# CPU and GPU builds compile the same sources with different -D flags, so they
# must use separate object files. Sharing them would silently link stale objects
# when switching between `make cpu` and `make gpu` without `make clean`.
OBJS_CPU = main.cpu.o cli.cpu.o base58.cpu.o cpu_worker.cpu.o
OBJS_GPU = main.gpu.o cli.gpu.o base58.gpu.o gpu_worker.gpu.o

# ── GPU architecture detection ─────────────────────────────────────────────
# Build for the cards actually installed; fall back to a portable fat binary.
INSTALLED_GPUS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//' | sort -u)

ifneq ($(INSTALLED_GPUS),)
    NATIVE_MAX := $(lastword $(sort $(INSTALLED_GPUS)))
    GENCODE_FLAGS := $(foreach a,$(INSTALLED_GPUS),-gencode arch=compute_$(a),code=sm_$(a)) \
                     -gencode arch=compute_$(NATIVE_MAX),code=compute_$(NATIVE_MAX)
else
    NVCC_ALL_ARCHS := $(shell $(NVCC) --list-gpu-code 2>/dev/null | grep -oE 'sm_[0-9]+' | sed 's/sm_//' | sort -n -u)
    ifeq ($(NVCC_ALL_ARCHS),)
        NVCC_ALL_ARCHS := $(shell $(NVCC) --help 2>/dev/null | grep -oE 'sm_[0-9]+' | sed 's/sm_//' | sort -n -u)
    endif
    ifeq ($(NVCC_ALL_ARCHS),)
        # Maxwell through Blackwell.
        NVCC_ALL_ARCHS := 50 52 60 61 70 75 80 86 87 89 90 100 120
    endif
    FULL_MAX := $(lastword $(NVCC_ALL_ARCHS))
    GENCODE_FLAGS := $(foreach a,$(NVCC_ALL_ARCHS),-gencode arch=compute_$(a),code=sm_$(a)) \
                     -gencode arch=compute_$(FULL_MAX),code=compute_$(FULL_MAX)
endif

NVCCFLAGS = -O3 -std=c++17 -Xcompiler -fopenmp -DUSE_CUDA \
            --expt-relaxed-constexpr -lineinfo --use_fast_math \
            -Wno-deprecated-gpu-targets \
            $(GENCODE_FLAGS)

all:
	@echo "Please specify a target:"
	@echo "  make cpu    - Build CPU only version"
	@echo "  make gpu    - Build GPU version (auto-detects your card)"
	@echo "  make clean  - Remove build files"
	@echo ""
	@if [ -n "$(INSTALLED_GPUS)" ]; then \
	    echo "  Detected GPU(s): sm_$(INSTALLED_GPUS) → fast native build"; \
	else \
	    echo "  No GPU detected → will build portable fat binary for all architectures"; \
	fi

cpu: $(OBJS_CPU)
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -o $(TARGET) $(OBJS_CPU) $(LDFLAGS)

gpu: LDFLAGS += -L/usr/local/cuda/lib64 -lcudart
gpu: $(OBJS_GPU)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJS_GPU) $(LDFLAGS)

# ── CPU objects (compiled with -DCPU_ONLY) ────────────────────────────────
main.cpu.o: main.cpp cli.h base58.h int256.h cpu_worker.h
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -c main.cpp -o $@

cli.cpu.o: cli.cpp cli.h base58.h
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -c cli.cpp -o $@

base58.cpu.o: base58.cpp base58.h sha256_rmd160.h
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -c base58.cpp -o $@

cpu_worker.cpu.o: cpu_worker.cpp cpu_worker.h secp256k1.h sha256_rmd160.h int256.h
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -c cpu_worker.cpp -o $@

# ── GPU objects (compiled with -DUSE_CUDA) ────────────────────────────────
main.gpu.o: main.cpp cli.h base58.h int256.h cpu_worker.h
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -c main.cpp -o $@

cli.gpu.o: cli.cpp cli.h base58.h
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -c cli.cpp -o $@

base58.gpu.o: base58.cpp base58.h sha256_rmd160.h
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -c base58.cpp -o $@

gpu_worker.gpu.o: gpu_worker.cu gpu_worker.cuh secp256k1.h sha256_rmd160.h int256.h
	$(NVCC) $(NVCCFLAGS) --ptxas-options=-v -c gpu_worker.cu -o $@

clean:
	rm -f *.o $(TARGET)
