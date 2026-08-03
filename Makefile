CXX = g++
NVCC = nvcc

CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp
LDFLAGS = -lpthread

TARGET = puzzle71

# ============= GPU Auto-Detection =============
# Detect GPU compute capability automatically via nvidia-smi
# Falls back to multi-arch build if detection fails

# Try to detect GPU arch automatically
GPU_DETECT := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')

# Map compute capability to sm_ arch
ifeq ($(GPU_DETECT),61)
  # Pascal: GTX 1050/1060/1070/1080
  AUTO_ARCH := sm_61
  GPU_NAME := Pascal (GTX 10xx)
else ifeq ($(GPU_DETECT),70)
  # Volta: V100, Titan V
  AUTO_ARCH := sm_70
  GPU_NAME := Volta (V100)
else ifeq ($(GPU_DETECT),75)
  # Turing: RTX 2060/2070/2080, T4, GTX 1660
  AUTO_ARCH := sm_75
  GPU_NAME := Turing (RTX 20xx/T4/GTX 16xx)
else ifeq ($(GPU_DETECT),80)
  # Ampere: A100, RTX 3090 (GA102 is 86, A100 is 80)
  AUTO_ARCH := sm_80
  GPU_NAME := Ampere (A100)
else ifeq ($(GPU_DETECT),86)
  # Ampere: RTX 3060/3070/3080/3090
  AUTO_ARCH := sm_86
  GPU_NAME := Ampere (RTX 30xx)
else ifeq ($(GPU_DETECT),87)
  # Ampere: Jetson Orin
  AUTO_ARCH := sm_87
  GPU_NAME := Ampere (Jetson Orin)
else ifeq ($(GPU_DETECT),89)
  # Ada Lovelace: RTX 4060/4070/4080/4090
  AUTO_ARCH := sm_89
  GPU_NAME := Ada Lovelace (RTX 40xx)
else ifeq ($(GPU_DETECT),90)
  # Hopper: H100
  AUTO_ARCH := sm_90
  GPU_NAME := Hopper (H100)
else ifeq ($(GPU_DETECT),90a)
  AUTO_ARCH := sm_90a
  GPU_NAME := Hopper (H100 SXM)
else ifeq ($(GPU_DETECT),100)
  # Blackwell: B100/B200
  AUTO_ARCH := sm_100
  GPU_NAME := Blackwell (B100/B200)
else
  # No GPU detected or unknown — build multi-arch for broad compatibility
  AUTO_ARCH := all-major
  GPU_NAME := Multi-arch (auto-detect failed)
endif

# Allow manual override: make gpu CUDA_ARCH=sm_86
CUDA_ARCH ?= $(AUTO_ARCH)

# Build gencode flags based on target
ifeq ($(CUDA_ARCH),all-major)
  # Multi-arch: covers Pascal through Ada Lovelace
  GENCODE_FLAGS := \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_89,code=compute_89
  ARCH_FLAGS := $(GENCODE_FLAGS)
else
  # Single target — fastest compilation, best optimization
  ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

NVCCFLAGS = -O3 -std=c++17 -Xcompiler -fopenmp $(ARCH_FLAGS) -DUSE_CUDA \
            --expt-relaxed-constexpr -lineinfo --use_fast_math \
            --maxrregcount=128 -Xptxas=-v

# ============= Build Targets =============

all: cpu

cpu: puzzle71_hunt.cpp
	@echo "[*] Building CPU version..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $^ $(LDFLAGS)
	@echo "[✓] CPU build complete: ./$(TARGET)"

gpu: puzzle71_hunt.cpp
	@echo "╔════════════════════════════════════════════╗"
	@echo "║  GPU Build - Auto-Detection Results        ║"
	@echo "╠════════════════════════════════════════════╣"
	@echo "║  Detected CC:  $(GPU_DETECT)"
	@echo "║  GPU Family:   $(GPU_NAME)"
	@echo "║  CUDA Arch:    $(CUDA_ARCH)"
	@echo "╚════════════════════════════════════════════╝"
	@echo ""
	cp $< puzzle71_hunt.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) puzzle71_hunt.cu $(LDFLAGS)
	rm -f puzzle71_hunt.cu
	@echo ""
	@echo "[✓] GPU build complete: ./$(TARGET)"
	@echo "[*] Run with: ./$(TARGET) --blocks 0 --tpb 0   (auto-tune)"

# Explicit arch targets for convenience
gpu-pascal: CUDA_ARCH = sm_61
gpu-pascal: gpu

gpu-turing: CUDA_ARCH = sm_75
gpu-turing: gpu

gpu-ampere: CUDA_ARCH = sm_86
gpu-ampere: gpu

gpu-ada: CUDA_ARCH = sm_89
gpu-ada: gpu

gpu-hopper: CUDA_ARCH = sm_90
gpu-hopper: gpu

gpu-all: CUDA_ARCH = all-major
gpu-all: gpu

info:
	@echo "=== GPU Detection ==="
	@echo "Compute Capability: $(GPU_DETECT)"
	@echo "Detected Family:    $(GPU_NAME)"
	@echo "CUDA Arch:          $(CUDA_ARCH)"
	@echo ""
	@echo "=== Available Targets ==="
	@echo "  make cpu        - Build CPU-only version"
	@echo "  make gpu        - Build GPU version (auto-detect arch)"
	@echo "  make gpu-pascal  - Force Pascal  (GTX 10xx)"
	@echo "  make gpu-turing  - Force Turing  (RTX 20xx, T4, GTX 16xx)"
	@echo "  make gpu-ampere  - Force Ampere  (RTX 30xx)"
	@echo "  make gpu-ada     - Force Ada     (RTX 40xx)"
	@echo "  make gpu-hopper  - Force Hopper  (H100)"
	@echo "  make gpu-all     - Multi-arch    (all GPUs, slower compile)"
	@echo ""
	@echo "  Override: make gpu CUDA_ARCH=sm_86"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not found)"

clean:
	rm -f $(TARGET) puzzle71_hunt.cu

.PHONY: all cpu gpu gpu-pascal gpu-turing gpu-ampere gpu-ada gpu-hopper gpu-all info clean
