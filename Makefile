CXX = g++
NVCC = nvcc

CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp
LDFLAGS = -lpthread

TARGET = puzzlesolver

# ============= GPU Auto-Detection =============
# Detect GPU compute capability automatically via nvidia-smi
# Falls back to multi-arch build if detection fails

# Try to detect GPU arch automatically
GPU_DETECT := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')

ifneq ($(GPU_DETECT),)
  AUTO_ARCH := sm_$(GPU_DETECT)
  GPU_NAME := NVIDIA GPU (CC $(GPU_DETECT))
  ifeq ($(GPU_DETECT),61)
    GPU_NAME := Pascal (GTX 10xx)
  else ifeq ($(GPU_DETECT),70)
    GPU_NAME := Volta (V100)
  else ifeq ($(GPU_DETECT),75)
    GPU_NAME := Turing (RTX 20xx/GTX 16xx)
  else ifeq ($(GPU_DETECT),80)
    GPU_NAME := Ampere (A100)
  else ifeq ($(GPU_DETECT),86)
    GPU_NAME := Ampere (RTX 30xx)
  else ifeq ($(GPU_DETECT),89)
    GPU_NAME := Ada Lovelace (RTX 40xx)
  else ifeq ($(GPU_DETECT),90)
    GPU_NAME := Hopper (H100)
  else ifeq ($(GPU_DETECT),100)
    GPU_NAME := Blackwell (B100/B200)
  else ifeq ($(GPU_DETECT),120)
    GPU_NAME := Blackwell (RTX 5060/5070/5080/5090)
  else ifeq ($(GPU_DETECT),120a)
    GPU_NAME := Blackwell (RTX 5060/5070/5080/5090)
  endif
else
  # No GPU detected or unknown — build multi-arch for broad compatibility
  AUTO_ARCH := all-major
  GPU_NAME := Multi-arch (auto-detect failed)
endif

# Allow manual override: make gpu CUDA_ARCH=sm_86
CUDA_ARCH ?= $(AUTO_ARCH)

# Query supported architectures from nvcc to prevent compilation errors on older/newer toolkits
NVCC_SUPPORTED := $(shell nvcc --help 2>/dev/null | grep -o 'sm_[0-9][0-9a-z]*' | sed 's/sm_//' | sort -u)
DEFAULT_ARCHS := 50 52 60 61 70 75 80 86 89 90 100 120
ALL_KNOWN_ARCHS := 30 35 37 50 52 53 60 61 62 70 72 75 80 86 87 89 90 90a 100 100a 120 120a

ifeq ($(strip $(NVCC_SUPPORTED)),)
  VALID_ARCHS := $(DEFAULT_ARCHS)
else
  VALID_ARCHS := $(filter $(ALL_KNOWN_ARCHS), $(NVCC_SUPPORTED))
endif

ifeq ($(CUDA_ARCH),all-major)
  # Multi-arch: 100% all NVIDIA GPUs supported by your installed nvcc compiler
  GENCODE_FLAGS := $(foreach arch,$(VALID_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
  HIGHEST_ARCH := $(lastword $(VALID_ARCHS))
  ifneq ($(HIGHEST_ARCH),)
    GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_ARCH),code=compute_$(HIGHEST_ARCH)
  endif
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

cpu: puzzlesolver.cpp
	@echo "[*] Building CPU version..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $^ $(LDFLAGS)
	@echo "[✓] CPU build complete: ./$(TARGET)"

gpu: puzzlesolver.cpp
	@echo "╔════════════════════════════════════════════╗"
	@echo "║  GPU Build - Auto-Detection Results        ║"
	@echo "╠════════════════════════════════════════════╣"
	@echo "║  Detected CC:  $(GPU_DETECT)"
	@echo "║  GPU Family:   $(GPU_NAME)"
	@echo "║  CUDA Arch:    $(CUDA_ARCH)"
	@echo "╚════════════════════════════════════════════╝"
	@echo ""
	cp $< puzzlesolver.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) puzzlesolver.cu $(LDFLAGS)
	rm -f puzzlesolver.cu
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
	@echo "  make gpu-turing  - Force Turing  (RTX 20xx, GTX 16xx)"
	@echo "  make gpu-ampere  - Force Ampere  (RTX 30xx)"
	@echo "  make gpu-ada     - Force Ada     (RTX 40xx)"
	@echo "  make gpu-hopper  - Force Hopper  (H100)"
	@echo "  make gpu-all     - Multi-arch    (all GPUs, slower compile)"
	@echo ""
	@echo "  Override: make gpu CUDA_ARCH=sm_86"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not found)"

clean:
	rm -f $(TARGET) puzzlesolver.cu

.PHONY: all cpu gpu gpu-pascal gpu-turing gpu-ampere gpu-ada gpu-hopper gpu-all info clean
