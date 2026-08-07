# 🔑 Bitcoin Puzzle Solver

An ultra high-performance, production-ready Bitcoin Puzzle Key Hunter designed to search for specific RIPEMD160 hashes (such as the infamous Puzzle 71). Based on original elliptic curve mathematics by Jean Luc Pons.

---

## 🛠️ How to Compile & Use

### 1. Building the Tool
Navigate to the project directory and compile using `make`.

**For CPU Only:**
```bash
make cpu
```

**For GPU (Auto-Detects and optimizes for your installed GPU):**
```bash
make gpu
```
*Note: `make gpu` automatically compiles a highly-optimized binary supporting all architectures your installed CUDA toolkit allows, ensuring maximum performance.*

### 2. Running the Solver
Once compiled, you can launch the solver.

**For default puzzle target:**
```bash
./puzzlesolver
```

**For custom range and target:**
```bash
./puzzlesolver -r 400000000000000000:7fffffffffffffffff -target f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8
```

---

## ⚙️ Command Line Arguments

The tool provides extensive options to tune the search to your exact hardware and target requirements:

- `-r`  `<min:max>`
   Defines the specific hex key range to scan. 
  *Example:* `-r 400000000000000000:7fffffffffffffffff`

- `-target <hash>`
   Sets a custom target RIPEMD160 hash to hunt for. Must be a 40-character hex string or a Base58 address starting with `1`.
  
  *Example:* `-target f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8`

- `-s`
   Runs the search sequentially starting from the minimum range. By default, the tool searches randomly.

- `-t <n>`
   Sets the number of CPU threads to use. (Only applicable when compiled/running in CPU mode).

- `-b <n>`
   Sets the batch size per thread. (Default is 4096 for CPU). GPU batch is fixed at compile time via `GPU_BATCH`.

- `-gpu <devices>`
   Selects which GPUs to use by device index.
  *Example:* `-gpu 035` uses GPUs 0, 3, and 5. Default: all GPUs.

- `--blocks <n>`
   Overrides the auto-tuned block count for the GPU kernel launch.

- `--tpb <n>`
   Sets threads per block. Must match the `THREADS_PER_BLOCK` compile-time setting (default 256).


---

---

## 🚀 NVIDIA GPU Support

The included dynamic `Makefile` communicates directly with your CUDA toolkit (`nvcc`) to perfectly optimize the executable for your hardware. It natively supports **every CUDA-capable GPU ever created**, ranging from legacy architectures all the way up to the latest bleeding-edge processors.

**Supported Architectures Include:**
- **Blackwell / Rubin (sm_100, sm_120):** RTX 50-series (5060, 5070, 5080, 5090), B100, B200
- **Hopper (sm_90):** H100, H200
- **Ada Lovelace (sm_89):** RTX 40-series
- **Ampere (sm_80, sm_86):** RTX 30-series, A100
- **Turing (sm_75):** RTX 20-series, GTX 16-series, T4
- **Volta (sm_70):** Titan V, V100
- **Pascal (sm_61):** GTX 10-series
- **Maxwell (sm_50, sm_52):** GTX 900-series
---

## 🎉 Success Output
When a matching key is found, the application halts immediately and logs the result to both the terminal and a `RESULT.txt` file, providing:
- The private key in standard hex format
- The Wallet Import Format (WIF) compressed key
- The associated legacy compressed Bitcoin address

---

## 📝 Changelog

**v2.2:**
- word-domain SHA-256 and RIPEMD-160 (no byte round-trips in the inner loop)
- splitmix64 random number generator for wider random coverage
- async dispatch pipeline with pinned memory for sustained GPU throughput
- volatile found-flag for immediate early exit on match
- real searched-key display
- optional windowed-comb scalar multiplication (`-DUSE_COMB=1`)

**v2.1:**
- fixed sequential boundary bug (tool now properly stops at `RANGE_MAX`).

**v2.0:**
- initial release.
