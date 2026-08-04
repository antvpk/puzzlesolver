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

**Example Command:**
```bash
./puzzlesolver --range 400000000000000000:7fffffffffffffffff -target f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8
```

---

## ⚙️ Command Line Arguments

The tool provides extensive options to tune the search to your exact hardware and target requirements:

- `-r` or `--range <min:max>`
  **Description:** Defines the specific hex key range to scan. 
  *Example:* `--range 400000000000000000:7fffffffffffffffff`

- `-target <hash>`
  **Description:** Sets a custom target RIPEMD160 hash to hunt for. Must be a 40-character hex string.
  *Example:* `-target f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8`

- `-s` or `--sequential`
  **Description:** Runs the search sequentially starting from the minimum range. By default, the tool searches randomly.

- `-t` or `--threads <num>`
  **Description:** Sets the number of CPU threads to use. (Only applicable when compiled/running in CPU mode).

- `-b` or `--batch <num>`
  **Description:** Sets the batch size per thread. (Default is 4096 for CPU, 1024 for GPU).

- `--blocks <num>`
  **Description:** Sets the CUDA block count manually. Leave as `0` for auto-tuning (GPU mode only).

- `--tpb <num>`
  **Description:** Sets the Threads Per Block manually. Leave as `0` for auto-tuning (GPU mode only).

---


---

## 🚀 100% NVIDIA GPU Support

The included dynamic `Makefile` communicates directly with your CUDA toolkit (`nvcc`) to perfectly optimize the executable for your hardware. It natively supports **every CUDA-capable GPU ever created**, ranging from legacy architectures all the way up to the latest bleeding-edge processors.

**Supported Architectures Include:**
- **Blackwell / Rubin (sm_100, sm_120):** RTX 50-series (5060, 5070, 5080, 5090), B100, B200
- **Hopper (sm_90):** H100, H200
- **Ada Lovelace (sm_89):** RTX 40-series
- **Ampere (sm_80, sm_86):** RTX 30-series, A100
- **Turing (sm_75):** RTX 20-series, GTX 16-series
- **Volta (sm_70):** Titan V, V100
- **Pascal (sm_61):** GTX 10-series
- **Maxwell (sm_50, sm_52):** GTX 900-series
- **Kepler (sm_30, sm_35):** GTX 700-series

---

## 🎉 Success Output
When a matching key is found, the application halts immediately and logs the result to both the terminal and a `RESULT.txt` file, providing:
- The private key in standard hex format
- The Wallet Import Format (WIF) compressed key
- The associated legacy compressed Bitcoin address
