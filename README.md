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

- `-bulk [FILE]`
   Bulk mode: match every scanned key against millions of funded addresses using a blocked bloom filter. Downloads the default address list automatically if no `FILE` is given.

- `-types <list>`
   Selects which address types bulk mode computes and matches. Letters in any order/separator:
   - `c` — compressed P2PKH + P2WPKH (bc1q). **This is the default and covers the puzzle and the vast majority of funded addresses.**
   - `u` — uncompressed P2PKH
   - `s` — P2SH-P2WPKH (3xxx)
   - `all` — shorthand for `cus`

   *Example:* `-bulk -types cs`

   Fewer types is faster: compressed-only computes one hash160 per key instead of three (roughly **3–4× fewer hashes**), and it also keeps disabled address types out of the bloom filter, so the false-positive rate stays low.


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

**v2.4:**
- **selectable bulk address types (`-types`), compressed-only by default:** the bulk kernel used to hash every key three ways (compressed, P2SH-P2WPKH, uncompressed = 4 SHA-256 blocks + 3 RIPEMD-160) even though the puzzle and almost all funded addresses are compressed P2PKH. It now computes only the enabled types, so the default compressed-only run does a single hash160 per key (~3–4× faster) and produces ~3× fewer bloom candidates. The P2SH path reuses the already-computed compressed hash160 as its redeem input, so enabling it adds only one SHA-256 block + one RIPEMD-160.
- **lower false-positive rate:** (a) fewer probes per key means fewer false-positive candidates; (b) P2TR (`bc1p`) addresses are no longer inserted — a taproot 32-byte key has no hash160, and the old 20-byte "proxy" was pure noise that only raised the FPR; (c) address types that no enabled search can match (e.g. `3xxx` when P2SH is off) are skipped at load time so they add no bits to the filter.
- **fixed an early-return bug that could mask a genuine multi-type hit:** the device checker used to stop at the first bloom-positive, so a false positive on one type could hide a real match on another. Each enabled type is now probed and reported independently, and the CPU still verifies every candidate exactly.
- normal (single-target) mode is unchanged: its GPU kernel (batch inversion, word-domain SHA/RIPEMD, early exit) is already near-optimal and its proven math was left intact.
- version banners now report v2.4.

**v2.3:**
- **bulk mode is much faster:** the bloom filter is now a *blocked* (cache-line) bloom filter, so each GPU probe touches a single 64-byte cache line instead of ~20–30 scattered global-memory loads per address type. Host (`bloomfilter.cpp`) and device (`gpu_worker.cu`) hashing are kept bit-for-bit identical.
- **fixed a keyspace-skip bug in bulk mode:** GPU threads used to bail out early once the per-dispatch candidate buffer filled, yet the host still counted their entire span as searched — silently skipping keys. Threads now always finish their assigned span, and the candidate buffer was enlarged so a coarse pre-filter cannot overflow it.
- per-dispatch reset now clears only the candidate counter instead of re-zeroing the whole result buffer every launch.
- bloom hash count is capped for the blocked layout; the CPU-side exact verification still removes every false positive.
- version banners now report the correct version.

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
