#include "gpu_worker.cuh"
#include "secp256k1.h"
#include "sha256_rmd160.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <mutex>
#include <random>
#include <vector>
#include "util.h"

extern std::mutex g_display_mutex;
extern uint256_t g_display_key;
extern bool g_display_valid;

// Report a CUDA failure with file:line instead of failing silently.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%s)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_err), #call);     \
        }                                                                     \
    } while (0)

// Same, but aborts the worker on failure (for fatal setup).
#define CUDA_CHECK_RET(call)                                                  \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%s)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_err), #call);     \
            return;                                                           \
        }                                                                     \
    } while (0)

// ═══════════════════════════════════════════════════════════════════════
//  Compile-time tuning
// ═══════════════════════════════════════════════════════════════════════

// GPU_BATCH — affine batch-inversion window. Sizes the per-thread array
// z_prod[GPU_BATCH] (32 B each), so 256 → 8 KB of local memory per thread.
// Per-point cost ≈ 4 + 270/GPU_BATCH field muls, so larger amortises the single
// modular inversion better but costs occupancy. 256 suits Turing through
// Blackwell; drop to 128 on cards with tight local-memory limits.
//   make gpu NVCCFLAGS+="-DGPU_BATCH=128"
#ifndef GPU_BATCH
#define GPU_BATCH 256
#endif

// THREADS_PER_BLOCK — must match the __launch_bounds__ below, so it is a
// compile-time constant. The -tpb runtime flag is validated against it.
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

// KEYS_PER_THREAD — keys each thread walks from its base per dispatch. One
// scalar multiplication sets up the base; subsequent keys are near-free. Larger
// = more throughput but fewer distinct random starting points per dispatch.
// Smaller = wider random scatter. Must be a multiple of GPU_BATCH.
//   4*GPU_BATCH = 1024 : throughput-oriented (default)
//   2*GPU_BATCH =  512 : 2x the random bases, a few % slower
//   1*GPU_BATCH =  256 : 4x the random bases
//   make gpu NVCCFLAGS+="-DKEYS_PER_THREAD=512"
#ifndef KEYS_PER_THREAD
#define KEYS_PER_THREAD (4 * GPU_BATCH)
#endif

// LB_MIN_BLOCKS_PER_SM — min_blocks_per_sm for __launch_bounds__, and the
// blocks-per-SM target for grid autotuning. Raising it forces ptxas to use
// fewer registers so more blocks co-reside; if the kernel needs more registers
// than that allows, the excess spills to local memory and throughput drops.
// The build prints the real register count (--ptxas-options=-v) — check it
// before raising this.
//   make gpu NVCCFLAGS+="-DLB_MIN_BLOCKS_PER_SM=3"
#ifndef LB_MIN_BLOCKS_PER_SM
#define LB_MIN_BLOCKS_PER_SM 2
#endif

// Per-dispatch result block, kept in device memory: every thread polls `found`
// at each batch boundary, and polling host memory over PCIe would be far slower.
struct DeviceResult {
    int       found;
    int       _pad;
    uint256_t found_key;
    uint256_t sample_key;   // a key actually searched this dispatch (thread 0's base)
};

// ═══════════════════════════════════════════════════════════════════════
//  Device-side globals
// ═══════════════════════════════════════════════════════════════════════
__device__ __constant__ uint32_t d_TARGET_H0;
// Target hash as 5 little-endian words, matching RIPEMD-160's native output.
// d_TARGET_W[0] is the same value as d_TARGET_H0.
__device__ __constant__ uint32_t d_TARGET_W[5];
__device__ __constant__ point_t  d_G_POINT;
// G_TABLE[i] = i·G, affine. Read from global memory, which is L1/L2-cached; the
// table is small and hot, so accesses hit cache after warm-up.
__device__ point_t d_G_TABLE[GPU_BATCH + 1];

// ═══════════════════════════════════════════════════════════════════════
//  Windowed scalar multiplication (optional)
// ═══════════════════════════════════════════════════════════════════════
//
// Every key is range_min + r with r bounded by the range width, so
//   key·G = (range_min·G) + r·G
// range_min·G is constant for the run (d_BASE_POINT); only r·G is per-thread.
// r·G uses a fixed-window comb over precomputed multiples, one table lookup and
// one add per window, no doublings.
//
// This trades ~3584 field muls for ~490 plus a per-lane table gather. On Ampere
// the gather costs more than it saves, because the generic path reads a single
// broadcast address while the comb reads 32 different addresses per warp.
// Disabled by default; benchmark before enabling.
//   make gpu NVCCFLAGS+="-DUSE_COMB=1"
#ifndef USE_COMB
#define USE_COMB 0
#endif

#ifndef WIN_BITS
#define WIN_BITS 4
#endif
#define WIN_SIZE    (1 << WIN_BITS)
#define MAX_WINDOWS ((256 + WIN_BITS - 1) / WIN_BITS)

#if USE_COMB
// 64 bytes exactly, 16-byte aligned: 2 cache sectors per gather and 4 clean
// 16-byte loads. point_t would be 72 bytes and unaligned (3 sectors). The
// infinity flag is unnecessary because index 0 is never read.
struct __align__(16) comb_entry_t {
    uint256_t x;
    uint256_t y;
};

__device__ point_t      d_BASE_POINT;
__device__ int          d_NUM_WINDOWS;
__device__ comb_entry_t d_WIN_TABLE[MAX_WINDOWS * WIN_SIZE];

// offset·G + base, affine.
//
// The trailing fp_inv cannot be folded into the caller's batch inversion:
// dx_i = base.x - G[i].x depends on base.x, which depends on this Z inverse, so
// the batch product cannot be formed until this inversion has completed.
__device__ __forceinline__ point_t
ec_mul_windowed(const uint256_t &offset, const point_t &base) {
    jpoint_t acc = affine_to_jacobian(base);
    const int nw = d_NUM_WINDOWS;

    for (int w = 0; w < nw; w++) {
        int bit  = w * WIN_BITS;
        int limb = bit >> 6;
        int sh   = bit & 63;

        uint32_t v = (uint32_t)((offset.d[limb] >> sh) & (WIN_SIZE - 1));
        // A window can straddle a limb boundary when 64 % WIN_BITS != 0.
        if (sh + WIN_BITS > 64 && limb < 3)
            v |= (uint32_t)((offset.d[limb + 1] << (64 - sh)) & (WIN_SIZE - 1));

        if (v == 0) continue;

        const comb_entry_t &e = d_WIN_TABLE[w * WIN_SIZE + v];
        point_t q; q.x = e.x; q.y = e.y; q.infinity = false;

        jpoint_t nxt;
        jpoint_add_affine(nxt, acc, q);
        acc = nxt;
    }
    return jacobian_to_affine(acc);
}
#endif // USE_COMB

// ═══════════════════════════════════════════════════════════════════════
//  Device helpers
// ═══════════════════════════════════════════════════════════════════════

// Pack a compressed pubkey (prefix ‖ big-endian x) straight into the 9 SHA-256
// message words, skipping the 33-byte intermediate buffer.
__device__ __forceinline__ void
pubkey_to_sha_words(const uint256_t &x, uint32_t prefix, uint32_t W[9]) {
    W[0] = (prefix << 24)                     | (uint32_t)((x.d[3] >> 40) & 0xFFFFFF);
    W[1] = (uint32_t)(x.d[3] >>  8);
    W[2] = (uint32_t)((x.d[3] & 0xFF) << 24)  | (uint32_t)((x.d[2] >> 40) & 0xFFFFFF);
    W[3] = (uint32_t)(x.d[2] >>  8);
    W[4] = (uint32_t)((x.d[2] & 0xFF) << 24)  | (uint32_t)((x.d[1] >> 40) & 0xFFFFFF);
    W[5] = (uint32_t)(x.d[1] >>  8);
    W[6] = (uint32_t)((x.d[1] & 0xFF) << 24)  | (uint32_t)((x.d[0] >> 40) & 0xFFFFFF);
    W[7] = (uint32_t)(x.d[0] >>  8);
    W[8] = (uint32_t)((x.d[0] & 0xFF) << 24)  | 0x00800000;
}

// Single-parity check: prefix determined by py's parity bit. Returns 1 on match.
// Stays in the word domain throughout — no byte buffers on this path.
__device__ __forceinline__ int
check_point_1way(const uint256_t &px, const uint256_t &py) {
    uint32_t W[9];
    pubkey_to_sha_words(px, (py.d[0] & 1) ? 0x03u : 0x02u, W);

    uint32_t sw[8];
    sha256_33_core(W, sw);

    uint32_t h[5];
    if (!hash160_from_sha_words(sw, h, d_TARGET_H0)) return 0;

    return (h[1] == d_TARGET_W[1] && h[2] == d_TARGET_W[2] &&
            h[3] == d_TARGET_W[3] && h[4] == d_TARGET_W[4]) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════════════════
//  Main search kernel
// ═══════════════════════════════════════════════════════════════════════
__global__
__launch_bounds__(THREADS_PER_BLOCK, LB_MIN_BLOCKS_PER_SM)
void search_kernel(uint256_t start_offset, uint64_t keys_per_thread,
                   DeviceResult *res,
                   bool is_sequential, uint256_t range_min, uint256_t diff,
                   int diff_bits, uint64_t seed) {

    // Tracked as offset = key - range_min; start_key is materialised for
    // reporting and for the generic scalar-multiply path.
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint256_t offset;
    uint256_t start_key;

    if (is_sequential) {
        offset = start_offset;
        u256_add64(offset, offset, tid * keys_per_thread);
    } else {
        // splitmix64: full 64-bit avalanche, so neighbouring tids and
        // neighbouring dispatch seeds land far apart rather than in clusters.
        uint64_t s = seed + tid * 0x9E3779B97F4A7C15ULL;
        const int limb        = diff_bits / 64;
        const int bit_in_limb = diff_bits % 64;
        uint64_t  mask        = (bit_in_limb == 63)
                                  ? 0xFFFFFFFFFFFFFFFFULL
                                  : ((1ULL << (bit_in_limb + 1)) - 1);
        while (true) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s += 0x9E3779B97F4A7C15ULL;
                uint64_t z = s;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                offset.d[i] = z ^ (z >> 31);
            }
            offset.d[limb] &= mask;
            for (int i = limb + 1; i < 4; i++) offset.d[i] = 0;
            if (u256_cmp(offset, diff) <= 0) break;
        }
    }

    u256_add(start_key, offset, range_min);

    // Report a genuinely-searched key so the live display reflects real work.
    if (tid == 0) res->sample_key = start_key;

    uint256_t base_key = start_key;

#if USE_COMB
    point_t base_point = ec_mul_windowed(offset, d_BASE_POINT);
#else
    point_t base_point = ec_mul(start_key, d_G_POINT);
#endif
    if (base_point.infinity) return;

    uint64_t checked = 0;

    // The only local-memory array: GPU_BATCH × 32 B (8 KB at GPU_BATCH=256).
    // dx_i and its zero-collision condition are recomputed in the backward pass
    // from a single fp_sub rather than stored, which would double this.
    uint256_t z_prod[GPU_BATCH];

    // Volatile: otherwise ptxas may hoist the load out of the loop, and a thread
    // already inside would never observe another thread's hit.
    volatile int *found_flag_v = &res->found;

    while (checked < keys_per_thread && !(*found_flag_v)) {
        int bc = GPU_BATCH;
        if ((uint64_t)(checked + bc) > keys_per_thread)
            bc = (int)(keys_per_thread - checked);

        // Forward pass: z_prod[i] = (base.x - G[1].x) × … × (base.x - G[i].x).
        // dx[i] == 0 when base.x == G_TABLE[i].x would zero the whole prefix
        // product and silently skip every key in the batch, so substitute 1 to
        // keep it invertible; those points are recomputed exactly below.
        for (int i = 1; i < bc; i++) {
            uint256_t dxi;
            fp_sub(dxi, base_point.x, d_G_TABLE[i].x);
            if (u256_is_zero(dxi)) { dxi.d[0]=1; dxi.d[1]=0; dxi.d[2]=0; dxi.d[3]=0; }
            if (i == 1) z_prod[1] = dxi;
            else        fp_mul(z_prod[i], z_prod[i-1], dxi);
        }

        // One modular inversion for the whole batch.
        uint256_t inv_prod;
        fp_inv(inv_prod, z_prod[bc - 1]);

        for (int i = bc - 1; i >= 1; i--) {
            uint256_t rx, ry;
            uint256_t dxi;
            fp_sub(dxi, base_point.x, d_G_TABLE[i].x);

            if (u256_is_zero(dxi)) {
                // Doubling or point-at-infinity: the affine slope formula is
                // undefined, so use a Jacobian add. inv_prod is deliberately not
                // updated — the forward pass folded a 1 in at this index.
                jpoint_t R;
                jpoint_add_affine(R, affine_to_jacobian(base_point), d_G_TABLE[i]);
                point_t rp = jacobian_to_affine(R);
                rx = rp.x; ry = rp.y;
            } else {
                uint256_t cur_inv;
                if (i > 1) {
                    fp_mul(cur_inv,  inv_prod, z_prod[i-1]);
                    fp_mul(inv_prod, inv_prod, dxi);
                } else {
                    cur_inv = inv_prod;
                }

                uint256_t dy, s, s2, tmp;
                fp_sub(dy,  base_point.y, d_G_TABLE[i].y);
                fp_mul(s,   dy, cur_inv);
                fp_sqr(s2,  s);
                fp_sub(tmp, s2, base_point.x);
                fp_sub(rx,  tmp, d_G_TABLE[i].x);
                fp_sub(tmp, base_point.x, rx);
                fp_mul(tmp, s, tmp);
                fp_sub(ry,  tmp, base_point.y);
            }

            if (check_point_1way(rx, ry)) {
                uint256_t k;
                u256_add64(k, base_key, (uint64_t)i);
                if (atomicExch(&res->found, 1) == 0) res->found_key = k;
                return;
            }
        }

        // The base point itself (i = 0).
        if (check_point_1way(base_point.x, base_point.y)) {
            if (atomicExch(&res->found, 1) == 0) res->found_key = base_key;
            return;
        }

        checked += (uint64_t)bc;

        // Advance base += GPU_BATCH·G with one Jacobian add instead of a full
        // scalar multiplication.
        if (bc == GPU_BATCH) {
            u256_add64(base_key, base_key, (uint64_t)GPU_BATCH);
            jpoint_t R;
            jpoint_add_affine(R, affine_to_jacobian(base_point), d_G_TABLE[GPU_BATCH]);
            base_point = jacobian_to_affine(R);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Host side
// ═══════════════════════════════════════════════════════════════════════
int get_cuda_device_count() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

std::vector<int> parse_gpu_devices(const std::string &s) {
    std::vector<int> devs;
    if (s.empty()) {
        int n = get_cuda_device_count();
        for (int i = 0; i < n; i++) devs.push_back(i);
        return devs;
    }
    for (char c : s)
        if (c >= '0' && c <= '9') devs.push_back(c - '0');
    return devs;
}

void init_gpu_worker() {}

void launch_gpu_worker(int device_id, uint256_t start_key, uint64_t keys_to_search,
                       std::atomic<uint64_t> &keys_checked, std::atomic<bool> &found_flag,
                       std::atomic<bool>     &running_flag,
                       uint256_t &found_key,  const uint8_t *target_hash, uint32_t target_h0,
                       int blocks, int tpb, uint64_t /*batch_ignored*/,
                       bool is_sequential, uint256_t range_min, uint256_t range_max) {
    // Spin rather than block while waiting on the GPU. Blocking-sync parks the
    // host thread in the driver and adds tens of microseconds of wake-up latency
    // between dispatches; with the shorter (more random) dispatches below, that
    // latency is now a measurable fraction of a dispatch. Costs one busy host
    // core per GPU, which this workload can spare.
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    CUDA_CHECK_RET(cudaSetDevice(device_id));

    // Build host-side G-table: G_TABLE[0]=∞, G_TABLE[i]=i·G
    point_t h_G_POINT = { SECP256K1_GX, SECP256K1_GY, false };
    // std::vector rather than new[]: the CUDA_CHECK_RET uploads below return
    // early on failure, which would leak a raw array.
    std::vector<point_t> g_table(GPU_BATCH + 1);
    point_t *h_G_TABLE = g_table.data();
    h_G_TABLE[0].infinity = true;
    {
        jpoint_t acc = affine_to_jacobian(h_G_POINT);
        h_G_TABLE[1] = h_G_POINT;
        for (int i = 2; i <= GPU_BATCH; i++) {
            jpoint_t nxt;
            jpoint_add_affine(nxt, acc, h_G_POINT);
            acc = nxt;
            h_G_TABLE[i] = jacobian_to_affine(acc);
        }
    }

    uint32_t h_target_w[5];
    for (int i = 0; i < 5; i++) {
        h_target_w[i] =  (uint32_t)target_hash[i*4]
                      | ((uint32_t)target_hash[i*4+1] <<  8)
                      | ((uint32_t)target_hash[i*4+2] << 16)
                      | ((uint32_t)target_hash[i*4+3] << 24);
    }

    CUDA_CHECK_RET(cudaMemcpyToSymbol(d_TARGET_H0, &target_h0, sizeof(uint32_t)));
    CUDA_CHECK_RET(cudaMemcpyToSymbol(d_TARGET_W, h_target_w, sizeof(h_target_w)));
    CUDA_CHECK_RET(cudaMemcpyToSymbol(d_G_POINT, &h_G_POINT, sizeof(point_t)));
    CUDA_CHECK_RET(cudaMemcpyToSymbol(d_G_TABLE, h_G_TABLE, sizeof(point_t) * (GPU_BATCH + 1)));

#if USE_COMB
    // Comb table over the offset scalar (key - range_min).
    uint256_t range_span;
    u256_sub(range_span, range_max, range_min);
    int span_bits = 255;
    while (span_bits >= 0 && u256_get_bit(range_span, span_bits) == 0) span_bits--;
    if (span_bits < 0) span_bits = 0;

    int scalar_bits = span_bits + 1;
    // Sequential mode can overshoot the range end within a dispatch by up to
    // total_threads × keys_per_thread. 24 bits covers any realistic geometry.
    // Random mode clamps offset ≤ diff in the RNG, so it needs no headroom.
    if (is_sequential) scalar_bits += 24;

    if (scalar_bits > 256) scalar_bits = 256;
    int num_windows  = (scalar_bits + WIN_BITS - 1) / WIN_BITS;
    if (num_windows < 1)            num_windows = 1;
    if (num_windows > MAX_WINDOWS)  num_windows = MAX_WINDOWS;

    point_t h_base_point;
    if (u256_is_zero(range_min)) {
        h_base_point.x = {0,0,0,0};
        h_base_point.y = {0,0,0,0};
        h_base_point.infinity = true;
    } else {
        h_base_point = ec_mul(range_min, h_G_POINT);
    }

    // win[w][v] = (v · 2^(WIN_BITS·w)) · G
    std::vector<comb_entry_t> win((size_t)num_windows * WIN_SIZE);
    {
        point_t step = h_G_POINT;
        for (int w = 0; w < num_windows; w++) {
            comb_entry_t *row = &win[(size_t)w * WIN_SIZE];
            row[0].x = {0,0,0,0};
            row[0].y = {0,0,0,0};
            row[1].x = step.x;
            row[1].y = step.y;

            jpoint_t acc = affine_to_jacobian(step);
            for (int v = 2; v < WIN_SIZE; v++) {
                jpoint_t nxt;
                jpoint_add_affine(nxt, acc, step);
                acc = nxt;
                point_t p = jacobian_to_affine(acc);
                row[v].x = p.x;
                row[v].y = p.y;
            }

            jpoint_t d = affine_to_jacobian(step);
            for (int i = 0; i < WIN_BITS; i++) { jpoint_t t; jpoint_double(t, d); d = t; }
            step = jacobian_to_affine(d);
        }
    }

    CUDA_CHECK_RET(cudaMemcpyToSymbol(d_BASE_POINT, &h_base_point, sizeof(point_t)));
    CUDA_CHECK_RET(cudaMemcpyToSymbol(d_NUM_WINDOWS, &num_windows, sizeof(int)));
    CUDA_CHECK_RET(cudaMemcpyToSymbol(d_WIN_TABLE, win.data(),
                                      sizeof(comb_entry_t) * win.size()));
#endif

    cudaDeviceProp prop;
    CUDA_CHECK_RET(cudaGetDeviceProperties(&prop, device_id));
    int cc = prop.major * 10 + prop.minor;
    (void)cc;

    // Reject a --tpb value that conflicts with the compile-time launch_bounds.
    if (tpb != 0 && tpb != THREADS_PER_BLOCK) {
        fprintf(stderr, "[GPU%d] --tpb %d rejected: kernel compiled for "
                "THREADS_PER_BLOCK=%d.  Rebuild with "
                "NVCCFLAGS+=\"-DTHREADS_PER_BLOCK=%d\".\n",
                device_id, tpb, THREADS_PER_BLOCK, tpb);
        return;
    }
    tpb = THREADS_PER_BLOCK;

    if (blocks == 0) {
        int bsm = LB_MIN_BLOCKS_PER_SM;
        blocks = prop.multiProcessorCount * bsm;
        int thr_per_sm = bsm * tpb;
        if (thr_per_sm > prop.maxThreadsPerMultiProcessor)
            blocks = (prop.maxThreadsPerMultiProcessor / tpb) * prop.multiProcessorCount;
    }

    uint64_t total_threads   = (uint64_t)blocks * tpb;
    uint64_t points_per_thread = KEYS_PER_THREAD;
    uint64_t points_per_launch = points_per_thread * total_threads;

    DeviceResult *d_res;
    CUDA_CHECK_RET(cudaMalloc(&d_res, sizeof(DeviceResult)));
    DeviceResult h_res_zero = {0};
    CUDA_CHECK(cudaMemcpy(d_res, &h_res_zero, sizeof(DeviceResult), cudaMemcpyHostToDevice));

    printf("[GPU%d] %s (sm_%d) | SMs=%d | %d blocks × %d tpb = %llu threads | "
           "GPU_BATCH=%d | kpt=%llu | bases/dispatch=%llu\n",
           device_id, prop.name, cc,
           prop.multiProcessorCount, blocks, tpb,
           (unsigned long long)total_threads,
           GPU_BATCH, (unsigned long long)points_per_thread,
           (unsigned long long)total_threads);
#if USE_COMB
    printf("[GPU%d] comb: %d windows × %d entries (%d-bit) = %.1f KB\n",
           device_id, num_windows, WIN_SIZE, WIN_BITS,
           sizeof(comb_entry_t) * win.size() / 1024.0);
#endif
    fflush(stdout);

    uint64_t current_searched = 0;

    uint256_t diff;
    u256_sub(diff, range_max, range_min);
    int diff_bits = 255;
    while (diff_bits >= 0 && u256_get_bit(diff, diff_bits) == 0) diff_bits--;
    if (diff_bits < 0) diff_bits = 0;

    // Dispatch pipeline: queue the next kernel before the host waits on the
    // current one, keeping the GPU fed. One in-order stream, results copied into
    // pinned host memory with an event per slot.
    cudaStream_t stream;
    CUDA_CHECK_RET(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    DeviceResult *h_res = nullptr;
    CUDA_CHECK_RET(cudaMallocHost(&h_res, 2 * sizeof(DeviceResult)));

    cudaEvent_t ev[2];
    CUDA_CHECK_RET(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming));
    CUDA_CHECK_RET(cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming));

    thread_local std::random_device rd;
    thread_local std::mt19937_64   gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFULL);

    int      slot            = 0;
    bool     pending[2]      = { false, false };
    uint64_t pending_keys[2] = { 0, 0 };

    // Retire one completed dispatch. Returns false if the search should stop.
    auto retire = [&](int s) -> bool {
        if (pending[s]) {
            // Must complete before h_res[s] is read: the copy into pinned host
            // memory is asynchronous.
            CUDA_CHECK(cudaEventSynchronize(ev[s]));

            current_searched += pending_keys[s];
            keys_checked.fetch_add(pending_keys[s], std::memory_order_relaxed);

            if (device_id == 0 && g_display_mutex.try_lock()) {
                g_display_key   = h_res[s].sample_key;
                g_display_valid = true;
                g_display_mutex.unlock();
            }

            if (h_res[s].found) {
                found_key = h_res[s].found_key;
                found_flag.store(true);
                return false;
            }
            pending[s] = false;
        }
        return true;
    };

    while (current_searched < keys_to_search && !found_flag.load() && running_flag.load()) {
        uint256_t cursor_offset;
        u256_sub(cursor_offset, start_key, range_min);
        if (is_sequential && u256_cmp(start_key, range_max) > 0) break;

        uint64_t remaining = keys_to_search - current_searched;
        uint64_t cur_ppt   = points_per_thread;
        if (remaining < points_per_launch) {
            cur_ppt = remaining / total_threads;
            if (cur_ppt == 0) cur_ppt = 1;
            if (cur_ppt >= (uint64_t)GPU_BATCH)
                cur_ppt = (cur_ppt / GPU_BATCH) * GPU_BATCH;
        }

        uint64_t seed = is_sequential ? 0 : dist(gen);

        search_kernel<<<blocks, tpb, 0, stream>>>(
            cursor_offset, cur_ppt, d_res,
            is_sequential, range_min, diff, diff_bits, seed);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(&h_res[slot], d_res, sizeof(DeviceResult),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(ev[slot], stream));

        pending[slot]      = true;
        pending_keys[slot] = cur_ppt * total_threads;

        // Advance the sequential cursor before waiting so the next dispatch can
        // be queued as soon as this one reports back.
        if (is_sequential)
            u256_add64(start_key, start_key, pending_keys[slot]);

        // Retire the previous dispatch while this one is still running.
        int other = slot ^ 1;
        if (!retire(other)) break;
        slot = other;
    }

    // Drain: a kernel may still be in flight; freeing device memory out from
    // under a running kernel would be a use-after-free on the device.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!found_flag.load()) {
        if (retire(slot ^ 1)) retire(slot);
    }

    cudaEventDestroy(ev[0]);
    cudaEventDestroy(ev[1]);
    cudaFreeHost(h_res);
    cudaStreamDestroy(stream);
    cudaFree(d_res);
}
