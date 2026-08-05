
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <signal.h>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_VAR __device__ __managed__
#else
#define HOST_DEVICE
#define HOST_DEVICE_VAR
#endif

#include "int256.h"
#include "secp256k1.h"
#include "sha256_rmd160.h"

// ============= Configuration =============

#define BATCH_SIZE 4096       // Affine batch inversion size (CPU)
#define GPU_BATCH_SIZE 1024   // GPU: larger batch = better fp_inv amortization
#define GPU_SHARED_TABLE 64   // First N entries cached in shared memory

#ifdef __CUDACC__
// GPU: device-side variables (written via cudaMemcpyToSymbol before kernel launch)
__device__ uint8_t d_TARGET_HASH[20];
__device__ uint32_t d_TARGET_H0;  // Precomputed first RIPEMD160 word for early reject
__device__ point_t d_G_POINT;
__device__ point_t d_G_TABLE[GPU_BATCH_SIZE + 1];

// Host-side copies (used by HOST_DEVICE functions when compiled for host path)
__attribute__((unused)) static uint8_t s_TARGET_HASH[20];
__attribute__((unused)) static uint32_t s_TARGET_H0;
__attribute__((unused)) static point_t s_G_POINT;

// __CUDA_ARCH__ is only defined inside device code, so HOST_DEVICE functions
// will pick the right variable depending on whether they're running on GPU or CPU
#define G_POINT       ([]() -> auto& { return d_G_POINT; }())
// Can't use lambdas in __device__ context. Use ternary-free approach:
#undef G_POINT

// Use #ifdef __CUDA_ARCH__ directly in functions instead of macros.
// For the GPU kernel, reference d_ vars directly.
// For macros used only in device code:
#define G_TABLE d_G_TABLE

#else
// CPU-only static variables
static uint8_t s_TARGET_HASH[20];
static uint32_t s_TARGET_H0;
static point_t s_G_POINT;
static point_t G_TABLE[BATCH_SIZE + 1];

#define G_POINT s_G_POINT
#define TARGET_HASH s_TARGET_HASH
#define TARGET_H0 s_TARGET_H0
#endif

// Range
static uint256_t RANGE_MIN;
static uint256_t RANGE_MAX;

// Statistics
static std::atomic<uint64_t> g_total_keys(0);
static std::atomic<bool> g_found(false);
static std::atomic<bool> g_running(true);
static std::mutex g_file_mutex;
static std::chrono::steady_clock::time_point g_start_time;

// Host-side copies for setup
static uint8_t h_TARGET_HASH[20];
static uint32_t h_TARGET_H0;
static point_t h_G_POINT;
static point_t h_G_TABLE[BATCH_SIZE + 1];

// ============= Utility Functions =============

static void hex_to_bytes(const char *hex, uint8_t *bytes, int len) {
    for (int i = 0; i < len; i++) {
        char hi = hex[2*i], lo = hex[2*i+1];
        uint8_t bhi = (hi >= '0' && hi <= '9') ? hi - '0' : (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10 : hi - 'A' + 10;
        uint8_t blo = (lo >= '0' && lo <= '9') ? lo - '0' : (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10 : lo - 'A' + 10;
        bytes[i] = (bhi << 4) | blo;
    }
}

static void bytes_to_hex(const uint8_t *bytes, char *hex, int len) {
    for (int i = 0; i < len; i++) sprintf(hex + 2*i, "%02x", bytes[i]);
    hex[2*len] = 0;
}

HOST_DEVICE inline void x_to_pubkey_bytes(const uint256_t &x, uint8_t *pubkey33, uint8_t prefix) {
    pubkey33[0] = prefix;
    for (int i = 0; i < 4; i++) {
        uint64_t v = x.d[3 - i];
        pubkey33[1 + i*8 + 0] = (v >> 56) & 0xFF;
        pubkey33[1 + i*8 + 1] = (v >> 48) & 0xFF;
        pubkey33[1 + i*8 + 2] = (v >> 40) & 0xFF;
        pubkey33[1 + i*8 + 3] = (v >> 32) & 0xFF;
        pubkey33[1 + i*8 + 4] = (v >> 24) & 0xFF;
        pubkey33[1 + i*8 + 5] = (v >> 16) & 0xFF;
        pubkey33[1 + i*8 + 6] = (v >> 8) & 0xFF;
        pubkey33[1 + i*8 + 7] = v & 0xFF;
    }
}

HOST_DEVICE inline bool hash160_check(const uint8_t *pubkey33) {
    uint8_t sha[32];
    sha256_33(pubkey33, sha);

    uint8_t h160[20];
    ripemd160_32(sha, h160);

#ifdef __CUDA_ARCH__
    const uint8_t *tgt = d_TARGET_HASH;
#else
    const uint8_t *tgt = s_TARGET_HASH;
#endif
    for (int i = 0; i < 20; i++) {
        if (h160[i] != tgt[i]) return false;
    }
    return true;
}

// Fast hash check with early RIPEMD160 rejection
HOST_DEVICE inline bool hash160_check_fast(const uint8_t *pubkey33) {
    uint8_t sha[32];
    sha256_33(pubkey33, sha);

#ifdef __CUDA_ARCH__
    uint32_t tgt_h0 = d_TARGET_H0;
    const uint8_t *tgt = d_TARGET_HASH;
#else
    uint32_t tgt_h0 = s_TARGET_H0;
    const uint8_t *tgt = s_TARGET_HASH;
#endif

    uint8_t h160[20];
    // Early reject: check first 4 bytes of RIPEMD160 before computing the rest
    if (!ripemd160_32_early_reject(sha, h160, tgt_h0)) return false;

    // First word matched! Check remaining 16 bytes
    for (int i = 4; i < 20; i++) {
        if (h160[i] != tgt[i]) return false;
    }
    return true;
}

HOST_DEVICE inline int check_point_1way(const uint256_t &px, const uint256_t &py, uint8_t *pubkey) {
    // Compressed public key prefix (0x02 for even Y, 0x03 for odd Y)
    uint8_t prefix = (py.d[0] & 1) ? 0x03 : 0x02;

    x_to_pubkey_bytes(px, pubkey, prefix);
    if (hash160_check_fast(pubkey)) return 1;

    return 0;
}

#include <string>
#include <vector>

static void cpu_sha256(const uint8_t *data, size_t len, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    
    size_t padded_len = len + 1;
    while ((padded_len % 64) != 56) padded_len++;
    
    std::vector<uint8_t> buffer(padded_len + 8, 0);
    memcpy(buffer.data(), data, len);
    buffer[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        buffer[padded_len + 7 - i] = (bit_len >> (i * 8)) & 0xFF;
    }
    
    for (size_t offset = 0; offset < buffer.size(); offset += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (buffer[offset + i*4] << 24) | (buffer[offset + i*4+1] << 16) |
                   (buffer[offset + i*4+2] << 8) | buffer[offset + i*4+3];
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = (w[i-15] >> 7 | w[i-15] << 25) ^ (w[i-15] >> 18 | w[i-15] << 14) ^ (w[i-15] >> 3);
            uint32_t s1 = (w[i-2] >> 17 | w[i-2] << 15) ^ (w[i-2] >> 19 | w[i-2] << 13) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = (e >> 6 | e << 26) ^ (e >> 11 | e << 21) ^ (e >> 25 | e << 7);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h + S1 + ch + k[i] + w[i];
            uint32_t S0 = (a >> 2 | a << 30) ^ (a >> 13 | a << 19) ^ (a >> 22 | a << 10);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (state[i] >> 24) & 0xFF;
        hash[i*4+1] = (state[i] >> 16) & 0xFF;
        hash[i*4+2] = (state[i] >> 8) & 0xFF;
        hash[i*4+3] = state[i] & 0xFF;
    }
}

static std::string base58_encode(const uint8_t *data, size_t len) {
    const char *ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> digits(len * 138 / 100 + 1, 0);
    size_t digitslen = 1;
    for (size_t i = 0; i < len; i++) {
        uint32_t carry = data[i];
        for (size_t j = 0; j < digitslen; j++) {
            carry += (uint32_t)digits[j] << 8;
            digits[j] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            digits[digitslen++] = carry % 58;
            carry /= 58;
        }
    }
    std::string result;
    for (size_t i = 0; i < len && data[i] == 0; i++) {
        result.push_back('1');
    }
    for (size_t i = 0; i < digitslen; i++) {
        result.push_back(ALPHABET[digits[digitslen - 1 - i]]);
    }
    return result;
}

static std::string encode_wif(const uint256_t &privkey, bool compressed) {
    uint8_t payload[34];
    payload[0] = 0x80;
    for (int i=0; i<4; i++) {
        uint64_t v = privkey.d[3 - i];
        for (int j=0; j<8; j++) {
            payload[1 + i*8 + j] = (v >> (56 - j*8)) & 0xFF;
        }
    }
    size_t len = 33;
    if (compressed) {
        payload[33] = 0x01;
        len = 34;
    }
    uint8_t h1[32], h2[32];
    cpu_sha256(payload, len, h1);
    cpu_sha256(h1, 32, h2);
    uint8_t full[38];
    memcpy(full, payload, len);
    memcpy(full + len, h2, 4);
    return base58_encode(full, len + 4);
}

static std::string encode_address(const uint8_t hash160[20]) {
    uint8_t payload[21];
    payload[0] = 0x00;
    memcpy(payload + 1, hash160, 20);
    uint8_t h1[32], h2[32];
    cpu_sha256(payload, 21, h1);
    cpu_sha256(h1, 32, h2);
    uint8_t full[25];
    memcpy(full, payload, 21);
    memcpy(full + 21, h2, 4);
    return base58_encode(full, 25);
}

static void save_result(const uint256_t &privkey) {
    std::lock_guard<std::mutex> lock(g_file_mutex);
    char hex_key[65];
    u256_to_hex(hex_key, privkey);
    char *trimmed = hex_key;
    while (*trimmed == '0' && *(trimmed+1) != '\0') trimmed++;

    std::string wif_compressed = encode_wif(privkey, true);
    std::string address = encode_address(h_TARGET_HASH);

    FILE *f = fopen("RESULT.txt", "a");
    if (f) {
        time_t now = time(NULL);
        char timebuf[64];
        strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        fprintf(f, "=== PUZZLE KEY FOUND ===\nTime: %s\nKey:  0x%s\nKey (WIF Compressed): %s\nAddress (Legacy Compressed): %s\n===\n\n",
                timebuf, trimmed, wif_compressed.c_str(), address.c_str());
        fclose(f);
    }

    printf("\n\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║         🎉 PUZZLE KEY FOUND! 🎉             ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Private Key: 0x%-28s  ║\n", trimmed);
    printf("║  Saved to: RESULT.txt                       ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");
}

// ============= Precomputation =============

static void precompute() {
    int table_size = BATCH_SIZE;
#ifdef __CUDACC__
    table_size = GPU_BATCH_SIZE;
#endif
    printf("[*] Precomputing %d G-table for affine batching...\n", table_size);
    
    h_G_POINT.infinity = false;
    h_G_POINT.x = SECP256K1_GX;
    h_G_POINT.y = SECP256K1_GY;

    // Compute target_h0 for early RIPEMD160 rejection
    h_TARGET_H0 = ((uint32_t)h_TARGET_HASH[0]) | ((uint32_t)h_TARGET_HASH[1] << 8) |
                  ((uint32_t)h_TARGET_HASH[2] << 16) | ((uint32_t)h_TARGET_HASH[3] << 24);

    // T[0] = infinity (we will handle i=0 specially in the batch)
    h_G_TABLE[0].infinity = true;
    
    jpoint_t acc = affine_to_jacobian(h_G_POINT);
    h_G_TABLE[1] = h_G_POINT;
    
    for (int i = 2; i <= table_size; i++) {
        jpoint_t next;
        jpoint_add_affine(next, acc, h_G_POINT);
        acc = next;
        h_G_TABLE[i] = jacobian_to_affine(acc);
    }

#ifdef __CUDACC__
    // Copy to device constant memory
    cudaMemcpyToSymbol(d_TARGET_HASH, h_TARGET_HASH, 20);
    cudaMemcpyToSymbol(d_TARGET_H0, &h_TARGET_H0, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_G_POINT, &h_G_POINT, sizeof(point_t));
    cudaMemcpyToSymbol(d_G_TABLE, h_G_TABLE, sizeof(point_t) * (GPU_BATCH_SIZE + 1));
    cudaDeviceSynchronize();
#else
    // CPU: copy to module-level statics
    G_POINT = h_G_POINT;
    memcpy(TARGET_HASH, h_TARGET_HASH, 20);
    TARGET_H0 = h_TARGET_H0;
    memcpy(G_TABLE, h_G_TABLE, sizeof(point_t) * (BATCH_SIZE + 1));
#endif
}

// ============= Search Worker (CPU) =============

#ifndef __CUDACC__
static void search_worker(uint256_t start_key, uint64_t total_keys) {
    uint8_t pubkey[33];

    point_t base_point = ec_mul(start_key, G_POINT);
    if (base_point.infinity) return;

    uint256_t base_key = start_key;
    uint64_t checked = 0;

    // We compute P + T[i] for i = 0 .. BATCH_SIZE-1
    // dx[i] = P.x - T[i].x
    static thread_local uint256_t dx[BATCH_SIZE];
    static thread_local uint256_t z_prod[BATCH_SIZE];
    static thread_local uint256_t dx_inv[BATCH_SIZE];

    while (checked < total_keys && g_running.load()) {
        int bc = BATCH_SIZE;
        if (checked + bc > total_keys) bc = (int)(total_keys - checked);

        // 1. Compute dx = P.x - T[i].x
        for (int i = 1; i < bc; i++) {
            fp_sub(dx[i], base_point.x, G_TABLE[i].x);
        }

        // 2. Batch invert dx
        z_prod[1] = dx[1];
        for (int i = 2; i < bc; i++) {
            fp_mul(z_prod[i], z_prod[i-1], dx[i]);
        }

        uint256_t inv_prod;
        fp_inv(inv_prod, z_prod[bc - 1]);

        for (int i = bc - 1; i > 1; i--) {
            fp_mul(dx_inv[i], inv_prod, z_prod[i-1]);
            fp_mul(inv_prod, inv_prod, dx[i]);
        }
        dx_inv[1] = inv_prod;

        // 3. Compute affine additions and check
        for (int i = 0; i < bc; i++) {
            uint256_t k;
            u256_add64(k, base_key, i);
            
            uint256_t rx, ry;
            
            if (i == 0) {
                // T[0] is infinity, so P + T[0] = P
                rx = base_point.x;
                ry = base_point.y;
            } else {
                // s = (P.y - T[i].y) / dx
                uint256_t dy, s, s2, temp;
                fp_sub(dy, base_point.y, G_TABLE[i].y);
                fp_mul(s, dy, dx_inv[i]);
                
                // rx = s^2 - P.x - T[i].x
                fp_sqr(s2, s);
                fp_sub(temp, s2, base_point.x);
                fp_sub(rx, temp, G_TABLE[i].x);
                
                // ry = s*(P.x - rx) - P.y
                fp_sub(temp, base_point.x, rx);
                fp_mul(temp, s, temp);
                fp_sub(ry, temp, base_point.y);
            }

            int match = check_point_1way(rx, ry, pubkey);
            if (match) {
                g_found.store(true);
                save_result(k);
                g_running.store(false);
                return;
            }
        }

        checked += bc;
        g_total_keys.fetch_add((uint64_t)bc, std::memory_order_relaxed);

        // Advance base_point by BATCH_SIZE * G
        if (bc == BATCH_SIZE) {
            u256_add64(base_key, base_key, BATCH_SIZE);
            jpoint_t R;
            jpoint_add_affine(R, affine_to_jacobian(base_point), G_TABLE[BATCH_SIZE]);
            base_point = jacobian_to_affine(R);
        }
    }
}
#endif

// ============= Display Thread =============

static void display_thread_func() {
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count() / 1000.0;
        if (elapsed < 0.1) continue;

        uint64_t total = g_total_keys.load();
        double speed = total / elapsed;

        const char *su = "keys/s";
        double sd = speed;
        if (speed >= 1e9) { sd = speed / 1e9; su = "Gkeys/s"; }
        else if (speed >= 1e6) { sd = speed / 1e6; su = "Mkeys/s"; }
        else if (speed >= 1e3) { sd = speed / 1e3; su = "Kkeys/s"; }

        const char *tu = "";
        double td = (double)total;
        if (total >= 1000000000000ULL) { td = total / 1e12; tu = "T"; }
        else if (total >= 1000000000ULL) { td = total / 1e9; tu = "G"; }
        else if (total >= 1000000ULL) { td = total / 1e6; tu = "M"; }
        else if (total >= 1000ULL) { td = total / 1e3; tu = "K"; }

        // \033[K clears the rest of the line so we don't leave trailing junk
        printf("\r[⚡] Speed: %.2f %s | Total: %.2f%s | %s\033[K",
               sd, su, td, tu,
               g_found.load() ? "FOUND ✅" : "Searching...");
        fflush(stdout);
    }
}

// ============= Signal Handler =============

static void signal_handler(int sig) {
    (void)sig;
    g_running.store(false);
}

// ============= Self-test =============

static bool self_test() {
    printf("[*] Self-test...\n");

    point_t G;
    G.infinity = false;
    G.x = SECP256K1_GX;
    G.y = SECP256K1_GY;

    uint8_t pubkey[33], h160[20];
    char hex[41];

    point_t p1 = ec_mul({1, 0, 0, 0}, G);
    x_to_pubkey_bytes(p1.x, pubkey, (p1.y.d[0] & 1) ? 0x03 : 0x02);
    hash160(pubkey, h160);
    bytes_to_hex(h160, hex, 20);
    if (strcmp(hex, "751e76e8199196d454941c45d1b3a323f1433bd6") != 0) {
        printf("  [FAIL] Key=1: %s\n", hex); return false;
    }

    point_t p2 = ec_mul({2, 0, 0, 0}, G);
    x_to_pubkey_bytes(p2.x, pubkey, (p2.y.d[0] & 1) ? 0x03 : 0x02);
    hash160(pubkey, h160);
    bytes_to_hex(h160, hex, 20);
    if (strcmp(hex, "06afd46bcdfd22ef94ac122aa11f241244a37ecc") != 0) {
        printf("  [FAIL] Key=2: %s\n", hex); return false;
    }

    // Test early reject function
    point_t p3 = ec_mul({1, 0, 0, 0}, G);
    x_to_pubkey_bytes(p3.x, pubkey, (p3.y.d[0] & 1) ? 0x03 : 0x02);
    uint8_t sha[32];
    sha256_33(pubkey, sha);
    uint8_t rmd_test[20];
    ripemd160_32(sha, rmd_test);
    uint32_t test_h0 = ((uint32_t)rmd_test[0]) | ((uint32_t)rmd_test[1] << 8) |
                       ((uint32_t)rmd_test[2] << 16) | ((uint32_t)rmd_test[3] << 24);
    uint8_t rmd_test2[20];
    bool should_pass = ripemd160_32_early_reject(sha, rmd_test2, test_h0);
    if (!should_pass) {
        printf("  [FAIL] Early reject returned false for matching hash\n"); return false;
    }
    // Test with wrong h0
    bool should_fail = ripemd160_32_early_reject(sha, rmd_test2, test_h0 ^ 0x12345678);
    if (should_fail) {
        printf("  [FAIL] Early reject returned true for non-matching hash\n"); return false;
    }

    printf("  [PASS] All checks ✓\n");
    return true;
}

static uint256_t random_start_in_range() {
    uint256_t diff;
    u256_sub(diff, RANGE_MAX, RANGE_MIN);
    
    int bits = 255;
    while (bits >= 0 && u256_get_bit(diff, bits) == 0) bits--;
    if (bits < 0) return RANGE_MIN;
    
    thread_local std::random_device rd;
    thread_local std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFULL);
    
    uint256_t r;
    while (true) {
        r.d[0] = dist(gen);
        r.d[1] = dist(gen);
        r.d[2] = dist(gen);
        r.d[3] = dist(gen);
        
        int limb = bits / 64;
        int bit_in_limb = bits % 64;
        uint64_t mask = (1ULL << (bit_in_limb + 1)) - 1;
        if (bit_in_limb == 63) mask = 0xFFFFFFFFFFFFFFFFULL;
        
        r.d[limb] &= mask;
        for (int i = limb + 1; i < 4; i++) r.d[i] = 0;
        
        if (u256_cmp(r, diff) <= 0) break;
    }
    
    u256_add(r, r, RANGE_MIN);
    return r;
}

#ifdef __CUDACC__

// ============= GPU Kernel =============
// Each thread processes GPU_BATCH_SIZE keys using batch affine addition
__global__ void search_kernel(uint256_t start_key_base, uint64_t keys_per_thread, 
                              int *found_flag, uint256_t *found_key) {
    // Cache first GPU_SHARED_TABLE entries in shared memory for fastest access
    __shared__ __align__(8) uint8_t raw_smem[sizeof(point_t) * (GPU_SHARED_TABLE + 1)];
    point_t *shared_gtable = (point_t*)raw_smem;
    
    int lid = threadIdx.x;
    // Cooperatively load shared table
    for (int i = lid; i <= GPU_SHARED_TABLE; i += blockDim.x) {
        shared_gtable[i] = d_G_TABLE[i];
    }
    __syncthreads();

    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint256_t start_key = start_key_base;
    u256_add64(start_key, start_key, tid * keys_per_thread);

    uint8_t pubkey[33];
    point_t base_point = ec_mul(start_key, d_G_POINT);
    if (base_point.infinity) return;

    uint256_t base_key = start_key;
    uint64_t checked = 0;

    // Reduced to a single array (32KB per thread for 1024 batch) to avoid massive VRAM spilling
    uint256_t z_prod[GPU_BATCH_SIZE];

    while (checked < keys_per_thread && !(*found_flag)) {
        int bc = GPU_BATCH_SIZE;
        if (checked + bc > keys_per_thread) bc = (int)(keys_per_thread - checked);

        // 1. Compute dx and forward z_prod in one pass
        for (int i = 1; i < bc; i++) {
            uint256_t dx_i;
            const point_t &gt = (i <= GPU_SHARED_TABLE) ? shared_gtable[i] : d_G_TABLE[i];
            fp_sub(dx_i, base_point.x, gt.x);
            if (i == 1) {
                z_prod[1] = dx_i;
            } else {
                fp_mul(z_prod[i], z_prod[i-1], dx_i);
            }
        }

        uint256_t inv_prod;
        fp_inv(inv_prod, z_prod[bc - 1]);

        // 2 & 3. Process backwards, computing inverse and affine points on the fly
        for (int i = bc - 1; i >= 1; i--) {
            uint256_t current_inv;
            
            // Recompute dx[i] - incredibly cheap compared to VRAM array access
            uint256_t dx_i;
            const point_t &gt = (i <= GPU_SHARED_TABLE) ? shared_gtable[i] : d_G_TABLE[i];
            fp_sub(dx_i, base_point.x, gt.x);

            if (i > 1) {
                fp_mul(current_inv, inv_prod, z_prod[i-1]);
                fp_mul(inv_prod, inv_prod, dx_i);
            } else {
                current_inv = inv_prod;
            }

            uint256_t rx, ry, dy, s, s2, temp;
            fp_sub(dy, base_point.y, gt.y);
            fp_mul(s, dy, current_inv);
            fp_sqr(s2, s);
            fp_sub(temp, s2, base_point.x);
            fp_sub(rx, temp, gt.x);
            fp_sub(temp, base_point.x, rx);
            fp_mul(temp, s, temp);
            fp_sub(ry, temp, base_point.y);

            int match = check_point_1way(rx, ry, pubkey);
            if (match) {
                uint256_t k;
                u256_add64(k, base_key, i);
                if (atomicExch((int*)found_flag, 1) == 0) {
                    *found_key = k;
                }
                return;
            }
        }

        // Handle i = 0 separately
        int match0 = check_point_1way(base_point.x, base_point.y, pubkey);
        if (match0) {
            if (atomicExch((int*)found_flag, 1) == 0) {
                *found_key = base_key;
            }
            return;
        }

        checked += bc;
        if (bc == GPU_BATCH_SIZE) {
            u256_add64(base_key, base_key, GPU_BATCH_SIZE);
            jpoint_t R;
            const point_t &gt_end = d_G_TABLE[GPU_BATCH_SIZE];
            jpoint_add_affine(R, affine_to_jacobian(base_point), gt_end);
            base_point = jacobian_to_affine(R);
        }
    }
}

#endif

int main(int argc, char **argv) {
    RANGE_MIN = {0, 0, 0, 0};
    RANGE_MIN.d[1] = 0x40; // Default: 0x400000000000000000
    RANGE_MAX = {0, 0, 0, 0};
    RANGE_MAX.d[1] = 0x7F;
    RANGE_MAX.d[0] = 0xFFFFFFFFFFFFFFFF; // Default: 0x7fffffffffffffffff

    hex_to_bytes("f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8", h_TARGET_HASH, 20);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!self_test()) {
        printf("[!] Self-test FAILED!\n");
        return 1;
    }

#ifndef __CUDACC__
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
#endif

    bool random_mode = true;
    uint64_t keys_per_thread = 4096; // Optimal default batch length
#ifdef __CUDACC__
    int blocks = 0;                  // 0 = auto-detect from GPU
    int threads_per_block = 0;       // 0 = auto-detect from GPU
#endif

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sequential") == 0)
            random_mode = false;
#ifndef __CUDACC__
        if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && i+1 < argc)
            num_threads = atoi(argv[++i]);
#endif
        if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch") == 0) && i+1 < argc)
            keys_per_thread = strtoull(argv[++i], NULL, 0);
#ifdef __CUDACC__
        if (strcmp(argv[i], "--blocks") == 0 && i+1 < argc)
            blocks = atoi(argv[++i]);
        if (strcmp(argv[i], "--tpb") == 0 && i+1 < argc)
            threads_per_block = atoi(argv[++i]);
#endif
        if (strcmp(argv[i], "-target") == 0 && i+1 < argc) {
            const char *t = argv[++i];
            if (strlen(t) == 40) hex_to_bytes(t, h_TARGET_HASH, 20);
        }
        if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--range") == 0) && i+1 < argc) {
            const char *r = argv[++i];
            char min_s[65] = {0}, max_s[65] = {0};
            const char *colon = strchr(r, ':');
            if (colon) {
                strncpy(min_s, r, colon - r);
                strcpy(max_s, colon + 1);
            } else {
                strcpy(min_s, r);
            }
            u256_from_hex(RANGE_MIN, min_s);
            if (strlen(max_s) > 0) u256_from_hex(RANGE_MAX, max_s);
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
#ifdef __CUDACC__
            printf("Usage: %s [-s] [-b N] [--blocks N] [--tpb N] [-target HASH] [-r MIN:MAX] [-h]\n", argv[0]);
            printf("  --blocks 0  = auto-detect optimal blocks from GPU\n");
            printf("  --tpb 0     = auto-detect optimal threads/block from GPU\n");
#else
            printf("Usage: %s [-t N] [-s] [-b N] [-target HASH] [-r MIN:MAX] [-h]\n", argv[0]);
#endif
            return 0;
        }
    }

    precompute();

    char target_hex_str[41] = {0};
    for (int i=0; i<20; i++) sprintf(&target_hex_str[i*2], "%02x", h_TARGET_HASH[i]);
    
    char min_hex_str[65] = {0};
    u256_to_hex(min_hex_str, RANGE_MIN);
    char max_hex_str[65] = {0};
    u256_to_hex(max_hex_str, RANGE_MAX);
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  🔑 Puzzle Keyhunt v2.0 - Ultra Performance CPU/GPU  ║\n");
    printf("║  Target: %-44s ║\n", target_hex_str);
    char *min_p = min_hex_str;
    while (*min_p == '0' && *(min_p+1) != '\0') min_p++;
    char *max_p = max_hex_str;
    while (*max_p == '0' && *(max_p+1) != '\0') max_p++;

    char range_buf[200];
    snprintf(range_buf, sizeof(range_buf), "0x%s : 0x%s", min_p, max_p);
    printf("║  Range:  %-44s ║\n", range_buf);
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    signal(SIGINT, signal_handler);

    g_start_time = std::chrono::steady_clock::now();
    std::thread display(display_thread_func);

    int round = 0;
    
#ifdef __CUDACC__
    // ============= GPU Auto-Detection & Tuning =============
    {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        const char *gpu_family = "Unknown";
        int cc = prop.major * 10 + prop.minor;
        if      (cc == 61) gpu_family = "Pascal";
        else if (cc == 70) gpu_family = "Volta";
        else if (cc == 75) gpu_family = "Turing";
        else if (cc == 80) gpu_family = "Ampere";
        else if (cc == 86) gpu_family = "Ampere";
        else if (cc == 87) gpu_family = "Ampere (Orin)";
        else if (cc == 89) gpu_family = "Ada Lovelace";
        else if (cc == 90) gpu_family = "Hopper";
        else if (cc >= 100) gpu_family = "Blackwell";

        printf("[*] GPU: %s (%s)\n", prop.name, gpu_family);
        printf("[*] SMs: %d | Max threads/SM: %d | Shared mem/SM: %zu KB\n",
               prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
               prop.sharedMemPerMultiprocessor / 1024);
        printf("[*] VRAM: %.1f GB | Compute: %d.%d\n",
               prop.totalGlobalMem / 1073741824.0, prop.major, prop.minor);

        // Auto-tune blocks and threads if set to 0
        if (threads_per_block == 0) {
            // Use 256 threads for most GPUs, 128 for very old ones
            if (cc >= 75) {
                threads_per_block = 256;
            } else {
                threads_per_block = 128;
            }
        }

        if (blocks == 0) {
            // Target: 2 blocks per SM for good occupancy without too much register pressure
            // For high-end GPUs (RTX 40xx, A100, H100), use more blocks per SM
            int blocks_per_sm = 2;
            if (cc >= 86) blocks_per_sm = 3;  // Ampere+ can handle more
            if (cc >= 89) blocks_per_sm = 4;  // Ada Lovelace has more resources
            if (cc >= 90) blocks_per_sm = 4;  // Hopper
            
            blocks = prop.multiProcessorCount * blocks_per_sm;

            // Ensure we don't exceed max threads per SM
            int threads_per_sm = blocks_per_sm * threads_per_block;
            if (threads_per_sm > prop.maxThreadsPerMultiProcessor) {
                blocks = (prop.maxThreadsPerMultiProcessor / threads_per_block) * prop.multiProcessorCount;
            }
        }

        printf("[*] Config: Blocks=%d | Threads/Block=%d | Total threads=%d\n",
               blocks, threads_per_block, blocks * threads_per_block);
    }

    printf("[*] GPU Mode | Mode: %s | Blocks: %d | Threads/Block: %d | Batch: %d | SharedTable: %d\n\n", 
           random_mode ? "Random" : "Sequential", blocks, threads_per_block, GPU_BATCH_SIZE, GPU_SHARED_TABLE);
    
    // Use device memory for found flags (not managed — faster)
    int *d_found_flag;
    uint256_t *d_found_key;
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_found_key, sizeof(uint256_t));
    int h_found_flag = 0;
    cudaMemset(d_found_flag, 0, sizeof(int));

    // Double-buffered streams for overlapping kernel execution with host work
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStream_t streams[2] = {stream1, stream2};
    int stream_idx = 0;

    while (g_running.load() && !h_found_flag) {
        round++;
        uint256_t start;
        if (random_mode) {
            start = random_start_in_range();
        } else {
            start = RANGE_MIN;
            uint64_t offset = (uint64_t)(round - 1) * blocks * threads_per_block * keys_per_thread;
            u256_add64(start, start, offset);
        }
        
        // Launch on alternating streams
        search_kernel<<<blocks, threads_per_block, 0, streams[stream_idx]>>>(
            start, keys_per_thread, d_found_flag, d_found_key);
        stream_idx = 1 - stream_idx;
        
        // Check for found every other round to overlap kernel with check
        if (round % 2 == 0) {
            cudaDeviceSynchronize();
            cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            g_total_keys.fetch_add((uint64_t)2 * blocks * threads_per_block * keys_per_thread, 
                                  std::memory_order_relaxed);
        }
        
        if (h_found_flag) {
            cudaDeviceSynchronize();
            uint256_t h_found_key;
            cudaMemcpy(&h_found_key, d_found_key, sizeof(uint256_t), cudaMemcpyDeviceToHost);
            g_found.store(true);
            save_result(h_found_key);
            break;
        }
    }
    
    // Final sync and check
    if (!h_found_flag) {
        cudaDeviceSynchronize();
        cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_found_flag) {
            uint256_t h_found_key;
            cudaMemcpy(&h_found_key, d_found_key, sizeof(uint256_t), cudaMemcpyDeviceToHost);
            g_found.store(true);
            save_result(h_found_key);
        }
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_found_flag);
    cudaFree(d_found_key);
#else
    printf("[*] CPU Mode | Threads: %d | Mode: %s | Affine Batch: %d\n\n",
           num_threads, random_mode ? "Random" : "Sequential", BATCH_SIZE);

    while (g_running.load() && !g_found.load()) {
        round++;
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
        #pragma omp parallel
        {
#else
        {
#endif
            uint256_t start;
            if (random_mode) {
                start = random_start_in_range();
            } else {
                start = RANGE_MIN;
                uint64_t tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                uint64_t offset = tid * keys_per_thread +
                                  (uint64_t)(round - 1) * num_threads * keys_per_thread;
                u256_add64(start, start, offset);
            }
            search_worker(start, keys_per_thread);
        }
    }
#endif

    g_running.store(false);
    display.join();

    auto end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - g_start_time).count() / 1000.0;
    uint64_t total = g_total_keys.load();

    printf("\n\n[📊] Total: %llu (%.2e) in %.1fs | Avg: %.2f Mkeys/s | %s\n",
           (unsigned long long)total, (double)total, total_time,
           total / total_time / 1e6,
           g_found.load() ? "FOUND ✅" : "Not found ❌");

    return g_found.load() ? 0 : 1;
}
