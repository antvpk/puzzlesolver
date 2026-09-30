#include "cpu_worker.h"
#include "secp256k1.h"
#include "sha256_rmd160.h"
#include <cstring>
#include <cstdio>
#include <immintrin.h> 
#include <thread>
#include <mutex>
#include "util.h"

extern std::mutex g_display_mutex;
extern uint256_t g_display_key;
extern bool g_display_valid;

static point_t G_POINT;

void init_cpu_worker() {
    G_POINT.infinity = false;
    G_POINT.x = SECP256K1_GX;
    G_POINT.y = SECP256K1_GY;
}

static inline void x_to_pubkey_bytes(const uint256_t &x, uint8_t *pubkey33, uint8_t prefix) {
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

static inline int check_point_1way(const uint256_t &px, const uint256_t &py, uint8_t *pubkey, const uint8_t* tgt, uint32_t tgt_h0) {
    uint8_t prefix = (py.d[0] & 1) ? 0x03 : 0x02;
    x_to_pubkey_bytes(px, pubkey, prefix);

    uint8_t sha[32];
    sha256_33(pubkey, sha);

    uint8_t h160[20];
    if (!ripemd160_32_early_reject(sha, h160, tgt_h0)) return 0;

    for (int i = 4; i < 20; i++) {
        if (h160[i] != tgt[i]) return 0;
    }
    return 1;
}

__attribute__((target_clones("default", "avx", "avx2", "avx512f")))
void cpu_search_worker(int thread_id, uint256_t start_key, uint64_t keys_to_search,
                       std::atomic<uint64_t>& keys_checked, std::atomic<bool>& found_flag,
                       std::atomic<bool>& running_flag,
                       uint256_t& found_key, const uint8_t* target_hash, uint32_t target_h0,
                       uint64_t batch_size, bool is_sequential, uint256_t range_min, uint256_t range_max) {

    point_t* local_G_TABLE = new point_t[batch_size + 1];
    local_G_TABLE[0].infinity = true;
    jpoint_t acc = affine_to_jacobian(G_POINT);
    local_G_TABLE[1] = G_POINT;
    for (uint64_t i = 2; i <= batch_size; i++) {
        jpoint_t next;
        jpoint_add_affine(next, acc, G_POINT);
        acc = next;
        local_G_TABLE[i] = jacobian_to_affine(acc);
    }

    uint8_t pubkey[33];
    point_t base_point = ec_mul(start_key, G_POINT);
    if (base_point.infinity) {
        delete[] local_G_TABLE;
        return;
    }

    uint256_t base_key = start_key;
    uint64_t checked = 0;

    uint256_t* dx = new uint256_t[batch_size];
    uint256_t* z_prod = new uint256_t[batch_size];
    uint256_t* dx_inv = new uint256_t[batch_size];
    bool* has_collision = new bool[batch_size];

    while (checked < keys_to_search && !found_flag.load(std::memory_order_relaxed) && running_flag.load(std::memory_order_relaxed)) {
        if (is_sequential && u256_cmp(base_key, range_max) > 0) {
            break;
        }
        uint64_t bc = batch_size;
        if (checked + bc > keys_to_search) bc = keys_to_search - checked;

        // Guard the batch-inversion "zero denominator" collision:
        // dx[i] == 0 happens when base.x == G_TABLE[i].x (i.e. base_key ≡ ±i
        // mod n). A single zero factor would zero the whole prefix product and
        // poison the entire batch, silently skipping every key in it. Substitute
        // 1 so the product stays invertible; the affected point is recomputed
        // exactly below via a full Jacobian add.
        for (uint64_t i = 1; i < bc; i++) {
            fp_sub(dx[i], base_point.x, local_G_TABLE[i].x);
            has_collision[i] = u256_is_zero(dx[i]);
            if (has_collision[i]) { dx[i].d[0]=1; dx[i].d[1]=0; dx[i].d[2]=0; dx[i].d[3]=0; }
        }

        z_prod[1] = dx[1];
        for (uint64_t i = 2; i < bc; i++) {
            fp_mul(z_prod[i], z_prod[i-1], dx[i]);
        }

        uint256_t inv_prod;
        fp_inv(inv_prod, z_prod[bc - 1]);

        for (uint64_t i = bc - 1; i > 1; i--) {
            fp_mul(dx_inv[i], inv_prod, z_prod[i-1]);
            fp_mul(inv_prod, inv_prod, dx[i]);
        }
        dx_inv[1] = inv_prod;

        for (uint64_t i = 0; i < bc; i++) {
            uint256_t k;
            u256_add64(k, base_key, i);
            uint256_t rx, ry;

            if (i == 0) {
                rx = base_point.x;
                ry = base_point.y;
            } else if (has_collision[i]) {
                // base.x == G_TABLE[i].x → doubling or point-at-infinity; the
                // affine slope formula is undefined. Fall back to a Jacobian add
                // which handles both cases correctly.
                jpoint_t R;
                jpoint_add_affine(R, affine_to_jacobian(base_point), local_G_TABLE[i]);
                point_t rp = jacobian_to_affine(R);
                rx = rp.x; ry = rp.y;
            } else {
                uint256_t dy, s, s2, temp;
                fp_sub(dy, base_point.y, local_G_TABLE[i].y);
                fp_mul(s, dy, dx_inv[i]);
                fp_sqr(s2, s);
                fp_sub(temp, s2, base_point.x);
                fp_sub(rx, temp, local_G_TABLE[i].x);
                fp_sub(temp, base_point.x, rx);
                fp_mul(temp, s, temp);
                fp_sub(ry, temp, base_point.y);
            }

            if (check_point_1way(rx, ry, pubkey, target_hash, target_h0)) {
                found_flag.store(true);
                found_key = k;
                delete[] dx; delete[] z_prod; delete[] dx_inv; delete[] has_collision; delete[] local_G_TABLE;
                return;
            }
        }

        checked += bc;
        keys_checked.fetch_add(bc, std::memory_order_relaxed);

        if (bc == batch_size) {
            if (is_sequential) {
                u256_add64(base_key, base_key, batch_size);
                jpoint_t R;
                jpoint_add_affine(R, affine_to_jacobian(base_point), local_G_TABLE[batch_size]);
                base_point = jacobian_to_affine(R);
            } else {
                base_key = random_start_in_range(range_min, range_max);
                base_point = ec_mul(base_key, G_POINT);
            }
            if (thread_id == 0 && g_display_mutex.try_lock()) {
                g_display_key = base_key;
                g_display_valid = true;
                g_display_mutex.unlock();
            }
        }
        std::this_thread::yield();
    }

    delete[] dx; delete[] z_prod; delete[] dx_inv; delete[] has_collision; delete[] local_G_TABLE;
}
