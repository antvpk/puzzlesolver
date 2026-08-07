#ifndef UTIL_H
#define UTIL_H

#include "int256.h"
#include <random>

inline uint256_t random_start_in_range(const uint256_t& min_r, const uint256_t& max_r) {
    uint256_t diff;
    u256_sub(diff, max_r, min_r);

    int bits = 255;
    while (bits >= 0 && u256_get_bit(diff, bits) == 0) bits--;
    if (bits < 0) return min_r;

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

    u256_add(r, r, min_r);
    return r;
}

#endif // UTIL_H
