/*
 * int256.h - 256-bit unsigned integer arithmetic
 * Optimized for secp256k1 field/scalar operations
 * Uses 4x uint64_t limbs (little-endian)
 */
#ifndef INT256_H
#define INT256_H

#include <cstdint>
#include <cstring>
#include <cstdio>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define DEVICE_FUNC __device__
#else
#define HOST_DEVICE
#define DEVICE_FUNC
#endif

struct uint256_t {
    uint64_t d[4]; // little-endian: d[0] = lowest 64 bits
};

// Compare: returns -1, 0, 1
HOST_DEVICE inline int u256_cmp(const uint256_t &a, const uint256_t &b) {
    for (int i = 3; i >= 0; i--) {
        if (a.d[i] < b.d[i]) return -1;
        if (a.d[i] > b.d[i]) return 1;
    }
    return 0;
}

HOST_DEVICE inline bool u256_is_zero(const uint256_t &a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3]) == 0;
}

HOST_DEVICE inline bool u256_is_even(const uint256_t &a) {
    return (a.d[0] & 1) == 0;
}

// a + b, returns carry
HOST_DEVICE inline uint64_t u256_add(uint256_t &r, const uint256_t &a, const uint256_t &b) {
#ifdef __CUDA_ARCH__
    uint64_t carry;
    asm volatile(
        "add.cc.u64      %0, %5, %9;\n\t"
        "addc.cc.u64     %1, %6, %10;\n\t"
        "addc.cc.u64     %2, %7, %11;\n\t"
        "addc.cc.u64     %3, %8, %12;\n\t"
        "addc.u64        %4, 0, 0;\n\t"
        : "=&l"(r.d[0]), "=&l"(r.d[1]), "=&l"(r.d[2]), "=&l"(r.d[3]), "=&l"(carry)
        : "l"(a.d[0]), "l"(a.d[1]), "l"(a.d[2]), "l"(a.d[3]),
          "l"(b.d[0]), "l"(b.d[1]), "l"(b.d[2]), "l"(b.d[3])
    );
    return carry;
#else
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t s = (__uint128_t)a.d[i] + b.d[i] + carry;
        r.d[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
    return carry;
#endif
}

// a - b, returns borrow
HOST_DEVICE inline uint64_t u256_sub(uint256_t &r, const uint256_t &a, const uint256_t &b) {
#ifdef __CUDA_ARCH__
    uint64_t borrow;
    asm volatile(
        "sub.cc.u64      %0, %5, %9;\n\t"
        "subc.cc.u64     %1, %6, %10;\n\t"
        "subc.cc.u64     %2, %7, %11;\n\t"
        "subc.cc.u64     %3, %8, %12;\n\t"
        "subc.u64        %4, 0, 0;\n\t"
        : "=&l"(r.d[0]), "=&l"(r.d[1]), "=&l"(r.d[2]), "=&l"(r.d[3]), "=&l"(borrow)
        : "l"(a.d[0]), "l"(a.d[1]), "l"(a.d[2]), "l"(a.d[3]),
          "l"(b.d[0]), "l"(b.d[1]), "l"(b.d[2]), "l"(b.d[3])
    );
    // PTX subc computes: dest = src1 - src2 - CF.
    // %4 = 0 - 0 - CF = -CF. If borrow occurred, %4 is 0xFFFFFFFFFFFFFFFF.
    return borrow & 1;
#else
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t s = (__uint128_t)a.d[i] - b.d[i] - borrow;
        r.d[i] = (uint64_t)s;
        borrow = (s >> 127) & 1; // check if negative
    }
    return borrow;
#endif
}

// r = a >> 1
HOST_DEVICE inline void u256_rshift1(uint256_t &r, const uint256_t &a) {
    r.d[0] = (a.d[0] >> 1) | (a.d[1] << 63);
    r.d[1] = (a.d[1] >> 1) | (a.d[2] << 63);
    r.d[2] = (a.d[2] >> 1) | (a.d[3] << 63);
    r.d[3] = a.d[3] >> 1;
}

// r = a << 1, returns carry
HOST_DEVICE inline uint64_t u256_lshift1(uint256_t &r, const uint256_t &a) {
    uint64_t carry = a.d[3] >> 63;
    r.d[3] = (a.d[3] << 1) | (a.d[2] >> 63);
    r.d[2] = (a.d[2] << 1) | (a.d[1] >> 63);
    r.d[1] = (a.d[1] << 1) | (a.d[0] >> 63);
    r.d[0] = a.d[0] << 1;
    return carry;
}

// 256x256 -> 512 bit multiplication
HOST_DEVICE inline void u256_mul_full(uint64_t r[8], const uint256_t &a, const uint256_t &b) {
#ifdef __CUDA_ARCH__
    for (int i = 0; i < 8; i++) r[i] = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo = a.d[i] * b.d[j];
            uint64_t hi = __umul64hi(a.d[i], b.d[j]);
            uint64_t sum = r[i+j] + lo;
            uint64_t c1 = (sum < r[i+j]) ? 1ULL : 0ULL;
            uint64_t sum2 = sum + carry;
            uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
            r[i+j] = sum2;
            carry = hi + c1 + c2;
        }
        r[i+4] = carry;
    }
#else
    __uint128_t carry = 0;
    for (int i = 0; i < 8; i++) r[i] = 0;

    for (int i = 0; i < 4; i++) {
        carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a.d[i] * b.d[j] + r[i+j] + carry;
            r[i+j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        r[i+4] = (uint64_t)carry;
    }
#endif
}

// 256-bit squaring -> 512 bits. Exploits symmetry: off-diag products computed once & doubled.
// Saves ~37% of 64-bit multiplies vs calling u256_mul_full(r, a, a).
HOST_DEVICE inline void u256_sqr_full(uint64_t r[8], const uint256_t &a) {
#ifdef __CUDA_ARCH__
    // Diagonal products: a[i] * a[i]
    uint64_t d0_lo = a.d[0] * a.d[0], d0_hi = __umul64hi(a.d[0], a.d[0]);
    uint64_t d1_lo = a.d[1] * a.d[1], d1_hi = __umul64hi(a.d[1], a.d[1]);
    uint64_t d2_lo = a.d[2] * a.d[2], d2_hi = __umul64hi(a.d[2], a.d[2]);
    uint64_t d3_lo = a.d[3] * a.d[3], d3_hi = __umul64hi(a.d[3], a.d[3]);

    // Off-diagonal products (computed once, will be doubled)
    uint64_t m01_lo = a.d[0] * a.d[1], m01_hi = __umul64hi(a.d[0], a.d[1]);
    uint64_t m02_lo = a.d[0] * a.d[2], m02_hi = __umul64hi(a.d[0], a.d[2]);
    uint64_t m03_lo = a.d[0] * a.d[3], m03_hi = __umul64hi(a.d[0], a.d[3]);
    uint64_t m12_lo = a.d[1] * a.d[2], m12_hi = __umul64hi(a.d[1], a.d[2]);
    uint64_t m13_lo = a.d[1] * a.d[3], m13_hi = __umul64hi(a.d[1], a.d[3]);
    uint64_t m23_lo = a.d[2] * a.d[3], m23_hi = __umul64hi(a.d[2], a.d[3]);

    // Accumulate cross terms into positions, then double, then add diagonals
    // Cross terms land at: m_ij at positions [i+j, i+j+1]
    // pos 1: m01_lo
    // pos 2: m01_hi + m02_lo
    // pos 3: m02_hi + m03_lo + m12_lo
    // pos 4: m03_hi + m12_hi + m13_lo
    // pos 5: m13_hi + m23_lo
    // pos 6: m23_hi

    uint64_t c1 = m01_lo;
    uint64_t c2_lo = m02_lo; uint64_t c2_hi = 0;
    c2_lo += m01_hi; c2_hi += (c2_lo < m01_hi) ? 1ULL : 0ULL;

    uint64_t c3_lo = m12_lo; uint64_t c3_hi = 0;
    c3_lo += m02_hi; c3_hi += (c3_lo < m02_hi) ? 1ULL : 0ULL;
    c3_lo += m03_lo; c3_hi += (c3_lo < m03_lo) ? 1ULL : 0ULL;

    uint64_t c4_lo = m13_lo; uint64_t c4_hi = 0;
    c4_lo += m03_hi; c4_hi += (c4_lo < m03_hi) ? 1ULL : 0ULL;
    c4_lo += m12_hi; c4_hi += (c4_lo < m12_hi) ? 1ULL : 0ULL;

    uint64_t c5_lo = m23_lo; uint64_t c5_hi = 0;
    c5_lo += m13_hi; c5_hi += (c5_lo < m13_hi) ? 1ULL : 0ULL;

    uint64_t c6 = m23_hi;

    // Double cross terms (shift left by 1 bit across the whole thing)
    uint64_t c7_carry = c6 >> 63;
    c6 = (c6 << 1) | (c5_hi >> 63);
    c5_hi = (c5_hi << 1) | (c5_lo >> 63);
    c5_lo = (c5_lo << 1) | (c4_hi >> 63);
    c4_hi = (c4_hi << 1) | (c4_lo >> 63);
    c4_lo = (c4_lo << 1) | (c3_hi >> 63);
    c3_hi = (c3_hi << 1) | (c3_lo >> 63);
    c3_lo = (c3_lo << 1) | (c2_hi >> 63);
    c2_hi = (c2_hi << 1) | (c2_lo >> 63);
    c2_lo = (c2_lo << 1) | (c1 >> 63);
    c1 = c1 << 1;

    // Now add diagonal terms
    // r[0] = d0_lo
    r[0] = d0_lo;

    // r[1] = c1 + d0_hi
    uint64_t s = c1 + d0_hi;
    uint64_t carry = (s < c1) ? 1ULL : 0ULL;
    r[1] = s;

    // r[2] = c2_lo + d1_lo + carry
    s = c2_lo + d1_lo;
    uint64_t cc = (s < c2_lo) ? 1ULL : 0ULL;
    s += carry; cc += (s < carry) ? 1ULL : 0ULL;
    r[2] = s;
    carry = c2_hi + cc;

    // r[3] = c3_lo + d1_hi + carry
    s = c3_lo + d1_hi;
    cc = (s < c3_lo) ? 1ULL : 0ULL;
    s += carry; cc += (s < carry) ? 1ULL : 0ULL;
    r[3] = s;
    carry = c3_hi + cc;

    // r[4] = c4_lo + d2_lo + carry
    s = c4_lo + d2_lo;
    cc = (s < c4_lo) ? 1ULL : 0ULL;
    s += carry; cc += (s < carry) ? 1ULL : 0ULL;
    r[4] = s;
    carry = c4_hi + cc;

    // r[5] = c5_lo + d2_hi + carry
    s = c5_lo + d2_hi;
    cc = (s < c5_lo) ? 1ULL : 0ULL;
    s += carry; cc += (s < carry) ? 1ULL : 0ULL;
    r[5] = s;
    carry = c5_hi + cc;

    // r[6] = c6 + d3_lo + carry
    s = c6 + d3_lo;
    cc = (s < c6) ? 1ULL : 0ULL;
    s += carry; cc += (s < carry) ? 1ULL : 0ULL;
    r[6] = s;
    carry = c7_carry + cc;

    // r[7] = d3_hi + carry
    r[7] = d3_hi + carry;
#else
    // CPU path: exploit __uint128_t for simplicity
    // Compute off-diagonal sum (will be doubled)
    __uint128_t cross[8] = {0};
    for (int i = 0; i < 4; i++) {
        __uint128_t carry128 = 0;
        for (int j = i + 1; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a.d[i] * a.d[j] + cross[i+j] + carry128;
            cross[i+j] = (uint64_t)prod;
            carry128 = prod >> 64;
        }
        cross[i+4] = carry128;
    }

    // Double the cross terms
    __uint128_t carry128 = 0;
    for (int i = 0; i < 8; i++) {
        __uint128_t val = (cross[i] << 1) | carry128;
        cross[i] = (uint64_t)val;
        carry128 = val >> 64;
    }

    // Add diagonal terms
    carry128 = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t diag = (__uint128_t)a.d[i] * a.d[i];
        __uint128_t s = cross[2*i] + (uint64_t)diag + carry128;
        r[2*i] = (uint64_t)s;
        carry128 = s >> 64;

        s = cross[2*i+1] + (uint64_t)(diag >> 64) + carry128;
        r[2*i+1] = (uint64_t)s;
        carry128 = s >> 64;
    }
#endif
}

// Get bit at position
HOST_DEVICE inline int u256_get_bit(const uint256_t &a, int pos) {
    return (a.d[pos / 64] >> (pos % 64)) & 1;
}

// Set from hex string (host only)
inline void u256_from_hex(uint256_t &r, const char *hex) {
    r = {0, 0, 0, 0};
    int len = strlen(hex);
    for (int i = 0; i < len && i < 64; i++) {
        int pos = len - 1 - i;
        char c = hex[pos];
        uint64_t nib;
        if (c >= '0' && c <= '9') nib = c - '0';
        else if (c >= 'a' && c <= 'f') nib = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') nib = c - 'A' + 10;
        else continue;
        r.d[i / 16] |= nib << ((i % 16) * 4);
    }
}

// Print as hex (host only)
inline void u256_to_hex(char *buf, const uint256_t &a) {
    sprintf(buf, "%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3], (unsigned long long)a.d[2],
        (unsigned long long)a.d[1], (unsigned long long)a.d[0]);
}

// Add uint64_t to uint256_t
HOST_DEVICE inline uint64_t u256_add64(uint256_t &r, const uint256_t &a, uint64_t b) {
#ifdef __CUDA_ARCH__
    uint64_t sum = a.d[0] + b;
    r.d[0] = sum;
    uint64_t carry = (sum < a.d[0]) ? 1ULL : 0ULL;
    for (int i = 1; i < 4; i++) {
        sum = a.d[i] + carry;
        r.d[i] = sum;
        carry = (sum < a.d[i]) ? 1ULL : 0ULL;
    }
    return carry;
#else
    __uint128_t s = (__uint128_t)a.d[0] + b;
    r.d[0] = (uint64_t)s;
    uint64_t carry = (uint64_t)(s >> 64);
    for (int i = 1; i < 4; i++) {
        s = (__uint128_t)a.d[i] + carry;
        r.d[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
    return carry;
#endif
}

#endif // INT256_H
