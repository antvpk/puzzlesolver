
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
    uint64_t d[4]; 
};

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

    return borrow & 1;
#else
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t s = (__uint128_t)a.d[i] - b.d[i] - borrow;
        r.d[i] = (uint64_t)s;
        borrow = (s >> 127) & 1; 
    }
    return borrow;
#endif
}

HOST_DEVICE inline void u256_rshift1(uint256_t &r, const uint256_t &a) {
    r.d[0] = (a.d[0] >> 1) | (a.d[1] << 63);
    r.d[1] = (a.d[1] >> 1) | (a.d[2] << 63);
    r.d[2] = (a.d[2] >> 1) | (a.d[3] << 63);
    r.d[3] = a.d[3] >> 1;
}

HOST_DEVICE inline uint64_t u256_lshift1(uint256_t &r, const uint256_t &a) {
    uint64_t carry = a.d[3] >> 63;
    r.d[3] = (a.d[3] << 1) | (a.d[2] >> 63);
    r.d[2] = (a.d[2] << 1) | (a.d[1] >> 63);
    r.d[1] = (a.d[1] << 1) | (a.d[0] >> 63);
    r.d[0] = a.d[0] << 1;
    return carry;
}

HOST_DEVICE inline void u256_mul_full(uint64_t r[8], const uint256_t &a, const uint256_t &b) {
    // Portable __uint128_t schoolbook multiply, used on both host and device.
    // nvcc supports __uint128_t in device code. CPU-verified against known
    // secp256k1 test vectors, so GPU and CPU produce bit-identical products.
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
}

HOST_DEVICE inline void u256_sqr_full(uint64_t r[8], const uint256_t &a) {
    // Portable __uint128_t squaring (cross terms doubled + diagonal), used on
    // both host and device. nvcc supports __uint128_t in device code.
    // CPU-verified, so GPU and CPU produce bit-identical results.
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

    __uint128_t carry128 = 0;
    for (int i = 0; i < 8; i++) {
        __uint128_t val = (cross[i] << 1) | carry128;
        cross[i] = (uint64_t)val;
        carry128 = val >> 64;
    }

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
}

HOST_DEVICE inline int u256_get_bit(const uint256_t &a, int pos) {
    return (a.d[pos / 64] >> (pos % 64)) & 1;
}

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

inline void u256_to_hex(char *buf, const uint256_t &a) {
    sprintf(buf, "%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3], (unsigned long long)a.d[2],
        (unsigned long long)a.d[1], (unsigned long long)a.d[0]);
}

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

#endif 
