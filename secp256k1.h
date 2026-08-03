/*
 * secp256k1.h - secp256k1 elliptic curve operations
 * With endomorphism support (lambda/beta) for 3x speedup
 * Jacobian coordinates for fast point operations
 */
#ifndef SECP256K1_H
#define SECP256K1_H

#include "int256.h"

// secp256k1 field prime: p = 2^256 - 2^32 - 977
#define SECP256K1_P ((uint256_t){0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL})

// secp256k1 order: n
#define SECP256K1_N ((uint256_t){0xD0364141ULL, 0xAF48A03BBFD25E8CULL, 0xFFFFFFFEBAAEDCE6ULL, 0xFFFFFFFFFFFFFFFFULL})

// Generator point G (affine)
#define SECP256K1_GX ((uint256_t){0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL})
#define SECP256K1_GY ((uint256_t){0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL})

// Endomorphism constants
// beta: cube root of 1 mod p (beta^3 = 1 mod p)
#define SECP256K1_BETA ((uint256_t){0xC1396C28719501EEULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL})

// lambda: cube root of 1 mod n (lambda^3 = 1 mod n)
// lambda * G = (beta*Gx, Gy)
#define SECP256K1_LAMBDA ((uint256_t){0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL, 0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL})

// Point in affine coordinates
struct point_t {
    uint256_t x, y;
    bool infinity;
};

// Point in Jacobian coordinates: (X, Y, Z) represents (X/Z^2, Y/Z^3)
struct jpoint_t {
    uint256_t X, Y, Z;
};

// ============= Field arithmetic mod p =============

// Reduce mod p using the special structure p = 2^256 - 2^32 - 977
HOST_DEVICE inline void fp_reduce(uint256_t &r, const uint64_t t[8]) {
    // Fast reduction for secp256k1
    // p = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
    // = 2^256 - 0x1000003D1
    // So r = t mod p = t_low + t_high * 0x1000003D1 (mod p)

    uint64_t c = 0x1000003D1ULL;

#ifdef __CUDA_ARCH__
    // GPU path: no __uint128_t, use __umul64hi intrinsic
    // carry = t[4] * c + t[0]
    uint64_t lo = t[4] * c;
    uint64_t hi = __umul64hi(t[4], c);
    uint64_t sum = lo + t[0];
    hi += (sum < lo) ? 1ULL : 0ULL;
    r.d[0] = sum;
    uint64_t carry_lo = hi;

    // carry + t[5] * c + t[1]
    lo = t[5] * c;
    hi = __umul64hi(t[5], c);
    sum = lo + t[1];
    hi += (sum < lo) ? 1ULL : 0ULL;
    sum += carry_lo;
    hi += (sum < carry_lo) ? 1ULL : 0ULL;
    r.d[1] = sum;
    carry_lo = hi;

    // carry + t[6] * c + t[2]
    lo = t[6] * c;
    hi = __umul64hi(t[6], c);
    sum = lo + t[2];
    hi += (sum < lo) ? 1ULL : 0ULL;
    sum += carry_lo;
    hi += (sum < carry_lo) ? 1ULL : 0ULL;
    r.d[2] = sum;
    carry_lo = hi;

    // carry + t[7] * c + t[3]
    lo = t[7] * c;
    hi = __umul64hi(t[7], c);
    sum = lo + t[3];
    hi += (sum < lo) ? 1ULL : 0ULL;
    sum += carry_lo;
    hi += (sum < carry_lo) ? 1ULL : 0ULL;
    r.d[3] = sum;
    uint64_t overflow = hi;

    // carry might still be nonzero, multiply by c again
    if (overflow) {
        lo = overflow * c;
        hi = __umul64hi(overflow, c);
        sum = r.d[0] + lo;
        uint64_t c2 = hi + ((sum < r.d[0]) ? 1ULL : 0ULL);
        r.d[0] = sum;
        for (int i = 1; i < 4 && c2; i++) {
            sum = r.d[i] + c2;
            c2 = (sum < r.d[i]) ? 1ULL : 0ULL;
            r.d[i] = sum;
        }
    }
#else
    // CPU path: use __uint128_t for efficiency
    __uint128_t carry = 0;

    // Multiply high part by c and add to low part
    carry = (__uint128_t)t[4] * c + t[0];
    r.d[0] = (uint64_t)carry;
    carry >>= 64;

    carry += (__uint128_t)t[5] * c + t[1];
    r.d[1] = (uint64_t)carry;
    carry >>= 64;

    carry += (__uint128_t)t[6] * c + t[2];
    r.d[2] = (uint64_t)carry;
    carry >>= 64;

    carry += (__uint128_t)t[7] * c + t[3];
    r.d[3] = (uint64_t)carry;
    carry >>= 64;

    // carry might still be nonzero, multiply by c again
    uint64_t overflow = (uint64_t)carry;
    if (overflow) {
        __uint128_t s = (__uint128_t)overflow * c + r.d[0];
        r.d[0] = (uint64_t)s;
        uint64_t c2 = (uint64_t)(s >> 64);
        for (int i = 1; i < 4 && c2; i++) {
            s = (__uint128_t)r.d[i] + c2;
            r.d[i] = (uint64_t)s;
            c2 = (uint64_t)(s >> 64);
        }
    }
#endif

    // Final reduction: if r >= p, subtract p
    if (u256_cmp(r, SECP256K1_P) >= 0) {
        u256_sub(r, r, SECP256K1_P);
    }
}

// Modular multiplication: r = a * b mod p
HOST_DEVICE inline void fp_mul(uint256_t &r, const uint256_t &a, const uint256_t &b) {
    uint64_t t[8];
    u256_mul_full(t, a, b);
    fp_reduce(r, t);
}

// Modular squaring: r = a^2 mod p
HOST_DEVICE inline void fp_sqr(uint256_t &r, const uint256_t &a) {
    uint64_t t[8];
    u256_sqr_full(t, a);
    fp_reduce(r, t);
}

// Modular addition: r = (a + b) mod p
HOST_DEVICE inline void fp_add(uint256_t &r, const uint256_t &a, const uint256_t &b) {
    uint64_t carry = u256_add(r, a, b);
    if (carry || u256_cmp(r, SECP256K1_P) >= 0) {
        u256_sub(r, r, SECP256K1_P);
    }
}

// Modular subtraction: r = (a - b) mod p
HOST_DEVICE inline void fp_sub(uint256_t &r, const uint256_t &a, const uint256_t &b) {
    uint64_t borrow = u256_sub(r, a, b);
    if (borrow) {
        u256_add(r, r, SECP256K1_P);
    }
}

// Modular negation: r = (-a) mod p
HOST_DEVICE inline void fp_neg(uint256_t &r, const uint256_t &a) {
    if (u256_is_zero(a)) {
        r = {0, 0, 0, 0};
    } else {
        u256_sub(r, SECP256K1_P, a);
    }
}

// Modular inverse using Fermat's little theorem: a^(p-2) mod p
// Optimized addition chain for secp256k1's prime
// p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// ~259 field multiplications instead of ~384 with naive binary method
HOST_DEVICE inline void fp_inv(uint256_t &r, const uint256_t &a) {
    uint256_t x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    // x2 = a^(2^2-1) = a^3
    fp_sqr(x2, a);
    fp_mul(x2, x2, a);

    // x3 = a^(2^3-1) = a^7
    fp_sqr(x3, x2);
    fp_mul(x3, x3, a);

    // x6 = a^(2^6-1)
    fp_sqr(x6, x3);
    for (int i = 1; i < 3; i++) fp_sqr(x6, x6);
    fp_mul(x6, x6, x3);

    // x9 = a^(2^9-1)
    fp_sqr(x9, x6);
    for (int i = 1; i < 3; i++) fp_sqr(x9, x9);
    fp_mul(x9, x9, x3);

    // x11 = a^(2^11-1)
    fp_sqr(x11, x9);
    for (int i = 1; i < 2; i++) fp_sqr(x11, x11);
    fp_mul(x11, x11, x2);

    // x22 = a^(2^22-1)
    fp_sqr(x22, x11);
    for (int i = 1; i < 11; i++) fp_sqr(x22, x22);
    fp_mul(x22, x22, x11);

    // x44 = a^(2^44-1)
    fp_sqr(x44, x22);
    for (int i = 1; i < 22; i++) fp_sqr(x44, x44);
    fp_mul(x44, x44, x22);

    // x88 = a^(2^88-1)
    fp_sqr(x88, x44);
    for (int i = 1; i < 44; i++) fp_sqr(x88, x88);
    fp_mul(x88, x88, x44);

    // x176 = a^(2^176-1)
    fp_sqr(x176, x88);
    for (int i = 1; i < 88; i++) fp_sqr(x176, x176);
    fp_mul(x176, x176, x88);

    // x220 = a^(2^220-1)
    fp_sqr(x220, x176);
    for (int i = 1; i < 44; i++) fp_sqr(x220, x220);
    fp_mul(x220, x220, x44);

    // x223 = a^(2^223-1)
    fp_sqr(x223, x220);
    for (int i = 1; i < 3; i++) fp_sqr(x223, x223);
    fp_mul(x223, x223, x3);

    // The final exponent is 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1 - 2
    // = p - 2
    // t1 = x223 << 23
    t1 = x223;
    for (int i = 0; i < 23; i++) fp_sqr(t1, t1);
    // t1 *= x22
    fp_mul(t1, t1, x22);
    // t1 <<= 5
    for (int i = 0; i < 5; i++) fp_sqr(t1, t1);
    // t1 *= a
    fp_mul(t1, t1, a);
    // t1 <<= 3
    for (int i = 0; i < 3; i++) fp_sqr(t1, t1);
    // t1 *= x2
    fp_mul(t1, t1, x2);
    // t1 <<= 2
    fp_sqr(t1, t1);
    fp_sqr(t1, t1);
    // Final multiply by a
    fp_mul(r, t1, a);
}

// ============= Point operations (Jacobian) =============

// Convert affine to Jacobian
HOST_DEVICE inline jpoint_t affine_to_jacobian(const point_t &p) {
    if (p.infinity) {
        jpoint_t r;
        r.X = {0, 0, 0, 0};
        r.Y = {1, 0, 0, 0};
        r.Z = {0, 0, 0, 0};
        return r;
    }
    return {p.x, p.y, {1, 0, 0, 0}};
}

// Convert Jacobian to affine
HOST_DEVICE inline point_t jacobian_to_affine(const jpoint_t &jp) {
    if (u256_is_zero(jp.Z)) {
        return {{0, 0, 0, 0}, {0, 0, 0, 0}, true};
    }
    point_t r;
    r.infinity = false;

    uint256_t zinv, zinv2, zinv3;
    fp_inv(zinv, jp.Z);
    fp_sqr(zinv2, zinv);
    fp_mul(zinv3, zinv2, zinv);

    fp_mul(r.x, jp.X, zinv2);
    fp_mul(r.y, jp.Y, zinv3);
    return r;
}

// Jacobian point doubling: R = 2*P
HOST_DEVICE inline void jpoint_double(jpoint_t &R, const jpoint_t &P) {
    if (u256_is_zero(P.Z)) {
        R.X = {0, 0, 0, 0};
        R.Y = {1, 0, 0, 0};
        R.Z = {0, 0, 0, 0};
        return;
    }

    uint256_t A, B, C, D, E, F;

    // A = Y^2
    fp_sqr(A, P.Y);
    // B = 4*X*A
    fp_mul(B, P.X, A);
    fp_add(B, B, B);
    fp_add(B, B, B);
    // C = 8*A^2
    fp_sqr(C, A);
    fp_add(C, C, C);
    fp_add(C, C, C);
    fp_add(C, C, C);
    // D = 3*X^2 (a=0 for secp256k1, so no +a*Z^4 term)
    fp_sqr(D, P.X);
    fp_add(E, D, D);
    fp_add(D, E, D);

    // X3 = D^2 - 2*B
    fp_sqr(R.X, D);
    fp_sub(R.X, R.X, B);
    fp_sub(R.X, R.X, B);

    // Z3 = 2*Y*Z (must compute BEFORE overwriting R.Y, in case R aliases P)
    fp_mul(R.Z, P.Y, P.Z);
    fp_add(R.Z, R.Z, R.Z);

    // Y3 = D*(B - X3) - C
    fp_sub(F, B, R.X);
    fp_mul(R.Y, D, F);
    fp_sub(R.Y, R.Y, C);
}

// Jacobian point addition: R = P + Q (P in Jacobian, Q in affine with Z=1)
// This is faster than general addition
HOST_DEVICE inline void jpoint_add_affine(jpoint_t &R, const jpoint_t &P, const point_t &Q) {
    if (Q.infinity) {
        R = P;
        return;
    }
    if (u256_is_zero(P.Z)) {
        R = affine_to_jacobian(Q);
        return;
    }

    uint256_t Z1Z1, U2, S2, H, HH, I, J, rr, V;

    // Z1Z1 = Z1^2
    fp_sqr(Z1Z1, P.Z);
    // U2 = X2 * Z1Z1
    fp_mul(U2, Q.x, Z1Z1);
    // S2 = Y2 * Z1 * Z1Z1
    fp_mul(S2, Q.y, P.Z);
    fp_mul(S2, S2, Z1Z1);

    // H = U2 - X1
    fp_sub(H, U2, P.X);
    // rr = S2 - Y1
    fp_sub(rr, S2, P.Y);

    // Check if P == Q (H == 0 && rr == 0) -> double
    if (u256_is_zero(H)) {
        if (u256_is_zero(rr)) {
            jpoint_double(R, P);
            return;
        }
        // P == -Q, return infinity
        R.X = {0, 0, 0, 0};
        R.Y = {1, 0, 0, 0};
        R.Z = {0, 0, 0, 0};
        return;
    }

    // HH = H^2
    fp_sqr(HH, H);
    // I = 4*HH
    fp_add(I, HH, HH);
    fp_add(I, I, I);
    // J = H * I
    fp_mul(J, H, I);
    // rr = 2*rr
    fp_add(rr, rr, rr);
    // V = X1 * I
    fp_mul(V, P.X, I);

    // Save P.Y and P.Z before overwriting R (in case R aliases P)
    uint256_t P_Y = P.Y;
    uint256_t P_Z = P.Z;

    // X3 = rr^2 - J - 2*V
    fp_sqr(R.X, rr);
    fp_sub(R.X, R.X, J);
    fp_sub(R.X, R.X, V);
    fp_sub(R.X, R.X, V);

    // Y3 = rr*(V - X3) - 2*Y1*J
    fp_sub(V, V, R.X);
    fp_mul(R.Y, rr, V);
    fp_mul(S2, P_Y, J);
    fp_sub(R.Y, R.Y, S2);
    fp_sub(R.Y, R.Y, S2);

    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    fp_add(R.Z, P_Z, H);
    fp_sqr(R.Z, R.Z);
    fp_sub(R.Z, R.Z, Z1Z1);
    fp_sub(R.Z, R.Z, HH);
}

// General Jacobian addition: R = P + Q (both Jacobian)
HOST_DEVICE inline void jpoint_add(jpoint_t &R, const jpoint_t &P, const jpoint_t &Q) {
    if (u256_is_zero(P.Z)) { R = Q; return; }
    if (u256_is_zero(Q.Z)) { R = P; return; }

    uint256_t Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V;

    fp_sqr(Z1Z1, P.Z);
    fp_sqr(Z2Z2, Q.Z);
    fp_mul(U1, P.X, Z2Z2);
    fp_mul(U2, Q.X, Z1Z1);
    fp_mul(S1, P.Y, Q.Z);
    fp_mul(S1, S1, Z2Z2);
    fp_mul(S2, Q.Y, P.Z);
    fp_mul(S2, S2, Z1Z1);

    fp_sub(H, U2, U1);
    fp_sub(rr, S2, S1);

    if (u256_is_zero(H)) {
        if (u256_is_zero(rr)) {
            jpoint_double(R, P);
            return;
        }
        R.X = {0, 0, 0, 0};
        R.Y = {1, 0, 0, 0};
        R.Z = {0, 0, 0, 0};
        return;
    }

    fp_add(I, H, H);
    fp_sqr(I, I);
    fp_mul(J, H, I);
    fp_add(rr, rr, rr);
    fp_mul(V, U1, I);

    fp_sqr(R.X, rr);
    fp_sub(R.X, R.X, J);
    fp_sub(R.X, R.X, V);
    fp_sub(R.X, R.X, V);

    fp_sub(V, V, R.X);
    fp_mul(R.Y, rr, V);
    fp_mul(S1, S1, J);
    fp_sub(R.Y, R.Y, S1);
    fp_sub(R.Y, R.Y, S1);

    fp_add(R.Z, P.Z, Q.Z);
    fp_sqr(R.Z, R.Z);
    fp_sub(R.Z, R.Z, Z1Z1);
    fp_sub(R.Z, R.Z, Z2Z2);
    fp_mul(R.Z, R.Z, H);
}

// Scalar multiplication: R = k * P (using double-and-add with affine P)
HOST_DEVICE inline point_t ec_mul(const uint256_t &k, const point_t &P) {
    jpoint_t R;
    R.X = {0, 0, 0, 0};
    R.Y = {1, 0, 0, 0};
    R.Z = {0, 0, 0, 0};

    // Find highest bit
    int highest = 255;
    while (highest >= 0 && u256_get_bit(k, highest) == 0) highest--;
    if (highest < 0) return {{0, 0, 0, 0}, {0, 0, 0, 0}, true};

    for (int i = highest; i >= 0; i--) {
        jpoint_double(R, R);
        if (u256_get_bit(k, i)) {
            jpoint_add_affine(R, R, P);
        }
    }

    return jacobian_to_affine(R);
}

// Endomorphism: given P=(x,y), returns (beta*x, y) which equals lambda*P
HOST_DEVICE inline point_t endomorphism(const point_t &P) {
    if (P.infinity) return P;
    point_t R;
    R.infinity = false;
    fp_mul(R.x, P.x, SECP256K1_BETA);
    R.y = P.y;
    return R;
}

// Apply endomorphism: given P=(x,y), returns (beta*x, y) = lambda*P
// and (beta^2*x, y) = lambda^2*P
// This gives us 3 points to check for each scalar multiplication!

#endif // SECP256K1_H
