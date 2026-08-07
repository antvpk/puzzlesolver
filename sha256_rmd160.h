
#ifndef SHA256_RMD160_H
#define SHA256_RMD160_H

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

HOST_DEVICE inline uint32_t sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

#define SHA256_CH(x,y,z) ((x & y) ^ (~x & z))
#define SHA256_MAJ(x,y,z) ((x & y) ^ (x & z) ^ (y & z))
#define SHA256_S0(x) (sha256_rotr(x,2) ^ sha256_rotr(x,13) ^ sha256_rotr(x,22))
#define SHA256_S1(x) (sha256_rotr(x,6) ^ sha256_rotr(x,11) ^ sha256_rotr(x,25))
#define SHA256_s0(x) (sha256_rotr(x,7) ^ sha256_rotr(x,18) ^ (x >> 3))
#define SHA256_s1(x) (sha256_rotr(x,17) ^ sha256_rotr(x,19) ^ (x >> 10))

#ifdef __CUDACC__
__constant__ const uint32_t d_sha256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
#endif
static const uint32_t h_sha256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#ifdef __CUDA_ARCH__
#define GET_K(i) d_sha256_K[i]
#else
#define GET_K(i) h_sha256_K[i]
#endif

#define SHA256_RND(a,b,c,d,e,f,g,h,i) \
    { uint32_t t1 = h + SHA256_S1(e) + SHA256_CH(e,f,g) + GET_K(i) + W[i]; \
      uint32_t t2 = SHA256_S0(a) + SHA256_MAJ(a,b,c); \
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2; }

// 32-bit endian swap. On device this is a single PRMT instruction.
HOST_DEVICE inline uint32_t bswap32_(uint32_t v) {
#ifdef __CUDA_ARCH__
    return __byte_perm(v, 0, 0x0123);
#else
    return ((v >> 24) & 0x000000FFu) | ((v >> 8) & 0x0000FF00u) |
           ((v <<  8) & 0x00FF0000u) | ((v << 24) & 0xFF000000u);
#endif
}

// Core SHA-256 over a 33-byte message, working entirely in 32-bit words.
//
// Win[0..8] are the message words that vary; W[9..15] are the fixed padding and
// the constant bit-length (264) for a 33-byte input. Output is the 8 state words
// in big-endian value form (out[0] is the first 4 digest bytes, MSB first).
//
// Callers holding the data as words (the GPU search path) use this directly and
// never materialise the 33 input or 32 output bytes.
HOST_DEVICE inline void sha256_33_core(const uint32_t Win[9], uint32_t out[8]) {
    uint32_t W[64];

    W[0] = Win[0]; W[1] = Win[1]; W[2] = Win[2]; W[3] = Win[3]; W[4] = Win[4];
    W[5] = Win[5]; W[6] = Win[6]; W[7] = Win[7]; W[8] = Win[8];
    W[9] = W[10] = W[11] = W[12] = W[13] = W[14] = 0;
    W[15] = 264;

    W[16] = SHA256_s1(W[14]) + SHA256_s0(W[1]) + W[0];
    W[17] = SHA256_s1(W[15]) + SHA256_s0(W[2]) + W[1];
    W[18] = SHA256_s1(W[16]) + SHA256_s0(W[3]) + W[2];
    W[19] = SHA256_s1(W[17]) + SHA256_s0(W[4]) + W[3];
    W[20] = SHA256_s1(W[18]) + SHA256_s0(W[5]) + W[4];
    W[21] = SHA256_s1(W[19]) + SHA256_s0(W[6]) + W[5];
    W[22] = SHA256_s1(W[20]) + W[15] + SHA256_s0(W[7]) + W[6];
    W[23] = SHA256_s1(W[21]) + W[16] + SHA256_s0(W[8]) + W[7];
    W[24] = SHA256_s1(W[22]) + W[17] + SHA256_s0(W[9]) + W[8];
    W[25] = SHA256_s1(W[23]) + W[18] + SHA256_s0(W[10]) + W[9];
    W[26] = SHA256_s1(W[24]) + W[19] + SHA256_s0(W[11]) + W[10];
    W[27] = SHA256_s1(W[25]) + W[20] + SHA256_s0(W[12]) + W[11];
    W[28] = SHA256_s1(W[26]) + W[21] + SHA256_s0(W[13]) + W[12];
    W[29] = SHA256_s1(W[27]) + W[22] + SHA256_s0(W[14]) + W[13];
    W[30] = SHA256_s1(W[28]) + W[23] + SHA256_s0(W[15]) + W[14];
    W[31] = SHA256_s1(W[29]) + W[24] + SHA256_s0(W[16]) + W[15];
    W[32] = SHA256_s1(W[30]) + W[25] + SHA256_s0(W[17]) + W[16];
    W[33] = SHA256_s1(W[31]) + W[26] + SHA256_s0(W[18]) + W[17];
    W[34] = SHA256_s1(W[32]) + W[27] + SHA256_s0(W[19]) + W[18];
    W[35] = SHA256_s1(W[33]) + W[28] + SHA256_s0(W[20]) + W[19];
    W[36] = SHA256_s1(W[34]) + W[29] + SHA256_s0(W[21]) + W[20];
    W[37] = SHA256_s1(W[35]) + W[30] + SHA256_s0(W[22]) + W[21];
    W[38] = SHA256_s1(W[36]) + W[31] + SHA256_s0(W[23]) + W[22];
    W[39] = SHA256_s1(W[37]) + W[32] + SHA256_s0(W[24]) + W[23];
    W[40] = SHA256_s1(W[38]) + W[33] + SHA256_s0(W[25]) + W[24];
    W[41] = SHA256_s1(W[39]) + W[34] + SHA256_s0(W[26]) + W[25];
    W[42] = SHA256_s1(W[40]) + W[35] + SHA256_s0(W[27]) + W[26];
    W[43] = SHA256_s1(W[41]) + W[36] + SHA256_s0(W[28]) + W[27];
    W[44] = SHA256_s1(W[42]) + W[37] + SHA256_s0(W[29]) + W[28];
    W[45] = SHA256_s1(W[43]) + W[38] + SHA256_s0(W[30]) + W[29];
    W[46] = SHA256_s1(W[44]) + W[39] + SHA256_s0(W[31]) + W[30];
    W[47] = SHA256_s1(W[45]) + W[40] + SHA256_s0(W[32]) + W[31];
    W[48] = SHA256_s1(W[46]) + W[41] + SHA256_s0(W[33]) + W[32];
    W[49] = SHA256_s1(W[47]) + W[42] + SHA256_s0(W[34]) + W[33];
    W[50] = SHA256_s1(W[48]) + W[43] + SHA256_s0(W[35]) + W[34];
    W[51] = SHA256_s1(W[49]) + W[44] + SHA256_s0(W[36]) + W[35];
    W[52] = SHA256_s1(W[50]) + W[45] + SHA256_s0(W[37]) + W[36];
    W[53] = SHA256_s1(W[51]) + W[46] + SHA256_s0(W[38]) + W[37];
    W[54] = SHA256_s1(W[52]) + W[47] + SHA256_s0(W[39]) + W[38];
    W[55] = SHA256_s1(W[53]) + W[48] + SHA256_s0(W[40]) + W[39];
    W[56] = SHA256_s1(W[54]) + W[49] + SHA256_s0(W[41]) + W[40];
    W[57] = SHA256_s1(W[55]) + W[50] + SHA256_s0(W[42]) + W[41];
    W[58] = SHA256_s1(W[56]) + W[51] + SHA256_s0(W[43]) + W[42];
    W[59] = SHA256_s1(W[57]) + W[52] + SHA256_s0(W[44]) + W[43];
    W[60] = SHA256_s1(W[58]) + W[53] + SHA256_s0(W[45]) + W[44];
    W[61] = SHA256_s1(W[59]) + W[54] + SHA256_s0(W[46]) + W[45];
    W[62] = SHA256_s1(W[60]) + W[55] + SHA256_s0(W[47]) + W[46];
    W[63] = SHA256_s1(W[61]) + W[56] + SHA256_s0(W[48]) + W[47];

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    SHA256_RND(a,b,c,d,e,f,g,h,0)
    SHA256_RND(a,b,c,d,e,f,g,h,1)
    SHA256_RND(a,b,c,d,e,f,g,h,2)
    SHA256_RND(a,b,c,d,e,f,g,h,3)
    SHA256_RND(a,b,c,d,e,f,g,h,4)
    SHA256_RND(a,b,c,d,e,f,g,h,5)
    SHA256_RND(a,b,c,d,e,f,g,h,6)
    SHA256_RND(a,b,c,d,e,f,g,h,7)
    SHA256_RND(a,b,c,d,e,f,g,h,8)
    SHA256_RND(a,b,c,d,e,f,g,h,9)
    SHA256_RND(a,b,c,d,e,f,g,h,10)
    SHA256_RND(a,b,c,d,e,f,g,h,11)
    SHA256_RND(a,b,c,d,e,f,g,h,12)
    SHA256_RND(a,b,c,d,e,f,g,h,13)
    SHA256_RND(a,b,c,d,e,f,g,h,14)
    SHA256_RND(a,b,c,d,e,f,g,h,15)
    SHA256_RND(a,b,c,d,e,f,g,h,16)
    SHA256_RND(a,b,c,d,e,f,g,h,17)
    SHA256_RND(a,b,c,d,e,f,g,h,18)
    SHA256_RND(a,b,c,d,e,f,g,h,19)
    SHA256_RND(a,b,c,d,e,f,g,h,20)
    SHA256_RND(a,b,c,d,e,f,g,h,21)
    SHA256_RND(a,b,c,d,e,f,g,h,22)
    SHA256_RND(a,b,c,d,e,f,g,h,23)
    SHA256_RND(a,b,c,d,e,f,g,h,24)
    SHA256_RND(a,b,c,d,e,f,g,h,25)
    SHA256_RND(a,b,c,d,e,f,g,h,26)
    SHA256_RND(a,b,c,d,e,f,g,h,27)
    SHA256_RND(a,b,c,d,e,f,g,h,28)
    SHA256_RND(a,b,c,d,e,f,g,h,29)
    SHA256_RND(a,b,c,d,e,f,g,h,30)
    SHA256_RND(a,b,c,d,e,f,g,h,31)
    SHA256_RND(a,b,c,d,e,f,g,h,32)
    SHA256_RND(a,b,c,d,e,f,g,h,33)
    SHA256_RND(a,b,c,d,e,f,g,h,34)
    SHA256_RND(a,b,c,d,e,f,g,h,35)
    SHA256_RND(a,b,c,d,e,f,g,h,36)
    SHA256_RND(a,b,c,d,e,f,g,h,37)
    SHA256_RND(a,b,c,d,e,f,g,h,38)
    SHA256_RND(a,b,c,d,e,f,g,h,39)
    SHA256_RND(a,b,c,d,e,f,g,h,40)
    SHA256_RND(a,b,c,d,e,f,g,h,41)
    SHA256_RND(a,b,c,d,e,f,g,h,42)
    SHA256_RND(a,b,c,d,e,f,g,h,43)
    SHA256_RND(a,b,c,d,e,f,g,h,44)
    SHA256_RND(a,b,c,d,e,f,g,h,45)
    SHA256_RND(a,b,c,d,e,f,g,h,46)
    SHA256_RND(a,b,c,d,e,f,g,h,47)
    SHA256_RND(a,b,c,d,e,f,g,h,48)
    SHA256_RND(a,b,c,d,e,f,g,h,49)
    SHA256_RND(a,b,c,d,e,f,g,h,50)
    SHA256_RND(a,b,c,d,e,f,g,h,51)
    SHA256_RND(a,b,c,d,e,f,g,h,52)
    SHA256_RND(a,b,c,d,e,f,g,h,53)
    SHA256_RND(a,b,c,d,e,f,g,h,54)
    SHA256_RND(a,b,c,d,e,f,g,h,55)
    SHA256_RND(a,b,c,d,e,f,g,h,56)
    SHA256_RND(a,b,c,d,e,f,g,h,57)
    SHA256_RND(a,b,c,d,e,f,g,h,58)
    SHA256_RND(a,b,c,d,e,f,g,h,59)
    SHA256_RND(a,b,c,d,e,f,g,h,60)
    SHA256_RND(a,b,c,d,e,f,g,h,61)
    SHA256_RND(a,b,c,d,e,f,g,h,62)
    SHA256_RND(a,b,c,d,e,f,g,h,63)

    out[0] = a + 0x6a09e667; out[1] = b + 0xbb67ae85;
    out[2] = c + 0x3c6ef372; out[3] = d + 0xa54ff53a;
    out[4] = e + 0x510e527f; out[5] = f + 0x9b05688c;
    out[6] = g + 0x1f83d9ab; out[7] = h + 0x5be0cd19;
}

// Byte-oriented wrapper, kept for the base58 / CPU-worker callers.
HOST_DEVICE inline void sha256_33(const uint8_t *msg, uint8_t *hash) {
    uint32_t Win[9];
    Win[0] = ((uint32_t)msg[0] << 24) | ((uint32_t)msg[1] << 16) | ((uint32_t)msg[2] << 8) | msg[3];
    Win[1] = ((uint32_t)msg[4] << 24) | ((uint32_t)msg[5] << 16) | ((uint32_t)msg[6] << 8) | msg[7];
    Win[2] = ((uint32_t)msg[8] << 24) | ((uint32_t)msg[9] << 16) | ((uint32_t)msg[10] << 8) | msg[11];
    Win[3] = ((uint32_t)msg[12] << 24) | ((uint32_t)msg[13] << 16) | ((uint32_t)msg[14] << 8) | msg[15];
    Win[4] = ((uint32_t)msg[16] << 24) | ((uint32_t)msg[17] << 16) | ((uint32_t)msg[18] << 8) | msg[19];
    Win[5] = ((uint32_t)msg[20] << 24) | ((uint32_t)msg[21] << 16) | ((uint32_t)msg[22] << 8) | msg[23];
    Win[6] = ((uint32_t)msg[24] << 24) | ((uint32_t)msg[25] << 16) | ((uint32_t)msg[26] << 8) | msg[27];
    Win[7] = ((uint32_t)msg[28] << 24) | ((uint32_t)msg[29] << 16) | ((uint32_t)msg[30] << 8) | msg[31];
    Win[8] = ((uint32_t)msg[32] << 24) | 0x00800000;

    uint32_t o[8];
    sha256_33_core(Win, o);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i*4 + 0] = (uint8_t)(o[i] >> 24);
        hash[i*4 + 1] = (uint8_t)(o[i] >> 16);
        hash[i*4 + 2] = (uint8_t)(o[i] >>  8);
        hash[i*4 + 3] = (uint8_t)(o[i]);
    }
}

#define RMD_ROTL(x,n) (((x) << (n)) | ((x) >> (32-(n))))
#define RMD_F(x,y,z) ((x) ^ (y) ^ (z))
#define RMD_G(x,y,z) (((x) & (y)) | (~(x) & (z)))
#define RMD_H(x,y,z) (((x) | ~(y)) ^ (z))
#define RMD_I(x,y,z) (((x) & (z)) | ((y) & ~(z)))
#define RMD_J(x,y,z) ((x) ^ ((y) | ~(z)))

#define RMD_ROUND(f, a, b, c, d, e, x, s, k) \
    { a += f(b, c, d) + x + k; a = RMD_ROTL(a, s) + e; c = RMD_ROTL(c, 10); }

HOST_DEVICE inline void ripemd160_32(const uint8_t *msg, uint8_t *hash) {
    uint32_t X[16];

    for (int i = 0; i < 8; i++) {
        X[i] = ((uint32_t)msg[i*4]) | ((uint32_t)msg[i*4+1] << 8) |
               ((uint32_t)msg[i*4+2] << 16) | ((uint32_t)msg[i*4+3] << 24);
    }
    X[8] = 0x00000080;
    X[9] = X[10] = X[11] = X[12] = X[13] = 0;
    X[14] = 256; 
    X[15] = 0;

    uint32_t al = 0x67452301, bl = 0xefcdab89, cl = 0x98badcfe, dl = 0x10325476, el = 0xc3d2e1f0;
    uint32_t ar = al, br = bl, cr = cl, dr = dl, er = el;

    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[ 0], 11, 0x00000000);
    RMD_ROUND(RMD_F, el, al, bl, cl, dl, X[ 1], 14, 0x00000000);
    RMD_ROUND(RMD_F, dl, el, al, bl, cl, X[ 2], 15, 0x00000000);
    RMD_ROUND(RMD_F, cl, dl, el, al, bl, X[ 3], 12, 0x00000000);
    RMD_ROUND(RMD_F, bl, cl, dl, el, al, X[ 4],  5, 0x00000000);
    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[ 5],  8, 0x00000000);
    RMD_ROUND(RMD_F, el, al, bl, cl, dl, X[ 6],  7, 0x00000000);
    RMD_ROUND(RMD_F, dl, el, al, bl, cl, X[ 7],  9, 0x00000000);
    RMD_ROUND(RMD_F, cl, dl, el, al, bl, X[ 8], 11, 0x00000000);
    RMD_ROUND(RMD_F, bl, cl, dl, el, al, X[ 9], 13, 0x00000000);
    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[10], 14, 0x00000000);
    RMD_ROUND(RMD_F, el, al, bl, cl, dl, X[11], 15, 0x00000000);
    RMD_ROUND(RMD_F, dl, el, al, bl, cl, X[12],  6, 0x00000000);
    RMD_ROUND(RMD_F, cl, dl, el, al, bl, X[13],  7, 0x00000000);
    RMD_ROUND(RMD_F, bl, cl, dl, el, al, X[14],  9, 0x00000000);
    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[15],  8, 0x00000000);

    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 7],  7, 0x5a827999);
    RMD_ROUND(RMD_G, dl, el, al, bl, cl, X[ 4],  6, 0x5a827999);
    RMD_ROUND(RMD_G, cl, dl, el, al, bl, X[13],  8, 0x5a827999);
    RMD_ROUND(RMD_G, bl, cl, dl, el, al, X[ 1], 13, 0x5a827999);
    RMD_ROUND(RMD_G, al, bl, cl, dl, el, X[10], 11, 0x5a827999);
    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 6],  9, 0x5a827999);
    RMD_ROUND(RMD_G, dl, el, al, bl, cl, X[15],  7, 0x5a827999);
    RMD_ROUND(RMD_G, cl, dl, el, al, bl, X[ 3], 15, 0x5a827999);
    RMD_ROUND(RMD_G, bl, cl, dl, el, al, X[12],  7, 0x5a827999);
    RMD_ROUND(RMD_G, al, bl, cl, dl, el, X[ 0], 12, 0x5a827999);
    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 9], 15, 0x5a827999);
    RMD_ROUND(RMD_G, dl, el, al, bl, cl, X[ 5],  9, 0x5a827999);
    RMD_ROUND(RMD_G, cl, dl, el, al, bl, X[ 2], 11, 0x5a827999);
    RMD_ROUND(RMD_G, bl, cl, dl, el, al, X[14],  7, 0x5a827999);
    RMD_ROUND(RMD_G, al, bl, cl, dl, el, X[11], 13, 0x5a827999);
    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 8], 12, 0x5a827999);

    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[ 3], 11, 0x6ed9eba1);
    RMD_ROUND(RMD_H, cl, dl, el, al, bl, X[10], 13, 0x6ed9eba1);
    RMD_ROUND(RMD_H, bl, cl, dl, el, al, X[14],  6, 0x6ed9eba1);
    RMD_ROUND(RMD_H, al, bl, cl, dl, el, X[ 4],  7, 0x6ed9eba1);
    RMD_ROUND(RMD_H, el, al, bl, cl, dl, X[ 9], 14, 0x6ed9eba1);
    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[15],  9, 0x6ed9eba1);
    RMD_ROUND(RMD_H, cl, dl, el, al, bl, X[ 8], 13, 0x6ed9eba1);
    RMD_ROUND(RMD_H, bl, cl, dl, el, al, X[ 1], 15, 0x6ed9eba1);
    RMD_ROUND(RMD_H, al, bl, cl, dl, el, X[ 2], 14, 0x6ed9eba1);
    RMD_ROUND(RMD_H, el, al, bl, cl, dl, X[ 7],  8, 0x6ed9eba1);
    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[ 0], 13, 0x6ed9eba1);
    RMD_ROUND(RMD_H, cl, dl, el, al, bl, X[ 6],  6, 0x6ed9eba1);
    RMD_ROUND(RMD_H, bl, cl, dl, el, al, X[13],  5, 0x6ed9eba1);
    RMD_ROUND(RMD_H, al, bl, cl, dl, el, X[11], 12, 0x6ed9eba1);
    RMD_ROUND(RMD_H, el, al, bl, cl, dl, X[ 5],  7, 0x6ed9eba1);
    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[12],  5, 0x6ed9eba1);

    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 1], 11, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, bl, cl, dl, el, al, X[ 9], 12, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, al, bl, cl, dl, el, X[11], 14, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, el, al, bl, cl, dl, X[10], 15, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, dl, el, al, bl, cl, X[ 0], 14, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 8], 15, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, bl, cl, dl, el, al, X[12],  9, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, al, bl, cl, dl, el, X[ 4],  8, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, el, al, bl, cl, dl, X[13],  9, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, dl, el, al, bl, cl, X[ 3], 14, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 7],  5, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, bl, cl, dl, el, al, X[15],  6, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, al, bl, cl, dl, el, X[14],  8, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, el, al, bl, cl, dl, X[ 5],  6, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, dl, el, al, bl, cl, X[ 6],  5, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 2], 12, 0x8f1bbcdc);

    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[ 4],  9, 0xa953fd4e);
    RMD_ROUND(RMD_J, al, bl, cl, dl, el, X[ 0], 15, 0xa953fd4e);
    RMD_ROUND(RMD_J, el, al, bl, cl, dl, X[ 5],  5, 0xa953fd4e);
    RMD_ROUND(RMD_J, dl, el, al, bl, cl, X[ 9], 11, 0xa953fd4e);
    RMD_ROUND(RMD_J, cl, dl, el, al, bl, X[ 7],  6, 0xa953fd4e);
    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[12],  8, 0xa953fd4e);
    RMD_ROUND(RMD_J, al, bl, cl, dl, el, X[ 2], 13, 0xa953fd4e);
    RMD_ROUND(RMD_J, el, al, bl, cl, dl, X[10], 12, 0xa953fd4e);
    RMD_ROUND(RMD_J, dl, el, al, bl, cl, X[14],  5, 0xa953fd4e);
    RMD_ROUND(RMD_J, cl, dl, el, al, bl, X[ 1], 12, 0xa953fd4e);
    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[ 3], 13, 0xa953fd4e);
    RMD_ROUND(RMD_J, al, bl, cl, dl, el, X[ 8], 14, 0xa953fd4e);
    RMD_ROUND(RMD_J, el, al, bl, cl, dl, X[11], 11, 0xa953fd4e);
    RMD_ROUND(RMD_J, dl, el, al, bl, cl, X[ 6],  8, 0xa953fd4e);
    RMD_ROUND(RMD_J, cl, dl, el, al, bl, X[15],  5, 0xa953fd4e);
    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[13],  6, 0xa953fd4e);

    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[ 5],  8, 0x50a28be6);
    RMD_ROUND(RMD_J, er, ar, br, cr, dr, X[14],  9, 0x50a28be6);
    RMD_ROUND(RMD_J, dr, er, ar, br, cr, X[ 7],  9, 0x50a28be6);
    RMD_ROUND(RMD_J, cr, dr, er, ar, br, X[ 0], 11, 0x50a28be6);
    RMD_ROUND(RMD_J, br, cr, dr, er, ar, X[ 9], 13, 0x50a28be6);
    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[ 2], 15, 0x50a28be6);
    RMD_ROUND(RMD_J, er, ar, br, cr, dr, X[11], 15, 0x50a28be6);
    RMD_ROUND(RMD_J, dr, er, ar, br, cr, X[ 4],  5, 0x50a28be6);
    RMD_ROUND(RMD_J, cr, dr, er, ar, br, X[13],  7, 0x50a28be6);
    RMD_ROUND(RMD_J, br, cr, dr, er, ar, X[ 6],  7, 0x50a28be6);
    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[15],  8, 0x50a28be6);
    RMD_ROUND(RMD_J, er, ar, br, cr, dr, X[ 8], 11, 0x50a28be6);
    RMD_ROUND(RMD_J, dr, er, ar, br, cr, X[ 1], 14, 0x50a28be6);
    RMD_ROUND(RMD_J, cr, dr, er, ar, br, X[10], 14, 0x50a28be6);
    RMD_ROUND(RMD_J, br, cr, dr, er, ar, X[ 3], 12, 0x50a28be6);
    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[12],  6, 0x50a28be6);

    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[ 6],  9, 0x5c4dd124);
    RMD_ROUND(RMD_I, dr, er, ar, br, cr, X[11], 13, 0x5c4dd124);
    RMD_ROUND(RMD_I, cr, dr, er, ar, br, X[ 3], 15, 0x5c4dd124);
    RMD_ROUND(RMD_I, br, cr, dr, er, ar, X[ 7],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, ar, br, cr, dr, er, X[ 0], 12, 0x5c4dd124);
    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[13],  8, 0x5c4dd124);
    RMD_ROUND(RMD_I, dr, er, ar, br, cr, X[ 5],  9, 0x5c4dd124);
    RMD_ROUND(RMD_I, cr, dr, er, ar, br, X[10], 11, 0x5c4dd124);
    RMD_ROUND(RMD_I, br, cr, dr, er, ar, X[14],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, ar, br, cr, dr, er, X[15],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[ 8], 12, 0x5c4dd124);
    RMD_ROUND(RMD_I, dr, er, ar, br, cr, X[12],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, cr, dr, er, ar, br, X[ 4],  6, 0x5c4dd124);
    RMD_ROUND(RMD_I, br, cr, dr, er, ar, X[ 9], 15, 0x5c4dd124);
    RMD_ROUND(RMD_I, ar, br, cr, dr, er, X[ 1], 13, 0x5c4dd124);
    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[ 2], 11, 0x5c4dd124);

    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[15],  9, 0x6d703ef3);
    RMD_ROUND(RMD_H, cr, dr, er, ar, br, X[ 5],  7, 0x6d703ef3);
    RMD_ROUND(RMD_H, br, cr, dr, er, ar, X[ 1], 15, 0x6d703ef3);
    RMD_ROUND(RMD_H, ar, br, cr, dr, er, X[ 3], 11, 0x6d703ef3);
    RMD_ROUND(RMD_H, er, ar, br, cr, dr, X[ 7],  8, 0x6d703ef3);
    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[14],  6, 0x6d703ef3);
    RMD_ROUND(RMD_H, cr, dr, er, ar, br, X[ 6],  6, 0x6d703ef3);
    RMD_ROUND(RMD_H, br, cr, dr, er, ar, X[ 9], 14, 0x6d703ef3);
    RMD_ROUND(RMD_H, ar, br, cr, dr, er, X[11], 12, 0x6d703ef3);
    RMD_ROUND(RMD_H, er, ar, br, cr, dr, X[ 8], 13, 0x6d703ef3);
    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[12],  5, 0x6d703ef3);
    RMD_ROUND(RMD_H, cr, dr, er, ar, br, X[ 2], 14, 0x6d703ef3);
    RMD_ROUND(RMD_H, br, cr, dr, er, ar, X[10], 13, 0x6d703ef3);
    RMD_ROUND(RMD_H, ar, br, cr, dr, er, X[ 0], 13, 0x6d703ef3);
    RMD_ROUND(RMD_H, er, ar, br, cr, dr, X[ 4],  7, 0x6d703ef3);
    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[13],  5, 0x6d703ef3);

    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[ 8], 15, 0x7a6d76e9);
    RMD_ROUND(RMD_G, br, cr, dr, er, ar, X[ 6],  5, 0x7a6d76e9);
    RMD_ROUND(RMD_G, ar, br, cr, dr, er, X[ 4],  8, 0x7a6d76e9);
    RMD_ROUND(RMD_G, er, ar, br, cr, dr, X[ 1], 11, 0x7a6d76e9);
    RMD_ROUND(RMD_G, dr, er, ar, br, cr, X[ 3], 14, 0x7a6d76e9);
    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[11], 14, 0x7a6d76e9);
    RMD_ROUND(RMD_G, br, cr, dr, er, ar, X[15],  6, 0x7a6d76e9);
    RMD_ROUND(RMD_G, ar, br, cr, dr, er, X[ 0], 14, 0x7a6d76e9);
    RMD_ROUND(RMD_G, er, ar, br, cr, dr, X[ 5],  6, 0x7a6d76e9);
    RMD_ROUND(RMD_G, dr, er, ar, br, cr, X[12],  9, 0x7a6d76e9);
    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[ 2], 12, 0x7a6d76e9);
    RMD_ROUND(RMD_G, br, cr, dr, er, ar, X[13],  9, 0x7a6d76e9);
    RMD_ROUND(RMD_G, ar, br, cr, dr, er, X[ 9], 12, 0x7a6d76e9);
    RMD_ROUND(RMD_G, er, ar, br, cr, dr, X[ 7],  5, 0x7a6d76e9);
    RMD_ROUND(RMD_G, dr, er, ar, br, cr, X[10], 15, 0x7a6d76e9);
    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[14],  8, 0x7a6d76e9);

    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[12],  8, 0x00000000);
    RMD_ROUND(RMD_F, ar, br, cr, dr, er, X[15],  5, 0x00000000);
    RMD_ROUND(RMD_F, er, ar, br, cr, dr, X[10], 12, 0x00000000);
    RMD_ROUND(RMD_F, dr, er, ar, br, cr, X[ 4],  9, 0x00000000);
    RMD_ROUND(RMD_F, cr, dr, er, ar, br, X[ 1], 12, 0x00000000);
    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[ 5],  5, 0x00000000);
    RMD_ROUND(RMD_F, ar, br, cr, dr, er, X[ 8], 14, 0x00000000);
    RMD_ROUND(RMD_F, er, ar, br, cr, dr, X[ 7],  6, 0x00000000);
    RMD_ROUND(RMD_F, dr, er, ar, br, cr, X[ 6],  8, 0x00000000);
    RMD_ROUND(RMD_F, cr, dr, er, ar, br, X[ 2], 13, 0x00000000);
    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[13],  6, 0x00000000);
    RMD_ROUND(RMD_F, ar, br, cr, dr, er, X[14],  5, 0x00000000);
    RMD_ROUND(RMD_F, er, ar, br, cr, dr, X[ 0], 15, 0x00000000);
    RMD_ROUND(RMD_F, dr, er, ar, br, cr, X[ 3], 13, 0x00000000);
    RMD_ROUND(RMD_F, cr, dr, er, ar, br, X[ 9], 11, 0x00000000);
    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[11], 11, 0x00000000);

    uint32_t h0 = 0xefcdab89 + cl + dr;
    uint32_t h1 = 0x98badcfe + dl + er;
    uint32_t h2 = 0x10325476 + el + ar;
    uint32_t h3 = 0xc3d2e1f0 + al + br;
    uint32_t h4 = 0x67452301 + bl + cr;

    hash[0] = h0; hash[1] = h0 >> 8; hash[2] = h0 >> 16; hash[3] = h0 >> 24;
    hash[4] = h1; hash[5] = h1 >> 8; hash[6] = h1 >> 16; hash[7] = h1 >> 24;
    hash[8] = h2; hash[9] = h2 >> 8; hash[10] = h2 >> 16; hash[11] = h2 >> 24;
    hash[12] = h3; hash[13] = h3 >> 8; hash[14] = h3 >> 16; hash[15] = h3 >> 24;
    hash[16] = h4; hash[17] = h4 >> 8; hash[18] = h4 >> 16; hash[19] = h4 >> 24;
}

// Core RIPEMD-160 over one block, taking the 16 message words directly.
// Returns whether the first output word matches target_first_word.
HOST_DEVICE inline bool ripemd160_X_early_reject(const uint32_t X[16], uint32_t out[5],
                                                 uint32_t target_first_word) {
    uint32_t al = 0x67452301, bl = 0xefcdab89, cl = 0x98badcfe, dl = 0x10325476, el = 0xc3d2e1f0;
    uint32_t ar = al, br = bl, cr = cl, dr = dl, er = el;

    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[ 0], 11, 0x00000000);
    RMD_ROUND(RMD_F, el, al, bl, cl, dl, X[ 1], 14, 0x00000000);
    RMD_ROUND(RMD_F, dl, el, al, bl, cl, X[ 2], 15, 0x00000000);
    RMD_ROUND(RMD_F, cl, dl, el, al, bl, X[ 3], 12, 0x00000000);
    RMD_ROUND(RMD_F, bl, cl, dl, el, al, X[ 4],  5, 0x00000000);
    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[ 5],  8, 0x00000000);
    RMD_ROUND(RMD_F, el, al, bl, cl, dl, X[ 6],  7, 0x00000000);
    RMD_ROUND(RMD_F, dl, el, al, bl, cl, X[ 7],  9, 0x00000000);
    RMD_ROUND(RMD_F, cl, dl, el, al, bl, X[ 8], 11, 0x00000000);
    RMD_ROUND(RMD_F, bl, cl, dl, el, al, X[ 9], 13, 0x00000000);
    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[10], 14, 0x00000000);
    RMD_ROUND(RMD_F, el, al, bl, cl, dl, X[11], 15, 0x00000000);
    RMD_ROUND(RMD_F, dl, el, al, bl, cl, X[12],  6, 0x00000000);
    RMD_ROUND(RMD_F, cl, dl, el, al, bl, X[13],  7, 0x00000000);
    RMD_ROUND(RMD_F, bl, cl, dl, el, al, X[14],  9, 0x00000000);
    RMD_ROUND(RMD_F, al, bl, cl, dl, el, X[15],  8, 0x00000000);
    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 7],  7, 0x5a827999);
    RMD_ROUND(RMD_G, dl, el, al, bl, cl, X[ 4],  6, 0x5a827999);
    RMD_ROUND(RMD_G, cl, dl, el, al, bl, X[13],  8, 0x5a827999);
    RMD_ROUND(RMD_G, bl, cl, dl, el, al, X[ 1], 13, 0x5a827999);
    RMD_ROUND(RMD_G, al, bl, cl, dl, el, X[10], 11, 0x5a827999);
    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 6],  9, 0x5a827999);
    RMD_ROUND(RMD_G, dl, el, al, bl, cl, X[15],  7, 0x5a827999);
    RMD_ROUND(RMD_G, cl, dl, el, al, bl, X[ 3], 15, 0x5a827999);
    RMD_ROUND(RMD_G, bl, cl, dl, el, al, X[12],  7, 0x5a827999);
    RMD_ROUND(RMD_G, al, bl, cl, dl, el, X[ 0], 12, 0x5a827999);
    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 9], 15, 0x5a827999);
    RMD_ROUND(RMD_G, dl, el, al, bl, cl, X[ 5],  9, 0x5a827999);
    RMD_ROUND(RMD_G, cl, dl, el, al, bl, X[ 2], 11, 0x5a827999);
    RMD_ROUND(RMD_G, bl, cl, dl, el, al, X[14],  7, 0x5a827999);
    RMD_ROUND(RMD_G, al, bl, cl, dl, el, X[11], 13, 0x5a827999);
    RMD_ROUND(RMD_G, el, al, bl, cl, dl, X[ 8], 12, 0x5a827999);
    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[ 3], 11, 0x6ed9eba1);
    RMD_ROUND(RMD_H, cl, dl, el, al, bl, X[10], 13, 0x6ed9eba1);
    RMD_ROUND(RMD_H, bl, cl, dl, el, al, X[14],  6, 0x6ed9eba1);
    RMD_ROUND(RMD_H, al, bl, cl, dl, el, X[ 4],  7, 0x6ed9eba1);
    RMD_ROUND(RMD_H, el, al, bl, cl, dl, X[ 9], 14, 0x6ed9eba1);
    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[15],  9, 0x6ed9eba1);
    RMD_ROUND(RMD_H, cl, dl, el, al, bl, X[ 8], 13, 0x6ed9eba1);
    RMD_ROUND(RMD_H, bl, cl, dl, el, al, X[ 1], 15, 0x6ed9eba1);
    RMD_ROUND(RMD_H, al, bl, cl, dl, el, X[ 2], 14, 0x6ed9eba1);
    RMD_ROUND(RMD_H, el, al, bl, cl, dl, X[ 7],  8, 0x6ed9eba1);
    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[ 0], 13, 0x6ed9eba1);
    RMD_ROUND(RMD_H, cl, dl, el, al, bl, X[ 6],  6, 0x6ed9eba1);
    RMD_ROUND(RMD_H, bl, cl, dl, el, al, X[13],  5, 0x6ed9eba1);
    RMD_ROUND(RMD_H, al, bl, cl, dl, el, X[11], 12, 0x6ed9eba1);
    RMD_ROUND(RMD_H, el, al, bl, cl, dl, X[ 5],  7, 0x6ed9eba1);
    RMD_ROUND(RMD_H, dl, el, al, bl, cl, X[12],  5, 0x6ed9eba1);
    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 1], 11, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, bl, cl, dl, el, al, X[ 9], 12, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, al, bl, cl, dl, el, X[11], 14, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, el, al, bl, cl, dl, X[10], 15, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, dl, el, al, bl, cl, X[ 0], 14, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 8], 15, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, bl, cl, dl, el, al, X[12],  9, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, al, bl, cl, dl, el, X[ 4],  8, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, el, al, bl, cl, dl, X[13],  9, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, dl, el, al, bl, cl, X[ 3], 14, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 7],  5, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, bl, cl, dl, el, al, X[15],  6, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, al, bl, cl, dl, el, X[14],  8, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, el, al, bl, cl, dl, X[ 5],  6, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, dl, el, al, bl, cl, X[ 6],  5, 0x8f1bbcdc);
    RMD_ROUND(RMD_I, cl, dl, el, al, bl, X[ 2], 12, 0x8f1bbcdc);
    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[ 4],  9, 0xa953fd4e);
    RMD_ROUND(RMD_J, al, bl, cl, dl, el, X[ 0], 15, 0xa953fd4e);
    RMD_ROUND(RMD_J, el, al, bl, cl, dl, X[ 5],  5, 0xa953fd4e);
    RMD_ROUND(RMD_J, dl, el, al, bl, cl, X[ 9], 11, 0xa953fd4e);
    RMD_ROUND(RMD_J, cl, dl, el, al, bl, X[ 7],  6, 0xa953fd4e);
    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[12],  8, 0xa953fd4e);
    RMD_ROUND(RMD_J, al, bl, cl, dl, el, X[ 2], 13, 0xa953fd4e);
    RMD_ROUND(RMD_J, el, al, bl, cl, dl, X[10], 12, 0xa953fd4e);
    RMD_ROUND(RMD_J, dl, el, al, bl, cl, X[14],  5, 0xa953fd4e);
    RMD_ROUND(RMD_J, cl, dl, el, al, bl, X[ 1], 12, 0xa953fd4e);
    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[ 3], 13, 0xa953fd4e);
    RMD_ROUND(RMD_J, al, bl, cl, dl, el, X[ 8], 14, 0xa953fd4e);
    RMD_ROUND(RMD_J, el, al, bl, cl, dl, X[11], 11, 0xa953fd4e);
    RMD_ROUND(RMD_J, dl, el, al, bl, cl, X[ 6],  8, 0xa953fd4e);
    RMD_ROUND(RMD_J, cl, dl, el, al, bl, X[15],  5, 0xa953fd4e);
    RMD_ROUND(RMD_J, bl, cl, dl, el, al, X[13],  6, 0xa953fd4e);

    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[ 5],  8, 0x50a28be6);
    RMD_ROUND(RMD_J, er, ar, br, cr, dr, X[14],  9, 0x50a28be6);
    RMD_ROUND(RMD_J, dr, er, ar, br, cr, X[ 7],  9, 0x50a28be6);
    RMD_ROUND(RMD_J, cr, dr, er, ar, br, X[ 0], 11, 0x50a28be6);
    RMD_ROUND(RMD_J, br, cr, dr, er, ar, X[ 9], 13, 0x50a28be6);
    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[ 2], 15, 0x50a28be6);
    RMD_ROUND(RMD_J, er, ar, br, cr, dr, X[11], 15, 0x50a28be6);
    RMD_ROUND(RMD_J, dr, er, ar, br, cr, X[ 4],  5, 0x50a28be6);
    RMD_ROUND(RMD_J, cr, dr, er, ar, br, X[13],  7, 0x50a28be6);
    RMD_ROUND(RMD_J, br, cr, dr, er, ar, X[ 6],  7, 0x50a28be6);
    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[15],  8, 0x50a28be6);
    RMD_ROUND(RMD_J, er, ar, br, cr, dr, X[ 8], 11, 0x50a28be6);
    RMD_ROUND(RMD_J, dr, er, ar, br, cr, X[ 1], 14, 0x50a28be6);
    RMD_ROUND(RMD_J, cr, dr, er, ar, br, X[10], 14, 0x50a28be6);
    RMD_ROUND(RMD_J, br, cr, dr, er, ar, X[ 3], 12, 0x50a28be6);
    RMD_ROUND(RMD_J, ar, br, cr, dr, er, X[12],  6, 0x50a28be6);
    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[ 6],  9, 0x5c4dd124);
    RMD_ROUND(RMD_I, dr, er, ar, br, cr, X[11], 13, 0x5c4dd124);
    RMD_ROUND(RMD_I, cr, dr, er, ar, br, X[ 3], 15, 0x5c4dd124);
    RMD_ROUND(RMD_I, br, cr, dr, er, ar, X[ 7],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, ar, br, cr, dr, er, X[ 0], 12, 0x5c4dd124);
    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[13],  8, 0x5c4dd124);
    RMD_ROUND(RMD_I, dr, er, ar, br, cr, X[ 5],  9, 0x5c4dd124);
    RMD_ROUND(RMD_I, cr, dr, er, ar, br, X[10], 11, 0x5c4dd124);
    RMD_ROUND(RMD_I, br, cr, dr, er, ar, X[14],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, ar, br, cr, dr, er, X[15],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[ 8], 12, 0x5c4dd124);
    RMD_ROUND(RMD_I, dr, er, ar, br, cr, X[12],  7, 0x5c4dd124);
    RMD_ROUND(RMD_I, cr, dr, er, ar, br, X[ 4],  6, 0x5c4dd124);
    RMD_ROUND(RMD_I, br, cr, dr, er, ar, X[ 9], 15, 0x5c4dd124);
    RMD_ROUND(RMD_I, ar, br, cr, dr, er, X[ 1], 13, 0x5c4dd124);
    RMD_ROUND(RMD_I, er, ar, br, cr, dr, X[ 2], 11, 0x5c4dd124);
    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[15],  9, 0x6d703ef3);
    RMD_ROUND(RMD_H, cr, dr, er, ar, br, X[ 5],  7, 0x6d703ef3);
    RMD_ROUND(RMD_H, br, cr, dr, er, ar, X[ 1], 15, 0x6d703ef3);
    RMD_ROUND(RMD_H, ar, br, cr, dr, er, X[ 3], 11, 0x6d703ef3);
    RMD_ROUND(RMD_H, er, ar, br, cr, dr, X[ 7],  8, 0x6d703ef3);
    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[14],  6, 0x6d703ef3);
    RMD_ROUND(RMD_H, cr, dr, er, ar, br, X[ 6],  6, 0x6d703ef3);
    RMD_ROUND(RMD_H, br, cr, dr, er, ar, X[ 9], 14, 0x6d703ef3);
    RMD_ROUND(RMD_H, ar, br, cr, dr, er, X[11], 12, 0x6d703ef3);
    RMD_ROUND(RMD_H, er, ar, br, cr, dr, X[ 8], 13, 0x6d703ef3);
    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[12],  5, 0x6d703ef3);
    RMD_ROUND(RMD_H, cr, dr, er, ar, br, X[ 2], 14, 0x6d703ef3);
    RMD_ROUND(RMD_H, br, cr, dr, er, ar, X[10], 13, 0x6d703ef3);
    RMD_ROUND(RMD_H, ar, br, cr, dr, er, X[ 0], 13, 0x6d703ef3);
    RMD_ROUND(RMD_H, er, ar, br, cr, dr, X[ 4],  7, 0x6d703ef3);
    RMD_ROUND(RMD_H, dr, er, ar, br, cr, X[13],  5, 0x6d703ef3);
    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[ 8], 15, 0x7a6d76e9);
    RMD_ROUND(RMD_G, br, cr, dr, er, ar, X[ 6],  5, 0x7a6d76e9);
    RMD_ROUND(RMD_G, ar, br, cr, dr, er, X[ 4],  8, 0x7a6d76e9);
    RMD_ROUND(RMD_G, er, ar, br, cr, dr, X[ 1], 11, 0x7a6d76e9);
    RMD_ROUND(RMD_G, dr, er, ar, br, cr, X[ 3], 14, 0x7a6d76e9);
    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[11], 14, 0x7a6d76e9);
    RMD_ROUND(RMD_G, br, cr, dr, er, ar, X[15],  6, 0x7a6d76e9);
    RMD_ROUND(RMD_G, ar, br, cr, dr, er, X[ 0], 14, 0x7a6d76e9);
    RMD_ROUND(RMD_G, er, ar, br, cr, dr, X[ 5],  6, 0x7a6d76e9);
    RMD_ROUND(RMD_G, dr, er, ar, br, cr, X[12],  9, 0x7a6d76e9);
    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[ 2], 12, 0x7a6d76e9);
    RMD_ROUND(RMD_G, br, cr, dr, er, ar, X[13],  9, 0x7a6d76e9);
    RMD_ROUND(RMD_G, ar, br, cr, dr, er, X[ 9], 12, 0x7a6d76e9);
    RMD_ROUND(RMD_G, er, ar, br, cr, dr, X[ 7],  5, 0x7a6d76e9);
    RMD_ROUND(RMD_G, dr, er, ar, br, cr, X[10], 15, 0x7a6d76e9);
    RMD_ROUND(RMD_G, cr, dr, er, ar, br, X[14],  8, 0x7a6d76e9);
    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[12],  8, 0x00000000);
    RMD_ROUND(RMD_F, ar, br, cr, dr, er, X[15],  5, 0x00000000);
    RMD_ROUND(RMD_F, er, ar, br, cr, dr, X[10], 12, 0x00000000);
    RMD_ROUND(RMD_F, dr, er, ar, br, cr, X[ 4],  9, 0x00000000);
    RMD_ROUND(RMD_F, cr, dr, er, ar, br, X[ 1], 12, 0x00000000);
    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[ 5],  5, 0x00000000);
    RMD_ROUND(RMD_F, ar, br, cr, dr, er, X[ 8], 14, 0x00000000);
    RMD_ROUND(RMD_F, er, ar, br, cr, dr, X[ 7],  6, 0x00000000);
    RMD_ROUND(RMD_F, dr, er, ar, br, cr, X[ 6],  8, 0x00000000);
    RMD_ROUND(RMD_F, cr, dr, er, ar, br, X[ 2], 13, 0x00000000);
    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[13],  6, 0x00000000);
    RMD_ROUND(RMD_F, ar, br, cr, dr, er, X[14],  5, 0x00000000);
    RMD_ROUND(RMD_F, er, ar, br, cr, dr, X[ 0], 15, 0x00000000);
    RMD_ROUND(RMD_F, dr, er, ar, br, cr, X[ 3], 13, 0x00000000);
    RMD_ROUND(RMD_F, cr, dr, er, ar, br, X[ 9], 11, 0x00000000);
    RMD_ROUND(RMD_F, br, cr, dr, er, ar, X[11], 11, 0x00000000);

    // out[] is always fully written so this is usable as a plain hash too.
    out[0] = 0xefcdab89 + cl + dr;
    out[1] = 0x98badcfe + dl + er;
    out[2] = 0x10325476 + el + ar;
    out[3] = 0xc3d2e1f0 + al + br;
    out[4] = 0x67452301 + bl + cr;
    return out[0] == target_first_word;
}

// Build the RIPEMD-160 message block from SHA-256's state words.
// SHA-256 emits big-endian words; RIPEMD-160 consumes little-endian.
HOST_DEVICE inline bool hash160_from_sha_words(const uint32_t sw[8], uint32_t out[5],
                                               uint32_t target_first_word) {
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) X[i] = bswap32_(sw[i]);
    X[8]  = 0x00000080;
    X[9]  = X[10] = X[11] = X[12] = X[13] = 0;
    X[14] = 256;
    X[15] = 0;
    return ripemd160_X_early_reject(X, out, target_first_word);
}

// Byte-oriented wrapper, kept for the CPU-worker callers.
HOST_DEVICE inline bool ripemd160_32_early_reject(const uint8_t *msg, uint8_t *hash,
                                                  uint32_t target_first_word) {
    uint32_t X[16];
    for (int i = 0; i < 8; i++) {
        X[i] = ((uint32_t)msg[i*4]) | ((uint32_t)msg[i*4+1] << 8) |
               ((uint32_t)msg[i*4+2] << 16) | ((uint32_t)msg[i*4+3] << 24);
    }
    X[8]  = 0x00000080;
    X[9]  = X[10] = X[11] = X[12] = X[13] = 0;
    X[14] = 256;
    X[15] = 0;

    uint32_t h[5];
    if (!ripemd160_X_early_reject(X, h, target_first_word)) return false;

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        hash[i*4 + 0] = (uint8_t)(h[i]);
        hash[i*4 + 1] = (uint8_t)(h[i] >>  8);
        hash[i*4 + 2] = (uint8_t)(h[i] >> 16);
        hash[i*4 + 3] = (uint8_t)(h[i] >> 24);
    }
    return true;
}

HOST_DEVICE inline void hash160(const uint8_t *pubkey33, uint8_t *out20) {
    uint8_t sha[32];
    sha256_33(pubkey33, sha);
    ripemd160_32(sha, out20);
}

#endif 
