#include "base58.h"
#include <cstring>
#include "sha256_rmd160.h"

static const char* ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const int8_t mapBase58[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,
    -1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,
    22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,
    -1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,
    47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
};

std::string base58_encode(const uint8_t *data, size_t len) {
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

bool base58_decode(const char *psz, std::vector<uint8_t>& vch) {
    while (*psz && (*psz == ' ' || *psz == '\t' || *psz == '\n' || *psz == '\r'))
        psz++;
    int zeroes = 0;
    while (*psz == '1') {
        zeroes++;
        psz++;
    }
    std::vector<uint8_t> b256(strlen(psz) * 733 / 1000 + 1);
    int length = 0;
    while (*psz && (*psz != ' ' && *psz != '\t' && *psz != '\n' && *psz != '\r')) {
        int carry = mapBase58[(uint8_t)*psz];
        if (carry == -1) return false;
        int i = 0;
        for (std::vector<uint8_t>::reverse_iterator it = b256.rbegin(); (carry != 0 || i < length) && (it != b256.rend()); ++it, ++i) {
            carry += 58 * (*it);
            *it = carry % 256;
            carry /= 256;
        }
        length = i;
        psz++;
    }
    while (*psz && (*psz == ' ' || *psz == '\t' || *psz == '\n' || *psz == '\r'))
        psz++;
    if (*psz != 0) return false;

    std::vector<uint8_t>::iterator it = b256.begin() + (b256.size() - length);
    vch.assign(zeroes, 0x00);
    while (it != b256.end())
        vch.push_back(*(it++));
    return true;
}

static void _cpu_sha256(const uint8_t *data, size_t len, uint8_t hash[32]) {
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

bool decode_address_to_hash160(const std::string& address, uint8_t* hash160_out) {
    std::vector<uint8_t> decoded;
    if (!base58_decode(address.c_str(), decoded)) {
        return false;
    }
    if (decoded.size() != 25) {
        return false;
    }

    uint8_t h1[32];
    uint8_t h2[32];
    _cpu_sha256(decoded.data(), 21, h1);
    _cpu_sha256(h1, 32, h2);

    if (memcmp(h2, decoded.data() + 21, 4) != 0) {
        return false;
    }

    memcpy(hash160_out, decoded.data() + 1, 20);
    return true;
}

std::string base58check_encode(const uint8_t *data, size_t len) {
    uint8_t h1[32];
    uint8_t h2[32];
    _cpu_sha256(data, len, h1);
    _cpu_sha256(h1, 32, h2);
    
    std::vector<uint8_t> payload(data, data + len);
    payload.insert(payload.end(), h2, h2 + 4);
    return base58_encode(payload.data(), payload.size());
}
