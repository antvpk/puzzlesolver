#include <iostream>
#include <vector>
#include <random>
#include "bloomfilter/bloomfilter.cpp"

__device__ __forceinline__ uint64_t d_murmur_mix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

__global__ void test_bloom(const uint64_t* bits, uint64_t num_bits, uint32_t num_hashes, int* hits) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Generate a pseudo-random hash160 for each thread
    uint8_t h160[20];
    for (int i=0; i<20; i++) {
        h160[i] = (uint8_t)((tid * 1234567 + i * 98765) >> (i%8));
    }
    
    uint64_t a, b;
    memcpy(&a, h160, 8);
    memcpy(&b, h160 + 8, 8);
    uint32_t c;
    memcpy(&c, h160 + 16, 4);
    a ^= (uint64_t)c << 32;

    uint64_t h1   = d_murmur_mix(a);   // selects the block
    uint64_t seed = d_murmur_mix(b);   // positions inside it
    // Splitmix64 stream, one mix per position — see BloomFilter::compute_indices
    // for why the double-hashing progression cannot be used here.

    uint64_t block_mask = (num_bits >> 9) - 1;
    uint64_t base_bit   = (h1 & block_mask) << 9;
    bool pass = true;
    uint64_t x = seed;
    for (uint32_t i = 0; i < num_hashes; i++) {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 27; z *= 0x94D049BB133111EBULL;
        z ^= z >> 31;
        uint32_t bit  = (uint32_t)z & 511u;
        uint64_t widx = (base_bit >> 6) + (bit >> 6);
        if (!(bits[widx] & (1ULL << (bit & 63)))) {
            pass = false;
            break;
        }
    }
    if (pass) atomicAdd(hits, 1);
}

int main() {
    BloomFilter bf;
    bf.init(200000000ULL, 1e-8);
    // Load a smaller subset to simulate some load
    load_addresses_to_bloom("Latest_Rich_Bitcoin_Address.txt", bf, 1);
    
    uint64_t bloom_num_bits = bf.get_num_bits();
    uint64_t bloom_bytes = (bloom_num_bits / 64) * 8;
    
    uint64_t* d_bloom;
    cudaMalloc(&d_bloom, bloom_bytes);
    cudaMemcpy(d_bloom, bf.get_bits(), bloom_bytes, cudaMemcpyHostToDevice);
    
    int* d_hits;
    cudaMalloc(&d_hits, sizeof(int));
    cudaMemset(d_hits, 0, sizeof(int));
    
    test_bloom<<<1000, 1024>>>(d_bloom, bloom_num_bits, bf.get_num_hashes(), d_hits);
    
    int h_hits = 0;
    cudaMemcpy(&h_hits, d_hits, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Tested 1024000 hashes. Hits: " << h_hits << std::endl;
    return 0;
}
