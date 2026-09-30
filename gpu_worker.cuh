#ifndef GPU_WORKER_CUH
#define GPU_WORKER_CUH

#include "int256.h"
#include <atomic>
#include <string>
#include <vector>

#ifdef USE_CUDA
void init_gpu_worker();

void launch_gpu_worker(int device_id, uint256_t start_key, uint64_t keys_to_search,
                       std::atomic<uint64_t>& keys_checked, std::atomic<bool>& found_flag,
                       std::atomic<bool>& running_flag,
                       uint256_t& found_key, const uint8_t* target_hash, uint32_t target_h0,
                       int blocks, int tpb, uint64_t batch_size,
                       bool is_sequential, uint256_t range_min, uint256_t range_max);

// Bulk mode: bloom filter based multi-address matching
struct BulkResult {
    uint8_t  privkey[32];   // Raw private key bytes (big-endian)
    uint8_t  h160[20];      // Matched hash160
    uint32_t addr_type;     // 0=P2PKH_C, 1=P2PKH_U, 2=P2SH-P2WPKH, 3=P2WPKH
};

void launch_gpu_bulk_worker(int device_id, uint256_t start_key, uint64_t keys_to_search,
                            std::atomic<uint64_t>& keys_checked,
                            std::atomic<bool>& found_flag,
                            std::atomic<bool>& running_flag,
                            const uint64_t* bloom_bits, uint64_t bloom_num_bits,
                            uint32_t bloom_num_hashes, uint32_t addr_type_mask,
                            int blocks, int tpb, uint64_t batch_size,
                            bool is_sequential, uint256_t range_min, uint256_t range_max,
                            // Callback for found results
                            void (*on_found)(const BulkResult& result));

std::vector<int> parse_gpu_devices(const std::string& gpu_string);
int get_cuda_device_count();
#else
inline void init_gpu_worker() {}

inline void launch_gpu_worker(int, uint256_t, uint64_t,
                       std::atomic<uint64_t>&, std::atomic<bool>&, std::atomic<bool>&,
                       uint256_t&, const uint8_t*, uint32_t,
                       int, int, uint64_t, bool, uint256_t, uint256_t) {}

struct BulkResult {
    uint8_t  privkey[32];
    uint8_t  h160[20];
    uint32_t addr_type;
};

inline void launch_gpu_bulk_worker(int, uint256_t, uint64_t,
                            std::atomic<uint64_t>&, std::atomic<bool>&, std::atomic<bool>&,
                            const uint64_t*, uint64_t, uint32_t, uint32_t,
                            int, int, uint64_t, bool, uint256_t, uint256_t,
                            void (*)(const BulkResult&)) {}

inline std::vector<int> parse_gpu_devices(const std::string&) { return {}; }
inline int get_cuda_device_count() { return 0; }
#endif

#endif
