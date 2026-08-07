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

std::vector<int> parse_gpu_devices(const std::string& gpu_string);
int get_cuda_device_count();
#else
inline void init_gpu_worker() {}

inline void launch_gpu_worker(int, uint256_t, uint64_t,
                       std::atomic<uint64_t>&, std::atomic<bool>&, std::atomic<bool>&,
                       uint256_t&, const uint8_t*, uint32_t,
                       int, int, uint64_t, bool, uint256_t, uint256_t) {}

inline std::vector<int> parse_gpu_devices(const std::string&) { return {}; }
inline int get_cuda_device_count() { return 0; }
#endif

#endif
