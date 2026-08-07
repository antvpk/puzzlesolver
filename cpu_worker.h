#ifndef CPU_WORKER_H
#define CPU_WORKER_H

#include "int256.h"
#include <atomic>

void init_cpu_worker();

void cpu_search_worker(int thread_id, uint256_t start_key, uint64_t keys_to_search,
                       std::atomic<uint64_t>& keys_checked, std::atomic<bool>& found_flag,
                       std::atomic<bool>& running_flag,
                       uint256_t& found_key, const uint8_t* target_hash, uint32_t target_h0,
                       uint64_t batch_size, bool is_sequential, uint256_t range_min, uint256_t range_max);

#endif
