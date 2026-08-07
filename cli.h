#ifndef CLI_H
#define CLI_H

#include <string>
#include <vector>
#include <cstdint>

struct Config {
    bool sequential;
    uint64_t batch_size;
    uint8_t target_hash[20];
    bool target_set;
    std::string target_str;
    std::string range_min_hex;
    std::string range_max_hex;

    std::string gpu_devices; 
    int cpu_threads;         
    bool use_cpu;
    bool use_gpu;

    int blocks;
    int tpb;
};

bool parse_cli(int argc, char** argv, Config& config);

#endif
