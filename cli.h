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

    // Bulk mode: bloom filter multi-address matching
    bool bulk_mode;
    std::string bulk_file;   // Path to address file (or empty to auto-download)

    // Bulk mode address-type selection bitmask (see BULK_TYPE_* below).
    // Default is compressed-only, the fastest and by far the most common case.
    uint32_t bulk_types;
};

// Bulk-mode address type bits. These match the addr_type codes emitted by the
// GPU kernel so a result of type t is enabled iff (bulk_types & (1u << t)).
#define BULK_TYPE_COMPRESSED    0x1u  // P2PKH compressed + P2WPKH (bc1q)
#define BULK_TYPE_UNCOMPRESSED  0x2u  // P2PKH uncompressed
#define BULK_TYPE_P2SH          0x4u  // P2SH-P2WPKH (3xxx)

bool parse_cli(int argc, char** argv, Config& config);

#endif
