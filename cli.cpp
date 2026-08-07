#include "cli.h"
#include "base58.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>

static void hex_to_bytes(const char *hex, uint8_t *bytes, int len) {
    for (int i = 0; i < len; i++) {
        char hi = hex[2*i], lo = hex[2*i+1];
        uint8_t bhi = (hi >= '0' && hi <= '9') ? hi - '0' : (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10 : hi - 'A' + 10;
        uint8_t blo = (lo >= '0' && lo <= '9') ? lo - '0' : (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10 : lo - 'A' + 10;
        bytes[i] = (bhi << 4) | blo;
    }
}

bool parse_cli(int argc, char** argv, Config& config) {

    config.sequential = false;
    config.batch_size = 4096;
    config.target_set = false;
    config.range_min_hex = "400000000000000000";
    config.range_max_hex = "7fffffffffffffffff";
    config.gpu_devices = "";
    config.cpu_threads = 0;
    config.blocks = 0;
    config.tpb = 0;
    config.use_cpu = false;
    config.use_gpu = true; 

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sequential") == 0) {
            config.sequential = true;
        }
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            config.cpu_threads = atoi(argv[++i]);
            config.use_cpu = true;
        }
        else if (strcmp(argv[i], "-gpu") == 0 && i + 1 < argc) {
            config.gpu_devices = argv[++i];
        }
        else if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch") == 0) && i + 1 < argc) {
            config.batch_size = strtoull(argv[++i], NULL, 0);
        }
        else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            config.blocks = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tpb") == 0 && i + 1 < argc) {
            config.tpb = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-target") == 0 && i + 1 < argc) {
            std::string t = argv[++i];
            if (t.length() == 40 && t.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos) {

                hex_to_bytes(t.c_str(), config.target_hash, 20);
                config.target_set = true;
                config.target_str = t;
            } else if (t[0] == '1') {

                if (decode_address_to_hash160(t, config.target_hash)) {
                    config.target_set = true;
                    config.target_str = t;
                } else {
                    fprintf(stderr, "[!] Invalid Base58 address provided for -target.\n");
                    return false;
                }
            } else {
                fprintf(stderr, "[!] Invalid -target argument. Must be 40-char hex or Base58 address starting with 1.\n");
                return false;
            }
        }
        else if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--range") == 0) && i + 1 < argc) {
            const char *r = argv[++i];
            const char *colon = strchr(r, ':');
            if (colon) {
                config.range_min_hex = std::string(r, colon - r);
                config.range_max_hex = std::string(colon + 1);
            } else {
                config.range_min_hex = r;
                config.range_max_hex = "";
            }
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n\n", argv[0]);
            printf("Options:\n");
            printf("  -target HASH/ADDR Specify target RIPEMD-160 hash or Base58 Address.\n");
            printf("  -r, --range M:X   Set the search range from MIN to MAX in hex.\n");
            printf("  -gpu ID_STRING    Select GPUs to use (e.g. -gpu 035). Default: all.\n");
            printf("  -t, --threads N   Number of CPU threads. Default: 0 (uses GPU only).\n");
            printf("  -s, --sequential  Run in sequential mode. Default is random mode.\n");
            printf("  -b, --batch N     CPU affine batch-inversion window (default: 4096).\n");
            printf("                    GPU batch is fixed at compile-time (GPU_BATCH=256).\n");
            printf("  -h, --help        Show this help message and exit.\n");
            return false;
        }
    }

    if (!config.target_set) {
        hex_to_bytes("f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8", config.target_hash, 20);
        config.target_set = true;
        config.target_str = "f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8";
    }

    return true;
}
