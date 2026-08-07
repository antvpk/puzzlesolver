#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <signal.h>
#include <random>
#include <fstream>

#include "cli.h"
#include "base58.h"
#include "int256.h"
#include "util.h"
#ifndef USE_CUDA
#include "cpu_worker.h"
#endif

#ifdef USE_CUDA
#include "gpu_worker.cuh"
#endif

static std::atomic<uint64_t> g_total_cpu_keys(0);
static std::atomic<uint64_t> g_total_gpu_keys(0);
static std::atomic<bool> g_found(false);
static std::atomic<bool> g_running(true);
static std::atomic<int> g_active_workers(0);
static std::mutex g_file_mutex;
static std::chrono::steady_clock::time_point g_start_time;

std::mutex g_display_mutex;
uint256_t g_display_key;
bool g_display_valid = false;

static uint256_t g_found_key;

static void signal_handler(int sig) {
    (void)sig;
    g_running.store(false);
}



int main(int argc, char** argv) {
    Config config;
    if (!parse_cli(argc, argv, config)) {
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    uint256_t range_min, range_max;
    u256_from_hex(range_min, config.range_min_hex.c_str());
    if (!config.range_max_hex.empty()) {
        u256_from_hex(range_max, config.range_max_hex.c_str());
    } else {
        u256_from_hex(range_max, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    }

    uint32_t target_h0 = ((uint32_t)config.target_hash[0]) | ((uint32_t)config.target_hash[1] << 8) |
                         ((uint32_t)config.target_hash[2] << 16) | ((uint32_t)config.target_hash[3] << 24);

    std::vector<int> gpu_devices;
#ifdef USE_CUDA
    if (config.use_gpu) {
        gpu_devices = parse_gpu_devices(config.gpu_devices);
    }
#endif

    if (gpu_devices.empty() && config.cpu_threads == 0) {
        printf("[*] No GPUs found. Falling back to CPU mode.\n");
        config.cpu_threads = 1;
        config.use_cpu = true;
    }

    char min_hex_str[65] = {0};
    u256_to_hex(min_hex_str, range_min);
    char max_hex_str[65] = {0};
    u256_to_hex(max_hex_str, range_max);

    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  🔑 puzzlesolver v2.1 - Ultra Performance CPU/GPU    ║\n");
    printf("║  Target: %-44s ║\n", config.target_str.c_str());
    char *min_p = min_hex_str;
    while (*min_p == '0' && *(min_p+1) != '\0') min_p++;
    char *max_p = max_hex_str;
    while (*max_p == '0' && *(max_p+1) != '\0') max_p++;

    char range_buf[200];
    snprintf(range_buf, sizeof(range_buf), "0x%s : 0x%s", min_p, max_p);
    printf("║  Range:  %-44s ║\n", range_buf);
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    if (!gpu_devices.empty()) {
        printf("[*] GPU Mode | Mode: %s | Blocks: %s | Threads/Block: %s\n",
               config.sequential ? "Sequential" : "Random",
               config.blocks == 0 ? "Auto" : std::to_string(config.blocks).c_str(),
               config.tpb == 0 ? "Auto" : std::to_string(config.tpb).c_str());
    }
    if (config.use_cpu && config.cpu_threads > 0) {
        printf("[*] CPU Mode | Threads: %d | Mode: %s | Affine Batch: %d\n",
               config.cpu_threads, config.sequential ? "Sequential" : "Random", (int)config.batch_size);
    }
    printf("\n");

    std::vector<std::thread> workers;

#ifndef USE_CUDA
    if (config.use_cpu && config.cpu_threads > 0) {
        init_cpu_worker();
        for (int i = 0; i < config.cpu_threads; i++) {
            uint256_t start_key;
            if (config.sequential) {
                uint256_t stagger;
                stagger.d[0] = 100000000000000ULL * i;
                stagger.d[1] = 0; stagger.d[2] = 0; stagger.d[3] = 0;
                u256_add(start_key, range_min, stagger);
            } else {
                start_key = random_start_in_range(range_min, range_max);
            }
            uint64_t keys_to_search = 0xFFFFFFFFFFFFFFFFULL; 
            g_active_workers++;
            workers.emplace_back([i, start_key, keys_to_search, config, range_min, range_max, target_h0]() {
                cpu_search_worker(i, start_key, keys_to_search,
                                 std::ref(g_total_cpu_keys), std::ref(g_found), std::ref(g_running), std::ref(g_found_key),
                                 config.target_hash, target_h0, config.batch_size,
                                 config.sequential, range_min, range_max);
                g_active_workers--;
            });
        }
    }
#endif

#ifdef USE_CUDA
    if (!gpu_devices.empty()) {
        init_gpu_worker();
        for (int device_id : gpu_devices) {
            uint256_t start_key = config.sequential ? range_min : random_start_in_range(range_min, range_max);
            uint64_t keys_to_search = 0xFFFFFFFFFFFFFFFFULL;
            g_active_workers++;
            workers.emplace_back([device_id, start_key, keys_to_search, config, range_min, range_max, target_h0]() {
                launch_gpu_worker(device_id, start_key, keys_to_search,
                                 std::ref(g_total_gpu_keys), std::ref(g_found), std::ref(g_running),
                                 std::ref(g_found_key), config.target_hash, target_h0,
                                 config.blocks, config.tpb, config.batch_size,
                                 config.sequential, range_min, range_max);
                g_active_workers--;
            });
        }
    }
#endif

    g_start_time = std::chrono::steady_clock::now();

    bool use_c = config.use_cpu && config.cpu_threads > 0;
    bool use_g = !gpu_devices.empty();

    while (g_running.load() && !g_found.load()) {
        if (g_active_workers.load() == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count() / 1000.0;
        if (elapsed < 0.1) continue;

        uint64_t total_cpu = g_total_cpu_keys.load();
        uint64_t total_gpu = g_total_gpu_keys.load();
        uint64_t total = total_cpu + total_gpu;
        
        double speed_cpu = total_cpu / elapsed;
        double speed_gpu = total_gpu / elapsed;

        const char *gsu = "keys/s";
        double gsd = speed_gpu;
        if (speed_gpu >= 1e9) { gsd = speed_gpu / 1e9; gsu = "Gkeys/s"; }
        else if (speed_gpu >= 1e6) { gsd = speed_gpu / 1e6; gsu = "Mkeys/s"; }
        else if (speed_gpu >= 1e3) { gsd = speed_gpu / 1e3; gsu = "Kkeys/s"; }

        const char *csu = "keys/s";
        double csd = speed_cpu;
        if (speed_cpu >= 1e9) { csd = speed_cpu / 1e9; csu = "Gkeys/s"; }
        else if (speed_cpu >= 1e6) { csd = speed_cpu / 1e6; csu = "Mkeys/s"; }
        else if (speed_cpu >= 1e3) { csd = speed_cpu / 1e3; csu = "Kkeys/s"; }


        const char *tu = "";
        double td = (double)total;
        if (total >= 1000000000000ULL) { td = total / 1e12; tu = "T"; }
        else if (total >= 1000000000ULL) { td = total / 1e9; tu = "G"; }
        else if (total >= 1000000ULL) { td = total / 1e6; tu = "M"; }
        else if (total >= 1000ULL) { td = total / 1e3; tu = "K"; }

        char current_key_str[65] = {0};
        if (g_display_mutex.try_lock()) {
            if (g_display_valid) {
                u256_to_hex(current_key_str, g_display_key);
            }
            g_display_mutex.unlock();
        }
        
        char *trim_key = current_key_str;
        while (*trim_key == '0' && *(trim_key+1) != '\0') trim_key++;

        char speed_buf[512];
        int slen = 0;
        
        if (use_c && use_g) {
            slen += snprintf(speed_buf + slen, sizeof(speed_buf) - slen, "\r[⚡] CPU: %.2f %s | GPU: %.2f %s", csd, csu, gsd, gsu);
        } else if (use_c) {
            slen += snprintf(speed_buf + slen, sizeof(speed_buf) - slen, "\r[⚡] CPU: %.2f %s", csd, csu);
        } else if (use_g) {
            slen += snprintf(speed_buf + slen, sizeof(speed_buf) - slen, "\r[⚡] GPU: %.2f %s", gsd, gsu);
        }
        
        slen += snprintf(speed_buf + slen, sizeof(speed_buf) - slen, " | Total: %.2f%s", td, tu);

        if (current_key_str[0] != '\0') {
            snprintf(speed_buf + slen, sizeof(speed_buf) - slen, " | Current: 0x%s \033[K", trim_key);
        } else {
            snprintf(speed_buf + slen, sizeof(speed_buf) - slen, " | %s\033[K", g_found.load() ? "FOUND ✅" : "Searching...");
        }
        printf("%s", speed_buf);
        fflush(stdout);
    }

    g_running.store(false);

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    printf("\n"); 

    auto end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - g_start_time).count() / 1000.0;
    uint64_t total_cpu = g_total_cpu_keys.load();
    uint64_t total_gpu = g_total_gpu_keys.load();
    uint64_t total = total_cpu + total_gpu;

    char final_stats[512];
    int fslen = snprintf(final_stats, sizeof(final_stats), "\n[📊] Total: %llu (%.2e) in %.1fs",
                          (unsigned long long)total, (double)total, total_time);
    if (use_c) {
        fslen += snprintf(final_stats + fslen, sizeof(final_stats) - fslen, " | Avg CPU: %.2f Mkeys/s", total_cpu / total_time / 1e6);
    }
    if (use_g) {
        fslen += snprintf(final_stats + fslen, sizeof(final_stats) - fslen, " | Avg GPU: %.2f Mkeys/s", total_gpu / total_time / 1e6);
    }
    snprintf(final_stats + fslen, sizeof(final_stats) - fslen, " | %s\n", g_found.load() ? "FOUND ✅" : "Not found ❌ (Stopped)");
    printf("%s", final_stats);

    if (g_found.load()) {
        char hex_key[65];
        u256_to_hex(hex_key, g_found_key);
        char *trimmed = hex_key;
        while (*trimmed == '0' && *(trimmed+1) != '\0') trimmed++;

        printf("\n\n");
        printf("╔══════════════════════════════════════════════╗\n");
        printf("║         🎉 PUZZLE KEY FOUND! 🎉             ║\n");
        printf("╠══════════════════════════════════════════════╣\n");
        printf("║  Private Key: 0x%-28s  ║\n", trimmed);
        printf("╚══════════════════════════════════════════════╝\n\n");
        
        uint8_t wif_data[34];
        wif_data[0] = 0x80;
        for (int i=0; i<4; i++) {
            uint64_t v = g_found_key.d[3 - i];
            for (int j=0; j<8; j++) {
                wif_data[1 + i*8 + j] = (v >> (56 - j*8)) & 0xFF;
            }
        }
        wif_data[33] = 0x01;
        std::string wif = base58check_encode(wif_data, 34);

        uint8_t addr_data[21];
        addr_data[0] = 0x00;
        memcpy(addr_data + 1, config.target_hash, 20);
        std::string addr = base58check_encode(addr_data, 21);
        
        std::ofstream outfile("RESULT.txt", std::ios_base::app);
        if (outfile.is_open()) {
            outfile << "Target Hash: " << config.target_str << "\n";
            outfile << "Address (Compressed): " << addr << "\n";
            outfile << "Private Key (Hex): 0x" << trimmed << "\n";
            outfile << "Private Key (WIF Compressed): " << wif << "\n";
            outfile << "----------------------------------------\n";
            outfile.close();
            printf("[*] Key successfully saved to RESULT.txt\n\n");
        } else {
            printf("[!] Failed to open RESULT.txt for saving.\n\n");
        }
    }

    return 0;
}
