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
#include <sys/stat.h>

#include "cli.h"
#include "base58.h"
#include "int256.h"
#include "util.h"
#include "bloomfilter/bloomfilter.h"
#ifndef USE_CUDA
#include "cpu_worker.h"
#endif

#include "gpu_worker.cuh"

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
static BloomFilter* g_bf = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    g_running.store(false);
}

// ================================================================
//  Bulk mode: WIF encoding + result formatting helpers
// ================================================================
static void bytes_to_hex_str(const uint8_t *data, int len, char *out) {
    const char hex[] = "0123456789abcdef";
    for (int i = 0; i < len; i++) {
        out[i*2]   = hex[data[i] >> 4];
        out[i*2+1] = hex[data[i] & 0xF];
    }
    out[len*2] = '\0';
}

static std::string privkey_to_wif_compressed(const uint8_t privkey[32]) {
    uint8_t data[34];
    data[0] = 0x80;
    memcpy(data + 1, privkey, 32);
    data[33] = 0x01; // compressed flag
    return base58check_encode(data, 34);
}

static std::string privkey_to_wif_uncompressed(const uint8_t privkey[32]) {
    uint8_t data[33];
    data[0] = 0x80;
    memcpy(data + 1, privkey, 32);
    return base58check_encode(data, 33);
}

static const char* addr_type_name(uint32_t type) {
    switch (type) {
        case 0: return "P2PKH(c)/P2WPKH";
        case 1: return "P2PKH(u)";
        case 2: return "P2SH-P2WPKH";
        case 3: return "P2WPKH";
        default: return "Unknown";
    }
}

// ================================================================
//  Bulk mode result callback
// ================================================================
static std::atomic<uint64_t> g_bloom_fp(0); // False positive counter

static void on_bulk_found(const BulkResult& result) {
    // ── CPU-side exact verification to eliminate bloom filter false positives ──
    if (g_bf && !g_bf->exact_test(result.h160)) {
        g_bloom_fp.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // ── Verified real match! ──
    g_found.store(true);
    std::lock_guard<std::mutex> lock(g_file_mutex);

    char hex_key[65];
    bytes_to_hex_str(result.privkey, 32, hex_key);

    std::string wif_c = privkey_to_wif_compressed(result.privkey);
    std::string wif_u = privkey_to_wif_uncompressed(result.privkey);

    char h160_hex[41];
    bytes_to_hex_str(result.h160, 20, h160_hex);

    // Compute the matching address from the hash160
    uint8_t addr_data[21];
    std::string matched_addr;
    if (result.addr_type == 0) {
        // Could be P2PKH compressed — show as legacy address
        addr_data[0] = 0x00;
        memcpy(addr_data + 1, result.h160, 20);
        matched_addr = base58check_encode(addr_data, 21);
    } else if (result.addr_type == 1) {
        // P2PKH uncompressed
        addr_data[0] = 0x00;
        memcpy(addr_data + 1, result.h160, 20);
        matched_addr = base58check_encode(addr_data, 21);
    } else if (result.addr_type == 2) {
        // P2SH
        addr_data[0] = 0x05;
        memcpy(addr_data + 1, result.h160, 20);
        matched_addr = base58check_encode(addr_data, 21);
    } else {
        matched_addr = std::string("h160:") + h160_hex;
    }

    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║        🎉 BULK MATCH FOUND! 🎉                     ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Type         : %-36s ║\n", addr_type_name(result.addr_type));
    printf("║  Address      : %-36s ║\n", matched_addr.c_str());
    printf("║  Hash160      : %s   ║\n", h160_hex);
    printf("║  Private Key  : %s ║\n", hex_key);
    printf("║  WIF (compr.) : %-36s ║\n", wif_c.c_str());
    printf("║  WIF (uncomp.): %-36s ║\n", wif_u.c_str());
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // Save to file
    std::ofstream outfile("RESULT.txt", std::ios_base::app);
    if (outfile.is_open()) {
        outfile << "========== BULK MATCH FOUND ==========\n";
        outfile << "Address Type: " << addr_type_name(result.addr_type) << "\n";
        outfile << "Address: " << matched_addr << "\n";
        outfile << "Hash160: " << h160_hex << "\n";
        outfile << "Private Key (Hex): " << hex_key << "\n";
        outfile << "WIF (Compressed): " << wif_c << "\n";
        outfile << "WIF (Uncompressed): " << wif_u << "\n";
        outfile << "=======================================\n\n";
        outfile.close();
        printf("[*] Result saved to RESULT.txt\n\n");
    }
}

// ================================================================
//  Bulk mode entry point
// ================================================================
static int run_bulk_mode(const Config& config) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  🔑 puzzlesolver v2.4 - BULK MODE (Bloom Filter)    ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    // Report which address types are active. Compressed-only (the default) is the
    // fastest and covers the puzzle / vast majority of funded addresses; extra
    // types cost proportionally more per-key hashing and add bloom candidates.
    {
        char types_buf[64]; int tlen = 0;
        if (config.bulk_types & BULK_TYPE_COMPRESSED)
            tlen += snprintf(types_buf + tlen, sizeof(types_buf) - tlen, "%sP2PKH(c)/P2WPKH", tlen ? ", " : "");
        if (config.bulk_types & BULK_TYPE_UNCOMPRESSED)
            tlen += snprintf(types_buf + tlen, sizeof(types_buf) - tlen, "%sP2PKH(u)", tlen ? ", " : "");
        if (config.bulk_types & BULK_TYPE_P2SH)
            tlen += snprintf(types_buf + tlen, sizeof(types_buf) - tlen, "%sP2SH-P2WPKH", tlen ? ", " : "");
        printf("[*] Address types: %s   (change with -types c/u/s or 'all')\n\n", types_buf);
    }

    // Step 1: Load or download address file
    std::string addr_file = config.bulk_file;

    if (addr_file.empty()) {
        // Auto-download the default file
        printf("[1/3] Downloading address database...\n");
        // Try each file
        for (int i = 0; i < NUM_ADDR_FILES; i++) {
            download_address_file(ADDR_FILES[i].filename, ADDR_FILES[i].url);
        }
        // Use all downloaded files
    } else {
        printf("[1/3] Using address file: %s\n", addr_file.c_str());
    }

    // Step 2: Build bloom filter
    printf("\n[2/3] Building bloom filter...\n");

    // Resolve the list of files to read (the download step above has already
    // run when no file was given).
    std::vector<std::string> addr_files;
    if (addr_file.empty()) {
        for (int i = 0; i < NUM_ADDR_FILES; i++) {
            std::string fn = ADDR_FILES[i].filename;
            if (fn.size() > 3 && fn.substr(fn.size() - 3) == ".gz")
                fn = fn.substr(0, fn.size() - 3);
            struct stat st;
            if (stat(fn.c_str(), &st) == 0) {
                addr_files.push_back(fn);
            } else {
                fprintf(stderr, "  [!] %s not found, skipping.\n", fn.c_str());
            }
        }
        if (addr_files.empty()) {
            fprintf(stderr, "[!] No address file available. Aborting.\n");
            return 1;
        }
    } else {
        addr_files.push_back(addr_file);
    }

    // Pass 1: count what will really be inserted and size the filter from it.
    // The old fixed 200M guess cost 1 GB of filter for a ~30M row file and
    // spread the same set bits over ~8x the address space, so every per-key
    // probe walked a far colder, far larger array than it needed to.
    printf("  Counting addresses...\n");
    uint64_t estimated = 0;
    for (const auto &fn : addr_files)
        estimated += count_addresses_in_file(fn, config.bulk_types);

    if (estimated == 0) {
        fprintf(stderr, "[!] No matchable addresses found. Aborting.\n");
        return 1;
    }

    BloomFilter bf;
    bf.init(estimated, 1e-8); // Very low FPR

    printf("  Bloom filter: %llu bits (%.1f MB), k=%u for ~%llu addresses\n",
           (unsigned long long)bf.get_num_bits(),
           bf.get_size_bytes() / (1024.0 * 1024.0),
           bf.get_num_hashes(), (unsigned long long)estimated);

    // Pass 2: decode and insert.
    uint64_t total_loaded = 0;
    for (const auto &fn : addr_files) {
        printf("  Loading %s ...\n", fn.c_str());
        total_loaded += load_addresses_to_bloom(fn, bf, config.bulk_types);
    }

    if (total_loaded == 0) {
        fprintf(stderr, "[!] No addresses loaded. Aborting.\n");
        return 1;
    }

    printf("  Total addresses in bloom filter: %llu\n", (unsigned long long)total_loaded);

    // Sort the exact hash set for CPU-side false-positive verification
    printf("  Sorting exact hashes for verification...\n");
    bf.sort_exact();
    g_bf = &bf;
    printf("  Done.\n\n");

    // Step 3: Launch GPU/CPU workers
    printf("[3/3] Starting search...\n\n");

    // Auto-set range for bulk mode: 249-bit to 256-bit private keys
    // (2^248 to 2^256-1) unless user explicitly set a range (overriding the default puzzle 67 range)
    uint256_t range_min, range_max;
    if (config.range_min_hex.empty() || config.range_min_hex == "1" || 
        (config.range_min_hex == "400000000000000000" && config.range_max_hex == "7fffffffffffffffff")) {
        // Default bulk range: 249-bit to 256-bit
        u256_from_hex(range_min, "0100000000000000000000000000000000000000000000000000000000000000");
        u256_from_hex(range_max, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
        printf("  Auto range (249→256 bit): 2^248 to secp256k1 order-1\n");
    } else {
        u256_from_hex(range_min, config.range_min_hex.c_str());
        if (!config.range_max_hex.empty()) {
            u256_from_hex(range_max, config.range_max_hex.c_str());
        } else {
            u256_from_hex(range_max, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
        }
    }

    char min_hex_str[65] = {0};
    u256_to_hex(min_hex_str, range_min);
    char max_hex_str[65] = {0};
    u256_to_hex(max_hex_str, range_max);

    char *min_p = min_hex_str;
    while (*min_p == '0' && *(min_p+1) != '\0') min_p++;
    char *max_p = max_hex_str;
    while (*max_p == '0' && *(max_p+1) != '\0') max_p++;
    printf("  Range: 0x%s : 0x%s\n", min_p, max_p);
    printf("  Addresses: %llu | Bloom: %.1f MB\n\n",
           (unsigned long long)total_loaded, bf.get_size_bytes() / (1024.0 * 1024.0));

    std::vector<int> gpu_devices;
#ifdef USE_CUDA
    if (config.use_gpu) {
        gpu_devices = parse_gpu_devices(config.gpu_devices);
    }
#endif

    if (gpu_devices.empty() && config.cpu_threads == 0) {
        printf("[*] No GPUs found. Falling back to CPU mode.\n");
        // For CPU bulk mode, we'd need a separate implementation
        fprintf(stderr, "[!] CPU bulk mode not yet implemented. Use GPU.\n");
        return 1;
    }

    std::vector<std::thread> workers;

    g_start_time = std::chrono::steady_clock::now();

#ifdef USE_CUDA
    if (!gpu_devices.empty()) {
        init_gpu_worker();
        for (int device_id : gpu_devices) {
            uint256_t start_key = config.sequential ? range_min : random_start_in_range(range_min, range_max);
            uint64_t keys_to_search = 0xFFFFFFFFFFFFFFFFULL;
            g_active_workers++;
            workers.emplace_back([device_id, start_key, keys_to_search, &config, range_min, range_max, &bf]() {
                launch_gpu_bulk_worker(device_id, start_key, keys_to_search,
                                       std::ref(g_total_gpu_keys), std::ref(g_found), std::ref(g_running),
                                       bf.get_bits(), bf.get_num_bits(), bf.get_num_hashes(),
                                       config.bulk_types,
                                       config.blocks, config.tpb, config.batch_size,
                                       config.sequential, range_min, range_max,
                                       on_bulk_found);
                g_active_workers--;
            });
        }
    }
#endif

    bool use_g = !gpu_devices.empty();

    while (g_running.load()) {
        if (g_active_workers.load() == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count() / 1000.0;
        if (elapsed < 0.1) continue;

        uint64_t total_gpu = g_total_gpu_keys.load();
        double speed_gpu = total_gpu / elapsed;

        const char *gsu = "keys/s";
        double gsd = speed_gpu;
        if (speed_gpu >= 1e9) { gsd = speed_gpu / 1e9; gsu = "Gkeys/s"; }
        else if (speed_gpu >= 1e6) { gsd = speed_gpu / 1e6; gsu = "Mkeys/s"; }
        else if (speed_gpu >= 1e3) { gsd = speed_gpu / 1e3; gsu = "Kkeys/s"; }

        const char *tu = "";
        double td = (double)total_gpu;
        if (total_gpu >= 1000000000000ULL) { td = total_gpu / 1e12; tu = "T"; }
        else if (total_gpu >= 1000000000ULL) { td = total_gpu / 1e9; tu = "G"; }
        else if (total_gpu >= 1000000ULL) { td = total_gpu / 1e6; tu = "M"; }
        else if (total_gpu >= 1000ULL) { td = total_gpu / 1e3; tu = "K"; }

        char current_key_str[65] = {0};
        if (g_display_mutex.try_lock()) {
            if (g_display_valid) {
                u256_to_hex(current_key_str, g_display_key);
            }
            g_display_mutex.unlock();
        }

        char *trim_key = current_key_str;
        while (*trim_key == '0' && *(trim_key+1) != '\0') trim_key++;

        uint64_t fp = g_bloom_fp.load(std::memory_order_relaxed);
        if (fp > 0) {
            printf("\r[⚡ BULK] GPU: %.2f %s | Total: %.2f%s | FP filtered: %llu | %s\033[K",
                   gsd, gsu, td, tu, (unsigned long long)fp,
                   current_key_str[0] ? trim_key : (g_found.load() ? "FOUND ✅" : "Searching..."));
        } else {
            printf("\r[⚡ BULK] GPU: %.2f %s | Total: %.2f%s | %s\033[K",
                   gsd, gsu, td, tu,
                   current_key_str[0] ? trim_key : (g_found.load() ? "FOUND ✅" : "Searching..."));
        }
        fflush(stdout);
    }

    g_running.store(false);

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    printf("\n");

    auto end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - g_start_time).count() / 1000.0;
    uint64_t total = g_total_gpu_keys.load();

    printf("\n[📊] Total: %llu (%.2e) in %.1fs",
           (unsigned long long)total, (double)total, total_time);
    if (use_g) {
        printf(" | Avg GPU: %.2f Mkeys/s", total / total_time / 1e6);
    }
    printf(" | %s\n", g_found.load() ? "FOUND ✅" : "Not found ❌ (Stopped)");

    return 0;
}

// ================================================================
//  Main
// ================================================================
int main(int argc, char** argv) {
    Config config;
    if (!parse_cli(argc, argv, config)) {
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // ── Bulk mode: bloom filter multi-address search ──
    if (config.bulk_mode) {
        return run_bulk_mode(config);
    }

    // ── Single-target mode (original behavior) ──
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
    printf("║  🔑 puzzlesolver v2.4 - Ultra Performance CPU/GPU    ║\n");
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
