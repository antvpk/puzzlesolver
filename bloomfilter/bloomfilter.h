#ifndef BLOOMFILTER_H
#define BLOOMFILTER_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <array>

// ================================================================
//  Bloom filter for bulk address matching
//  - Host side: build from address files, export raw bit array
//  - GPU side:  upload bit array, probe in kernel
// ================================================================

// Bloom filter parameters
// A blocked filter must size itself from the BLOCK density, not from the
// classic scattered-bit formula: see BloomFilter::init. k is fixed at 20 —
// high enough to keep the false-positive rate at target, low enough that the
// 512-bit block stays only about a third full and a probe exits after a couple
// of iterations.
#define BLOOM_DEFAULT_NUM_HASHES    20

// Compact struct for GPU upload - just the bit array and parameters
struct BloomFilterGPU {
    uint64_t *bits;          // Device pointer to bit array
    uint64_t  num_bits;      // Total bits in filter
    uint64_t  num_bits_mask; // num_bits - 1 (must be power of 2)
    uint32_t  num_hashes;    // Number of hash functions (k)
};

// Host-side bloom filter
class BloomFilter {
public:
    BloomFilter();
    ~BloomFilter();

    // Initialize with expected element count and desired FPR
    void init(uint64_t expected_elements, double fpr = 1e-7);

    // Initialize with explicit parameters
    void init_explicit(uint64_t num_bits, uint32_t num_hashes);

    // Insert a 20-byte hash160 into the bloom filter
    void insert(const uint8_t h160[20]);

    // Test if a 20-byte hash160 might be in the bloom filter (GPU fallback)
    bool test(const uint8_t h160[20]) const;

    // Verify exactly if a hash160 is in the database (CPU double-check)
    bool exact_test(const uint8_t h160[20]) const;
    void sort_exact(); // Call after all inserts

    // Get raw bit array for GPU upload
    const uint64_t* get_bits() const { return bits_; }
    uint64_t get_num_bits() const { return num_bits_; }
    uint64_t get_num_words() const { return num_words_; }
    uint32_t get_num_hashes() const { return num_hashes_; }
    uint64_t get_size_bytes() const { return num_words_ * sizeof(uint64_t); }
    uint64_t get_count() const { return count_; }

private:
    uint64_t *bits_;
    uint64_t  num_bits_;
    uint64_t  num_words_;
    uint32_t  num_hashes_;
    uint64_t  count_;
    
    std::vector<std::array<uint8_t, 20>> exact_hashes_;

    void compute_indices(const uint8_t h160[20], uint64_t *indices) const;
};

// ================================================================
//  Address file management
// ================================================================

// Address database file URLs (TSV format: address\tbalance)
struct AddressFileInfo {
    const char *url;
    const char *filename;
    const char *description;
};

// Known address database files
// These are the standard Bitcoin rich-list / funded-address dumps.
extern const AddressFileInfo ADDR_FILES[];
extern const int NUM_ADDR_FILES;

// Download an address file if not already present. Returns true on success.
bool download_address_file(const std::string &filename, const std::string &url);

// Address-type bits for load_addresses_to_bloom's type_mask. These mirror the
// BULK_TYPE_* bits in cli.h and the GPU kernel's addr_type codes.
//   0x1 = compressed P2PKH + P2WPKH (bc1q)   0x2 = uncompressed P2PKH   0x4 = P2SH
// An address is inserted only if a type the search actually computes can match it,
// so disabled types add no noise to the filter and do not raise its FPR.

// Count the addresses in a file that load_addresses_to_bloom would insert for
// the same type_mask, without decoding or inserting anything. Size the filter
// with this before init() so it is built for the real element count.
// Returns 0 (and prints why) if the file cannot be opened.
uint64_t count_addresses_in_file(const std::string &filepath, uint32_t type_mask);

// Load addresses from a file into a bloom filter, keeping only those an enabled
// search type can match. type_mask uses the bits above.
// Supported inputs per line: one address, or TSV/CSV (address<sep>balance).
//   1xxx (P2PKH)  -> inserted if compressed OR uncompressed is enabled
//   3xxx (P2SH)   -> inserted if P2SH is enabled
//   bc1q (P2WPKH) -> inserted if compressed is enabled (same hash160 as compressed P2PKH)
//   bc1p (P2TR)   -> always skipped (a 32-byte taproot key has no hash160 to match;
//                    the old 20-byte "proxy" was pure noise that only raised the FPR)
// Returns the number of addresses inserted.
uint64_t load_addresses_to_bloom(const std::string &filepath, BloomFilter &bf,
                                 uint32_t type_mask);

// Decode a matchable Bitcoin address to its 20-byte hash160.
// For P2PKH/P2SH: extracts hash160 from base58check.
// For P2WPKH (bc1q): extracts 20-byte witness program.
// P2TR (bc1p) is rejected: its 32-byte witness program is not a hash160, so it
// can never match and must not be inserted into the filter.
// Returns true on success.
bool decode_address_to_h160(const std::string &addr, uint8_t h160[20]);

// Bech32/Bech32m decode to witness program
bool bech32_decode_to_program(const std::string &addr, std::vector<uint8_t> &program, int &witver);

#endif // BLOOMFILTER_H
