#ifndef BASE58_H
#define BASE58_H

#include <string>
#include <vector>
#include <cstdint>
#include <stddef.h>

std::string base58_encode(const uint8_t *data, size_t len);
bool base58_decode(const char *psz, std::vector<uint8_t>& vch);
std::string base58check_encode(const uint8_t *data, size_t len);

bool decode_address_to_hash160(const std::string& address, uint8_t* hash160_out);

#endif
