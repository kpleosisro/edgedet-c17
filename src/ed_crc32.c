#include "ed_internal.h"

uint32_t ed_crc32(const void *data, size_t size) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xffffffffu;
    size_t i;
    for (i = 0; i < size; ++i) {
        uint32_t x = (crc ^ p[i]) & 0xffu;
        unsigned bit;
        for (bit = 0; bit < 8u; ++bit)
            x = (x >> 1u) ^ (0xedb88320u & (0u - (x & 1u)));
        crc = (crc >> 8u) ^ x;
    }
    return crc ^ 0xffffffffu;
}
