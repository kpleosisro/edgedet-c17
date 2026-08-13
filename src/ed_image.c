#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#include "ed_internal.h"
#include <limits.h>
#include <stdlib.h>

ed_status ed_decode_image(const uint8_t *bytes, size_t byte_count,
                          uint8_t **rgb, uint32_t *width, uint32_t *height) {
    int w, h, channels;
    stbi_uc *pixels;
    if (!bytes || !byte_count || byte_count > INT_MAX || !rgb || !width || !height)
        return ED_ERR_ARGUMENT;
    *rgb = NULL;
    pixels = stbi_load_from_memory(bytes, (int)byte_count, &w, &h, &channels, 3);
    if (!pixels || w <= 0 || h <= 0) return ED_ERR_FORMAT;
    *rgb = pixels;
    *width = (uint32_t)w;
    *height = (uint32_t)h;
    return ED_OK;
}
