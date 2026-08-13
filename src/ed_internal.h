#ifndef ED_INTERNAL_H
#define ED_INTERNAL_H

#include "edgedet.h"
#include <stdio.h>

#define ED_EDM_MAGIC 0x314d4445u /* EDM1 */
#define ED_EDB_MAGIC 0x31424445u /* EDB1 */
#define ED_ENDIAN_TAG 0x01020304u
#define ED_MODEL_FLAG_FROZEN_BACKBONE 1u
#define ED_ANN_FLAG_IGNORE 1u
#define ED_DTYPE_FP16 4u
#define ED_DTYPE_INT8_STORAGE 5u
#define ED_DTYPE_INT4_STORAGE 6u
#define ED_MAX_TENSOR_RANK 4u
#define ED_PICODET_LEVELS 4u

#if defined(_MSC_VER)
#pragma pack(push, 1)
#define ED_PACKED
#else
#define ED_PACKED __attribute__((packed))
#endif

typedef struct ED_PACKED {
    uint32_t magic;
    uint16_t major;
    uint16_t minor;
    uint32_t endian_tag;
    uint32_t header_bytes;
    uint32_t architecture;
    uint32_t precision;
    uint32_t flags;
    uint32_t tensor_count;
    uint32_t class_count;
    uint32_t metadata_bytes;
    uint64_t tensor_table_offset;
    uint64_t class_table_offset;
    uint64_t metadata_offset;
    uint64_t payload_offset;
    uint64_t file_bytes;
    uint8_t payload_sha256[32];
    uint8_t source_sha256[32];
    uint32_t header_crc32;
} ed_edm_header;

typedef struct ED_PACKED {
    char name[ED_TENSOR_NAME_BYTES];
    uint32_t dtype;
    uint32_t rank;
    uint32_t dims[ED_MAX_TENSOR_RANK];
    uint64_t data_offset;
    uint64_t data_bytes;
    uint32_t data_crc32;
    uint32_t flags;
} ed_edm_tensor_disk;

typedef struct ED_PACKED {
    uint32_t magic;
    uint16_t major;
    uint16_t minor;
    uint32_t endian_tag;
    uint32_t header_bytes;
    uint32_t record_count;
    uint32_t class_count;
    uint64_t record_table_offset;
    uint64_t class_table_offset;
    uint64_t data_offset;
    uint64_t file_bytes;
    uint32_t header_crc32;
    uint8_t reserved[12];
} ed_edb_header;

typedef struct ED_PACKED {
    uint64_t image_offset;
    uint32_t image_bytes;
    uint32_t image_crc32;
    uint32_t width;
    uint32_t height;
    uint64_t annotation_offset;
    uint32_t annotation_count;
    uint32_t record_crc32;
} ed_edb_record_disk;

typedef struct ED_PACKED {
    float x1, y1, x2, y2;
    uint16_t class_id;
    uint16_t flags;
} ed_edb_annotation_disk;

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

typedef struct {
    char name[ED_TENSOR_NAME_BYTES];
    uint32_t dtype;
    uint32_t rank;
    uint32_t dims[ED_MAX_TENSOR_RANK];
    uint64_t data_bytes;
    void *data;
    uint32_t flags;
} ed_tensor;

struct ed_model {
    uint32_t architecture;
    uint32_t precision;
    uint32_t flags;
    uint32_t class_count;
    char (*class_names)[ED_CLASS_NAME_BYTES];
    uint32_t tensor_count;
    ed_tensor *tensors;
    char *metadata;
    uint32_t metadata_bytes;
    uint8_t source_sha256[32];
};

struct ed_dataset {
    const uint8_t *file_data;
    size_t file_bytes;
    void *file_handle;
    void *mapping_handle;
    ed_edb_header header;
    const ed_edb_record_disk *records;
    const char (*class_names)[ED_CLASS_NAME_BYTES];
};

typedef struct {
    const uint8_t *data;
    size_t size;
    void *file_handle;
    void *mapping_handle;
} ed_mapped_file;

uint32_t ed_crc32(const void *data, size_t size);
void ed_sha256(const void *data, size_t size, uint8_t digest[32]);
int ed_digest_equal(const uint8_t a[32], const uint8_t b[32]);
uint64_t ed_monotonic_ms(void);
uint64_t ed_monotonic_ns(void);
ed_status ed_read_entire_file(const char *path, uint8_t **data, size_t *size);
ed_status ed_write_entire_file(const char *path, const void *data, size_t size);
ed_status ed_mapped_file_open(const char *path, ed_mapped_file *file);
void ed_mapped_file_close(ed_mapped_file *file);
ed_tensor *ed_find_tensor(ed_model *model, const char *name);
const ed_tensor *ed_find_tensor_const(const ed_model *model, const char *name);
int ed_find_source_class(const ed_model *model, const char *name);
int ed_cpu_has_avx2_fma(void);
int ed_cpu_has_neon(void);
typedef void (*ed_parallel_fn)(void *context,size_t begin,size_t end);
ed_status ed_parallel_for(size_t count,ed_parallel_fn function,void *context);
#if defined(ED_HAVE_AVX2_IMPL)
void ed_conv2d_f32_avx2_range(const float*,uint32_t,uint32_t,uint32_t,const float*,const float*,uint32_t,uint32_t,uint32_t,uint32_t,float*,size_t,size_t);
void ed_depthwise_conv2d_f32_avx2_range(const float*,uint32_t,uint32_t,uint32_t,const float*,const float*,uint32_t,uint32_t,uint32_t,float*,size_t,size_t);
#endif
#if defined(ED_HAVE_NEON_IMPL)
void ed_conv2d_f32_neon_range(const float*,uint32_t,uint32_t,uint32_t,const float*,const float*,uint32_t,uint32_t,uint32_t,uint32_t,float*,size_t,size_t);
void ed_depthwise_conv2d_f32_neon_range(const float*,uint32_t,uint32_t,uint32_t,const float*,const float*,uint32_t,uint32_t,uint32_t,float*,size_t,size_t);
#endif
ed_status ed_decode_image(const uint8_t *bytes, size_t byte_count,
                          uint8_t **rgb, uint32_t *width, uint32_t *height);
ed_status ed_dataset_record(const ed_dataset *dataset, uint32_t index,
                            const uint8_t **image_bytes, uint32_t *image_size,
                            const ed_edb_annotation_disk **annotations,
                            uint32_t *annotation_count, uint32_t *width,
                            uint32_t *height);
ed_status ed_prepare_input_320(const ed_image *image, float **out_input);
void ed_runtime_power_mark_begin(void);
void ed_runtime_power_mark_end(void);

#endif
