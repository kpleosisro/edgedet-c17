#ifndef EDGEDET_H
#define EDGEDET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ED_VERSION_MAJOR 0u
#define ED_VERSION_MINOR 1u
#define ED_MAX_CLASSES 256u
#define ED_MAX_DETECTIONS 512u
#define ED_CLASS_NAME_BYTES 32u
#define ED_TENSOR_NAME_BYTES 64u

typedef enum {
    ED_OK = 0,
    ED_ERR_ARGUMENT,
    ED_ERR_IO,
    ED_ERR_FORMAT,
    ED_ERR_CHECKSUM,
    ED_ERR_MEMORY,
    ED_ERR_UNSUPPORTED,
    ED_ERR_DEADLINE,
    ED_ERR_INTERNAL
} ed_status;

typedef enum {
    ED_PRECISION_FP32 = 1,
    ED_PRECISION_INT8 = 2,
    ED_PRECISION_INT4 = 3
} ed_precision;

typedef enum {
    ED_STORAGE_FP32 = 1,
    ED_STORAGE_FP16 = 2,
    ED_STORAGE_INT8 = 3,
    ED_STORAGE_INT4 = 4
} ed_storage;

typedef enum {
    ED_ARCH_PICODET_S_320 = 1
} ed_architecture;

typedef enum {
    ED_TRAIN_OUTPUTS = 1,
    ED_TRAIN_HEAD = 2,
    ED_TRAIN_CLASSIFICATION = 3
} ed_train_scope;

typedef enum {
    ED_HEAD_ADAPTER_INDEPENDENT = 0,
    ED_HEAD_ADAPTER_CROSS_LEVEL = 1
} ed_head_adapter;

typedef enum {
    ED_SAMPLE_AUTO = 0,
    ED_SAMPLE_FULL = 1,
    ED_SAMPLE_ZOOM = 2,
    ED_SAMPLE_MIXED = 3,
    ED_SAMPLE_MOSAIC = 4,
    ED_SAMPLE_TILE = 5,
    ED_SAMPLE_ADAPT = 6
} ed_sample_mode;

typedef enum {
    ED_INIT_AUTO = 0,
    ED_INIT_BEST_SINGLE = 1,
    ED_INIT_EQUAL_BLEND = 2,
    ED_INIT_AP_WEIGHTED = 3,
    ED_INIT_PROTOTYPE = 4,
    ED_INIT_PROTOTYPE_RESIDUAL = 5
} ed_class_init_mode;

typedef enum {
    ED_PROTO_RIDGE = 0,
    ED_PROTO_DIAG = 1,
    ED_PROTO_LDA = 2
} ed_proto_solver;

typedef struct ed_model ed_model;
typedef struct ed_dataset ed_dataset;

typedef struct {
    const uint8_t *rgb;
    uint32_t width;
    uint32_t height;
    uint32_t stride_bytes;
} ed_image;

typedef struct {
    float x1, y1, x2, y2;
    float score;
    uint32_t class_id;
} ed_detection;

typedef struct {
    ed_detection *items;
    size_t capacity;
    size_t count;
} ed_detection_list;

typedef struct {
    uint64_t budget_ms;
    uint32_t threads;
    uint64_t seed;
    ed_train_scope scope;
    uint32_t mosaic_size;
    uint32_t accumulation;
    float learning_rate;
    float final_learning_rate;
    float momentum;
    float weight_decay;
    const char *checkpoint_prefix;
    uint32_t feature_cache_samples;
    uint32_t max_optimizer_steps;
    ed_sample_mode sample_mode;
    float ema_decay;
    float wise_mix;
    float dfl_lr_scale;
    uint32_t restart_after_steps;
    ed_storage checkpoint_storage;
    const float *class_lr_scale;
    const float *class_wise_mix;
    uint32_t class_loss_norm;
    uint32_t hard_neg_boost;
    ed_head_adapter head_adapter;
    float cross_level_mix;
    uint32_t nesterov;
    uint32_t roi_head;
    uint32_t picofeat_adapter;
} ed_train_config;

typedef struct {
    uint64_t elapsed_ms;
    uint64_t images_seen;
    uint64_t mosaics_seen;
    uint64_t optimizer_steps;
    uint64_t trainable_parameters;
    float final_loss;
    int deadline_respected;
    uint32_t sample_mode_used;
    float median_box_side;
    float assign_recall;
    float assign_mean_iou;
} ed_train_report;

typedef struct {
    uint32_t class_count;
    float per_class_ap[ED_MAX_CLASSES];
    float per_class_recall[ED_MAX_CLASSES];
    float map50;
    float map50_11point;
} ed_map_report;

typedef struct {
    uint32_t class_count;
    uint32_t source_count[ED_MAX_CLASSES];
    uint32_t source_class[ED_MAX_CLASSES][4];
    float source_affinity[ED_MAX_CLASSES][4];
    float blended_affinity[ED_MAX_CLASSES];
    uint32_t calibrated_objects[ED_MAX_CLASSES];
    uint32_t init_mode_used;
    uint32_t proto_solver_used;
    uint32_t proto_positives[ED_MAX_CLASSES];
    uint32_t proto_negatives[ED_MAX_CLASSES];
    float fold_map50;
    float fold_ap[ED_MAX_CLASSES];
    float classifier_norm[ED_MAX_CLASSES];
} ed_class_calibration_report;

const char *ed_status_string(ed_status status);

ed_status ed_model_load(const char *path, ed_model **out_model);
ed_status ed_model_save(const ed_model *model, const char *path);
ed_status ed_model_save_fp16(const ed_model *model, const char *path);
ed_status ed_model_save_int8(const ed_model *model, const char *path);
ed_status ed_model_save_int4(const ed_model *model, const char *path);
ed_status ed_model_save_storage(const ed_model *model, const char *path,
                                ed_storage storage);
void ed_model_free(ed_model *model);
uint32_t ed_model_class_count(const ed_model *model);
const char *ed_model_class_name(const ed_model *model, uint32_t class_id);
ed_status ed_model_remap_classes(ed_model *model, const char *const *class_names,
                                 uint32_t class_count);
ed_status ed_model_remap_class_indices(ed_model *model,
                                       const char *const *class_names,
                                       const uint32_t *source_class,
                                       uint32_t class_count);
ed_status ed_model_remap_class_blends(ed_model *model,
                                      const char *const *class_names,
                                      const uint32_t *source_class,
                                      const uint32_t *source_count,
                                      uint32_t class_count);
ed_status ed_model_calibrate_classes(ed_model *model,
                                     const ed_dataset *dataset,
                                     uint32_t samples_per_class,
                                     uint64_t seed,
                                     ed_class_calibration_report *report);
ed_status ed_model_calibrate_classes_ex(ed_model *model,
                                        const ed_dataset *dataset,
                                        uint32_t samples_per_class,
                                        uint64_t seed,
                                        ed_class_init_mode init_mode,
                                        ed_proto_solver proto_solver,
                                        ed_class_calibration_report *report);
ed_status ed_model_remap_class_heads(ed_model *model,
                                     const char *const *class_names,
                                     uint32_t class_count,
                                     const float *weights,
                                     const float *biases);
size_t ed_process_peak_rss(void);
ed_status ed_runtime_set_threads(uint32_t threads);
uint32_t ed_runtime_threads(void);
ed_status ed_runtime_set_power_limit_w(float watts);
float ed_runtime_power_limit_w(void);
ed_status ed_runtime_set_busy_core_watts(float watts);
float ed_runtime_busy_core_watts(void);
void ed_runtime_power_apply(void);
float ed_runtime_last_average_w(void);
float ed_runtime_last_sleep_ms(void);

ed_status ed_dataset_load(const char *path, ed_dataset **out_dataset);
void ed_dataset_free(ed_dataset *dataset);
uint64_t ed_dataset_count(const ed_dataset *dataset);

ed_status ed_predict(const ed_model *model, const ed_image *image,
                     float score_threshold, float nms_threshold,
                     ed_detection_list *detections);
ed_status ed_predict_auto(const ed_model *model, const ed_image *image,
                          float score_threshold, float nms_threshold,
                          ed_detection_list *detections);
ed_status ed_predict_tiled(const ed_model *model, const ed_image *image,
                           float score_threshold, float nms_threshold,
                           uint32_t tiles_x, uint32_t tiles_y,
                           float tile_overlap,
                           int include_full_image,
                           ed_detection_list *detections);
ed_status ed_train(ed_model *model, const ed_dataset *dataset,
                   const ed_train_config *config, ed_train_report *report);
ed_status ed_evaluate_map50(const ed_model *model, const ed_dataset *dataset,
                            float score_threshold, float nms_threshold,
                            ed_map_report *report);
ed_status ed_evaluate_map50_tiled(const ed_model *model,
                                  const ed_dataset *dataset,
                                  float score_threshold,
                                  float nms_threshold,
                                  uint32_t tiles_x, uint32_t tiles_y,
                                  float tile_overlap,
                                  int include_full_image,
                                  ed_map_report *report);
ed_status ed_evaluate_map50_ex(const ed_model *model, const ed_dataset *dataset,
                               float score_threshold, float nms_threshold,
                               uint32_t tiles_x, uint32_t tiles_y,
                               float tile_overlap, int include_full_image,
                               float soft_nms_sigma, ed_map_report *report);

/* Low-level deterministic kernels are public for embedded integration/tests. */
void ed_conv2d_f32(const float *input, uint32_t in_h, uint32_t in_w,
                   uint32_t in_c, const float *weights, const float *bias,
                   uint32_t out_c, uint32_t kernel, uint32_t stride,
                   uint32_t padding, float *output);
void ed_depthwise_conv2d_f32(const float *input, uint32_t in_h, uint32_t in_w,
                             uint32_t channels, const float *weights,
                             const float *bias, uint32_t kernel,
                             uint32_t stride, uint32_t padding, float *output);
void ed_hardswish_f32(float *values, size_t count);
void ed_dfl_decode_f32(const float *logits, uint32_t locations,
                       uint32_t reg_max, float *distances);
size_t ed_nms(ed_detection *items, size_t count, float iou_threshold);
size_t ed_nms_soft(ed_detection *items, size_t count, float sigma,
                   float score_threshold);
uint64_t ed_picodet_forward_flops(void);
float ed_estimate_energy_mj(uint32_t forwards, float watts, float latency_ms);

#ifdef __cplusplus
}
#endif
#endif
