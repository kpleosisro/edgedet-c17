#ifndef ED_HW_H
#define ED_HW_H

#include "edgedet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* FPGA-model compute: integer add/sub/mul/shift/compare + ROM lookups.
   Host golden (ED_COMPUTE_HOST) is unchanged FP32. */

typedef enum {
    ED_COMPUTE_HOST = 0,
    ED_COMPUTE_FPGA_MODEL = 1
} ed_compute;

typedef int16_t ed_hw_q;
typedef int8_t ed_hw_w;
typedef int64_t ed_hw_acc;

#define ED_HW_Q_MAX 32767
#define ED_HW_Q_MIN (-32768)
#define ED_HW_LUT 256

typedef struct {
    int32_t mult;
    int32_t shift;
} ed_hw_scale;

ed_status ed_runtime_set_compute(ed_compute compute);
ed_compute ed_runtime_compute(void);

/* Verilog: acc <= acc + $signed(a) * $signed(b); */
ed_hw_acc ed_hw_mac(ed_hw_acc acc, ed_hw_q a, ed_hw_w b);
/* Verilog: y <= a + b; */
int32_t ed_hw_add32(int32_t a, int32_t b);
/* Verilog: y <= a - b; */
int32_t ed_hw_sub32(int32_t a, int32_t b);
/* Verilog: sat to signed 16. */
ed_hw_q ed_hw_sat16(ed_hw_acc value);
/* Verilog: y <= (acc * mult) >>> shift; then sat16. */
ed_hw_q ed_hw_requant(ed_hw_acc acc, ed_hw_scale scale);
/* Verilog: y <= (x < 0) ? 0 : x; */
ed_hw_q ed_hw_relu(ed_hw_q x);
/* Verilog: y <= (x < lo) ? lo : ((x > hi) ? hi : x); */
ed_hw_q ed_hw_clip(ed_hw_q x, ed_hw_q lo, ed_hw_q hi);
/* Hard-sigmoid: clip(x/6 + 0.5, 0, 1). x is Q8 (real = x/256). Output Q15. */
ed_hw_q ed_hw_hard_sigmoid_q8(ed_hw_q x_q8);
/* ROM logistic. x_q8 real = x/256. Output Q15 in [0, 32767]. */
ed_hw_q ed_hw_sigmoid_lut(ed_hw_q x_q8);
/* ROM exp for DFL. x_q8 is (logit-max) usually <= 0. Output Q10. */
int32_t ed_hw_exp_lut(ed_hw_q x_q8);
/* ROM sqrt. x_q15 in [0, 32767] meaning [0, 1]. Output Q15. */
ed_hw_q ed_hw_sqrt_lut(ed_hw_q x_q15);

void ed_hw_lut_init(void);
const int16_t *ed_hw_sigmoid_rom(void);
const int32_t *ed_hw_exp_rom(void);
const int16_t *ed_hw_sqrt_rom(void);

void ed_hw_conv2d(const ed_hw_q *input, uint32_t in_h, uint32_t in_w, uint32_t in_c,
                  const ed_hw_w *weights, const int32_t *bias, uint32_t out_c,
                  uint32_t kernel, uint32_t stride, uint32_t padding,
                  ed_hw_acc *acc);
void ed_hw_depthwise_conv2d(const ed_hw_q *input, uint32_t in_h, uint32_t in_w,
                            uint32_t channels, const ed_hw_w *weights,
                            const int32_t *bias, uint32_t kernel, uint32_t stride,
                            uint32_t padding, ed_hw_acc *acc);

void ed_hw_requant_buffer(const ed_hw_acc *acc, size_t n, ed_hw_scale scale, ed_hw_q *out);
ed_hw_scale ed_hw_scale_from_max(ed_hw_acc max_abs, int32_t target);

ed_status ed_hw_predict(const ed_model *model, const ed_image *image,
                        float score_threshold, float nms_threshold,
                        ed_detection_list *detections);
ed_status ed_hw_train(ed_model *model, const ed_dataset *dataset,
                      const ed_train_config *config, ed_train_report *report);

#ifdef __cplusplus
}
#endif
#endif
