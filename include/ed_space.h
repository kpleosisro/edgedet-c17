#ifndef ED_SPACE_H
#define ED_SPACE_H

#include "edgedet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* FPGA-only flight frame: no JPEG, no .edb, no malloc in this API.
   Ground tools may fill the RGB buffer from a file; the spacecraft fills
   it from the camera. */

#define ED_SPACE_SIZE 320u
#define ED_SPACE_TEACHER_MAX 8u

typedef struct {
    float x1, y1, x2, y2;
    float score;
    uint32_t class_id;
} ed_space_box;

typedef struct {
    uint8_t rgb[ED_SPACE_SIZE * ED_SPACE_SIZE * 3u];
    uint8_t gray[ED_SPACE_SIZE * ED_SPACE_SIZE];
    ed_space_box teacher[ED_SPACE_TEACHER_MAX];
    uint32_t teacher_count;
    ed_space_box det[ED_MAX_DETECTIONS];
    uint32_t det_count;
} ed_space_frame;

ed_status ed_space_frame_init(ed_space_frame *frame);
/* Nearest-neighbor resize into the static 320x320 slot. */
ed_status ed_space_ingest_rgb(ed_space_frame *frame,
                              const uint8_t *rgb, uint32_t width, uint32_t height,
                              uint32_t stride_bytes);
/* Integer plate teacher: white lobes on a black field. Classes 0..n-1 left to right. */
ed_status ed_space_teacher(ed_space_frame *frame);
/* Integer detector using existing FPGA-model forward. */
ed_status ed_space_forward(ed_space_frame *frame, const ed_model *model,
                           float score_threshold, float nms_threshold);
/* Head-only integer update from teacher boxes. No-op if teacher_count==0. */
ed_status ed_space_adapt_head(ed_space_frame *frame, ed_model *model);
/* One camera tick: ingest (already done) -> teacher -> forward -> adapt. */
ed_status ed_space_step(ed_space_frame *frame, ed_model *model,
                        int do_forward, int do_adapt);

#ifdef __cplusplus
}
#endif
#endif
