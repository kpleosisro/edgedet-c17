#ifndef ED_GRAPH_H
#define ED_GRAPH_H

#include "ed_internal.h"

typedef enum {
    ED_OP_CONV = 1, ED_OP_ADD, ED_OP_CLIP, ED_OP_MUL, ED_OP_DIV,
    ED_OP_GLOBAL_AVG, ED_OP_RELU, ED_OP_HARD_SIGMOID,
    ED_OP_RESIZE_NEAREST, ED_OP_CONCAT_CHANNEL, ED_OP_SIGMOID, ED_OP_SQRT
} ed_graph_op;

/* References: -1=image, <=-2 model tensor (-2-index), >=0 node output. */
typedef struct {
    uint16_t op, input_count;
    int32_t input[4];
    uint16_t out_h, out_w, out_c;
    uint16_t kernel, stride, padding, groups;
} ed_graph_node;

typedef struct { float *data; uint32_t h,w,c; } ed_activation;

extern const ed_graph_node ed_picodet_nodes[];
extern const uint32_t ed_picodet_node_count;
extern const char *const ed_picodet_tensor_names[];
extern const uint32_t ed_picodet_tensor_count;
extern const int32_t ed_picodet_cls_nodes[ED_PICODET_LEVELS];
extern const int32_t ed_picodet_reg_nodes[ED_PICODET_LEVELS];
extern const int32_t ed_picodet_feature_nodes[ED_PICODET_LEVELS];
extern const int32_t ed_picodet_raw_cls_nodes[ED_PICODET_LEVELS];
extern const uint32_t ed_picodet_strides[ED_PICODET_LEVELS];

ed_status ed_graph_execute(const ed_model *model,const float *image,ed_activation **out);
ed_status ed_graph_apply_context(const ed_model *model,ed_activation *activations);
ed_status ed_graph_apply_spatial(const ed_model *model,ed_activation *activations);
ed_status ed_graph_apply_quality(const ed_model *model,ed_activation *activations);
void ed_graph_activations_free(ed_activation *a);
#endif
