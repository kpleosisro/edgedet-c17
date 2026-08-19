#include "ed_space.h"
#include "ed_hw.h"
#include "ed_internal.h"
#include <string.h>

ed_status ed_space_frame_init(ed_space_frame *frame){
    if(!frame)return ED_ERR_ARGUMENT;
    memset(frame,0,sizeof(*frame));
    return ED_OK;
}

ed_status ed_space_ingest_rgb(ed_space_frame *frame,const uint8_t *rgb,
                              uint32_t width,uint32_t height,uint32_t stride_bytes){
    uint32_t y,x;
    if(!frame||!rgb||!width||!height||stride_bytes<width*3u)return ED_ERR_ARGUMENT;
    for(y=0;y<ED_SPACE_SIZE;++y){
        uint32_t sy=(uint32_t)(((uint64_t)y*height)/ED_SPACE_SIZE);
        if(sy>=height)sy=height-1u;
        for(x=0;x<ED_SPACE_SIZE;++x){
            uint32_t sx=(uint32_t)(((uint64_t)x*width)/ED_SPACE_SIZE);
            const uint8_t *p;uint8_t *d;uint32_t g;
            if(sx>=width)sx=width-1u;
            p=rgb+(size_t)sy*stride_bytes+(size_t)sx*3u;
            d=frame->rgb+((size_t)y*ED_SPACE_SIZE+x)*3u;
            d[0]=p[0];d[1]=p[1];d[2]=p[2];
            g=((uint32_t)p[0]+(uint32_t)p[1]+(uint32_t)p[2])/3u;
            frame->gray[(size_t)y*ED_SPACE_SIZE+x]=(uint8_t)g;
        }
    }
    frame->teacher_count=0;frame->det_count=0;
    return ED_OK;
}

ed_status ed_space_forward(ed_space_frame *frame,const ed_model *model,
                           float score_threshold,float nms_threshold){
    ed_image im;ed_detection tmp[ED_MAX_DETECTIONS];ed_detection_list list={tmp,ED_MAX_DETECTIONS,0};
    ed_status s;uint32_t i;ed_compute old;
    if(!frame||!model)return ED_ERR_ARGUMENT;
    im.rgb=frame->rgb;im.width=ED_SPACE_SIZE;im.height=ED_SPACE_SIZE;im.stride_bytes=ED_SPACE_SIZE*3u;
    old=ed_runtime_compute();
    (void)ed_runtime_set_compute(ED_COMPUTE_FPGA_MODEL);
    s=ed_hw_predict(model,&im,score_threshold,nms_threshold,&list);
    (void)ed_runtime_set_compute(old);
    if(s!=ED_OK)return s;
    frame->det_count=0;
    for(i=0;i<list.count&&frame->det_count<ED_MAX_DETECTIONS;++i){
        ed_space_box *b=&frame->det[frame->det_count++];
        b->x1=tmp[i].x1;b->y1=tmp[i].y1;b->x2=tmp[i].x2;b->y2=tmp[i].y2;
        b->score=tmp[i].score;b->class_id=tmp[i].class_id;
    }
    return ED_OK;
}

ed_status ed_space_step(ed_space_frame *frame,ed_model *model,int do_forward,int do_adapt){
    ed_status s;
    if(!frame)return ED_ERR_ARGUMENT;
    s=ed_space_teacher(frame);if(s!=ED_OK)return s;
    if(do_forward){
        if(!model)return ED_ERR_ARGUMENT;
        s=ed_space_forward(frame,model,0.25f,0.42f);if(s!=ED_OK)return s;
    }
    if(do_adapt){
        if(!model)return ED_ERR_ARGUMENT;
        s=ed_space_adapt_head(frame,model);if(s!=ED_OK)return s;
    }
    return ED_OK;
}
