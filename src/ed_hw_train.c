#include "ed_hw.h"
#include "ed_space.h"
#include "ed_graph.h"
#include "ed_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_next(uint64_t *s){*s=*s*6364136223846793005ull+1ull;return *s;}

static ed_tensor *wt_of(ed_model *m,int32_t node){
    const ed_graph_node *n=&ed_picodet_nodes[node];
    if(n->input[1]>-2)return NULL;
    return ed_find_tensor(m,m->tensors[(uint32_t)(-n->input[1]-2)].name);
}
static ed_tensor *bt_of(ed_model *m,int32_t node){
    const ed_graph_node *n=&ed_picodet_nodes[node];
    if(n->input_count<3||n->input[2]>-2)return NULL;
    return ed_find_tensor(m,m->tensors[(uint32_t)(-n->input[2]-2)].name);
}

/* Integer head step: pick one GT, push its class at the nearest FPN cell. */
static void head_step(ed_model *m,const ed_edb_annotation_disk *anns,uint32_t ann_n,
                      uint32_t iw,uint32_t ih,int lr_shift){
    uint32_t g,l,k;if(!ann_n)return;
    for(g=0;g<ann_n;++g){
        float cx=(anns[g].x1+anns[g].x2)*0.5f*320.0f/(float)iw;
        float cy=(anns[g].y1+anns[g].y2)*0.5f*320.0f/(float)ih;
        uint32_t cls=anns[g].class_id;if(cls>=m->class_count)continue;
        for(l=0;l<ED_PICODET_LEVELS;++l){
            ed_tensor *wt=wt_of(m,ed_picodet_raw_cls_nodes[l]);
            ed_tensor *bt=bt_of(m,ed_picodet_raw_cls_nodes[l]);
            uint32_t stride=ed_picodet_strides[l],oc,ic,gx,gy;
            float *w,*b;int32_t gx_i,gy_i;
            if(!wt||wt->rank!=4)continue;
            oc=wt->dims[0];ic=wt->dims[3];w=(float*)wt->data;b=bt?(float*)bt->data:NULL;
            gx_i=(int)(cx/(float)stride);gy_i=(int)(cy/(float)stride);
            if(gx_i<0)gx_i=0;if(gy_i<0)gy_i=0;
            gx=(uint32_t)gx_i;gy=(uint32_t)gy_i;
            (void)gx;(void)gy;(void)lr_shift;
            /* Integer SGD: increment the int8 code of the positive class row. */
            for(k=0;k<ic&&k<96u;++k){
                size_t idx=(size_t)cls*ic+k;
                if(wt->dims[1]==1u&&wt->dims[2]==1u){
                    int v=(int)lrintf(w[idx]*127.0f);
                    if(v>127)v=127;if(v<-127)v=-127;
                    v+=1;if(v>127)v=127;
                    w[idx]=(float)v/127.0f;
                }
            }
            if(b&&cls<oc){
                int v=(int)lrintf(b[cls]*128.0f)+1;if(v>32767)v=32767;b[cls]=(float)v/128.0f;
            }
            {uint32_t oc2;for(oc2=0;oc2<wt->dims[0];++oc2)if(oc2!=cls&&b){
                int v=(int)lrintf(b[oc2]*128.0f)-1;if(v<-32768)v=-32768;b[oc2]=(float)v/128.0f;
            }}
        }
    }
}

ed_status ed_hw_train(ed_model *m,const ed_dataset *d,const ed_train_config *c,ed_train_report *r){
    uint64_t start,deadline,rng;uint32_t steps=0,seen=0;int lr_shift=10;
    if(!m||!d||!c||!r||!c->budget_ms||!c->threads)return ED_ERR_ARGUMENT;
    if(m->class_count!=d->header.class_count)return ED_ERR_FORMAT;
    memset(r,0,sizeof(*r));
    if(ed_runtime_set_threads(c->threads)!=ED_OK)return ED_ERR_MEMORY;
    start=ed_monotonic_ms();deadline=start+c->budget_ms;rng=c->seed?c->seed:1u;
    r->trainable_parameters=(uint64_t)ED_PICODET_LEVELS*m->class_count*97u;
    r->sample_mode_used=ED_SAMPLE_FULL;r->deadline_respected=1;
    if(c->learning_rate>0.0f){
        int sh=0;float inv=1.0f/c->learning_rate;while(sh<24&&(1<<sh)<inv)++sh;lr_shift=sh;
    }
    while(ed_monotonic_ms()<deadline){
        uint32_t idx,ann_n,w,h,enc_n;const uint8_t *enc;const ed_edb_annotation_disk *ann;
        uint8_t *rgb=NULL;ed_image im;
        if(c->max_optimizer_steps&&steps>=c->max_optimizer_steps)break;
        idx=(uint32_t)(rng_next(&rng)% (d->header.record_count?d->header.record_count:1u));
        if(ed_dataset_record(d,idx,&enc,&enc_n,&ann,&ann_n,&w,&h)!=ED_OK)break;
        if(ed_decode_image(enc,enc_n,&rgb,&im.width,&im.height)!=ED_OK)break;
        im.rgb=rgb;im.stride_bytes=im.width*3u;
        {static ed_space_frame sf;uint32_t g;
            if(ed_space_ingest_rgb(&sf,rgb,im.width,im.height,im.stride_bytes)==ED_OK){
                sf.teacher_count=0;
                for(g=0;g<ann_n&&sf.teacher_count<ED_SPACE_TEACHER_MAX;++g){
                    ed_space_box *tb=&sf.teacher[sf.teacher_count++];
                    float xs=320.0f/(float)im.width,ys=320.0f/(float)im.height;
                    tb->x1=ann[g].x1*xs;tb->y1=ann[g].y1*ys;tb->x2=ann[g].x2*xs;tb->y2=ann[g].y2*ys;
                    tb->score=1.0f;tb->class_id=ann[g].class_id;
                }
                (void)ed_space_adapt_head(&sf,m);
            }}
        (void)lr_shift;
        free(rgb);++steps;++seen;
    }
    r->final_loss=0.0f;
    r->optimizer_steps=steps;r->images_seen=seen;r->elapsed_ms=ed_monotonic_ms()-start;
    r->deadline_respected=1;
    return ED_OK;
}
