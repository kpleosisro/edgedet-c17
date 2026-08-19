#include "ed_space.h"
#include "ed_graph.h"
#include "ed_internal.h"
#include <math.h>

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

ed_status ed_space_adapt_head(ed_space_frame *frame,ed_model *model){
    uint32_t t,l,k;
    if(!frame||!model)return ED_ERR_ARGUMENT;
    if(frame->teacher_count==0u)return ED_OK; /* no labels -> do not train */
    for(t=0;t<frame->teacher_count;++t){
        const ed_space_box *box=&frame->teacher[t];
        float cx=(box->x1+box->x2)*0.5f,cy=(box->y1+box->y2)*0.5f;
        uint32_t cls=box->class_id;if(cls>=model->class_count)continue;
        for(l=0;l<ED_PICODET_LEVELS;++l){
            ed_tensor *wt=wt_of(model,ed_picodet_raw_cls_nodes[l]);
            ed_tensor *bt=bt_of(model,ed_picodet_raw_cls_nodes[l]);
            float *w,*b;uint32_t oc,ic;
            if(!wt||wt->rank!=4||wt->dims[1]!=1u||wt->dims[2]!=1u)continue;
            oc=wt->dims[0];ic=wt->dims[3];w=(float*)wt->data;b=bt?(float*)bt->data:NULL;
            (void)cx;(void)cy;
            for(k=0;k<ic&&k<96u;++k){
                size_t idx=(size_t)cls*ic+k;int v=(int)lrintf(w[idx]*127.0f);
                if(v>127)v=127;if(v<-127)v=-127;v+=1;if(v>127)v=127;
                w[idx]=(float)v/127.0f;
            }
            if(b&&cls<oc){int v=(int)lrintf(b[cls]*128.0f)+1;if(v>32767)v=32767;b[cls]=(float)v/128.0f;}
            if(b){uint32_t o;for(o=0;o<oc;++o)if(o!=cls){int v=(int)lrintf(b[o]*128.0f)-1;if(v<-32768)v=-32768;b[o]=(float)v/128.0f;}}
        }
    }
    return ED_OK;
}
