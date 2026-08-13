#include "ed_internal.h"
#include "ed_graph.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ED_HEAD_CHANNELS 96u
#define ED_REG_BINS 8u
#define ED_LEVEL_LOCATIONS (40u*40u+20u*20u+10u*10u+5u*5u)
#define ED_ASSIGN_MAX 20u
#define ED_SMALL_SIDE 32.0f
#define ED_FEATURE_ADAPTER_RANK 8u

typedef struct {
    float *gw,*gb,*vw,*vb,*ema_w,*ema_b,*init_w,*init_b,*before_w,*before_b;
    ed_tensor *w,*b;
} head_parameter;
typedef head_parameter context_parameter;
typedef head_parameter spatial_parameter;
typedef head_parameter quality_parameter;
typedef struct {int gt;float quality,target;} assignment;
typedef struct {float x1,y1,x2,y2;uint32_t class_id;} train_box;
typedef struct {
    float iou_power,center_radius,small_boost;
    uint32_t topk_small,topk_large;
} assign_cfg;

static uint64_t rng_next(uint64_t *state){uint64_t x=*state;x^=x>>12;x^=x<<25;x^=x>>27;*state=x;return x*UINT64_C(2685821657736338717);}
static float rng_unit(uint64_t *state){return (float)(rng_next(state)>>11)*(1.0f/9007199254740992.0f);}
static int cmp_float(const void *a,const void *b){float x=*(const float*)a,y=*(const float*)b;return x<y?-1:x>y?1:0;}
static float clampf(float v,float lo,float hi){if(v<lo)return lo;if(v>hi)return hi;return v;}
static uint32_t clampu(uint32_t v,uint32_t lo,uint32_t hi){if(v<lo)return lo;if(v>hi)return hi;return v;}

static float overlap(float ax1,float ay1,float ax2,float ay2,float bx1,float by1,float bx2,float by2){
    float x1=ax1>bx1?ax1:bx1,y1=ay1>by1?ay1:by1,x2=ax2<bx2?ax2:bx2,y2=ay2<by2?ay2:by2,w=x2-x1,h=y2-y1,inter,den;
    if(w<=0.0f||h<=0.0f)return 0.0f;inter=w*h;den=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter;return den>0.0f?inter/den:0.0f;
}
static float giou_loss(const float box[4],const train_box *g){
    float iou=overlap(box[0],box[1],box[2],box[3],g->x1,g->y1,g->x2,g->y2);
    float cx1=box[0]<g->x1?box[0]:g->x1,cy1=box[1]<g->y1?box[1]:g->y1;
    float cx2=box[2]>g->x2?box[2]:g->x2,cy2=box[3]>g->y2?box[3]:g->y2;
    float c=(cx2-cx1)*(cy2-cy1),a=(box[2]-box[0])*(box[3]-box[1]),b=(g->x2-g->x1)*(g->y2-g->y1);
    float inter=iou*(a+b)/(1.0f+iou);
    return 1.0f-(iou-(c-(a+b-inter))/(c+1e-9f));
}
static void decode_one(const float *logits,float cx,float cy,float stride,float box[4],float dist[4],float probs[32]){
    uint32_t side,k;
    for(side=0;side<4u;++side){
        const float *q=logits+side*ED_REG_BINS;float m=q[0],sum=0.0f,e=0.0f;
        for(k=1;k<ED_REG_BINS;++k)if(q[k]>m)m=q[k];
        for(k=0;k<ED_REG_BINS;++k){float p=expf(q[k]-m);probs[side*ED_REG_BINS+k]=p;sum+=p;}
        for(k=0;k<ED_REG_BINS;++k){float p=probs[side*ED_REG_BINS+k]/sum;probs[side*ED_REG_BINS+k]=p;e+=p*(float)k;}
        dist[side]=e;
    }
    box[0]=cx-dist[0]*stride;box[1]=cy-dist[1]*stride;box[2]=cx+dist[2]*stride;box[3]=cy+dist[3]*stride;
}
static int load_boxes(const ed_edb_annotation_disk *a,uint32_t count,uint32_t w,uint32_t h,train_box **out,uint32_t *out_n){
    train_box *b=(train_box*)malloc((count?count:1u)*sizeof(*b));uint32_t i,n=0;
    if(!b)return 0;
    for(i=0;i<count;++i)if(!(a[i].flags&ED_ANN_FLAG_IGNORE)){
        b[n].x1=a[i].x1*320.0f/(float)w;b[n].y1=a[i].y1*320.0f/(float)h;
        b[n].x2=a[i].x2*320.0f/(float)w;b[n].y2=a[i].y2*320.0f/(float)h;
        b[n].class_id=a[i].class_id;++n;
    }
    *out=b;*out_n=n;return 1;
}
static int ignored_location(const ed_edb_annotation_disk *a,uint32_t n,uint32_t w,uint32_t h,float x,float y){
    uint32_t i;
    for(i=0;i<n;++i)if((a[i].flags&ED_ANN_FLAG_IGNORE)&&x>=a[i].x1*320.0f/(float)w&&x<a[i].x2*320.0f/(float)w&&y>=a[i].y1*320.0f/(float)h&&y<a[i].y2*320.0f/(float)h)return 1;
    return 0;
}
static float dataset_mean_aspect(const ed_dataset *d){
    uint32_t i,n=d->header.record_count;float sum=0.0f,count=0.0f;
    if(n>64u)n=64u;
    for(i=0;i<n;++i){
        const ed_edb_record_disk *r=&d->records[i];
        if(r->width==0u||r->height==0u)continue;
        sum+=(float)r->width/(float)r->height;++count;
    }
    return count>0.0f?sum/count:1.0f;
}
static float dataset_median_side(const ed_dataset *d){
    float *sides=NULL,median=0.0f;uint32_t i,j,n=0,cap=0;
    for(i=0;i<d->header.record_count;++i){
        const ed_edb_record_disk *r=&d->records[i];
        const ed_edb_annotation_disk *a=(const ed_edb_annotation_disk*)(d->file_data+r->annotation_offset);
        for(j=0;j<r->annotation_count;++j){
            float bw,bh,side;if(a[j].flags&ED_ANN_FLAG_IGNORE||r->width==0u||r->height==0u)continue;
            bw=(a[j].x2-a[j].x1)*320.0f/(float)r->width;bh=(a[j].y2-a[j].y1)*320.0f/(float)r->height;
            side=sqrtf((bw>0.0f?bw:0.0f)*(bh>0.0f?bh:0.0f));
            if(n==cap){uint32_t next=cap?cap*2u:256u;void *p=realloc(sides,(size_t)next*sizeof(*sides));if(!p){free(sides);return 0.0f;}sides=(float*)p;cap=next;}
            sides[n++]=side;
        }
    }
    if(n){qsort(sides,n,sizeof(*sides),cmp_float);median=sides[n/2u];}
    free(sides);return median;
}
static uint32_t resolve_policy(const ed_train_config *c,float median,float aspect){
    if(c->sample_mode>=ED_SAMPLE_FULL&&c->sample_mode<=ED_SAMPLE_ADAPT)return (uint32_t)c->sample_mode;
    if(c->mosaic_size==4u)return ED_SAMPLE_MOSAIC;
    if(c->mosaic_size==1u)return ED_SAMPLE_FULL;
    /* Mosaic shrinks already-small boxes. Zoom crops around objects instead. */
    (void)aspect;
    if(median<48.0f)return ED_SAMPLE_ZOOM;
    return ED_SAMPLE_FULL;
}
static assign_cfg resolve_assign(float median){
    assign_cfg a;
    /* Official PicoDet SimOTA (score * IoU^6, center-in-box, top-13).
       Extra positives and IoU^2 assignment hurt this frozen-head adapter. */
    (void)median;
    a.iou_power=6.0f;a.center_radius=0.0f;a.small_boost=0.0f;a.topk_small=13u;a.topk_large=13u;
    return a;
}

static int parameter_init(ed_model *m,head_parameter *p){
    static const char *wn[8]={"conv2d_84.w_0","conv2d_85.w_0","conv2d_88.w_0","conv2d_89.w_0","conv2d_92.w_0","conv2d_93.w_0","conv2d_96.w_0","conv2d_97.w_0"};
    static const char *bn[8]={"1166","1167","1169","1170","1172","1173","1175","1176"};
    uint32_t i;
    for(i=0;i<8u;++i){
        size_t nw,nb;
        p[i].w=ed_find_tensor(m,wn[i]);p[i].b=ed_find_tensor(m,bn[i]);if(!p[i].w||!p[i].b)return 0;
        nw=(size_t)p[i].w->data_bytes/4u;nb=(size_t)p[i].b->data_bytes/4u;
        p[i].gw=(float*)calloc(nw,sizeof(float));p[i].gb=(float*)calloc(nb,sizeof(float));
        p[i].vw=(float*)calloc(nw,sizeof(float));p[i].vb=(float*)calloc(nb,sizeof(float));
        p[i].ema_w=(float*)calloc(nw,sizeof(float));p[i].ema_b=(float*)calloc(nb,sizeof(float));
        p[i].init_w=(float*)malloc(nw*sizeof(float));p[i].init_b=(float*)malloc(nb*sizeof(float));
        p[i].before_w=(float*)calloc(nw,sizeof(float));p[i].before_b=(float*)calloc(nb,sizeof(float));
        if(!p[i].gw||!p[i].gb||!p[i].vw||!p[i].vb||!p[i].ema_w||!p[i].ema_b||!p[i].init_w||!p[i].init_b||!p[i].before_w||!p[i].before_b)return 0;
        memcpy(p[i].init_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].init_b,p[i].b->data,nb*sizeof(float));
        memcpy(p[i].ema_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].ema_b,p[i].b->data,nb*sizeof(float));
    }
    return 1;
}
static int context_model_add(ed_model*m){
    static const char *wn[ED_PICODET_LEVELS]={"ed.roi.0.w","ed.roi.1.w","ed.roi.2.w","ed.roi.3.w"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.roi.0.b","ed.roi.1.b","ed.roi.2.b","ed.roi.3.b"};
    ed_tensor *grown;uint32_t l,old;if(ed_find_tensor(m,wn[0])){for(l=0;l<ED_PICODET_LEVELS;++l)if(!ed_find_tensor(m,wn[l])||!ed_find_tensor(m,bn[l]))return 0;return 1;}old=m->tensor_count;grown=(ed_tensor*)realloc(m->tensors,(size_t)(old+2u*ED_PICODET_LEVELS)*sizeof(*grown));if(!grown)return 0;m->tensors=grown;memset(m->tensors+old,0,2u*ED_PICODET_LEVELS*sizeof(*grown));
    for(l=0;l<ED_PICODET_LEVELS;++l){ed_tensor*w=&m->tensors[old+2u*l],*b=&m->tensors[old+2u*l+1u];strcpy(w->name,wn[l]);w->dtype=ED_PRECISION_FP32;w->rank=2;w->dims[0]=m->class_count;w->dims[1]=1;w->dims[2]=1;w->dims[3]=ED_HEAD_CHANNELS;w->data_bytes=(uint64_t)m->class_count*ED_HEAD_CHANNELS*sizeof(float);w->data=calloc(1,(size_t)w->data_bytes);w->flags=1u;strcpy(b->name,bn[l]);b->dtype=ED_PRECISION_FP32;b->rank=1;b->dims[0]=m->class_count;b->dims[1]=b->dims[2]=b->dims[3]=1;b->data_bytes=(uint64_t)m->class_count*sizeof(float);b->data=calloc(1,(size_t)b->data_bytes);b->flags=1u;if(!w->data||!b->data){m->tensor_count=old+2u*ED_PICODET_LEVELS;return 0;}}
    m->tensor_count=old+2u*ED_PICODET_LEVELS;return 1;
}
static int context_parameter_init(ed_model*m,context_parameter*p){
    static const char *wn[ED_PICODET_LEVELS]={"ed.roi.0.w","ed.roi.1.w","ed.roi.2.w","ed.roi.3.w"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.roi.0.b","ed.roi.1.b","ed.roi.2.b","ed.roi.3.b"};
    uint32_t i;for(i=0;i<ED_PICODET_LEVELS;++i){size_t nw,nb;p[i].w=ed_find_tensor(m,wn[i]);p[i].b=ed_find_tensor(m,bn[i]);if(!p[i].w||!p[i].b)return 0;nw=(size_t)p[i].w->data_bytes/4u;nb=(size_t)p[i].b->data_bytes/4u;p[i].gw=(float*)calloc(nw,sizeof(float));p[i].gb=(float*)calloc(nb,sizeof(float));p[i].vw=(float*)calloc(nw,sizeof(float));p[i].vb=(float*)calloc(nb,sizeof(float));p[i].ema_w=(float*)calloc(nw,sizeof(float));p[i].ema_b=(float*)calloc(nb,sizeof(float));p[i].init_w=(float*)malloc(nw*sizeof(float));p[i].init_b=(float*)malloc(nb*sizeof(float));p[i].before_w=(float*)calloc(nw,sizeof(float));p[i].before_b=(float*)calloc(nb,sizeof(float));if(!p[i].gw||!p[i].gb||!p[i].vw||!p[i].vb||!p[i].ema_w||!p[i].ema_b||!p[i].init_w||!p[i].init_b||!p[i].before_w||!p[i].before_b)return 0;memcpy(p[i].init_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].init_b,p[i].b->data,nb*sizeof(float));memcpy(p[i].ema_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].ema_b,p[i].b->data,nb*sizeof(float));}return 1;
}
static int spatial_model_add(ed_model*m){
    static const char *wn[ED_PICODET_LEVELS]={"ed.picofeat.0.a","ed.picofeat.1.a","ed.picofeat.2.a","ed.picofeat.3.a"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.picofeat.0.b","ed.picofeat.1.b","ed.picofeat.2.b","ed.picofeat.3.b"};
    ed_tensor *grown;uint32_t l,old;if(ed_find_tensor(m,wn[0])){for(l=0;l<ED_PICODET_LEVELS;++l)if(!ed_find_tensor(m,wn[l])||!ed_find_tensor(m,bn[l]))return 0;return 1;}old=m->tensor_count;grown=(ed_tensor*)realloc(m->tensors,(size_t)(old+2u*ED_PICODET_LEVELS)*sizeof(*grown));if(!grown)return 0;m->tensors=grown;memset(m->tensors+old,0,2u*ED_PICODET_LEVELS*sizeof(*grown));
    for(l=0;l<ED_PICODET_LEVELS;++l){ed_tensor*w=&m->tensors[old+2u*l],*b=&m->tensors[old+2u*l+1u];size_t i,n;strcpy(w->name,wn[l]);w->dtype=ED_PRECISION_FP32;w->rank=4;w->dims[0]=m->class_count;w->dims[1]=ED_FEATURE_ADAPTER_RANK;w->dims[2]=1;w->dims[3]=ED_HEAD_CHANNELS;w->data_bytes=(uint64_t)m->class_count*ED_FEATURE_ADAPTER_RANK*ED_HEAD_CHANNELS*sizeof(float);w->data=malloc((size_t)w->data_bytes);w->flags=1u;strcpy(b->name,bn[l]);b->dtype=ED_PRECISION_FP32;b->rank=2;b->dims[0]=m->class_count;b->dims[1]=ED_FEATURE_ADAPTER_RANK;b->dims[2]=b->dims[3]=1;b->data_bytes=(uint64_t)m->class_count*ED_FEATURE_ADAPTER_RANK*sizeof(float);b->data=calloc(1,(size_t)b->data_bytes);b->flags=1u;if(!w->data||!b->data){m->tensor_count=old+2u*ED_PICODET_LEVELS;return 0;}n=(size_t)w->data_bytes/sizeof(float);for(i=0;i<n;++i)((float*)w->data)[i]=((float)((i*37u+l*101u)%257u)/256.0f-0.5f)*0.10f;}
    m->tensor_count=old+2u*ED_PICODET_LEVELS;return 1;
}
static int spatial_parameter_init(ed_model*m,spatial_parameter*p){
    static const char *wn[ED_PICODET_LEVELS]={"ed.picofeat.0.a","ed.picofeat.1.a","ed.picofeat.2.a","ed.picofeat.3.a"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.picofeat.0.b","ed.picofeat.1.b","ed.picofeat.2.b","ed.picofeat.3.b"};
    uint32_t i;for(i=0;i<ED_PICODET_LEVELS;++i){size_t nw,nb;p[i].w=ed_find_tensor(m,wn[i]);p[i].b=ed_find_tensor(m,bn[i]);if(!p[i].w||!p[i].b)return 0;nw=(size_t)p[i].w->data_bytes/4u;nb=(size_t)p[i].b->data_bytes/4u;p[i].gw=(float*)calloc(nw,sizeof(float));p[i].gb=(float*)calloc(nb,sizeof(float));p[i].vw=(float*)calloc(nw,sizeof(float));p[i].vb=(float*)calloc(nb,sizeof(float));p[i].ema_w=(float*)calloc(nw,sizeof(float));p[i].ema_b=(float*)calloc(nb,sizeof(float));p[i].init_w=(float*)malloc(nw*sizeof(float));p[i].init_b=(float*)malloc(nb*sizeof(float));p[i].before_w=(float*)calloc(nw,sizeof(float));p[i].before_b=(float*)calloc(nb,sizeof(float));if(!p[i].gw||!p[i].gb||!p[i].vw||!p[i].vb||!p[i].ema_w||!p[i].ema_b||!p[i].init_w||!p[i].init_b||!p[i].before_w||!p[i].before_b)return 0;memcpy(p[i].init_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].init_b,p[i].b->data,nb*sizeof(float));memcpy(p[i].ema_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].ema_b,p[i].b->data,nb*sizeof(float));}return 1;
}
static int quality_model_add(ed_model*m){
    static const char *wn[ED_PICODET_LEVELS]={"ed.quality.0.w","ed.quality.1.w","ed.quality.2.w","ed.quality.3.w"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.quality.0.b","ed.quality.1.b","ed.quality.2.b","ed.quality.3.b"};
    ed_tensor *grown;uint32_t l,old;if(ed_find_tensor(m,wn[0])){for(l=0;l<ED_PICODET_LEVELS;++l)if(!ed_find_tensor(m,wn[l])||!ed_find_tensor(m,bn[l]))return 0;return 1;}old=m->tensor_count;grown=(ed_tensor*)realloc(m->tensors,(size_t)(old+2u*ED_PICODET_LEVELS)*sizeof(*grown));if(!grown)return 0;m->tensors=grown;memset(m->tensors+old,0,2u*ED_PICODET_LEVELS*sizeof(*grown));
    for(l=0;l<ED_PICODET_LEVELS;++l){ed_tensor*w=&m->tensors[old+2u*l],*b=&m->tensors[old+2u*l+1u];strcpy(w->name,wn[l]);w->dtype=ED_PRECISION_FP32;w->rank=2;w->dims[0]=m->class_count;w->dims[1]=ED_HEAD_CHANNELS;w->dims[2]=w->dims[3]=1u;w->data_bytes=(uint64_t)m->class_count*ED_HEAD_CHANNELS*sizeof(float);w->data=calloc(1,(size_t)w->data_bytes);w->flags=1u;strcpy(b->name,bn[l]);b->dtype=ED_PRECISION_FP32;b->rank=1;b->dims[0]=m->class_count;b->dims[1]=b->dims[2]=b->dims[3]=1u;b->data_bytes=(uint64_t)m->class_count*sizeof(float);b->data=calloc(1,(size_t)b->data_bytes);b->flags=1u;if(!w->data||!b->data){m->tensor_count=old+2u*ED_PICODET_LEVELS;return 0;}}
    m->tensor_count=old+2u*ED_PICODET_LEVELS;return 1;
}
static int quality_parameter_init(ed_model*m,quality_parameter*p){
    static const char *wn[ED_PICODET_LEVELS]={"ed.quality.0.w","ed.quality.1.w","ed.quality.2.w","ed.quality.3.w"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.quality.0.b","ed.quality.1.b","ed.quality.2.b","ed.quality.3.b"};
    uint32_t i;for(i=0;i<ED_PICODET_LEVELS;++i){size_t nw,nb;p[i].w=ed_find_tensor(m,wn[i]);p[i].b=ed_find_tensor(m,bn[i]);if(!p[i].w||!p[i].b)return 0;nw=(size_t)p[i].w->data_bytes/4u;nb=(size_t)p[i].b->data_bytes/4u;p[i].gw=(float*)calloc(nw,sizeof(float));p[i].gb=(float*)calloc(nb,sizeof(float));p[i].vw=(float*)calloc(nw,sizeof(float));p[i].vb=(float*)calloc(nb,sizeof(float));p[i].ema_w=(float*)calloc(nw,sizeof(float));p[i].ema_b=(float*)calloc(nb,sizeof(float));p[i].init_w=(float*)malloc(nw*sizeof(float));p[i].init_b=(float*)malloc(nb*sizeof(float));p[i].before_w=(float*)calloc(nw,sizeof(float));p[i].before_b=(float*)calloc(nb,sizeof(float));if(!p[i].gw||!p[i].gb||!p[i].vw||!p[i].vb||!p[i].ema_w||!p[i].ema_b||!p[i].init_w||!p[i].init_b||!p[i].before_w||!p[i].before_b)return 0;memcpy(p[i].init_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].init_b,p[i].b->data,nb*sizeof(float));memcpy(p[i].ema_w,p[i].w->data,nw*sizeof(float));memcpy(p[i].ema_b,p[i].b->data,nb*sizeof(float));}return 1;
}
static void parameter_free(head_parameter *p){
    uint32_t i;for(i=0;i<8u;++i){free(p[i].gw);free(p[i].gb);free(p[i].vw);free(p[i].vb);free(p[i].ema_w);free(p[i].ema_b);free(p[i].init_w);free(p[i].init_b);free(p[i].before_w);free(p[i].before_b);}
}
static void parameter_zero_grad(head_parameter *p){
    uint32_t i;for(i=0;i<8u;++i){memset(p[i].gw,0,(size_t)p[i].w->data_bytes);memset(p[i].gb,0,(size_t)p[i].b->data_bytes);}
}
static void context_parameter_free(context_parameter*p){uint32_t i;for(i=0;i<ED_PICODET_LEVELS;++i){free(p[i].gw);free(p[i].gb);free(p[i].vw);free(p[i].vb);free(p[i].ema_w);free(p[i].ema_b);free(p[i].init_w);free(p[i].init_b);free(p[i].before_w);free(p[i].before_b);}}
static void context_parameter_zero_grad(context_parameter*p){uint32_t i;for(i=0;i<ED_PICODET_LEVELS;++i){memset(p[i].gw,0,(size_t)p[i].w->data_bytes);memset(p[i].gb,0,(size_t)p[i].b->data_bytes);}}
static void spatial_parameter_free(spatial_parameter*p){context_parameter_free(p);}
static void spatial_parameter_zero_grad(spatial_parameter*p){context_parameter_zero_grad(p);}
static void quality_parameter_free(quality_parameter*p){context_parameter_free(p);}
static void quality_parameter_zero_grad(quality_parameter*p){context_parameter_zero_grad(p);}
static float schedule_lr(const ed_train_config *c,uint64_t step,uint64_t elapsed_ms,uint64_t *warmup_end_ms){
    float lr,progress;
    if(step<10u)return c->learning_rate*((float)step+1.0f)/10.0f;
    if(c->max_optimizer_steps>10u)progress=(float)(step-10u)/(float)(c->max_optimizer_steps-10u);
    else{if(*warmup_end_ms==0u)*warmup_end_ms=elapsed_ms;progress=c->budget_ms>*warmup_end_ms?(float)(elapsed_ms-*warmup_end_ms)/(float)(c->budget_ms-*warmup_end_ms):1.0f;}
    if(progress>1.0f)progress=1.0f;
    lr=c->final_learning_rate+0.5f*(c->learning_rate-c->final_learning_rate)*(1.0f+cosf(3.14159265359f*progress));
    return lr;
}
static void parameter_step_cross_level(head_parameter *p,const ed_train_config *c,float lr,float ema_decay,uint32_t class_count){
    uint32_t oc,k,l;
    for(oc=0;oc<class_count;++oc){
        float lr_c=lr*(c->class_lr_scale?c->class_lr_scale[oc]:1.0f);
        for(k=0;k<ED_HEAD_CHANNELS;++k){
            size_t index=(size_t)oc*ED_HEAD_CHANNELS+k;float consensus=0.0f,mean_weight=0.0f;
            for(l=0;l<ED_PICODET_LEVELS;++l){head_parameter *cp=&p[l*2u];consensus+=cp->gw[index]/(float)c->accumulation;mean_weight+=((float*)cp->w->data)[index];}
            consensus=consensus/(float)ED_PICODET_LEVELS+c->weight_decay*mean_weight/(float)ED_PICODET_LEVELS;
            for(l=0;l<ED_PICODET_LEVELS;++l){head_parameter *cp=&p[l*2u];float *weight=(float*)cp->w->data;float local=cp->gw[index]/(float)c->accumulation+c->weight_decay*weight[index];float gradient=(1.0f-c->cross_level_mix)*local+c->cross_level_mix*consensus,update;cp->vw[index]=c->momentum*cp->vw[index]+gradient;update=c->nesterov?gradient+c->momentum*cp->vw[index]:cp->vw[index];weight[index]-=lr_c*update;cp->ema_w[index]=ema_decay*cp->ema_w[index]+(1.0f-ema_decay)*weight[index];}
        }
        for(l=0;l<ED_PICODET_LEVELS;++l){head_parameter *cp=&p[l*2u];float *bias=(float*)cp->b->data;float gradient=cp->gb[oc]/(float)c->accumulation,update;cp->vb[oc]=c->momentum*cp->vb[oc]+gradient;update=c->nesterov?gradient+c->momentum*cp->vb[oc]:cp->vb[oc];bias[oc]-=lr_c*update;cp->ema_b[oc]=ema_decay*cp->ema_b[oc]+(1.0f-ema_decay)*bias[oc];}
    }
}
static float parameter_step(head_parameter *p,const ed_train_config *c,uint64_t step,uint64_t elapsed_ms,uint64_t *warmup_end_ms,float ema_decay,uint32_t class_count){
    uint32_t i;float base=schedule_lr(c,step,elapsed_ms,warmup_end_ms),dfl=c->dfl_lr_scale>0.0f?c->dfl_lr_scale:1.0f;
    if(c->scope==ED_TRAIN_QUALITY)return base;
    if(c->head_adapter==ED_HEAD_ADAPTER_CROSS_LEVEL)parameter_step_cross_level(p,c,base,ema_decay,class_count);
    for(i=0;i<8u;++i){
        size_t j,nw=(size_t)p[i].w->data_bytes/4u,nb=(size_t)p[i].b->data_bytes/4u;
        float *w=(float*)p[i].w->data,*b=(float*)p[i].b->data,lr=base;
        if(c->scope==ED_TRAIN_CLASSIFICATION&&(i&1u))continue;
        if(c->head_adapter==ED_HEAD_ADAPTER_CROSS_LEVEL&&!(i&1u))continue;
        if(i&1u)lr=base*dfl;
        if(!(i&1u)&&c->class_lr_scale&&class_count>0u){
            uint32_t oc;size_t stride=nw/class_count;
            for(oc=0;oc<class_count;++oc){
                float lr_c=lr*c->class_lr_scale[oc];size_t begin=(size_t)oc*stride,end=begin+stride,k;
                for(k=begin;k<end&&k<nw;++k){float g=p[i].gw[k]/(float)c->accumulation+c->weight_decay*w[k],update;p[i].vw[k]=c->momentum*p[i].vw[k]+g;update=c->nesterov?g+c->momentum*p[i].vw[k]:p[i].vw[k];w[k]-=lr_c*update;p[i].ema_w[k]=ema_decay*p[i].ema_w[k]+(1.0f-ema_decay)*w[k];}
                if(oc<nb){float g=p[i].gb[oc]/(float)c->accumulation,update;p[i].vb[oc]=c->momentum*p[i].vb[oc]+g;update=c->nesterov?g+c->momentum*p[i].vb[oc]:p[i].vb[oc];b[oc]-=lr_c*update;p[i].ema_b[oc]=ema_decay*p[i].ema_b[oc]+(1.0f-ema_decay)*b[oc];}
            }
            continue;
        }
        for(j=0;j<nw;++j){float g=p[i].gw[j]/(float)c->accumulation+c->weight_decay*w[j],update;p[i].vw[j]=c->momentum*p[i].vw[j]+g;update=c->nesterov?g+c->momentum*p[i].vw[j]:p[i].vw[j];w[j]-=lr*update;p[i].ema_w[j]=ema_decay*p[i].ema_w[j]+(1.0f-ema_decay)*w[j];}
        for(j=0;j<nb;++j){float g=p[i].gb[j]/(float)c->accumulation,update;p[i].vb[j]=c->momentum*p[i].vb[j]+g;update=c->nesterov?g+c->momentum*p[i].vb[j]:p[i].vb[j];b[j]-=lr*update;p[i].ema_b[j]=ema_decay*p[i].ema_b[j]+(1.0f-ema_decay)*b[j];}
    }return base;
}
static void context_parameter_step(context_parameter*p,const ed_train_config*c,float lr,float ema_decay,uint32_t class_count){uint32_t l,oc,k;for(l=0;l<ED_PICODET_LEVELS;++l){float*w=(float*)p[l].w->data,*b=(float*)p[l].b->data;for(oc=0;oc<class_count;++oc){float rate=lr*(c->class_lr_scale?c->class_lr_scale[oc]:1.0f);for(k=0;k<ED_HEAD_CHANNELS;++k){size_t i=(size_t)oc*ED_HEAD_CHANNELS+k;float g=p[l].gw[i]/(float)c->accumulation+c->weight_decay*w[i];p[l].vw[i]=c->momentum*p[l].vw[i]+g;w[i]-=rate*p[l].vw[i];p[l].ema_w[i]=ema_decay*p[l].ema_w[i]+(1.0f-ema_decay)*w[i];}{float g=p[l].gb[oc]/(float)c->accumulation;p[l].vb[oc]=c->momentum*p[l].vb[oc]+g;b[oc]-=rate*p[l].vb[oc];p[l].ema_b[oc]=ema_decay*p[l].ema_b[oc]+(1.0f-ema_decay)*b[oc];}}}}
static void spatial_parameter_step(spatial_parameter*p,const ed_train_config*c,float lr,float ema_decay,uint32_t class_count){uint32_t l,oc;for(l=0;l<ED_PICODET_LEVELS;++l){float*w=(float*)p[l].w->data,*b=(float*)p[l].b->data;size_t wstride=((size_t)p[l].w->data_bytes/4u)/class_count,bstride=((size_t)p[l].b->data_bytes/4u)/class_count;for(oc=0;oc<class_count;++oc){float rate=lr*(c->class_lr_scale?c->class_lr_scale[oc]:1.0f);size_t k,begin=(size_t)oc*wstride,end=begin+wstride;for(k=begin;k<end;++k){float g=p[l].gw[k]/(float)c->accumulation+c->weight_decay*w[k];p[l].vw[k]=c->momentum*p[l].vw[k]+g;w[k]-=rate*p[l].vw[k];p[l].ema_w[k]=ema_decay*p[l].ema_w[k]+(1.0f-ema_decay)*w[k];}begin=(size_t)oc*bstride;end=begin+bstride;for(k=begin;k<end;++k){float g=p[l].gb[k]/(float)c->accumulation;p[l].vb[k]=c->momentum*p[l].vb[k]+g;b[k]-=rate*p[l].vb[k];p[l].ema_b[k]=ema_decay*p[l].ema_b[k]+(1.0f-ema_decay)*b[k];}}}}
static void quality_parameter_step(quality_parameter*p,const ed_train_config*c,float lr,float ema_decay,uint32_t class_count){context_parameter_step(p,c,lr,ema_decay,class_count);}
static void parameter_finalize(head_parameter *p,const ed_train_config *c,int use_ema,float wise){
    uint32_t i,class_count=p[0].b?p[0].b->dims[0]:0u;
    if(c->scope==ED_TRAIN_QUALITY)return;
    if(wise<0.0f)wise=0.0f;if(wise>1.0f)wise=1.0f;
    for(i=0;i<8u;++i){
        size_t j,nw=(size_t)p[i].w->data_bytes/4u,nb=(size_t)p[i].b->data_bytes/4u;
        float *w=(float*)p[i].w->data,*b=(float*)p[i].b->data;
        if(c->scope==ED_TRAIN_CLASSIFICATION&&(i&1u))continue;
        if(!(i&1u)&&c->class_wise_mix&&class_count>0u){
            uint32_t oc;size_t stride=nw/class_count;
            for(oc=0;oc<class_count;++oc){
                float wc=c->class_wise_mix[oc];size_t begin=(size_t)oc*stride,end=begin+stride,k;
                if(wc<0.0f)wc=0.0f;if(wc>1.0f)wc=1.0f;
                for(k=begin;k<end&&k<nw;++k){float trained=use_ema?p[i].ema_w[k]:w[k];w[k]=wc*trained+(1.0f-wc)*p[i].init_w[k];}
                if(oc<nb){float trained=use_ema?p[i].ema_b[oc]:b[oc];b[oc]=wc*trained+(1.0f-wc)*p[i].init_b[oc];}
            }
            continue;
        }
        for(j=0;j<nw;++j){float trained=use_ema?p[i].ema_w[j]:w[j];w[j]=wise*trained+(1.0f-wise)*p[i].init_w[j];}
        for(j=0;j<nb;++j){float trained=use_ema?p[i].ema_b[j]:b[j];b[j]=wise*trained+(1.0f-wise)*p[i].init_b[j];}
    }
}
static void context_parameter_finalize(context_parameter*p,int use_ema,float wise){uint32_t l;for(l=0;l<ED_PICODET_LEVELS;++l){size_t i,nw=(size_t)p[l].w->data_bytes/4u,nb=(size_t)p[l].b->data_bytes/4u;float*w=(float*)p[l].w->data,*b=(float*)p[l].b->data;for(i=0;i<nw;++i){float trained=use_ema?p[l].ema_w[i]:w[i];w[i]=wise*trained+(1.0f-wise)*p[l].init_w[i];}for(i=0;i<nb;++i){float trained=use_ema?p[l].ema_b[i]:b[i];b[i]=wise*trained+(1.0f-wise)*p[l].init_b[i];}}}
static void spatial_parameter_finalize(spatial_parameter*p,int use_ema,float wise){context_parameter_finalize(p,use_ema,wise);}
static void quality_parameter_finalize(quality_parameter*p,int use_ema,float wise){context_parameter_finalize(p,use_ema,wise);}

static void parameter_restart(head_parameter *p,const ed_train_config *c,int use_ema,float wise){
    uint32_t i;parameter_finalize(p,c,use_ema,wise);
    for(i=0;i<8u;++i){size_t nw=(size_t)p[i].w->data_bytes/sizeof(float),nb=(size_t)p[i].b->data_bytes/sizeof(float);float *w=(float*)p[i].w->data,*b=(float*)p[i].b->data;
        memcpy(p[i].init_w,w,nw*sizeof(float));memcpy(p[i].init_b,b,nb*sizeof(float));memcpy(p[i].ema_w,w,nw*sizeof(float));memcpy(p[i].ema_b,b,nb*sizeof(float));memset(p[i].vw,0,nw*sizeof(float));memset(p[i].vb,0,nb*sizeof(float));
    }
}
static void context_parameter_restart(context_parameter*p,int use_ema,float wise){uint32_t l;context_parameter_finalize(p,use_ema,wise);for(l=0;l<ED_PICODET_LEVELS;++l){size_t nw=(size_t)p[l].w->data_bytes/4u,nb=(size_t)p[l].b->data_bytes/4u;float*w=(float*)p[l].w->data,*b=(float*)p[l].b->data;memcpy(p[l].init_w,w,nw*sizeof(float));memcpy(p[l].init_b,b,nb*sizeof(float));memcpy(p[l].ema_w,w,nw*sizeof(float));memcpy(p[l].ema_b,b,nb*sizeof(float));memset(p[l].vw,0,nw*sizeof(float));memset(p[l].vb,0,nb*sizeof(float));}}
static void spatial_parameter_restart(spatial_parameter*p,int use_ema,float wise){context_parameter_restart(p,use_ema,wise);}
static void quality_parameter_restart(quality_parameter*p,int use_ema,float wise){context_parameter_restart(p,use_ema,wise);}

static int record_has_class(const ed_dataset *d,uint32_t index,uint32_t class_id){
    const uint8_t *bytes;const ed_edb_annotation_disk *a;uint32_t bytes_n,n,w,h,i;
    if(ed_dataset_record(d,index,&bytes,&bytes_n,&a,&n,&w,&h)!=ED_OK)return 0;
    for(i=0;i<n;++i)if(!(a[i].flags&ED_ANN_FLAG_IGNORE)&&a[i].class_id==class_id)return 1;
    return 0;
}
static ed_status make_mosaic(const ed_dataset *d,uint64_t *rng,uint32_t sample_slot,ed_image *image,ed_edb_annotation_disk **out_a,uint32_t *out_n){
    uint8_t *canvas=(uint8_t*)calloc(320u*320u*3u,1);ed_edb_annotation_disk *all=NULL;uint32_t count=0,capacity=0,slot;
    if(!canvas)return ED_ERR_MEMORY;
    for(slot=0;slot<4u;++slot){
        uint32_t target=(sample_slot*4u+slot)%d->header.class_count,index=(uint32_t)(rng_next(rng)%d->header.record_count),attempt;
        const uint8_t *encoded;const ed_edb_annotation_disk *a;uint32_t encoded_n,n,w,h,j;uint8_t *rgb=NULL;ed_status s;
        for(attempt=0;attempt<64u&&!record_has_class(d,index,target);++attempt)index=(uint32_t)(rng_next(rng)%d->header.record_count);
        s=ed_dataset_record(d,index,&encoded,&encoded_n,&a,&n,&w,&h);if(s!=ED_OK){free(canvas);free(all);return s;}
        s=ed_decode_image(encoded,encoded_n,&rgb,&w,&h);if(s!=ED_OK){free(canvas);free(all);return s;}
        {uint32_t y,x,ox=(slot&1u)*160u,oy=(slot>>1u)*160u;
            for(y=0;y<160u;++y){uint32_t sy=(uint32_t)(((uint64_t)y*h)/160u);for(x=0;x<160u;++x){uint32_t sx=(uint32_t)(((uint64_t)x*w)/160u);memcpy(canvas+((size_t)(oy+y)*320u+ox+x)*3u,rgb+((size_t)sy*w+sx)*3u,3u);}}
            for(j=0;j<n;++j){ed_edb_annotation_disk z=a[j];if(count==capacity){uint32_t next=capacity?capacity*2u:32u;void *p=realloc(all,(size_t)next*sizeof(*all));if(!p){free(rgb);free(canvas);free(all);return ED_ERR_MEMORY;}all=(ed_edb_annotation_disk*)p;capacity=next;}
                z.x1=z.x1*160.0f/(float)w+(float)ox;z.x2=z.x2*160.0f/(float)w+(float)ox;z.y1=z.y1*160.0f/(float)h+(float)oy;z.y2=z.y2*160.0f/(float)h+(float)oy;all[count++]=z;}}
        free(rgb);
    }
    image->rgb=canvas;image->width=320u;image->height=320u;image->stride_bytes=960u;*out_a=all;*out_n=count;return ED_OK;
}
static ed_status make_single(const ed_dataset *d,uint64_t *rng,uint32_t sample_slot,ed_image *image,ed_edb_annotation_disk **out_a,uint32_t *out_n){
    uint32_t target=sample_slot%d->header.class_count,index=(uint32_t)(rng_next(rng)%d->header.record_count),attempt,encoded_n,n,w,h;const uint8_t *encoded;const ed_edb_annotation_disk *a;uint8_t *rgb=NULL;ed_edb_annotation_disk *copy;ed_status s;
    for(attempt=0;attempt<64u&&!record_has_class(d,index,target);++attempt)index=(uint32_t)(rng_next(rng)%d->header.record_count);
    s=ed_dataset_record(d,index,&encoded,&encoded_n,&a,&n,&w,&h);if(s!=ED_OK)return s;
    s=ed_decode_image(encoded,encoded_n,&rgb,&w,&h);if(s!=ED_OK)return s;
    copy=(ed_edb_annotation_disk*)malloc((n?n:1u)*sizeof(*copy));if(!copy){free(rgb);return ED_ERR_MEMORY;}
    if(n)memcpy(copy,a,(size_t)n*sizeof(*copy));
    image->rgb=rgb;image->width=w;image->height=h;image->stride_bytes=w*3u;*out_a=copy;*out_n=n;return ED_OK;
}
static ed_status make_zoom(const ed_dataset *d,uint64_t *rng,ed_image *image,ed_edb_annotation_disk **out_a,uint32_t *out_n){
    uint32_t index=(uint32_t)(rng_next(rng)%d->header.record_count),encoded_n,n,w,h,i,pick=0,valid=0;const uint8_t *encoded;const ed_edb_annotation_disk *a;uint8_t *rgb=NULL,*crop=NULL;ed_edb_annotation_disk *kept=NULL;ed_status s;
    uint32_t cw,ch,x0,y0,out_count=0;float frac;
    s=ed_dataset_record(d,index,&encoded,&encoded_n,&a,&n,&w,&h);if(s!=ED_OK)return s;
    s=ed_decode_image(encoded,encoded_n,&rgb,&w,&h);if(s!=ED_OK)return s;
    for(i=0;i<n;++i)if(!(a[i].flags&ED_ANN_FLAG_IGNORE)){if((rng_next(rng)%(++valid))==0u)pick=i;}
    frac=0.52f+0.38f*rng_unit(rng);
    cw=clampu((uint32_t)((float)w*frac+0.5f),32u,w);ch=clampu((uint32_t)((float)h*frac+0.5f),32u,h);
    if(valid){
        float cx=0.5f*(a[pick].x1+a[pick].x2),cy=0.5f*(a[pick].y1+a[pick].y2);
        float jx=(rng_unit(rng)-0.5f)*0.25f*(float)cw,jy=(rng_unit(rng)-0.5f)*0.25f*(float)ch;
        int ix=(int)(cx-0.5f*(float)cw+jx),iy=(int)(cy-0.5f*(float)ch+jy);
        if(ix<0)ix=0;if(iy<0)iy=0;if(ix+(int)cw>(int)w)ix=(int)(w-cw);if(iy+(int)ch>(int)h)iy=(int)(h-ch);
        x0=(uint32_t)ix;y0=(uint32_t)iy;
    }else{
        x0=w>cw?(uint32_t)(rng_next(rng)%(w-cw+1u)):0u;
        y0=h>ch?(uint32_t)(rng_next(rng)%(h-ch+1u)):0u;
    }
    crop=(uint8_t*)malloc((size_t)cw*ch*3u);kept=(ed_edb_annotation_disk*)malloc((n?n:1u)*sizeof(*kept));
    if(!crop||!kept){free(crop);free(kept);free(rgb);return ED_ERR_MEMORY;}
    for(i=0;i<ch;++i)memcpy(crop+(size_t)i*cw*3u,rgb+((size_t)(y0+i)*w+x0)*3u,(size_t)cw*3u);
    for(i=0;i<n;++i){
        ed_edb_annotation_disk z=a[i];float rw,rh;
        z.x1-=(float)x0;z.y1-=(float)y0;z.x2-=(float)x0;z.y2-=(float)y0;
        if(z.x1<0.0f)z.x1=0.0f;if(z.y1<0.0f)z.y1=0.0f;
        if(z.x2>(float)cw)z.x2=(float)cw;if(z.y2>(float)ch)z.y2=(float)ch;
        rw=z.x2-z.x1;rh=z.y2-z.y1;if(rw<2.0f||rh<2.0f)continue;
        kept[out_count++]=z;
    }
    free(rgb);image->rgb=crop;image->width=cw;image->height=ch;image->stride_bytes=cw*3u;*out_a=kept;*out_n=out_count;return ED_OK;
}
static void infer_tile_box(uint32_t img_w,uint32_t img_h,uint32_t tiles_x,uint32_t tiles_y,float overlap,uint32_t tx,uint32_t ty,uint32_t *x0,uint32_t *y0,uint32_t *x1,uint32_t *y1){
    float xden=(float)tiles_x-(float)(tiles_x>0u?tiles_x-1u:0u)*overlap;
    float yden=(float)tiles_y-(float)(tiles_y>0u?tiles_y-1u:0u)*overlap;
    uint32_t tile_w=(uint32_t)ceilf((float)img_w/xden),tile_h=(uint32_t)ceilf((float)img_h/yden);
    if(tile_w>img_w)tile_w=img_w;if(tile_h>img_h)tile_h=img_h;
    *x0=tiles_x<=1u?0u:(uint32_t)(((uint64_t)tx*(img_w-tile_w))/(tiles_x-1u));
    *y0=tiles_y<=1u?0u:(uint32_t)(((uint64_t)ty*(img_h-tile_h))/(tiles_y-1u));
    *x1=*x0+tile_w;*y1=*y0+tile_h;if(*x1>img_w)*x1=img_w;if(*y1>img_h)*y1=img_h;
}
static ed_status make_infer_tile(const ed_dataset *d,uint64_t *rng,uint32_t sample_slot,ed_image *image,ed_edb_annotation_disk **out_a,uint32_t *out_n){
    uint32_t index=(uint32_t)(rng_next(rng)%d->header.record_count),encoded_n,n,w,h,i,out_count=0;
    const uint8_t *encoded;const ed_edb_annotation_disk *a;uint8_t *rgb=NULL,*crop=NULL;ed_edb_annotation_disk *kept=NULL;ed_status s;
    uint32_t tiles_x=2u,tiles_y=1u,tx,ty,x0,y0,x1,y1,cw,ch;
    if((rng_next(rng)%4u)==0u)return make_single(d,rng,sample_slot,image,out_a,out_n);
    s=ed_dataset_record(d,index,&encoded,&encoded_n,&a,&n,&w,&h);if(s!=ED_OK)return s;
    s=ed_decode_image(encoded,encoded_n,&rgb,&w,&h);if(s!=ED_OK)return s;
    if((float)h>(float)w*1.6f){tiles_x=1u;tiles_y=2u;}
    tx=tiles_x>1u?(uint32_t)(rng_next(rng)%tiles_x):0u;
    ty=tiles_y>1u?(uint32_t)(rng_next(rng)%tiles_y):0u;
    infer_tile_box(w,h,tiles_x,tiles_y,0.06f,tx,ty,&x0,&y0,&x1,&y1);
    cw=x1-x0;ch=y1-y0;
    crop=(uint8_t*)malloc((size_t)cw*ch*3u);kept=(ed_edb_annotation_disk*)malloc((n?n:1u)*sizeof(*kept));
    if(!crop||!kept){free(crop);free(kept);free(rgb);return ED_ERR_MEMORY;}
    for(i=0;i<ch;++i)memcpy(crop+(size_t)i*cw*3u,rgb+((size_t)(y0+i)*w+x0)*3u,(size_t)cw*3u);
    for(i=0;i<n;++i){
        ed_edb_annotation_disk z=a[i];float rw,rh;
        z.x1-=(float)x0;z.y1-=(float)y0;z.x2-=(float)x0;z.y2-=(float)y0;
        if(z.x1<0.0f)z.x1=0.0f;if(z.y1<0.0f)z.y1=0.0f;
        if(z.x2>(float)cw)z.x2=(float)cw;if(z.y2>(float)ch)z.y2=(float)ch;
        rw=z.x2-z.x1;rh=z.y2-z.y1;if(rw<2.0f||rh<2.0f)continue;
        kept[out_count++]=z;
    }
    free(rgb);image->rgb=crop;image->width=cw;image->height=ch;image->stride_bytes=cw*3u;*out_a=kept;*out_n=out_count;return ED_OK;
}
static ed_status make_sample(const ed_dataset *d,uint64_t *rng,uint32_t policy,uint32_t slot,ed_image *image,ed_edb_annotation_disk **out_a,uint32_t *out_n){
    if(policy==ED_SAMPLE_ADAPT){
        uint32_t phase=slot%5u;
        if(phase<=2u)return make_mosaic(d,rng,slot,image,out_a,out_n);
        if(phase==3u)return make_single(d,rng,slot,image,out_a,out_n);
        return make_infer_tile(d,rng,slot,image,out_a,out_n);
    }
    if(policy==ED_SAMPLE_MOSAIC)return make_mosaic(d,rng,slot,image,out_a,out_n);
    if(policy==ED_SAMPLE_ZOOM)return make_zoom(d,rng,image,out_a,out_n);
    if(policy==ED_SAMPLE_TILE)return make_infer_tile(d,rng,slot,image,out_a,out_n);
    if(policy==ED_SAMPLE_MIXED)return (rng_next(rng)%5u)==0u?make_zoom(d,rng,image,out_a,out_n):make_single(d,rng,slot,image,out_a,out_n);
    return make_single(d,rng,slot,image,out_a,out_n);
}

typedef struct {uint32_t gt_seen,gt_hit;float iou_sum;} assign_stats;
static float train_activations(ed_model *m,const ed_image *image,const ed_edb_annotation_disk *anns,uint32_t ann_n,head_parameter *p,context_parameter *context,spatial_parameter *spatial,quality_parameter *quality,const ed_activation *a,const assign_cfg *cfg,const ed_train_config *cfg_train,assign_stats *st){
    train_box *gt=NULL;uint32_t gt_n=0,l,total=0,g;assignment *as=NULL;float *class_target=NULL;float loss=0.0f,assigned_score_sum=0.0f;uint32_t positives=0;uint32_t *pos_per_class=NULL;
    if(!load_boxes(anns,ann_n,image->width,image->height,&gt,&gt_n)||(gt_n==0u&&!(cfg_train&&cfg_train->aligned_loss))){free(gt);return 0.0f;}
    {uint32_t pi;for(pi=0;pi<8u;++pi){memcpy(p[pi].before_w,p[pi].gw,(size_t)p[pi].w->data_bytes);memcpy(p[pi].before_b,p[pi].gb,(size_t)p[pi].b->data_bytes);}}
    if(context){uint32_t pi;for(pi=0;pi<ED_PICODET_LEVELS;++pi){memcpy(context[pi].before_w,context[pi].gw,(size_t)context[pi].w->data_bytes);memcpy(context[pi].before_b,context[pi].gb,(size_t)context[pi].b->data_bytes);}}
    if(spatial){uint32_t pi;for(pi=0;pi<ED_PICODET_LEVELS;++pi){memcpy(spatial[pi].before_w,spatial[pi].gw,(size_t)spatial[pi].w->data_bytes);memcpy(spatial[pi].before_b,spatial[pi].gb,(size_t)spatial[pi].b->data_bytes);}}
    if(quality){uint32_t pi;for(pi=0;pi<ED_PICODET_LEVELS;++pi){memcpy(quality[pi].before_w,quality[pi].gw,(size_t)quality[pi].w->data_bytes);memcpy(quality[pi].before_b,quality[pi].gb,(size_t)quality[pi].b->data_bytes);}}
    as=(assignment*)malloc(ED_LEVEL_LOCATIONS*sizeof(*as));class_target=(float*)calloc((size_t)ED_LEVEL_LOCATIONS*m->class_count,sizeof(*class_target));
    pos_per_class=(uint32_t*)calloc(m->class_count,sizeof(*pos_per_class));if(!as||!class_target||!pos_per_class)goto done;
    for(g=0;g<ED_LEVEL_LOCATIONS;++g){as[g].gt=-1;as[g].quality=0.0f;as[g].target=0.0f;}
    for(g=0;g<gt_n;++g){
        float best_metric[ED_ASSIGN_MAX],best_iou[ED_ASSIGN_MAX],gw,gh,side,gcx,gcy;
        uint32_t best_index[ED_ASSIGN_MAX],best_n=0,base=0,topk;
        memset(best_metric,0,sizeof(best_metric));memset(best_iou,0,sizeof(best_iou));memset(best_index,0,sizeof(best_index));
        gw=gt[g].x2-gt[g].x1;gh=gt[g].y2-gt[g].y1;side=sqrtf((gw>0.0f?gw:0.0f)*(gh>0.0f?gh:0.0f));
        gcx=0.5f*(gt[g].x1+gt[g].x2);gcy=0.5f*(gt[g].y1+gt[g].y2);
        topk=side<ED_SMALL_SIDE?cfg->topk_small:cfg->topk_large;if(topk>ED_ASSIGN_MAX)topk=ED_ASSIGN_MAX;if(topk==0u)topk=1u;
        for(l=0;l<ED_PICODET_LEVELS;++l){
            const ed_activation *sc=&a[ed_picodet_cls_nodes[l]],*rg=&a[ed_picodet_reg_nodes[l]];
            uint32_t y,x;float stride=(float)ed_picodet_strides[l],radius=cfg->center_radius*stride;
            for(y=0;y<rg->h;++y)for(x=0;x<rg->w;++x){
                float cx=((float)x+0.5f)*stride,cy=((float)y+0.5f)*stride,box[4],dist[4],prob[32],iou,metric,score,dx,dy;
                uint32_t idx=base+y*rg->w+x,k,pos;int inside,near;
                inside=cx>=gt[g].x1&&cx<=gt[g].x2&&cy>=gt[g].y1&&cy<=gt[g].y2;
                dx=cx-gcx;if(dx<0.0f)dx=-dx;dy=cy-gcy;if(dy<0.0f)dy=-dy;
                near=radius>0.0f&&dx<=radius&&dy<=radius;
                if(!inside&&!near)continue;
                decode_one(rg->data+((size_t)y*rg->w+x)*32u,cx,cy,stride,box,dist,prob);
                iou=overlap(box[0],box[1],box[2],box[3],gt[g].x1,gt[g].y1,gt[g].x2,gt[g].y2);
                if(!inside&&iou<0.05f)continue;
                score=sc->data[((size_t)y*rg->w+x)*m->class_count+gt[g].class_id];
                metric=(score>0.0f?score:0.0f)*powf(iou>0.0f?iou:0.0f,cfg->iou_power);
                if(best_n<topk){pos=best_n++;}
                else{pos=0;for(k=1;k<topk;++k)if(best_metric[k]<best_metric[pos])pos=k;if(metric<=best_metric[pos])continue;}
                best_metric[pos]=metric;best_iou[pos]=iou;best_index[pos]=idx;
            }
            base+=rg->h*rg->w;
        }
        {int hit=0;float best=0.0f,max_metric=0.0f,max_iou=0.0f;
            for(l=0;l<best_n;++l)if(best_metric[l]>0.0f){if(best_metric[l]>max_metric)max_metric=best_metric[l];if(best_iou[l]>max_iou)max_iou=best_iou[l];}
            for(l=0;l<best_n;++l)if(best_metric[l]>0.0f){float target=cfg_train&&cfg_train->aligned_loss?best_metric[l]*max_iou/(max_metric+1e-9f):best_iou[l];size_t ct=(size_t)best_index[l]*m->class_count+gt[g].class_id;if(!(cfg_train&&cfg_train->aligned_loss)&&best_iou[l]>class_target[ct])class_target[ct]=best_iou[l];if(best_iou[l]>as[best_index[l]].quality){as[best_index[l]].gt=(int)g;as[best_index[l]].quality=best_iou[l];as[best_index[l]].target=target;}if(best_iou[l]>best)best=best_iou[l];hit=1;}
            if(st){++st->gt_seen;if(hit){++st->gt_hit;st->iou_sum+=best;}}}
    }
    if(cfg_train&&cfg_train->aligned_loss)for(g=0;g<ED_LEVEL_LOCATIONS;++g)if(as[g].gt>=0){uint32_t class_id=gt[as[g].gt].class_id;if(class_id<m->class_count)class_target[(size_t)g*m->class_count+class_id]=as[g].target;}
    total=0;
    for(l=0;l<ED_PICODET_LEVELS;++l){
        const ed_activation *feat=&a[ed_picodet_feature_nodes[l]],*raw=&a[ed_picodet_raw_cls_nodes[l]],*rg=&a[ed_picodet_reg_nodes[l]];
        head_parameter *cp=&p[l*2u],*rp=&p[l*2u+1u];uint32_t y,x,c,k;float stride=(float)ed_picodet_strides[l];
        for(y=0;y<rg->h;++y)for(x=0;x<rg->w;++x){
            uint32_t local=y*rg->w+x;assignment q=as[total+local];float cx=((float)x+0.5f)*stride,cy=((float)y+0.5f)*stride;
            const float *f=feat->data+(size_t)local*ED_HEAD_CHANNELS;int ignore=ignored_location(anns,ann_n,image->width,image->height,cx,cy);
            float box_w=1.0f,feature_mean[ED_HEAD_CHANNELS]={0};
            if(context){float roi_box[4],roi_dist[4],roi_prob[32];int x0,y0,x1,y1,iy,ix;uint32_t count=0;decode_one(rg->data+(size_t)local*32u,cx,cy,stride,roi_box,roi_dist,roi_prob);x0=(int)floorf(roi_box[0]/stride);y0=(int)floorf(roi_box[1]/stride);x1=(int)ceilf(roi_box[2]/stride);y1=(int)ceilf(roi_box[3]/stride);if(x0<0)x0=0;if(y0<0)y0=0;if(x1>(int)feat->w)x1=(int)feat->w;if(y1>(int)feat->h)y1=(int)feat->h;if(x1<=x0)x1=x0+1;if(y1<=y0)y1=y0+1;for(iy=y0;iy<y1;++iy)for(ix=x0;ix<x1;++ix){const float*fp=feat->data+((size_t)iy*feat->w+(uint32_t)ix)*ED_HEAD_CHANNELS;for(k=0;k<ED_HEAD_CHANNELS;++k)feature_mean[k]+=fp[k];++count;}for(k=0;k<ED_HEAD_CHANNELS;++k)feature_mean[k]/=(float)count;}
            if(q.gt>=0&&cfg->small_boost>0.0f){
                float bw=gt[q.gt].x2-gt[q.gt].x1,bh=gt[q.gt].y2-gt[q.gt].y1,side=sqrtf((bw>0.0f?bw:0.0f)*(bh>0.0f?bh:0.0f));
                box_w=sqrtf(ED_SMALL_SIDE/clampf(side,8.0f,ED_SMALL_SIDE));
            }
            for(c=0;c<m->class_count;++c){
                float z=raw->data[(size_t)local*m->class_count+c],pred=1.0f/(1.0f+expf(-z));
                float target=class_target[((size_t)total+local)*m->class_count+c];
                float weight,grad,quality_grad=0.0f;
                if(ignore&&q.gt<0)continue;
                if(cfg_train&&cfg_train->aligned_loss){
                    const ed_activation *score=&a[ed_picodet_cls_nodes[l]];float aligned=clampf(score->data[(size_t)local*m->class_count+c],1e-6f,1.0f-1e-6f),dlds;
                    weight=target>0.0f?target:0.75f*aligned*aligned;
                    if(target>0.0f)dlds=weight*(aligned-target)/(aligned*(1.0f-aligned));
                    else dlds=0.75f*(-2.0f*aligned*logf(1.0f-aligned)+aligned*aligned/(1.0f-aligned));
                    if(cfg_train->hard_neg_boost&&target<=0.0f&&q.gt>=0&&gt[q.gt].class_id!=c){float boost=aligned>0.25f?2.0f*aligned:aligned;if(boost>weight){dlds*=boost/(weight+1e-9f);weight=boost;}}
                    grad=dlds*0.5f*aligned*(1.0f-pred);
                    if(quality){float adjusted_q=clampf((aligned*aligned)/(pred+1e-12f),1e-6f,1.0f-1e-6f);quality_grad=dlds*0.5f*aligned*(1.0f-adjusted_q);}
                    loss-=weight*(target*logf(aligned)+(1.0f-target)*logf(1.0f-aligned));
                }else{
                    weight=target>0.0f?target*box_w:0.75f*pred*pred;
                    if(cfg_train&&cfg_train->hard_neg_boost&&target<=0.0f&&q.gt>=0&&gt[q.gt].class_id!=c){float boost=pred>0.25f?2.0f*pred:pred;if(boost>weight)weight=boost;}
                    grad=weight*(pred-target);
                    loss-=weight*(target*logf(pred+1e-7f)+(1.0f-target)*logf(1.0f-pred+1e-7f));
                }
                cp->gb[c]+=grad;for(k=0;k<ED_HEAD_CHANNELS;++k)cp->gw[(size_t)c*ED_HEAD_CHANNELS+k]+=grad*f[k];if(context){context[l].gb[c]+=grad;for(k=0;k<ED_HEAD_CHANNELS;++k)context[l].gw[(size_t)c*ED_HEAD_CHANNELS+k]+=grad*feature_mean[k];}
                if(spatial){uint32_t ar;const float*aw=(const float*)spatial[l].w->data+((size_t)c*ED_FEATURE_ADAPTER_RANK)*ED_HEAD_CHANNELS;const float*bw=(const float*)spatial[l].b->data+(size_t)c*ED_FEATURE_ADAPTER_RANK;for(ar=0;ar<ED_FEATURE_ADAPTER_RANK;++ar){float h=0.0f;size_t bi=(size_t)c*ED_FEATURE_ADAPTER_RANK+ar;for(k=0;k<ED_HEAD_CHANNELS;++k)h+=aw[(size_t)ar*ED_HEAD_CHANNELS+k]*f[k];if(h>0.0f){spatial[l].gb[bi]+=grad*h;for(k=0;k<ED_HEAD_CHANNELS;++k)spatial[l].gw[((size_t)c*ED_FEATURE_ADAPTER_RANK+ar)*ED_HEAD_CHANNELS+k]+=grad*bw[ar]*f[k];}}}
                if(quality){quality[l].gb[c]+=quality_grad;for(k=0;k<ED_HEAD_CHANNELS;++k)quality[l].gw[(size_t)c*ED_HEAD_CHANNELS+k]+=quality_grad*f[k];}
            }
            if(q.gt>=0){
                if(gt[q.gt].class_id<m->class_count)++pos_per_class[gt[q.gt].class_id];
                float box[4],dist[4],prob[32],target_dist[4],grad_dist[4];const float *logits=rg->data+(size_t)local*32u;
                float reg_weight=cfg_train&&cfg_train->aligned_loss?q.target*box_w:box_w;
                float giou_scale=cfg_train&&cfg_train->aligned_loss?2.5f:1.0f,dfl_scale=cfg_train&&cfg_train->aligned_loss?0.5f:1.0f;
                ++positives;if(cfg_train&&cfg_train->aligned_loss)assigned_score_sum+=q.target;decode_one(logits,cx,cy,stride,box,dist,prob);
                target_dist[0]=(cx-gt[q.gt].x1)/stride;target_dist[1]=(cy-gt[q.gt].y1)/stride;
                target_dist[2]=(gt[q.gt].x2-cx)/stride;target_dist[3]=(gt[q.gt].y2-cy)/stride;
                loss+=giou_scale*reg_weight*giou_loss(box,&gt[q.gt]);
                for(c=0;c<4u;++c){
                    float old=dist[c],eps=0.01f,plus[4],minus[4];
                    memcpy(plus,box,sizeof(box));memcpy(minus,box,sizeof(box));
                    if(c==0u){plus[0]-=eps*stride;minus[0]+=eps*stride;}
                    if(c==1u){plus[1]-=eps*stride;minus[1]+=eps*stride;}
                    if(c==2u){plus[2]+=eps*stride;minus[2]-=eps*stride;}
                    if(c==3u){plus[3]+=eps*stride;minus[3]-=eps*stride;}
                    grad_dist[c]=giou_scale*reg_weight*(giou_loss(plus,&gt[q.gt])-giou_loss(minus,&gt[q.gt]))/(2.0f*eps);
                    for(k=0;k<ED_REG_BINS;++k){
                        float td=target_dist[c];uint32_t lo=(uint32_t)(td<0.0f?0.0f:(td>7.0f?7.0f:floorf(td))),hi=lo<7u?lo+1u:lo;
                        float target=(k==lo?(float)hi-td:0.0f)+(k==hi?td-(float)lo:0.0f);if(lo==hi)target=k==lo?1.0f:0.0f;
                        {float chain=cfg_train&&cfg_train->aligned_loss?1.0f:0.5f;float grad=dfl_scale*reg_weight*(prob[c*8u+k]-target)+chain*grad_dist[c]*prob[c*8u+k]*((float)k-old);uint32_t oc=c*8u+k;
                            loss-=dfl_scale*reg_weight*target*logf(prob[c*8u+k]+1e-7f);rp->gb[oc]+=grad;
                            {uint32_t ic;for(ic=0;ic<ED_HEAD_CHANNELS;++ic)rp->gw[(size_t)oc*ED_HEAD_CHANNELS+ic]+=grad*f[ic];}}
                    }
                }
            }
        }
        total+=rg->h*rg->w;
    }
    if(positives){
        uint32_t pi;float normalizer=cfg_train&&cfg_train->aligned_loss?(assigned_score_sum>1.0f?assigned_score_sum:1.0f):(float)positives,scale=1.0f/normalizer;
        for(pi=0;pi<8u;++pi){
            size_t j;
            if(cfg_train&&cfg_train->class_loss_norm&&!(pi&1u)&&m->class_count>0u){
                uint32_t oc;size_t stride=((size_t)p[pi].w->data_bytes/4u)/m->class_count;
                for(oc=0;oc<m->class_count;++oc){
                    float cs=1.0f/(float)(pos_per_class[oc]?pos_per_class[oc]:1u);size_t begin=(size_t)oc*stride,end=begin+stride,k;
                    for(k=begin;k<end&&k<(size_t)p[pi].w->data_bytes/4u;++k){float delta=(p[pi].gw[k]-p[pi].before_w[k])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;p[pi].gw[k]=p[pi].before_w[k]+delta;}
                    if(oc<(size_t)p[pi].b->data_bytes/4u){float delta=(p[pi].gb[oc]-p[pi].before_b[oc])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;p[pi].gb[oc]=p[pi].before_b[oc]+delta;}
                }
                continue;
            }
            for(j=0;j<p[pi].w->data_bytes/4u;++j){float delta=(p[pi].gw[j]-p[pi].before_w[j])*scale;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;p[pi].gw[j]=p[pi].before_w[j]+delta;}
            for(j=0;j<p[pi].b->data_bytes/4u;++j){float delta=(p[pi].gb[j]-p[pi].before_b[j])*scale;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;p[pi].gb[j]=p[pi].before_b[j]+delta;}
        }
        loss*=scale;
        if(context){uint32_t cl,oc,k;for(cl=0;cl<ED_PICODET_LEVELS;++cl)for(oc=0;oc<m->class_count;++oc){float cs=cfg_train&&cfg_train->class_loss_norm?1.0f/(float)(pos_per_class[oc]?pos_per_class[oc]:1u):scale;for(k=0;k<ED_HEAD_CHANNELS;++k){size_t i=(size_t)oc*ED_HEAD_CHANNELS+k;float delta=(context[cl].gw[i]-context[cl].before_w[i])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;context[cl].gw[i]=context[cl].before_w[i]+delta;}{float delta=(context[cl].gb[oc]-context[cl].before_b[oc])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;context[cl].gb[oc]=context[cl].before_b[oc]+delta;}}}
        if(spatial){uint32_t sl,oc;for(sl=0;sl<ED_PICODET_LEVELS;++sl){size_t wstride=((size_t)spatial[sl].w->data_bytes/4u)/m->class_count,bstride=((size_t)spatial[sl].b->data_bytes/4u)/m->class_count;for(oc=0;oc<m->class_count;++oc){float cs=cfg_train&&cfg_train->class_loss_norm?1.0f/(float)(pos_per_class[oc]?pos_per_class[oc]:1u):scale;size_t k,begin=(size_t)oc*wstride,end=begin+wstride;for(k=begin;k<end;++k){float delta=(spatial[sl].gw[k]-spatial[sl].before_w[k])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;spatial[sl].gw[k]=spatial[sl].before_w[k]+delta;}begin=(size_t)oc*bstride;end=begin+bstride;for(k=begin;k<end;++k){float delta=(spatial[sl].gb[k]-spatial[sl].before_b[k])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;spatial[sl].gb[k]=spatial[sl].before_b[k]+delta;}}}}
        if(quality){uint32_t ql,oc,k;for(ql=0;ql<ED_PICODET_LEVELS;++ql)for(oc=0;oc<m->class_count;++oc){float cs=cfg_train&&cfg_train->class_loss_norm?1.0f/(float)(pos_per_class[oc]?pos_per_class[oc]:1u):scale;size_t begin=(size_t)oc*ED_HEAD_CHANNELS,end=begin+ED_HEAD_CHANNELS;for(k=(uint32_t)begin;k<(uint32_t)end;++k){float delta=(quality[ql].gw[k]-quality[ql].before_w[k])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;quality[ql].gw[k]=quality[ql].before_w[k]+delta;}{float delta=(quality[ql].gb[oc]-quality[ql].before_b[oc])*cs;if(delta>10.0f)delta=10.0f;if(delta<-10.0f)delta=-10.0f;quality[ql].gb[oc]=quality[ql].before_b[oc]+delta;}}}
    }
done:free(pos_per_class);free(class_target);free(as);free(gt);return loss;
}

static float train_image(ed_model *m,const ed_image *image,const ed_edb_annotation_disk *anns,uint32_t ann_n,head_parameter *p,context_parameter *context,spatial_parameter *spatial,quality_parameter *quality,const assign_cfg *cfg,const ed_train_config *cfg_train,assign_stats *st){
    float *input=NULL;ed_activation *a=NULL;float loss=0.0f;
    if(ed_prepare_input_320(image,&input)==ED_OK&&ed_graph_execute(m,input,&a)==ED_OK)loss=train_activations(m,image,anns,ann_n,p,context,spatial,quality,a,cfg,cfg_train,st);
    ed_graph_activations_free(a);free(input);return loss;
}

typedef struct {
    float *feature[ED_PICODET_LEVELS];
    float *quality[ED_PICODET_LEVELS];
    uint32_t h[ED_PICODET_LEVELS],w[ED_PICODET_LEVELS];
    ed_edb_annotation_disk *annotations;
    uint32_t annotation_count,width,height;
} feature_cache_sample;
static int32_t quality_probability_node(uint32_t level){
    int32_t cls=ed_picodet_cls_nodes[level],mul=ed_picodet_nodes[cls].input[0],left=ed_picodet_nodes[mul].input[0],right=ed_picodet_nodes[mul].input[1],raw=ed_picodet_raw_cls_nodes[level];
    if(ed_picodet_nodes[left].op!=ED_OP_SIGMOID||ed_picodet_nodes[right].op!=ED_OP_SIGMOID)return -1;
    if(ed_picodet_nodes[left].input[0]==raw)return right;
    if(ed_picodet_nodes[right].input[0]==raw)return left;
    return -1;
}
static void feature_cache_sample_free(feature_cache_sample *s){
    uint32_t l;if(!s)return;for(l=0;l<ED_PICODET_LEVELS;++l){free(s->feature[l]);free(s->quality[l]);}free(s->annotations);memset(s,0,sizeof(*s));
}
static ed_status feature_cache_sample_build(const ed_model *m,const ed_dataset *d,uint64_t *rng,uint32_t policy,uint32_t slot,feature_cache_sample *s){
    ed_image image;ed_edb_annotation_disk *anns=NULL;uint32_t ann_n=0,l;float *input=NULL;ed_activation *a=NULL;ed_status status;
    memset(s,0,sizeof(*s));
    status=make_sample(d,rng,policy,slot,&image,&anns,&ann_n);if(status!=ED_OK)return status;
    status=ed_prepare_input_320(&image,&input);if(status==ED_OK)status=ed_graph_execute(m,input,&a);
    if(status==ED_OK){
        for(l=0;l<ED_PICODET_LEVELS;++l){
            const ed_activation *f=&a[ed_picodet_feature_nodes[l]];int32_t qn=quality_probability_node(l);const ed_activation *q;
            if(qn<0){status=ED_ERR_FORMAT;break;}q=&a[qn];
            if(f->c!=ED_HEAD_CHANNELS||q->h!=f->h||q->w!=f->w||q->c!=1u){status=ED_ERR_FORMAT;break;}
            s->h[l]=f->h;s->w[l]=f->w;
            s->feature[l]=(float*)malloc((size_t)f->h*f->w*f->c*sizeof(float));
            s->quality[l]=(float*)malloc((size_t)q->h*q->w*sizeof(float));
            if(!s->feature[l]||!s->quality[l]){status=ED_ERR_MEMORY;break;}
            memcpy(s->feature[l],f->data,(size_t)f->h*f->w*f->c*sizeof(float));
            memcpy(s->quality[l],q->data,(size_t)q->h*q->w*sizeof(float));
        }
    }
    if(status==ED_OK){s->annotations=anns;s->annotation_count=ann_n;s->width=image.width;s->height=image.height;anns=NULL;}
    free((void*)image.rgb);free(anns);free(input);ed_graph_activations_free(a);
    if(status!=ED_OK)feature_cache_sample_free(s);
    return status;
}
static float train_feature_cache_sample(ed_model *m,const feature_cache_sample *s,head_parameter *p,context_parameter *context,spatial_parameter *spatial,quality_parameter *quality_adapter,const assign_cfg *cfg,const ed_train_config *cfg_train,assign_stats *st){
    ed_activation *a=(ed_activation*)calloc(ed_picodet_node_count,sizeof(*a));
    ed_image image;float loss=0.0f;uint32_t l;
    if(!a)return 0.0f;
    memset(&image,0,sizeof(image));image.width=s->width;image.height=s->height;
    for(l=0;l<ED_PICODET_LEVELS;++l){
        uint32_t h=s->h[l],w=s->w[l],locations=h*w,c;
        int32_t quality_node=quality_probability_node(l);
        ed_activation *feature=&a[ed_picodet_feature_nodes[l]],*raw=&a[ed_picodet_raw_cls_nodes[l]],*reg=&a[ed_picodet_reg_nodes[l]],*score=&a[ed_picodet_cls_nodes[l]],*quality;
        if(quality_node<0)goto done;quality=&a[quality_node];
        feature->data=s->feature[l];feature->h=h;feature->w=w;feature->c=ED_HEAD_CHANNELS;
        quality->data=s->quality[l];quality->h=h;quality->w=w;quality->c=1u;
        raw->h=score->h=reg->h=h;raw->w=score->w=reg->w=w;raw->c=score->c=m->class_count;reg->c=32u;
        raw->data=(float*)malloc((size_t)locations*m->class_count*sizeof(float));
        score->data=(float*)malloc((size_t)locations*m->class_count*sizeof(float));
        reg->data=(float*)malloc((size_t)locations*32u*sizeof(float));
        if(!raw->data||!score->data||!reg->data)goto done;
        ed_conv2d_f32(feature->data,h,w,ED_HEAD_CHANNELS,(const float*)p[l*2u].w->data,(const float*)p[l*2u].b->data,m->class_count,1u,1u,0u,raw->data);
        ed_conv2d_f32(feature->data,h,w,ED_HEAD_CHANNELS,(const float*)p[l*2u+1u].w->data,(const float*)p[l*2u+1u].b->data,32u,1u,1u,0u,reg->data);
        for(c=0;c<locations*m->class_count;++c){float sigmoid=1.0f/(1.0f+expf(-raw->data[c]));score->data[c]=sqrtf(sigmoid*s->quality[l][c/m->class_count]);}
    }
    if(ed_graph_apply_spatial(m,a)==ED_OK&&ed_graph_apply_context(m,a)==ED_OK&&ed_graph_apply_quality(m,a)==ED_OK)loss=train_activations(m,&image,s->annotations,s->annotation_count,p,context,spatial,quality_adapter,a,cfg,cfg_train,st);
done:
    for(l=0;l<ED_PICODET_LEVELS;++l){free(a[ed_picodet_raw_cls_nodes[l]].data);free(a[ed_picodet_reg_nodes[l]].data);free(a[ed_picodet_cls_nodes[l]].data);}
    free(a);return loss;
}

ed_status ed_train(ed_model *m,const ed_dataset *d,const ed_train_config *c,ed_train_report *r){
    head_parameter parameters[8];context_parameter context[ED_PICODET_LEVELS],*context_ptr=NULL;spatial_parameter spatial[ED_PICODET_LEVELS],*spatial_ptr=NULL;quality_parameter quality[ED_PICODET_LEVELS],*quality_ptr=NULL;feature_cache_sample *cache=NULL;ed_train_config phase_config;uint64_t start,deadline,rng,training_rng_start=0,next_checkpoint=10000u,warmup_end_ms=0,phase_start_ms=0,phase_steps=0;
    uint32_t accumulated=0,cache_count=0,policy;float loss=0.0f,median,ema_decay,wise;assign_cfg cfg;ed_status status=ED_OK;assign_stats stats={0};
    if(!m||!d||!c||!r||c->budget_ms==0u||c->threads==0u||(c->mosaic_size!=0u&&c->mosaic_size!=1u&&c->mosaic_size!=4u)||c->accumulation==0u||c->feature_cache_samples>256u)return ED_ERR_ARGUMENT;
    if(c->scope!=ED_TRAIN_OUTPUTS&&c->scope!=ED_TRAIN_CLASSIFICATION&&c->scope!=ED_TRAIN_QUALITY)return ED_ERR_ARGUMENT;
    if(c->restart_after_steps&&(!c->max_optimizer_steps||c->restart_after_steps>=c->max_optimizer_steps))return ED_ERR_ARGUMENT;
    if((unsigned)c->sample_mode>ED_SAMPLE_ADAPT||(c->checkpoint_storage&&(c->checkpoint_storage<ED_STORAGE_FP32||c->checkpoint_storage>ED_STORAGE_INT4))||(unsigned)c->head_adapter>ED_HEAD_ADAPTER_CROSS_LEVEL||(c->head_adapter==ED_HEAD_ADAPTER_CROSS_LEVEL&&(c->cross_level_mix<=0.0f||c->cross_level_mix>1.0f))||c->nesterov>1u||c->roi_head>1u||c->picofeat_adapter>1u||c->aligned_loss>1u||c->quality_adapter>1u||(c->quality_adapter&&!c->aligned_loss)||(c->scope==ED_TRAIN_QUALITY&&!c->quality_adapter))return ED_ERR_ARGUMENT;
    if(m->class_count!=d->header.class_count)return ED_ERR_FORMAT;
    {uint32_t ci;for(ci=0;ci<m->class_count;++ci)if(strncmp(m->class_names[ci],d->class_names[ci],ED_CLASS_NAME_BYTES)!=0)return ED_ERR_FORMAT;}
    if(ed_runtime_set_threads(c->threads)!=ED_OK)return ED_ERR_MEMORY;
    memset(r,0,sizeof(*r));memset(parameters,0,sizeof(parameters));memset(context,0,sizeof(context));memset(spatial,0,sizeof(spatial));memset(quality,0,sizeof(quality));
    if((c->roi_head&&!context_model_add(m))||(c->picofeat_adapter&&!spatial_model_add(m))||(c->quality_adapter&&!quality_model_add(m)))return ED_ERR_MEMORY;
    if(c->roi_head){if(!context_parameter_init(m,context)){context_parameter_free(context);return ED_ERR_MEMORY;}context_ptr=context;}
    if(c->picofeat_adapter){if(!spatial_parameter_init(m,spatial)){spatial_parameter_free(spatial);context_parameter_free(context);return ED_ERR_MEMORY;}spatial_ptr=spatial;}
    if(c->quality_adapter){if(!quality_parameter_init(m,quality)){quality_parameter_free(quality);spatial_parameter_free(spatial);context_parameter_free(context);return ED_ERR_MEMORY;}quality_ptr=quality;}
    if(!parameter_init(m,parameters)){quality_parameter_free(quality);spatial_parameter_free(spatial);context_parameter_free(context);parameter_free(parameters);return ED_ERR_FORMAT;}
    r->trainable_parameters=0;
    if(c->scope!=ED_TRAIN_QUALITY){if(c->head_adapter==ED_HEAD_ADAPTER_CROSS_LEVEL&&c->cross_level_mix>=0.999f){uint32_t i;r->trainable_parameters=(uint64_t)m->class_count*(ED_HEAD_CHANNELS+ED_PICODET_LEVELS);if(c->scope!=ED_TRAIN_CLASSIFICATION)for(i=1u;i<8u;i+=2u)r->trainable_parameters+=(parameters[i].w->data_bytes+parameters[i].b->data_bytes)/4u;}
        else{uint32_t i;for(i=0;i<8u;++i)if(c->scope!=ED_TRAIN_CLASSIFICATION||!(i&1u))r->trainable_parameters+=(parameters[i].w->data_bytes+parameters[i].b->data_bytes)/4u;}}
    if(context_ptr&&c->scope!=ED_TRAIN_QUALITY)r->trainable_parameters+=(uint64_t)ED_PICODET_LEVELS*m->class_count*(ED_HEAD_CHANNELS+1u);
    if(spatial_ptr&&c->scope!=ED_TRAIN_QUALITY)r->trainable_parameters+=(uint64_t)ED_PICODET_LEVELS*m->class_count*ED_FEATURE_ADAPTER_RANK*(ED_HEAD_CHANNELS+1u);
    if(quality_ptr)r->trainable_parameters+=(uint64_t)ED_PICODET_LEVELS*m->class_count*(ED_HEAD_CHANNELS+1u);
    median=dataset_median_side(d);policy=resolve_policy(c,median,dataset_mean_aspect(d));cfg=resolve_assign(median);
    r->sample_mode_used=policy;r->median_box_side=median;
    ema_decay=(c->ema_decay>0.0f&&c->ema_decay<1.0f)?c->ema_decay:1.0f;
    wise=(c->wise_mix>0.0f&&c->wise_mix<1.0f)?c->wise_mix:1.0f;
    phase_config=*c;if(c->restart_after_steps)phase_config.max_optimizer_steps=c->restart_after_steps;
    start=ed_monotonic_ms();deadline=start+c->budget_ms;rng=c->seed?c->seed:1u;parameter_zero_grad(parameters);if(context_ptr)context_parameter_zero_grad(context);if(spatial_ptr)spatial_parameter_zero_grad(spatial);if(quality_ptr)quality_parameter_zero_grad(quality);
    if(c->feature_cache_samples){
        uint64_t cache_start=ed_monotonic_ms(),cache_reserve=c->budget_ms/5u,cache_deadline;
        if(cache_reserve<1000u)cache_reserve=1000u;
        cache_deadline=deadline>cache_reserve?deadline-cache_reserve:start;
        cache=(feature_cache_sample*)calloc(c->feature_cache_samples,sizeof(*cache));
        if(!cache){status=ED_ERR_MEMORY;goto done;}
        while(cache_count<c->feature_cache_samples){
            uint64_t now=ed_monotonic_ms(),estimate=cache_count?(now-cache_start+cache_count-1u)/cache_count:1000u;
            if(estimate<1u)estimate=1u;if(now+estimate>=cache_deadline)break;
            (void)rng_next(&rng);
            status=feature_cache_sample_build(m,d,&rng,policy,cache_count,&cache[cache_count]);
            if(status!=ED_OK)goto done;++cache_count;
        }
        r->feature_cache_samples=cache_count;r->feature_cache_ms=ed_monotonic_ms()-cache_start;
        if(cache_count==0u){status=ED_ERR_DEADLINE;goto done;}
    }
    training_rng_start=rng;
    while(ed_monotonic_ms()+1000u<deadline&&(!c->max_optimizer_steps||r->optimizer_steps<c->max_optimizer_steps)){
        ed_edb_annotation_disk *anns=NULL;uint32_t ann_n=0;ed_image image;
        if(cache_count){uint32_t selected=(uint32_t)(rng_next(&rng)%cache_count);loss=train_feature_cache_sample(m,&cache[selected],parameters,context_ptr,spatial_ptr,quality_ptr,&cfg,c,&stats);}
        else{
            status=make_sample(d,&rng,policy,(uint32_t)r->mosaics_seen,&image,&anns,&ann_n);if(status!=ED_OK)break;
            loss=train_image(m,&image,anns,ann_n,parameters,context_ptr,spatial_ptr,quality_ptr,&cfg,c,&stats);free((void*)image.rgb);free(anns);
        }
        r->images_seen+=(policy==ED_SAMPLE_MOSAIC||policy==ED_SAMPLE_ADAPT)?4u:1u;++r->mosaics_seen;++accumulated;
        if(accumulated==c->accumulation){
            float step_lr=parameter_step(parameters,&phase_config,phase_steps,ed_monotonic_ms()-start-phase_start_ms,&warmup_end_ms,ema_decay,m->class_count);if(context_ptr&&c->scope!=ED_TRAIN_QUALITY)context_parameter_step(context,c,step_lr,ema_decay,m->class_count);if(spatial_ptr&&c->scope!=ED_TRAIN_QUALITY)spatial_parameter_step(spatial,c,step_lr,ema_decay,m->class_count);if(quality_ptr)quality_parameter_step(quality,c,step_lr,ema_decay,m->class_count);
            parameter_zero_grad(parameters);if(context_ptr)context_parameter_zero_grad(context);if(spatial_ptr)spatial_parameter_zero_grad(spatial);if(quality_ptr)quality_parameter_zero_grad(quality);accumulated=0;++r->optimizer_steps;++phase_steps;
            if(c->restart_after_steps&&r->optimizer_steps==c->restart_after_steps){parameter_restart(parameters,c,ema_decay<1.0f,wise);if(context_ptr&&c->scope!=ED_TRAIN_QUALITY)context_parameter_restart(context,ema_decay<1.0f,wise);if(spatial_ptr&&c->scope!=ED_TRAIN_QUALITY)spatial_parameter_restart(spatial,ema_decay<1.0f,wise);if(quality_ptr)quality_parameter_restart(quality,ema_decay<1.0f,wise);phase_config=*c;phase_config.max_optimizer_steps=c->max_optimizer_steps-c->restart_after_steps;phase_steps=0;phase_start_ms=ed_monotonic_ms()-start;warmup_end_ms=0;rng=training_rng_start;}
        }
        if(c->checkpoint_prefix&&next_checkpoint<=40000u&&ed_monotonic_ms()-start>=next_checkpoint){
            char path[1024];uint32_t i;uint64_t elapsed=ed_monotonic_ms()-start,checkpoint_ms=(elapsed/10000u)*10000u;
            if(checkpoint_ms>40000u)checkpoint_ms=40000u;
            for(i=0;i<8u;++i){memcpy(parameters[i].before_w,parameters[i].w->data,(size_t)parameters[i].w->data_bytes);memcpy(parameters[i].before_b,parameters[i].b->data,(size_t)parameters[i].b->data_bytes);}
            if(context_ptr)for(i=0;i<ED_PICODET_LEVELS;++i){memcpy(context[i].before_w,context[i].w->data,(size_t)context[i].w->data_bytes);memcpy(context[i].before_b,context[i].b->data,(size_t)context[i].b->data_bytes);}parameter_finalize(parameters,c,ema_decay<1.0f,wise);if(context_ptr)context_parameter_finalize(context,ema_decay<1.0f,wise);
            if(spatial_ptr)for(i=0;i<ED_PICODET_LEVELS;++i){memcpy(spatial[i].before_w,spatial[i].w->data,(size_t)spatial[i].w->data_bytes);memcpy(spatial[i].before_b,spatial[i].b->data,(size_t)spatial[i].b->data_bytes);}if(spatial_ptr)spatial_parameter_finalize(spatial,ema_decay<1.0f,wise);
            if(quality_ptr)for(i=0;i<ED_PICODET_LEVELS;++i){memcpy(quality[i].before_w,quality[i].w->data,(size_t)quality[i].w->data_bytes);memcpy(quality[i].before_b,quality[i].b->data,(size_t)quality[i].b->data_bytes);}if(quality_ptr)quality_parameter_finalize(quality,ema_decay<1.0f,wise);
            snprintf(path,sizeof(path),"%s.%llus.edm",c->checkpoint_prefix,(unsigned long long)(checkpoint_ms/1000u));
            status=ed_model_save_storage(m,path,c->checkpoint_storage?c->checkpoint_storage:ED_STORAGE_FP32);if(status!=ED_OK)break;
            for(i=0;i<8u;++i){memcpy(parameters[i].w->data,parameters[i].before_w,(size_t)parameters[i].w->data_bytes);memcpy(parameters[i].b->data,parameters[i].before_b,(size_t)parameters[i].b->data_bytes);}
            if(context_ptr)for(i=0;i<ED_PICODET_LEVELS;++i){memcpy(context[i].w->data,context[i].before_w,(size_t)context[i].w->data_bytes);memcpy(context[i].b->data,context[i].before_b,(size_t)context[i].b->data_bytes);}
            if(spatial_ptr)for(i=0;i<ED_PICODET_LEVELS;++i){memcpy(spatial[i].w->data,spatial[i].before_w,(size_t)spatial[i].w->data_bytes);memcpy(spatial[i].b->data,spatial[i].before_b,(size_t)spatial[i].b->data_bytes);}
            if(quality_ptr)for(i=0;i<ED_PICODET_LEVELS;++i){memcpy(quality[i].w->data,quality[i].before_w,(size_t)quality[i].w->data_bytes);memcpy(quality[i].b->data,quality[i].before_b,(size_t)quality[i].b->data_bytes);}
            next_checkpoint=checkpoint_ms+10000u;
        }
    }
    if(status==ED_OK){parameter_finalize(parameters,c,ema_decay<1.0f,wise);if(context_ptr&&c->scope!=ED_TRAIN_QUALITY)context_parameter_finalize(context,ema_decay<1.0f,wise);if(spatial_ptr&&c->scope!=ED_TRAIN_QUALITY)spatial_parameter_finalize(spatial,ema_decay<1.0f,wise);if(quality_ptr)quality_parameter_finalize(quality,ema_decay<1.0f,wise);}
done:
    r->elapsed_ms=ed_monotonic_ms()-start;r->final_loss=loss;r->deadline_respected=r->elapsed_ms<=c->budget_ms;
    r->assign_recall=stats.gt_seen?((float)stats.gt_hit/(float)stats.gt_seen):0.0f;
    r->assign_mean_iou=(stats.gt_hit?stats.iou_sum/(float)stats.gt_hit:0.0f);
    if(cache){uint32_t i;for(i=0;i<cache_count;++i)feature_cache_sample_free(&cache[i]);free(cache);}
    quality_parameter_free(quality);spatial_parameter_free(spatial);context_parameter_free(context);parameter_free(parameters);return status;
}
