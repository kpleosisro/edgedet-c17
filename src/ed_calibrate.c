#include "ed_graph.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ED_HEAD_CHANNELS 96u
#define ED_CALIBRATION_TOP_PER_IMAGE 32u
#define ED_CALIBRATION_SCORE_FLOOR 0.01f
#define ED_CALIBRATION_STRONG_AFFINITY 0.60f
#define ED_CALIBRATION_NOVEL_BLEND 4u
#define ED_CALIB_TOPK 13u
#define ED_CALIB_POS_CAP 768u
#define ED_CALIB_NEG_CAP 1536u
#define ED_CALIB_STORE_PER_IMAGE 192u
#define ED_CALIB_FOLD_MOD 5u
#define ED_PROTO_MIN_POS 4u
#define ED_PROTO_RESIDUAL 0.50f
#define ED_CALIB_IOU_POWER 6.0f

typedef struct {
    ed_detection detection;
    uint32_t image_slot;
    uint32_t source_class;
} calibration_detection;

typedef struct {
    float feat[ED_HEAD_CHANNELS];
    float quality;
    uint32_t image_slot;
} bank_item;

typedef struct {
    bank_item *items;
    uint32_t count,capacity;
} feature_bank;

typedef struct {
    float feat[ED_HEAD_CHANNELS];
    float quality,x1,y1,x2,y2;
    uint32_t image_slot,level;
} stored_loc;

typedef struct {
    float feat[ED_HEAD_CHANNELS];
    float quality,x1,y1,x2,y2,cx,cy,max_cls;
    int assigned;
} loc_tmp;

static float calibration_iou(float ax1,float ay1,float ax2,float ay2,float bx1,float by1,float bx2,float by2){
    float x1=ax1>bx1?ax1:bx1,y1=ay1>by1?ay1:by1,x2=ax2<bx2?ax2:bx2,y2=ay2<by2?ay2:by2;
    float w=x2-x1,h=y2-y1,inter,den;if(w<=0.0f||h<=0.0f)return 0.0f;inter=w*h;den=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter;return den>0.0f?inter/den:0.0f;
}
static int score_desc(const void *a,const void *b){
    float x=((const calibration_detection*)a)->detection.score,y=((const calibration_detection*)b)->detection.score;return x<y?1:(x>y?-1:0);
}
static int record_has_target(const ed_dataset *d,uint32_t index,uint32_t target){
    const uint8_t *bytes;const ed_edb_annotation_disk *a;uint32_t bytes_n,n,w,h,i;
    if(ed_dataset_record(d,index,&bytes,&bytes_n,&a,&n,&w,&h)!=ED_OK)return 0;
    for(i=0;i<n;++i)if(!(a[i].flags&ED_ANN_FLAG_IGNORE)&&a[i].class_id==target)return 1;
    return 0;
}
static int selected_contains(const uint32_t *selected,uint32_t count,uint32_t index){uint32_t i;for(i=0;i<count;++i)if(selected[i]==index)return 1;return 0;}
static int fold_select(uint32_t slot){return (slot%ED_CALIB_FOLD_MOD)==0u;}
static float clampf(float v,float lo,float hi){if(v<lo)return lo;if(v>hi)return hi;return v;}
static float vec_dot(const float *a,const float *b,uint32_t n){uint32_t i;float s=0.0f;for(i=0;i<n;++i)s+=a[i]*b[i];return s;}
static float vec_l2(const float *a,uint32_t n){return sqrtf(vec_dot(a,a,n));}

static int append_prediction(calibration_detection **items,size_t *count,size_t *capacity,const ed_detection *d,uint32_t image_slot,uint32_t source){
    if(*count==*capacity){size_t next=*capacity?*capacity*2u:4096u;void *p=realloc(*items,next*sizeof(**items));if(!p)return 0;*items=(calibration_detection*)p;*capacity=next;}
    (*items)[*count].detection=*d;(*items)[*count].image_slot=image_slot;(*items)[*count].source_class=source;++*count;return 1;
}
static void top_insert(ed_detection *items,uint32_t *count,const ed_detection *candidate){
    uint32_t i,minimum=0;if(*count<ED_CALIBRATION_TOP_PER_IMAGE){items[(*count)++]=*candidate;return;}
    for(i=1;i<*count;++i)if(items[i].score<items[minimum].score)minimum=i;
    if(candidate->score>items[minimum].score)items[minimum]=*candidate;
}
static int32_t quality_node(uint32_t level){
    int32_t cls=ed_picodet_cls_nodes[level],mul=ed_picodet_nodes[cls].input[0];
    int32_t left=ed_picodet_nodes[mul].input[0],right=ed_picodet_nodes[mul].input[1],raw=ed_picodet_raw_cls_nodes[level];
    if(ed_picodet_nodes[left].op!=ED_OP_SIGMOID||ed_picodet_nodes[right].op!=ED_OP_SIGMOID)return -1;
    if(ed_picodet_nodes[left].input[0]==raw)return right;
    if(ed_picodet_nodes[right].input[0]==raw)return left;
    return -1;
}
static int bank_init(feature_bank *b,uint32_t cap){
    b->items=(bank_item*)malloc((size_t)cap*sizeof(*b->items));b->count=0;b->capacity=cap;return b->items!=NULL;
}
static void bank_free(feature_bank *b){free(b->items);b->items=NULL;b->count=b->capacity=0;}
static void bank_insert(feature_bank *b,const float *feat,float quality,uint32_t image_slot){
    uint32_t i,worst=0;
    if(!b->items)return;
    if(b->count<b->capacity){
        memcpy(b->items[b->count].feat,feat,ED_HEAD_CHANNELS*sizeof(float));
        b->items[b->count].quality=quality;b->items[b->count].image_slot=image_slot;++b->count;return;
    }
    for(i=1;i<b->count;++i)if(b->items[i].quality<b->items[worst].quality)worst=i;
    if(quality>b->items[worst].quality){
        memcpy(b->items[worst].feat,feat,ED_HEAD_CHANNELS*sizeof(float));
        b->items[worst].quality=quality;b->items[worst].image_slot=image_slot;
    }
}
static void store_insert(stored_loc *items,uint32_t *count,uint32_t cap,const stored_loc *cand){
    uint32_t i,worst=0;
    if(*count<cap){items[(*count)++]=*cand;return;}
    for(i=1;i<*count;++i)if(items[i].quality<items[worst].quality)worst=i;
    if(cand->quality>items[worst].quality)items[worst]=*cand;
}
static uint32_t gather_bank(const feature_bank *b,int fit_only,float *out){
    uint32_t i,n=0;
    for(i=0;i<b->count;++i){
        if(fit_only&&fold_select(b->items[i].image_slot))continue;
        if(out)memcpy(out+(size_t)n*ED_HEAD_CHANNELS,b->items[i].feat,ED_HEAD_CHANNELS*sizeof(float));
        ++n;
    }
    return n;
}

static int solve_d(uint32_t n,double *A,double *b,double *x){
    uint32_t i,j,k;
    for(i=0;i<n;++i){
        uint32_t pivot=i;double best=fabs(A[(size_t)i*n+i]),swap;
        for(j=i+1u;j<n;++j){double v=fabs(A[(size_t)j*n+i]);if(v>best){best=v;pivot=j;}}
        if(best<1e-12)return 0;
        if(pivot!=i){
            for(k=i;k<n;++k){swap=A[(size_t)i*n+k];A[(size_t)i*n+k]=A[(size_t)pivot*n+k];A[(size_t)pivot*n+k]=swap;}
            swap=b[i];b[i]=b[pivot];b[pivot]=swap;
        }
        {
            double diag=A[(size_t)i*n+i];
            for(k=i;k<n;++k)A[(size_t)i*n+k]/=diag;
            b[i]/=diag;
        }
        for(j=0;j<n;++j)if(j!=i){
            double f=A[(size_t)j*n+i];
            if(f==0.0)continue;
            for(k=i;k<n;++k)A[(size_t)j*n+k]-=f*A[(size_t)i*n+k];
            b[j]-=f*b[i];
        }
    }
    for(i=0;i<n;++i)x[i]=b[i];
    return 1;
}
static void standardize(const float *pos,uint32_t np,const float *neg,uint32_t nn,float *mean,float *istd){
    uint32_t i,k,n=np+nn;
    for(k=0;k<ED_HEAD_CHANNELS;++k){mean[k]=0.0f;istd[k]=0.0f;}
    if(n==0u){for(k=0;k<ED_HEAD_CHANNELS;++k)istd[k]=1.0f;return;}
    for(i=0;i<np;++i)for(k=0;k<ED_HEAD_CHANNELS;++k)mean[k]+=pos[(size_t)i*ED_HEAD_CHANNELS+k];
    for(i=0;i<nn;++i)for(k=0;k<ED_HEAD_CHANNELS;++k)mean[k]+=neg[(size_t)i*ED_HEAD_CHANNELS+k];
    for(k=0;k<ED_HEAD_CHANNELS;++k)mean[k]/=(float)n;
    for(i=0;i<np;++i)for(k=0;k<ED_HEAD_CHANNELS;++k){float d=pos[(size_t)i*ED_HEAD_CHANNELS+k]-mean[k];istd[k]+=d*d;}
    for(i=0;i<nn;++i)for(k=0;k<ED_HEAD_CHANNELS;++k){float d=neg[(size_t)i*ED_HEAD_CHANNELS+k]-mean[k];istd[k]+=d*d;}
    for(k=0;k<ED_HEAD_CHANNELS;++k){float s=sqrtf(istd[k]/((float)n+1e-6f));istd[k]=1.0f/(s+1e-3f);}
}
static int fit_ridge(const float *pos,uint32_t np,const float *neg,uint32_t nn,float *w,float *bias){
    const uint32_t dim=ED_HEAD_CHANNELS+1u;uint32_t i,r,c;double *A,*rhs,*x;float mean[ED_HEAD_CHANNELS],istd[ED_HEAD_CHANNELS];
    if(np<ED_PROTO_MIN_POS||nn==0u)return 0;
    standardize(pos,np,neg,nn,mean,istd);
    A=(double*)calloc((size_t)dim*dim,sizeof(double));rhs=(double*)calloc(dim,sizeof(double));x=(double*)calloc(dim,sizeof(double));
    if(!A||!rhs||!x){free(A);free(rhs);free(x);return 0;}
    for(i=0;i<np+nn;++i){
        const float *src=i<np?pos+(size_t)i*ED_HEAD_CHANNELS:neg+(size_t)(i-np)*ED_HEAD_CHANNELS;
        double y=i<np?1.0:-1.0,row[ED_HEAD_CHANNELS+1u];
        for(c=0;c<ED_HEAD_CHANNELS;++c)row[c]=(double)((src[c]-mean[c])*istd[c]);
        row[ED_HEAD_CHANNELS]=1.0;
        for(r=0;r<dim;++r){rhs[r]+=y*row[r];for(c=0;c<dim;++c)A[(size_t)r*dim+c]+=row[r]*row[c];}
    }
    {double trace=0.0;for(r=0;r<dim;++r)trace+=A[(size_t)r*dim+r];trace=(trace/(double)dim)+1.0;for(r=0;r<dim;++r)A[(size_t)r*dim+r]+=trace;}
    if(!solve_d(dim,A,rhs,x)){free(A);free(rhs);free(x);return 0;}
    *bias=(float)x[ED_HEAD_CHANNELS];
    for(c=0;c<ED_HEAD_CHANNELS;++c){w[c]=(float)x[c]*istd[c];*bias-=w[c]*mean[c];}
    free(A);free(rhs);free(x);return 1;
}
static int fit_diag(const float *pos,uint32_t np,const float *neg,uint32_t nn,float *w,float *bias){
    float mp[ED_HEAD_CHANNELS],mn[ED_HEAD_CHANNELS],vp[ED_HEAD_CHANNELS],vn[ED_HEAD_CHANNELS];uint32_t i,k;
    if(np<ED_PROTO_MIN_POS)return 0;
    for(k=0;k<ED_HEAD_CHANNELS;++k){mp[k]=0.0f;mn[k]=0.0f;vp[k]=0.0f;vn[k]=0.0f;}
    for(i=0;i<np;++i)for(k=0;k<ED_HEAD_CHANNELS;++k)mp[k]+=pos[(size_t)i*ED_HEAD_CHANNELS+k];
    for(k=0;k<ED_HEAD_CHANNELS;++k)mp[k]/=(float)np;
    if(nn){for(i=0;i<nn;++i)for(k=0;k<ED_HEAD_CHANNELS;++k)mn[k]+=neg[(size_t)i*ED_HEAD_CHANNELS+k];for(k=0;k<ED_HEAD_CHANNELS;++k)mn[k]/=(float)nn;}
    for(i=0;i<np;++i)for(k=0;k<ED_HEAD_CHANNELS;++k){float d=pos[(size_t)i*ED_HEAD_CHANNELS+k]-mp[k];vp[k]+=d*d;}
    for(k=0;k<ED_HEAD_CHANNELS;++k)vp[k]/=(float)(np>1u?np-1u:1u);
    if(nn>1u){for(i=0;i<nn;++i)for(k=0;k<ED_HEAD_CHANNELS;++k){float d=neg[(size_t)i*ED_HEAD_CHANNELS+k]-mn[k];vn[k]+=d*d;}for(k=0;k<ED_HEAD_CHANNELS;++k)vn[k]/=(float)(nn-1u);}
    *bias=0.0f;
    for(k=0;k<ED_HEAD_CHANNELS;++k){
        float var=0.5f*(vp[k]+(nn?vn[k]:vp[k]))+1e-4f;
        w[k]=(mp[k]-mn[k])/var;*bias+=-0.5f*w[k]*(mp[k]+mn[k]);
    }
    return 1;
}
static int fit_lda(const float *pos,uint32_t np,const float *neg,uint32_t nn,float *w,float *bias){
    float mp[ED_HEAD_CHANNELS],mn[ED_HEAD_CHANNELS];uint32_t i,r,c;double *A,*rhs,*x,df;
    if(np<ED_PROTO_MIN_POS||nn<2u)return 0;
    for(c=0;c<ED_HEAD_CHANNELS;++c){mp[c]=0.0f;mn[c]=0.0f;}
    for(i=0;i<np;++i)for(c=0;c<ED_HEAD_CHANNELS;++c)mp[c]+=pos[(size_t)i*ED_HEAD_CHANNELS+c];
    for(i=0;i<nn;++i)for(c=0;c<ED_HEAD_CHANNELS;++c)mn[c]+=neg[(size_t)i*ED_HEAD_CHANNELS+c];
    for(c=0;c<ED_HEAD_CHANNELS;++c){mp[c]/=(float)np;mn[c]/=(float)nn;}
    A=(double*)calloc((size_t)ED_HEAD_CHANNELS*ED_HEAD_CHANNELS,sizeof(double));rhs=(double*)calloc(ED_HEAD_CHANNELS,sizeof(double));x=(double*)calloc(ED_HEAD_CHANNELS,sizeof(double));
    if(!A||!rhs||!x){free(A);free(rhs);free(x);return 0;}
    for(i=0;i<np;++i)for(r=0;r<ED_HEAD_CHANNELS;++r){double dr=(double)(pos[(size_t)i*ED_HEAD_CHANNELS+r]-mp[r]);for(c=0;c<ED_HEAD_CHANNELS;++c)A[(size_t)r*ED_HEAD_CHANNELS+c]+=dr*(double)(pos[(size_t)i*ED_HEAD_CHANNELS+c]-mp[c]);}
    for(i=0;i<nn;++i)for(r=0;r<ED_HEAD_CHANNELS;++r){double dr=(double)(neg[(size_t)i*ED_HEAD_CHANNELS+r]-mn[r]);for(c=0;c<ED_HEAD_CHANNELS;++c)A[(size_t)r*ED_HEAD_CHANNELS+c]+=dr*(double)(neg[(size_t)i*ED_HEAD_CHANNELS+c]-mn[c]);}
    df=(double)(np+nn-2u);if(df<1.0)df=1.0;
    {double trace=0.0;for(r=0;r<ED_HEAD_CHANNELS;++r){for(c=0;c<ED_HEAD_CHANNELS;++c)A[(size_t)r*ED_HEAD_CHANNELS+c]/=df;trace+=A[(size_t)r*ED_HEAD_CHANNELS+r];}
        trace=0.1*(trace/(double)ED_HEAD_CHANNELS)+1e-3;for(r=0;r<ED_HEAD_CHANNELS;++r)A[(size_t)r*ED_HEAD_CHANNELS+r]+=trace;}
    for(c=0;c<ED_HEAD_CHANNELS;++c)rhs[c]=(double)(mp[c]-mn[c]);
    if(!solve_d(ED_HEAD_CHANNELS,A,rhs,x)){free(A);free(rhs);free(x);return 0;}
    *bias=0.0f;for(c=0;c<ED_HEAD_CHANNELS;++c){w[c]=(float)x[c];*bias+=-0.5f*w[c]*(mp[c]+mn[c]);}
    free(A);free(rhs);free(x);return 1;
}
static int fit_classifier(ed_proto_solver solver,const float *pos,uint32_t np,const float *neg,uint32_t nn,float *w,float *bias){
    if(solver==ED_PROTO_LDA&&fit_lda(pos,np,neg,nn,w,bias))return 1;
    if(solver==ED_PROTO_DIAG&&fit_diag(pos,np,neg,nn,w,bias))return 1;
    if(fit_ridge(pos,np,neg,nn,w,bias))return 1;
    if(fit_diag(pos,np,neg,nn,w,bias))return 1;
    if(np==0u)return 0;
    {uint32_t i,k;for(k=0;k<ED_HEAD_CHANNELS;++k)w[k]=0.0f;
        for(i=0;i<np;++i)for(k=0;k<ED_HEAD_CHANNELS;++k)w[k]+=pos[(size_t)i*ED_HEAD_CHANNELS+k];
        for(k=0;k<ED_HEAD_CHANNELS;++k)w[k]/=(float)np;
        *bias=-0.5f*vec_dot(w,w,ED_HEAD_CHANNELS);}
    return 1;
}
static void match_norm(float *w,float *bias,const float *ref_w,const uint32_t *sources,uint32_t nsrc){
    float proto=vec_l2(w,ED_HEAD_CHANNELS),ref=0.0f;uint32_t k;
    if(nsrc==0u)return;
    for(k=0;k<nsrc;++k)ref+=vec_l2(ref_w+(size_t)sources[k]*ED_HEAD_CHANNELS,ED_HEAD_CHANNELS);
    ref/=(float)nsrc;
    if(proto>1e-8f&&ref>0.0f){float s=ref/proto;uint32_t i;for(i=0;i<ED_HEAD_CHANNELS;++i)w[i]*=s;*bias*=s;}
}
static void calibrate_bias(float *w,float *bias,const float *pos,uint32_t np,const float *neg,uint32_t nn,float target_pos){
    float mp=0.0f;uint32_t i;
    if(np==0u)return;
    for(i=0;i<np;++i)mp+=vec_dot(w,pos+(size_t)i*ED_HEAD_CHANNELS,ED_HEAD_CHANNELS);
    mp=mp/(float)np+*bias;
    *bias+=target_pos-mp;
    (void)neg;(void)nn;
}
static int fit_source_mix(const float *pos,uint32_t np,const float *neg,uint32_t nn,
                          const float *src_w,const float *src_b,const uint32_t *sources,uint32_t nsrc,
                          float *dst_w,float *dst_b){
    uint32_t dim=nsrc+1u,i,r,c,k;double *A,*rhs,*x;
    if(np<ED_PROTO_MIN_POS||nn==0u||nsrc==0u||nsrc>4u)return 0;
    A=(double*)calloc((size_t)dim*dim,sizeof(double));rhs=(double*)calloc(dim,sizeof(double));x=(double*)calloc(dim,sizeof(double));
    if(!A||!rhs||!x){free(A);free(rhs);free(x);return 0;}
    for(i=0;i<np+nn;++i){
        const float *src=i<np?pos+(size_t)i*ED_HEAD_CHANNELS:neg+(size_t)(i-np)*ED_HEAD_CHANNELS;
        double y=i<np?1.0:-1.0,row[5];
        for(k=0;k<nsrc;++k)row[k]=(double)(vec_dot(src_w+(size_t)sources[k]*ED_HEAD_CHANNELS,src,ED_HEAD_CHANNELS)+src_b[sources[k]]);
        row[nsrc]=1.0;
        for(r=0;r<dim;++r){rhs[r]+=y*row[r];for(c=0;c<dim;++c)A[(size_t)r*dim+c]+=row[r]*row[c];}
    }
    {double trace=0.0;for(r=0;r<dim;++r)trace+=A[(size_t)r*dim+r];trace=trace/(double)dim+1.0;for(r=0;r<dim;++r)A[(size_t)r*dim+r]+=trace;}
    if(!solve_d(dim,A,rhs,x)){free(A);free(rhs);free(x);return 0;}
    for(c=0;c<ED_HEAD_CHANNELS;++c){float s=0.0f;for(k=0;k<nsrc;++k)s+=(float)x[k]*src_w[(size_t)sources[k]*ED_HEAD_CHANNELS+c];dst_w[c]=s;}
    *dst_b=(float)x[nsrc];for(k=0;k<nsrc;++k)*dst_b+=(float)x[k]*src_b[sources[k]];
    free(A);free(rhs);free(x);return 1;
}
static void blend_from_sources(const float *src_w,const float *src_b,const uint32_t *sources,const float *affinity,uint32_t nsrc,int equal,float *dst_w,float *dst_b){
    float wsum=0.0f;uint32_t k,ic;
    if(equal)wsum=(float)nsrc;
    else{for(k=0;k<nsrc;++k)if(affinity[k]>0.0f)wsum+=affinity[k];if(wsum<=1e-8f){equal=1;wsum=(float)nsrc;}}
    if(nsrc==0u||wsum<=0.0f){memset(dst_w,0,ED_HEAD_CHANNELS*sizeof(float));*dst_b=-4.59511985f;return;}
    for(ic=0;ic<ED_HEAD_CHANNELS;++ic){float s=0.0f;for(k=0;k<nsrc;++k){float wk=equal?1.0f:(affinity[k]>0.0f?affinity[k]:0.0f);s+=wk*src_w[(size_t)sources[k]*ED_HEAD_CHANNELS+ic];}dst_w[ic]=s/wsum;}
    {float s=0.0f;for(k=0;k<nsrc;++k){float wk=equal?1.0f:(affinity[k]>0.0f?affinity[k]:0.0f);s+=wk*src_b[sources[k]];}*dst_b=s/wsum;}
}

static ed_status collect_record(const ed_model *m,const ed_dataset *d,uint32_t record_index,uint32_t image_slot,
                               calibration_detection **predictions,size_t *prediction_count,size_t *prediction_capacity,
                               feature_bank *pos,feature_bank *neg,stored_loc *store,uint32_t *store_count){
    const uint8_t *encoded;const ed_edb_annotation_disk *anns;uint32_t encoded_n,ann_n,w,h,l,s,loc_n=0,g;uint8_t *rgb=NULL;float *input=NULL;ed_activation *a=NULL;
    ed_detection *top=NULL;uint32_t *top_count=NULL;loc_tmp *locs=NULL;ed_status status;
    status=ed_dataset_record(d,record_index,&encoded,&encoded_n,&anns,&ann_n,&w,&h);if(status!=ED_OK)return status;
    status=ed_decode_image(encoded,encoded_n,&rgb,&w,&h);if(status!=ED_OK)return status;
    {ed_image image={rgb,w,h,w*3u};status=ed_prepare_input_320(&image,&input);}if(status==ED_OK)status=ed_graph_execute(m,input,&a);
    top=(ed_detection*)calloc((size_t)m->class_count*ED_CALIBRATION_TOP_PER_IMAGE,sizeof(*top));
    top_count=(uint32_t*)calloc(m->class_count,sizeof(*top_count));
    locs=(loc_tmp*)calloc(40u*40u+20u*20u+10u*10u+5u*5u,sizeof(*locs));
    if(!top||!top_count||!locs)status=ED_ERR_MEMORY;
    if(status==ED_OK)for(l=0;l<ED_PICODET_LEVELS;++l){
        const ed_activation *score=&a[ed_picodet_cls_nodes[l]],*reg=&a[ed_picodet_reg_nodes[l]],*feat=&a[ed_picodet_feature_nodes[l]];
        int32_t qn=quality_node(l);const ed_activation *q;uint32_t y,x,locations=reg->h*reg->w;float *dist;float stride=(float)ed_picodet_strides[l];
        if(qn<0||feat->c!=ED_HEAD_CHANNELS){status=ED_ERR_FORMAT;break;}
        q=&a[qn];dist=(float*)malloc((size_t)locations*4u*sizeof(float));if(!dist){status=ED_ERR_MEMORY;break;}
        ed_dfl_decode_f32(reg->data,locations,7u,dist);
        for(y=0;y<reg->h;++y)for(x=0;x<reg->w;++x){
            uint32_t local=y*reg->w+x;const float *qv=dist+(size_t)local*4u;loc_tmp *z=&locs[loc_n];float cx=((float)x+0.5f)*stride,cy=((float)y+0.5f)*stride,best=0.0f;
            memcpy(z->feat,feat->data+(size_t)local*ED_HEAD_CHANNELS,ED_HEAD_CHANNELS*sizeof(float));
            z->quality=q->data[local];z->cx=cx;z->cy=cy;z->assigned=-1;
            z->x1=(cx-qv[0]*stride)*(float)w/320.0f;z->y1=(cy-qv[1]*stride)*(float)h/320.0f;
            z->x2=(cx+qv[2]*stride)*(float)w/320.0f;z->y2=(cy+qv[3]*stride)*(float)h/320.0f;
            for(s=0;s<m->class_count;++s){float confidence=score->data[(size_t)local*m->class_count+s];ed_detection det;
                if(confidence>best)best=confidence;
                if(confidence<ED_CALIBRATION_SCORE_FLOOR)continue;
                det.x1=z->x1;det.y1=z->y1;det.x2=z->x2;det.y2=z->y2;det.score=confidence;det.class_id=s;
                top_insert(top+(size_t)s*ED_CALIBRATION_TOP_PER_IMAGE,&top_count[s],&det);
            }
            z->max_cls=best;++loc_n;
        }
        free(dist);
    }
    if(status==ED_OK)for(s=0;s<m->class_count;++s){ed_detection *items=top+(size_t)s*ED_CALIBRATION_TOP_PER_IMAGE;uint32_t i;size_t kept=ed_nms(items,top_count[s],0.6f);
        for(i=0;i<kept;++i)if(!append_prediction(predictions,prediction_count,prediction_capacity,&items[i],image_slot,s)){status=ED_ERR_MEMORY;break;}
        if(status!=ED_OK)break;
    }
    if(status==ED_OK){
        uint32_t local_n=0,level_base[ED_PICODET_LEVELS],lv;
        level_base[0]=0u;for(lv=1;lv<ED_PICODET_LEVELS;++lv)level_base[lv]=level_base[lv-1u]+a[ed_picodet_reg_nodes[lv-1u]].h*a[ed_picodet_reg_nodes[lv-1u]].w;
        for(g=0;g<ann_n;++g)if(!(anns[g].flags&ED_ANN_FLAG_IGNORE)&&anns[g].class_id<d->header.class_count){
            float gx1=anns[g].x1*320.0f/(float)w,gy1=anns[g].y1*320.0f/(float)h,gx2=anns[g].x2*320.0f/(float)w,gy2=anns[g].y2*320.0f/(float)h;
            float best_m[ED_CALIB_TOPK];uint32_t best_i[ED_CALIB_TOPK],best_n=0,i,k,slot;
            memset(best_m,0,sizeof(best_m));memset(best_i,0,sizeof(best_i));
            for(i=0;i<loc_n;++i){
                float box320[4],iou,metric;
                if(locs[i].cx<gx1||locs[i].cx>gx2||locs[i].cy<gy1||locs[i].cy>gy2)continue;
                box320[0]=locs[i].x1*320.0f/(float)w;box320[1]=locs[i].y1*320.0f/(float)h;box320[2]=locs[i].x2*320.0f/(float)w;box320[3]=locs[i].y2*320.0f/(float)h;
                iou=calibration_iou(box320[0],box320[1],box320[2],box320[3],gx1,gy1,gx2,gy2);
                metric=(locs[i].quality>0.0f?locs[i].quality:0.0f)*powf(iou>0.0f?iou:0.0f,ED_CALIB_IOU_POWER);
                if(best_n<ED_CALIB_TOPK)slot=best_n++;
                else{slot=0;for(k=1;k<ED_CALIB_TOPK;++k)if(best_m[k]<best_m[slot])slot=k;if(metric<=best_m[slot])continue;}
                best_m[slot]=metric;best_i[slot]=i;
            }
            for(i=0;i<best_n;++i)if(best_m[i]>0.0f){
                uint32_t idx=best_i[i];
                bank_insert(&pos[anns[g].class_id],locs[idx].feat,best_m[i],image_slot);
                if(locs[idx].assigned<0||best_m[i]>0.0f)locs[idx].assigned=(int)anns[g].class_id;
            }
        }
        for(g=0;g<loc_n;++g){
            uint32_t target;
            if(locs[g].quality<0.12f&&locs[g].max_cls<0.12f)continue;
            for(target=0;target<d->header.class_count;++target){
                float q=locs[g].quality;
                if(locs[g].assigned==(int)target)continue;
                if(locs[g].assigned>=0)q+=1.0f;
                bank_insert(&neg[target],locs[g].feat,q,image_slot);
            }
            {stored_loc cand;memcpy(cand.feat,locs[g].feat,sizeof(cand.feat));cand.quality=locs[g].quality>locs[g].max_cls?locs[g].quality:locs[g].max_cls;
                cand.x1=locs[g].x1;cand.y1=locs[g].y1;cand.x2=locs[g].x2;cand.y2=locs[g].y2;cand.image_slot=image_slot;cand.level=0;
                for(lv=1;lv<ED_PICODET_LEVELS;++lv)if(g>=level_base[lv])cand.level=lv;
                store_insert(store+*store_count,&local_n,ED_CALIB_STORE_PER_IMAGE,&cand);}
        }
        *store_count+=local_n;
    }
    free(locs);free(top_count);free(top);ed_graph_activations_free(a);free(input);free(rgb);return status;
}

static float calibration_ap(const ed_dataset *d,const uint32_t *selected,uint32_t selected_count,const calibration_detection *all,size_t all_count,uint32_t target,uint32_t source,uint32_t *object_count){
    calibration_detection *predictions=NULL;uint8_t **matched=NULL;size_t count=0,i,k;uint32_t image,tp=0,fp=0,total_gt=0;float *recall=NULL,*precision=NULL,previous=0.0f,ap=0.0f;
    for(i=0;i<all_count;++i)if(all[i].source_class==source)++count;
    predictions=(calibration_detection*)malloc((count?count:1u)*sizeof(*predictions));matched=(uint8_t**)calloc(selected_count,sizeof(*matched));
    if(!predictions||!matched)goto done;
    count=0;for(i=0;i<all_count;++i)if(all[i].source_class==source)predictions[count++]=all[i];qsort(predictions,count,sizeof(*predictions),score_desc);
    recall=(float*)malloc((count?count:1u)*sizeof(float));precision=(float*)malloc((count?count:1u)*sizeof(float));if(!recall||!precision)goto done;
    for(image=0;image<selected_count;++image){const ed_edb_record_disk *r=&d->records[selected[image]];const ed_edb_annotation_disk *a=(const ed_edb_annotation_disk*)(d->file_data+r->annotation_offset);uint32_t j;matched[image]=(uint8_t*)calloc(r->annotation_count?r->annotation_count:1u,1);if(!matched[image])goto done;for(j=0;j<r->annotation_count;++j)if(!(a[j].flags&ED_ANN_FLAG_IGNORE)&&a[j].class_id==target)++total_gt;}
    k=0;for(i=0;i<count;++i){const calibration_detection *p=&predictions[i];const ed_edb_record_disk *r=&d->records[selected[p->image_slot]];const ed_edb_annotation_disk *a=(const ed_edb_annotation_disk*)(d->file_data+r->annotation_offset);uint32_t j,best=UINT32_MAX;float best_iou=0.0f;
        for(j=0;j<r->annotation_count;++j)if(!(a[j].flags&ED_ANN_FLAG_IGNORE)&&a[j].class_id==target){float iou=calibration_iou(p->detection.x1,p->detection.y1,p->detection.x2,p->detection.y2,a[j].x1,a[j].y1,a[j].x2,a[j].y2);if(iou>best_iou){best_iou=iou;best=j;}}
        if(best!=UINT32_MAX&&best_iou>=0.5f&&!matched[p->image_slot][best]){matched[p->image_slot][best]=1u;++tp;}else ++fp;recall[k]=total_gt?(float)tp/(float)total_gt:0.0f;precision[k]=(float)tp/(float)(tp+fp);++k;
    }
    if(k){for(i=k-1;i>0;--i)if(precision[i-1]<precision[i])precision[i-1]=precision[i];for(i=0;i<k;++i){float delta=recall[i]-previous;if(delta>0.0f)ap+=delta*precision[i];previous=recall[i];}}
done:
    if(object_count)*object_count=total_gt;
    if(matched){for(image=0;image<selected_count;++image)free(matched[image]);}
    free(matched);free(predictions);free(recall);free(precision);return ap;
}

static float score_heads_ap(const ed_dataset *d,const uint32_t *selected,uint32_t selected_count,const stored_loc *store,uint32_t store_count,
                            const float *weights,const float *biases,uint32_t class_count,int select_fold,float *per_class){
    calibration_detection *predictions=NULL;size_t count=0,capacity=0,i;uint32_t c;float map=0.0f;
    for(i=0;i<store_count;++i){
        const stored_loc *z=&store[i];uint32_t cls;
        if(select_fold&&!fold_select(z->image_slot))continue;
        if(!select_fold&&fold_select(z->image_slot))continue;
        for(cls=0;cls<class_count;++cls){
            const float *w=weights+((size_t)z->level*class_count+cls)*ED_HEAD_CHANNELS;
            float logit=vec_dot(w,z->feat,ED_HEAD_CHANNELS)+biases[(size_t)z->level*class_count+cls];
            float prob=1.0f/(1.0f+expf(-clampf(logit,-20.0f,20.0f)));float score=sqrtf(prob*(z->quality>0.0f?z->quality:0.0f));
            ed_detection det;if(score<ED_CALIBRATION_SCORE_FLOOR)continue;
            det.x1=z->x1;det.y1=z->y1;det.x2=z->x2;det.y2=z->y2;det.score=score;det.class_id=cls;
            if(!append_prediction(&predictions,&count,&capacity,&det,z->image_slot,cls)){free(predictions);return 0.0f;}
        }
    }
    for(c=0;c<class_count;++c){
        float ap=calibration_ap(d,selected,selected_count,predictions,count,c,c,NULL);
        if(per_class)per_class[c]=ap;
        map+=ap/(float)class_count;
    }
    free(predictions);return map;
}

static int snapshot_cls(const ed_model *m,float **out_w,float **out_b,uint32_t *old_count){
    uint32_t level;
    *old_count=m->class_count;
    for(level=0;level<ED_PICODET_LEVELS;++level){
        const ed_graph_node *n=&ed_picodet_nodes[ed_picodet_raw_cls_nodes[level]];const ed_tensor *wt,*bt;
        if(n->op!=ED_OP_CONV||n->input_count<3||n->input[1]>-2||n->input[2]>-2)return 0;
        wt=&m->tensors[-n->input[1]-2];bt=&m->tensors[-n->input[2]-2];
        if(wt->dims[0]!=m->class_count||wt->dims[3]!=ED_HEAD_CHANNELS||bt->dims[0]!=m->class_count)return 0;
        out_w[level]=(float*)malloc(wt->data_bytes);out_b[level]=(float*)malloc(bt->data_bytes);
        if(!out_w[level]||!out_b[level])return 0;
        memcpy(out_w[level],wt->data,wt->data_bytes);memcpy(out_b[level],bt->data,bt->data_bytes);
    }
    return 1;
}

static void build_heads(ed_class_init_mode mode,ed_proto_solver solver,int fit_only,
                        float *const src_w[ED_PICODET_LEVELS],float *const src_b[ED_PICODET_LEVELS],
                        const ed_class_calibration_report *report,const feature_bank *pos,const feature_bank *neg,
                        uint32_t class_count,float *weights,float *biases,uint32_t *used_pos,uint32_t *used_neg){
    uint32_t target,level;
    for(target=0;target<class_count;++target){
        uint32_t nsrc=report->source_count[target],strong=report->source_affinity[target][0]>=ED_CALIBRATION_STRONG_AFFINITY?1u:0u;
        uint32_t use_n=nsrc;int use_proto=0;
        float *xp=NULL,*xn=NULL;uint32_t np,nn;
        if(mode==ED_INIT_BEST_SINGLE)use_n=1u;
        else if(strong&&mode!=ED_INIT_PROTOTYPE)use_n=1u;
        np=gather_bank(&pos[target],fit_only,NULL);nn=gather_bank(&neg[target],fit_only,NULL);
        if(used_pos)used_pos[target]=np;
        if(used_neg)used_neg[target]=nn;
        if((mode==ED_INIT_PROTOTYPE||mode==ED_INIT_PROTOTYPE_RESIDUAL)&&np>=ED_PROTO_MIN_POS){
            xp=(float*)malloc((size_t)(np?np:1u)*ED_HEAD_CHANNELS*sizeof(float));
            xn=(float*)malloc((size_t)(nn?nn:1u)*ED_HEAD_CHANNELS*sizeof(float));
            if(xp&&xn){gather_bank(&pos[target],fit_only,xp);gather_bank(&neg[target],fit_only,xn);use_proto=1;}
        }
        for(level=0;level<ED_PICODET_LEVELS;++level){
            float *dw=weights+((size_t)level*class_count+target)*ED_HEAD_CHANNELS;float *db=biases+(size_t)level*class_count+target;
            if(mode==ED_INIT_BEST_SINGLE||(strong&&mode!=ED_INIT_PROTOTYPE&&mode!=ED_INIT_AP_WEIGHTED)){
                blend_from_sources(src_w[level],src_b[level],report->source_class[target],report->source_affinity[target],1u,1,dw,db);
            }else if(mode==ED_INIT_AP_WEIGHTED){
                blend_from_sources(src_w[level],src_b[level],report->source_class[target],report->source_affinity[target],use_n,0,dw,db);
            }else if((mode==ED_INIT_PROTOTYPE||mode==ED_INIT_PROTOTYPE_RESIDUAL)&&use_proto&&!strong){
                float blend_w[ED_HEAD_CHANNELS],blend_b,pw[ED_HEAD_CHANNELS],pb;int mixed;
                blend_from_sources(src_w[level],src_b[level],report->source_class[target],report->source_affinity[target],use_n,1,blend_w,&blend_b);
                mixed=fit_source_mix(xp,np,xn,nn,src_w[level],src_b[level],report->source_class[target],use_n,pw,&pb);
                if(!mixed){
                    if(!fit_classifier(solver,xp,np,xn,nn,pw,&pb)){memcpy(dw,blend_w,sizeof(blend_w));*db=blend_b;continue;}
                    match_norm(pw,&pb,src_w[level],report->source_class[target],use_n?use_n:1u);
                    calibrate_bias(pw,&pb,xp,np,xn,nn,1.10f);
                }
                if(mode==ED_INIT_PROTOTYPE_RESIDUAL){
                    uint32_t ic;for(ic=0;ic<ED_HEAD_CHANNELS;++ic)dw[ic]=(1.0f-ED_PROTO_RESIDUAL)*blend_w[ic]+ED_PROTO_RESIDUAL*pw[ic];
                    *db=(1.0f-ED_PROTO_RESIDUAL)*blend_b+ED_PROTO_RESIDUAL*pb;
                }else{memcpy(dw,pw,ED_HEAD_CHANNELS*sizeof(float));*db=pb;}
                match_norm(dw,db,src_w[level],report->source_class[target],use_n?use_n:1u);
            }else{
                blend_from_sources(src_w[level],src_b[level],report->source_class[target],report->source_affinity[target],use_n,1,dw,db);
            }
        }
        free(xp);free(xn);
    }
}

ed_status ed_model_calibrate_classes_ex(ed_model *m,const ed_dataset *d,uint32_t samples_per_class,uint64_t seed,
                                        ed_class_init_mode init_mode,ed_proto_solver proto_solver,ed_class_calibration_report *report){
    uint32_t *selected=NULL,selected_count=0,selected_capacity,target,taken,scanned,start,s,old_count=0,store_count=0,store_cap;
    calibration_detection *predictions=NULL;size_t prediction_count=0,prediction_capacity=0;const char *names[ED_MAX_CLASSES];
    feature_bank *pos=NULL,*neg=NULL;stored_loc *store=NULL;float *src_w[ED_PICODET_LEVELS]={0},*src_b[ED_PICODET_LEVELS]={0};
    float *weights=NULL,*biases=NULL;ed_status status=ED_OK;uint32_t level;
    if(!m||!d||!report||samples_per_class==0u||d->header.class_count==0u||d->header.class_count>ED_MAX_CLASSES||m->class_count==0u||m->class_count>ED_MAX_CLASSES)return ED_ERR_ARGUMENT;
    if((unsigned)init_mode>ED_INIT_PROTOTYPE_RESIDUAL||(unsigned)proto_solver>ED_PROTO_LDA)return ED_ERR_ARGUMENT;
    selected_capacity=d->header.class_count*samples_per_class;selected=(uint32_t*)malloc((size_t)selected_capacity*sizeof(*selected));if(!selected)return ED_ERR_MEMORY;
    memset(report,0,sizeof(*report));report->class_count=d->header.class_count;report->init_mode_used=(uint32_t)init_mode;report->proto_solver_used=(uint32_t)proto_solver;
    pos=(feature_bank*)calloc(d->header.class_count,sizeof(*pos));neg=(feature_bank*)calloc(d->header.class_count,sizeof(*neg));
    if(!pos||!neg){status=ED_ERR_MEMORY;goto done;}
    for(target=0;target<d->header.class_count;++target)if(!bank_init(&pos[target],ED_CALIB_POS_CAP)||!bank_init(&neg[target],ED_CALIB_NEG_CAP)){status=ED_ERR_MEMORY;goto done;}
    for(target=0;target<d->header.class_count;++target){start=(uint32_t)((seed+target*2654435761u)%d->header.record_count);taken=0;scanned=0;
        while(taken<samples_per_class&&scanned<d->header.record_count){uint32_t index=(start+scanned)%d->header.record_count;++scanned;if(!record_has_target(d,index,target))continue;++taken;if(!selected_contains(selected,selected_count,index))selected[selected_count++]=index;}
        if(taken==0u){status=ED_ERR_FORMAT;goto done;}
    }
    store_cap=selected_count*ED_CALIB_STORE_PER_IMAGE;store=(stored_loc*)malloc((size_t)store_cap*sizeof(*store));if(!store){status=ED_ERR_MEMORY;goto done;}
    for(s=0;s<selected_count;++s){status=collect_record(m,d,selected[s],s,&predictions,&prediction_count,&prediction_capacity,pos,neg,store,&store_count);if(status!=ED_OK)goto done;}
    for(target=0;target<d->header.class_count;++target){uint32_t k,limit=m->class_count<ED_CALIBRATION_NOVEL_BLEND?m->class_count:ED_CALIBRATION_NOVEL_BLEND;
        for(k=0;k<ED_CALIBRATION_NOVEL_BLEND;++k){report->source_class[target][k]=UINT32_MAX;report->source_affinity[target][k]=-1.0f;}
        for(s=0;s<m->class_count;++s){float ap=calibration_ap(d,selected,selected_count,predictions,prediction_count,target,s,s==0u?&report->calibrated_objects[target]:NULL);for(k=0;k<limit;++k)if(ap>report->source_affinity[target][k]){uint32_t move;for(move=limit-1u;move>k;--move){report->source_affinity[target][move]=report->source_affinity[target][move-1u];report->source_class[target][move]=report->source_class[target][move-1u];}report->source_affinity[target][k]=ap;report->source_class[target][k]=s;break;}}
        report->source_count[target]=report->source_affinity[target][0]>=ED_CALIBRATION_STRONG_AFFINITY?1u:limit;report->blended_affinity[target]=report->source_affinity[target][0];
        names[target]=d->class_names[target];
    }
    if(!snapshot_cls(m,src_w,src_b,&old_count)){status=ED_ERR_FORMAT;goto done;}(void)old_count;
    weights=(float*)calloc((size_t)ED_PICODET_LEVELS*d->header.class_count*ED_HEAD_CHANNELS,sizeof(float));
    biases=(float*)calloc((size_t)ED_PICODET_LEVELS*d->header.class_count,sizeof(float));
    if(!weights||!biases){status=ED_ERR_MEMORY;goto done;}
    if(init_mode==ED_INIT_AUTO){
        static const ed_class_init_mode cand[]={ED_INIT_BEST_SINGLE,ED_INIT_EQUAL_BLEND,ED_INIT_AP_WEIGHTED,ED_INIT_PROTOTYPE,ED_INIT_PROTOTYPE_RESIDUAL};
        float best_map=-1.0f;ed_class_init_mode best=ED_INIT_EQUAL_BLEND;uint32_t ci;
        for(ci=0;ci<sizeof(cand)/sizeof(cand[0]);++ci){
            float fold_ap[ED_MAX_CLASSES],map;
            build_heads(cand[ci],proto_solver,1,src_w,src_b,report,pos,neg,d->header.class_count,weights,biases,NULL,NULL);
            map=score_heads_ap(d,selected,selected_count,store,store_count,weights,biases,d->header.class_count,1,fold_ap);
            if(map>best_map){best_map=map;best=cand[ci];memcpy(report->fold_ap,fold_ap,(size_t)d->header.class_count*sizeof(float));}
        }
        report->fold_map50=best_map;report->init_mode_used=(uint32_t)best;
        build_heads(best,proto_solver,0,src_w,src_b,report,pos,neg,d->header.class_count,weights,biases,report->proto_positives,report->proto_negatives);
    }else{
        build_heads(init_mode,proto_solver,0,src_w,src_b,report,pos,neg,d->header.class_count,weights,biases,report->proto_positives,report->proto_negatives);
        report->fold_map50=score_heads_ap(d,selected,selected_count,store,store_count,weights,biases,d->header.class_count,1,report->fold_ap);
        report->init_mode_used=(uint32_t)init_mode;
    }
    for(target=0;target<d->header.class_count;++target)report->classifier_norm[target]=vec_l2(weights+(size_t)target*ED_HEAD_CHANNELS,ED_HEAD_CHANNELS);
    status=ed_model_remap_class_heads(m,names,d->header.class_count,weights,biases);
done:
    free(predictions);free(selected);free(store);free(weights);free(biases);
    if(pos){for(target=0;target<d->header.class_count;++target)bank_free(&pos[target]);free(pos);}
    if(neg){for(target=0;target<d->header.class_count;++target)bank_free(&neg[target]);free(neg);}
    for(level=0;level<ED_PICODET_LEVELS;++level){free(src_w[level]);free(src_b[level]);}
    return status;
}

ed_status ed_model_calibrate_classes(ed_model *m,const ed_dataset *d,uint32_t samples_per_class,uint64_t seed,ed_class_calibration_report *report){
    return ed_model_calibrate_classes_ex(m,d,samples_per_class,seed,ED_INIT_EQUAL_BLEND,ED_PROTO_RIDGE,report);
}
