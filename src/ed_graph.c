#include "ed_graph.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define ED_FEATURE_ADAPTER_RANK 8u

static const float *ref_data(const ed_model*m,const float*image,const ed_activation*a,int32_t ref){
    if(ref==-1)return image;if(ref<=-2){uint32_t i=(uint32_t)(-ref-2);return i<m->tensor_count?(const float*)m->tensors[i].data:NULL;}return (uint32_t)ref<ed_picodet_node_count?a[ref].data:NULL;
}
static void ref_shape(const ed_model*m,const ed_activation*a,int32_t ref,uint32_t*h,uint32_t*w,uint32_t*c){
    if(ref==-1){*h=320;*w=320;*c=3;return;}if(ref<=-2){const ed_tensor*t=&m->tensors[-ref-2];*h=t->rank>2?t->dims[t->rank-3]:1;*w=t->rank>1?t->dims[t->rank-2]:1;*c=t->dims[t->rank-1];return;}*h=a[ref].h;*w=a[ref].w;*c=a[ref].c;
}
static size_t elems(uint32_t h,uint32_t w,uint32_t c){return(size_t)h*w*c;}
static ed_status alloc_act(ed_activation*x,uint32_t h,uint32_t w,uint32_t c){x->h=h;x->w=w;x->c=c;x->data=(float*)malloc(elems(h,w,c)*sizeof(float));return x->data?ED_OK:ED_ERR_MEMORY;}
static float bcast(const float*p,uint32_t ph,uint32_t pw,uint32_t pc,uint32_t y,uint32_t x,uint32_t c){uint32_t iy=ph==1?0:y,ix=pw==1?0:x,ic=pc==1?0:c;return p[((size_t)iy*pw+ix)*pc+ic];}
static int32_t quality_node(uint32_t level){int32_t cls=ed_picodet_cls_nodes[level],mul=ed_picodet_nodes[cls].input[0],left=ed_picodet_nodes[mul].input[0],right=ed_picodet_nodes[mul].input[1],raw=ed_picodet_raw_cls_nodes[level];if(ed_picodet_nodes[left].op!=ED_OP_SIGMOID||ed_picodet_nodes[right].op!=ED_OP_SIGMOID)return -1;if(ed_picodet_nodes[left].input[0]==raw)return right;if(ed_picodet_nodes[right].input[0]==raw)return left;return -1;}
ed_status ed_graph_apply_spatial(const ed_model*m,ed_activation*a){
    static const char *wn[ED_PICODET_LEVELS]={"ed.picofeat.0.a","ed.picofeat.1.a","ed.picofeat.2.a","ed.picofeat.3.a"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.picofeat.0.b","ed.picofeat.1.b","ed.picofeat.2.b","ed.picofeat.3.b"};
    uint32_t l;if(!m||!a)return ED_ERR_ARGUMENT;
    for(l=0;l<ED_PICODET_LEVELS;++l){const ed_tensor*w=ed_find_tensor_const(m,wn[l]),*b=ed_find_tensor_const(m,bn[l]);ed_activation*f=&a[ed_picodet_feature_nodes[l]],*raw=&a[ed_picodet_raw_cls_nodes[l]],*score=&a[ed_picodet_cls_nodes[l]];int32_t qn;uint32_t y,x,c,r,k;
        if(!w&&!b)continue;if(!w||!b||w->dims[0]!=m->class_count||w->dims[1]!=ED_FEATURE_ADAPTER_RANK||w->dims[3]!=96u||b->dims[0]!=m->class_count||b->dims[1]!=ED_FEATURE_ADAPTER_RANK||f->c!=96u||raw->c!=m->class_count)return ED_ERR_FORMAT;qn=quality_node(l);if(qn<0)return ED_ERR_FORMAT;
        for(y=0;y<raw->h;++y)for(x=0;x<raw->w;++x){const float*fp=f->data+((size_t)y*f->w+x)*96u;for(c=0;c<m->class_count;++c){float correction=0.0f,p,q;size_t i=((size_t)y*raw->w+x)*m->class_count+c;for(r=0;r<ED_FEATURE_ADAPTER_RANK;++r){float h=0.0f;const float*aw=(const float*)w->data+((size_t)c*ED_FEATURE_ADAPTER_RANK+r)*96u;for(k=0;k<96u;++k)h+=aw[k]*fp[k];if(h>0.0f)correction+=((const float*)b->data)[(size_t)c*ED_FEATURE_ADAPTER_RANK+r]*h;}raw->data[i]+=correction;p=1.0f/(1.0f+expf(-raw->data[i]));q=a[qn].data[(size_t)y*raw->w+x];score->data[i]=sqrtf(p*q);}}
    }return ED_OK;
}
ed_status ed_graph_apply_context(const ed_model*m,ed_activation*a){
    static const char *wn[ED_PICODET_LEVELS]={"ed.roi.0.w","ed.roi.1.w","ed.roi.2.w","ed.roi.3.w"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.roi.0.b","ed.roi.1.b","ed.roi.2.b","ed.roi.3.b"};
    uint32_t l;if(!m||!a)return ED_ERR_ARGUMENT;
    for(l=0;l<ED_PICODET_LEVELS;++l){const ed_tensor*w=ed_find_tensor_const(m,wn[l]),*b=ed_find_tensor_const(m,bn[l]);ed_activation*f=&a[ed_picodet_feature_nodes[l]],*raw=&a[ed_picodet_raw_cls_nodes[l]],*reg=&a[ed_picodet_reg_nodes[l]],*score=&a[ed_picodet_cls_nodes[l]];int32_t qn;float *dist;uint32_t y,x,c,k;
        if(!w&&!b)continue;if(!w||!b||w->dims[0]!=m->class_count||w->dims[3]!=96u||b->dims[0]!=m->class_count||f->c!=96u||raw->c!=m->class_count)return ED_ERR_FORMAT;qn=quality_node(l);if(qn<0)return ED_ERR_FORMAT;dist=(float*)malloc((size_t)reg->h*reg->w*4u*sizeof(float));if(!dist)return ED_ERR_MEMORY;ed_dfl_decode_f32(reg->data,reg->h*reg->w,7u,dist);
        for(y=0;y<raw->h;++y)for(x=0;x<raw->w;++x){const float*d=dist+((size_t)y*raw->w+x)*4u;float mean[96]={0};int x0=(int)floorf((float)x+0.5f-d[0]),y0=(int)floorf((float)y+0.5f-d[1]),x1=(int)ceilf((float)x+0.5f+d[2]),y1=(int)ceilf((float)y+0.5f+d[3]),iy,ix;uint32_t count=0;if(x0<0)x0=0;if(y0<0)y0=0;if(x1>(int)f->w)x1=(int)f->w;if(y1>(int)f->h)y1=(int)f->h;if(x1<=x0)x1=x0+1;if(y1<=y0)y1=y0+1;for(iy=y0;iy<y1;++iy)for(ix=x0;ix<x1;++ix){const float*fp=f->data+((size_t)iy*f->w+(uint32_t)ix)*96u;for(k=0;k<96u;++k)mean[k]+=fp[k];++count;}for(k=0;k<96u;++k)mean[k]/=(float)count;
            for(c=0;c<m->class_count;++c){size_t i=((size_t)y*raw->w+x)*m->class_count+c;float z=((const float*)b->data)[c],p,q;for(k=0;k<96u;++k)z+=((const float*)w->data)[(size_t)c*96u+k]*mean[k];raw->data[i]+=z;p=1.0f/(1.0f+expf(-raw->data[i]));q=a[qn].data[(size_t)y*raw->w+x];score->data[i]=sqrtf(p*q);}}
        free(dist);
    }return ED_OK;
}
ed_status ed_graph_apply_quality(const ed_model*m,ed_activation*a){
    static const char *wn[ED_PICODET_LEVELS]={"ed.quality.0.w","ed.quality.1.w","ed.quality.2.w","ed.quality.3.w"};
    static const char *bn[ED_PICODET_LEVELS]={"ed.quality.0.b","ed.quality.1.b","ed.quality.2.b","ed.quality.3.b"};
    uint32_t l;if(!m||!a)return ED_ERR_ARGUMENT;
    for(l=0;l<ED_PICODET_LEVELS;++l){const ed_tensor*w=ed_find_tensor_const(m,wn[l]),*b=ed_find_tensor_const(m,bn[l]);ed_activation*f=&a[ed_picodet_feature_nodes[l]],*raw=&a[ed_picodet_raw_cls_nodes[l]],*score=&a[ed_picodet_cls_nodes[l]];int32_t qn;uint32_t y,x,c,k;
        if(!w&&!b)continue;if(!w||!b||w->rank!=2u||w->dims[0]!=m->class_count||w->dims[1]!=96u||w->data_bytes!=(uint64_t)m->class_count*96u*sizeof(float)||b->rank!=1u||b->dims[0]!=m->class_count||b->data_bytes!=(uint64_t)m->class_count*sizeof(float)||f->c!=96u||raw->c!=m->class_count)return ED_ERR_FORMAT;qn=quality_node(l);if(qn<0)return ED_ERR_FORMAT;
        for(y=0;y<raw->h;++y)for(x=0;x<raw->w;++x){const float*fp=f->data+((size_t)y*f->w+x)*96u;float base_q=a[qn].data[(size_t)y*raw->w+x];base_q=base_q<1e-6f?1e-6f:(base_q>1.0f-1e-6f?1.0f-1e-6f:base_q);for(c=0;c<m->class_count;++c){size_t i=((size_t)y*raw->w+x)*m->class_count+c;float z=logf(base_q/(1.0f-base_q))+((const float*)b->data)[c],p,q;for(k=0;k<96u;++k)z+=((const float*)w->data)[(size_t)c*96u+k]*fp[k];q=1.0f/(1.0f+expf(-z));p=1.0f/(1.0f+expf(-raw->data[i]));score->data[i]=sqrtf(p*q);}}
    }return ED_OK;
}

ed_status ed_graph_execute(const ed_model*m,const float*image,ed_activation**out){
    ed_activation*a;uint32_t i;if(!m||!image||!out)return ED_ERR_ARGUMENT;*out=NULL;
    if(m->tensor_count<ed_picodet_tensor_count)return ED_ERR_FORMAT;
    for(i=0;i<ed_picodet_tensor_count;++i)if(strcmp(m->tensors[i].name,ed_picodet_tensor_names[i])!=0)return ED_ERR_FORMAT;
    a=(ed_activation*)calloc(ed_picodet_node_count,sizeof(*a));if(!a)return ED_ERR_MEMORY;
    ed_runtime_power_mark_begin();
    for(i=0;i<ed_picodet_node_count;++i){
        const ed_graph_node*n=&ed_picodet_nodes[i];const float*p0=ref_data(m,image,a,n->input[0]);uint32_t h0,w0,c0,h,w,c;
        ref_shape(m,a,n->input[0],&h0,&w0,&c0);h=n->out_h?n->out_h:h0;w=n->out_w?n->out_w:w0;c=n->out_c?n->out_c:c0;
        if(n->op==ED_OP_CONV){const ed_tensor*wt=&m->tensors[-n->input[1]-2];c=wt->dims[0];h=(h0+2u*n->padding-n->kernel)/n->stride+1u;w=(w0+2u*n->padding-n->kernel)/n->stride+1u;}
        if(n->op==ED_OP_GLOBAL_AVG){h=1;w=1;}if(n->op==ED_OP_RESIZE_NEAREST){h=h0*2u;w=w0*2u;}
        if(n->op==ED_OP_CONCAT_CHANNEL){uint32_t h1,w1,c1;ref_shape(m,a,n->input[1],&h1,&w1,&c1);if(h0!=h1||w0!=w1){ed_runtime_power_mark_end();ed_graph_activations_free(a);return ED_ERR_FORMAT;}c=c0+c1;}
        if(alloc_act(&a[i],h,w,c)!=ED_OK){ed_runtime_power_mark_end();ed_graph_activations_free(a);return ED_ERR_MEMORY;}
        if(n->op==ED_OP_CONV){
            const float*wp=ref_data(m,image,a,n->input[1]);const float*bp=n->input_count>2?ref_data(m,image,a,n->input[2]):NULL;
            if(n->groups==c0)ed_depthwise_conv2d_f32(p0,h0,w0,c0,wp,bp,n->kernel,n->stride,n->padding,a[i].data);
            else ed_conv2d_f32(p0,h0,w0,c0,wp,bp,c,n->kernel,n->stride,n->padding,a[i].data);
        } else if(n->op==ED_OP_GLOBAL_AVG){uint32_t y,x,ch;for(ch=0;ch<c0;++ch){float s=0;for(y=0;y<h0;++y)for(x=0;x<w0;++x)s+=p0[((size_t)y*w0+x)*c0+ch];a[i].data[ch]=s/(float)(h0*w0);}}
        else if(n->op==ED_OP_RELU){size_t z,N=elems(h,w,c);for(z=0;z<N;++z)a[i].data[z]=p0[z]>0?p0[z]:0;}
        else if(n->op==ED_OP_HARD_SIGMOID){size_t z,N=elems(h,w,c);for(z=0;z<N;++z){float v=p0[z]/6.0f+0.5f;a[i].data[z]=v<0?0:(v>1?1:v);}}
        else if(n->op==ED_OP_SIGMOID){size_t z,N=elems(h,w,c);for(z=0;z<N;++z)a[i].data[z]=1.0f/(1.0f+expf(-p0[z]));}
        else if(n->op==ED_OP_SQRT){size_t z,N=elems(h,w,c);for(z=0;z<N;++z)a[i].data[z]=sqrtf(p0[z]>0?p0[z]:0);}
        else if(n->op==ED_OP_RESIZE_NEAREST){uint32_t y,x,ch;for(y=0;y<h;++y)for(x=0;x<w;++x)for(ch=0;ch<c;++ch)a[i].data[((size_t)y*w+x)*c+ch]=p0[((size_t)(y/2u)*w0+x/2u)*c+ch];}
        else if(n->op==ED_OP_CONCAT_CHANNEL){const float*p1=ref_data(m,image,a,n->input[1]);uint32_t y,x;for(y=0;y<h;++y)for(x=0;x<w;++x){float*q=a[i].data+((size_t)y*w+x)*c;memcpy(q,p0+((size_t)y*w+x)*c0,c0*sizeof(float));memcpy(q+c0,p1+((size_t)y*w+x)*(c-c0),(c-c0)*sizeof(float));}}
        else if(n->op==ED_OP_ADD||n->op==ED_OP_MUL||n->op==ED_OP_DIV){const float*p1=ref_data(m,image,a,n->input[1]);uint32_t h1,w1,c1,y,x,ch;ref_shape(m,a,n->input[1],&h1,&w1,&c1);for(y=0;y<h;++y)for(x=0;x<w;++x)for(ch=0;ch<c;++ch){float u=bcast(p0,h0,w0,c0,y,x,ch),v=bcast(p1,h1,w1,c1,y,x,ch);a[i].data[((size_t)y*w+x)*c+ch]=n->op==ED_OP_ADD?u+v:(n->op==ED_OP_MUL?u*v:u/v);}}
        else if(n->op==ED_OP_CLIP){const float*lo=ref_data(m,image,a,n->input[1]);const float*hi=ref_data(m,image,a,n->input[2]);size_t z,N=elems(h,w,c);for(z=0;z<N;++z){float v=p0[z];if(v<*lo)v=*lo;if(v>*hi)v=*hi;a[i].data[z]=v;}}
        else {ed_runtime_power_mark_end();ed_graph_activations_free(a);return ED_ERR_UNSUPPORTED;}
    }
    if(ed_graph_apply_spatial(m,a)!=ED_OK||ed_graph_apply_context(m,a)!=ED_OK||ed_graph_apply_quality(m,a)!=ED_OK){ed_runtime_power_mark_end();ed_graph_activations_free(a);return ED_ERR_FORMAT;}*out=a;ed_runtime_power_mark_end();return ED_OK;
}
void ed_graph_activations_free(ed_activation*a){uint32_t i;if(!a)return;for(i=0;i<ed_picodet_node_count;++i)free(a[i].data);free(a);}
