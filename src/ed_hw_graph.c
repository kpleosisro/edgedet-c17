#include "ed_hw.h"
#include "ed_graph.h"
#include "ed_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ed_hw_q *q;
    uint32_t h,w,c;
    float scale;
} hw_act;

typedef struct {
    ed_hw_w *q;
    int32_t *bias;
    float *sw;
    uint32_t oc,k,ic,depthwise;
} hw_wpack;

static uint32_t out_dim(uint32_t n,uint32_t k,uint32_t s,uint32_t p){return(n+2u*p-k)/s+1u;}
static size_t elems(uint32_t h,uint32_t w,uint32_t c){return(size_t)h*w*c;}

static void free_acts(hw_act *a){
    uint32_t i;if(!a)return;
    for(i=0;i<ed_picodet_node_count;++i)free(a[i].q);
    free(a);
}
static int alloc_act(hw_act *x,uint32_t h,uint32_t w,uint32_t c){
    x->h=h;x->w=w;x->c=c;x->scale=1.0f;
    x->q=(ed_hw_q*)calloc(elems(h,w,c),sizeof(ed_hw_q));
    return x->q!=NULL;
}

static const float *tensor_f32(const ed_model *m,int32_t ref){
    uint32_t i;if(ref>-2)return NULL;i=(uint32_t)(-ref-2);
    return i<m->tensor_count?(const float*)m->tensors[i].data:NULL;
}
static const ed_tensor *tensor_ref(const ed_model *m,int32_t ref){
    uint32_t i;if(ref>-2)return NULL;i=(uint32_t)(-ref-2);
    return i<m->tensor_count?&m->tensors[i]:NULL;
}

static void quant_weight(const float *w,uint32_t oc,uint32_t k,uint32_t ic,
                         hw_wpack *p,int depthwise){
    uint32_t o,n=k*k*ic;p->oc=oc;p->k=k;p->ic=ic;p->depthwise=depthwise?1u:0u;
    p->q=(ed_hw_w*)malloc((size_t)oc*n);p->sw=(float*)calloc(oc,sizeof(float));
    p->bias=NULL;if(!p->q||!p->sw)return;
    for(o=0;o<oc;++o){
        const float *row=w+(size_t)o*n;uint32_t i;float mx=0.0f;
        for(i=0;i<n;++i){float a=row[i]<0?-row[i]:row[i];if(a>mx)mx=a;}
        p->sw[o]=mx>0.0f?mx/127.0f:1.0f;
        for(i=0;i<n;++i){
            int v=(int)lrintf(row[i]/p->sw[o]);
            if(v>127)v=127;if(v<-127)v=-127;p->q[(size_t)o*n+i]=(ed_hw_w)v;
        }
    }
}
static void quant_bias(const float *b,uint32_t oc,const float *sw,float sx,int32_t **out){
    uint32_t o;*out=(int32_t*)calloc(oc,sizeof(int32_t));if(!*out||!b)return;
    for(o=0;o<oc;++o){
        float den=sx*sw[o];int32_t q=0;
        if(den>0.0f){double v=b[o]/den;if(v>2147483647.0)v=2147483647.0;if(v<-2147483647.0)v=-2147483647.0;q=(int32_t)lrint(v);}
        (*out)[o]=q;
    }
}

static void requant_from_acc(ed_hw_acc *acc,size_t n,float acc_scale,hw_act *out){
    size_t i;ed_hw_acc mx=0;ed_hw_scale sc;
    for(i=0;i<n;++i){ed_hw_acc a=acc[i]<0?-acc[i]:acc[i];if(a>mx)mx=a;}
    sc=ed_hw_scale_from_max(mx,ED_HW_Q_MAX);
    ed_hw_requant_buffer(acc,n,sc,out->q);
    out->scale=acc_scale*(float)((uint32_t)1u<<(uint32_t)(sc.shift>0?sc.shift:0));
    if(sc.shift<0)out->scale=acc_scale/(float)((uint32_t)1u<<(uint32_t)(-sc.shift));
}

static ed_status prepare_q_input(const ed_image *im,hw_act *out){
    uint32_t y,x,c;static const float mean[3]={0.485f,0.456f,0.406f},std[3]={0.229f,0.224f,0.225f};
    if(!alloc_act(out,320,320,3))return ED_ERR_MEMORY;
    out->scale=1.0f/256.0f;
    for(y=0;y<320u;++y){
        float sy=((float)y+0.5f)*(float)im->height/320.0f-0.5f;int y0=(int)floorf(sy);float fy=sy-(float)y0;int y1=y0+1;
        if(y0<0)y0=0;if(y1>=(int)im->height)y1=(int)im->height-1;
        for(x=0;x<320u;++x){
            float sx=((float)x+0.5f)*(float)im->width/320.0f-0.5f;int x0=(int)floorf(sx);float fx=sx-(float)x0;int x1=x0+1;
            const uint8_t *p00,*p01,*p10,*p11;
            if(x0<0)x0=0;if(x1>=(int)im->width)x1=(int)im->width-1;
            p00=im->rgb+(size_t)y0*im->stride_bytes+(size_t)x0*3u;
            p01=im->rgb+(size_t)y0*im->stride_bytes+(size_t)x1*3u;
            p10=im->rgb+(size_t)y1*im->stride_bytes+(size_t)x0*3u;
            p11=im->rgb+(size_t)y1*im->stride_bytes+(size_t)x1*3u;
            for(c=0;c<3u;++c){
                float v=(1.0f-fy)*((1.0f-fx)*p00[c]+fx*p01[c])+fy*((1.0f-fx)*p10[c]+fx*p11[c]);
                float nrm=(v/255.0f-mean[c])/std[c];
                int q=(int)lrintf(nrm*256.0f);
                if(q>ED_HW_Q_MAX)q=ED_HW_Q_MAX;if(q<ED_HW_Q_MIN)q=ED_HW_Q_MIN;
                out->q[((size_t)y*320u+x)*3u+c]=(ed_hw_q)q;
            }
        }
    }
    return ED_OK;
}

static hw_act *ref_act(hw_act *image,hw_act *a,int32_t ref){
    if(ref==-1)return image;if(ref<=-2)return NULL;
    return (uint32_t)ref<ed_picodet_node_count?&a[ref]:NULL;
}

static void unary_from_float_fn(const hw_act *in,hw_act *out,int kind){
    size_t i,n=elems(out->h,out->w,out->c);
    for(i=0;i<n;++i){
        float real=(float)in->q[i]*in->scale;ed_hw_q q8,r;
        q8=(ed_hw_q)ed_hw_sat16((ed_hw_acc)lrintf(real*256.0f));
        if(kind==0)r=ed_hw_relu(in->q[i]);
        else if(kind==1)r=ed_hw_hard_sigmoid_q8(q8);
        else if(kind==2)r=ed_hw_sigmoid_lut(q8);
        else{
            ed_hw_q q15=(ed_hw_q)ed_hw_sat16((ed_hw_acc)lrintf((real<0?0:real)*32767.0f));
            r=ed_hw_sqrt_lut(q15);
        }
        out->q[i]=(kind==0)?r:r;
    }
    if(kind==0)out->scale=in->scale;
    else if(kind==3)out->scale=1.0f/32767.0f;
    else out->scale=1.0f/32767.0f;
}

static ed_status graph_hw(const ed_model *m,const ed_image *im,hw_act **out_acts){
    hw_act *a,*image=NULL;uint32_t i;
    if(!m||!im||!out_acts)return ED_ERR_ARGUMENT;*out_acts=NULL;
    ed_hw_lut_init();
    a=(hw_act*)calloc(ed_picodet_node_count,sizeof(*a));
    image=(hw_act*)calloc(1,sizeof(*image));
    if(!a||!image){free(a);free(image);return ED_ERR_MEMORY;}
    if(prepare_q_input(im,image)!=ED_OK){free(a);free(image->q);free(image);return ED_ERR_MEMORY;}
    for(i=0;i<ed_picodet_node_count;++i){
        const ed_graph_node *n=&ed_picodet_nodes[i];
        hw_act *p0=ref_act(image,a,n->input[0]);
        uint32_t h=n->out_h?n->out_h:(p0?p0->h:1),w=n->out_w?n->out_w:(p0?p0->w:1),c=n->out_c?n->out_c:(p0?p0->c:1);
        if(n->op==ED_OP_CONV){
            const ed_tensor *wt=tensor_ref(m,n->input[1]);
            if(!wt){free_acts(a);free(image->q);free(image);return ED_ERR_FORMAT;}
            c=wt->dims[0];h=out_dim(p0->h,n->kernel,n->stride,n->padding);w=out_dim(p0->w,n->kernel,n->stride,n->padding);
        }
        if(n->op==ED_OP_GLOBAL_AVG){h=1;w=1;}
        if(n->op==ED_OP_RESIZE_NEAREST){h=p0->h*2u;w=p0->w*2u;c=p0->c;}
        if(n->op==ED_OP_CONCAT_CHANNEL){
            hw_act *p1=ref_act(image,a,n->input[1]);if(!p1||p0->h!=p1->h||p0->w!=p1->w){free_acts(a);free(image->q);free(image);return ED_ERR_FORMAT;}
            h=p0->h;w=p0->w;c=p0->c+p1->c;
        }
        if(!alloc_act(&a[i],h,w,c)){free_acts(a);free(image->q);free(image);return ED_ERR_MEMORY;}
        if(n->op==ED_OP_CONV){
            const ed_tensor *wt=tensor_ref(m,n->input[1]);const float *bp=n->input_count>2?tensor_f32(m,n->input[2]):NULL;
            hw_wpack pack={0};ed_hw_acc *acc;uint32_t oc=wt->dims[0],k=n->kernel,ic=p0->c;
            int dw=n->groups==p0->c&&n->groups>1u;
            if(dw){
                uint32_t ch,ky,kx;const float *wf=(const float*)wt->data;
                pack.oc=ic;pack.k=k;pack.ic=ic;pack.depthwise=1;
                pack.q=(ed_hw_w*)malloc((size_t)k*k*ic);pack.sw=(float*)calloc(ic,sizeof(float));
                if(pack.q&&pack.sw)for(ch=0;ch<ic;++ch){
                    float mx=0.0f;
                    for(ky=0;ky<k;++ky)for(kx=0;kx<k;++kx){
                        float v=wf[((size_t)ky*k+kx)*ic+ch];if(v<0)v=-v;if(v>mx)mx=v;
                    }
                    pack.sw[ch]=mx>0.0f?mx/127.0f:1.0f;
                    for(ky=0;ky<k;++ky)for(kx=0;kx<k;++kx){
                        int q=(int)lrintf(wf[((size_t)ky*k+kx)*ic+ch]/pack.sw[ch]);
                        if(q>127)q=127;if(q<-127)q=-127;
                        pack.q[((size_t)ky*k+kx)*ic+ch]=(ed_hw_w)q;
                    }
                }
            }else quant_weight((const float*)wt->data,oc,k,ic,&pack,0);
            if(bp)quant_bias(bp,dw?ic:oc,pack.sw,p0->scale,&pack.bias);
            acc=(ed_hw_acc*)malloc(elems(h,w,c)*sizeof(*acc));
            if(!acc||!pack.q){free(acc);free(pack.q);free(pack.sw);free(pack.bias);free_acts(a);free(image->q);free(image);return ED_ERR_MEMORY;}
            if(dw)ed_hw_depthwise_conv2d(p0->q,p0->h,p0->w,p0->c,pack.q,pack.bias,k,n->stride,n->padding,acc);
            else ed_hw_conv2d(p0->q,p0->h,p0->w,p0->c,pack.q,pack.bias,oc,k,n->stride,n->padding,acc);
            {size_t N=elems(h,w,c),zi;uint32_t o=dw?p0->c:oc;float *tmp=(float*)malloc(N*sizeof(float)),mx=0.0f;
                if(!tmp){free(acc);free(pack.q);free(pack.sw);free(pack.bias);free_acts(a);free(image->q);free(image);return ED_ERR_MEMORY;}
                for(zi=0;zi<N;++zi){
                    float sw=pack.sw[zi%o],r=(float)acc[zi]*p0->scale*sw,ar=r<0?-r:r;
                    tmp[zi]=r;if(ar>mx)mx=ar;
                }
                a[i].scale=mx>0.0f?mx/(float)ED_HW_Q_MAX:1.0f;
                for(zi=0;zi<N;++zi)a[i].q[zi]=ed_hw_sat16((ed_hw_acc)lrintf(tmp[zi]/a[i].scale));
                free(tmp);}
            free(acc);free(pack.q);free(pack.sw);free(pack.bias);
        }else if(n->op==ED_OP_RELU){unary_from_float_fn(p0,&a[i],0);}
        else if(n->op==ED_OP_HARD_SIGMOID){unary_from_float_fn(p0,&a[i],1);}
        else if(n->op==ED_OP_SIGMOID){unary_from_float_fn(p0,&a[i],2);}
        else if(n->op==ED_OP_SQRT){unary_from_float_fn(p0,&a[i],3);}
        else if(n->op==ED_OP_GLOBAL_AVG){
            uint32_t ch,y,x,hw=p0->h*p0->w;
            for(ch=0;ch<p0->c;++ch){
                ed_hw_acc s=0;for(y=0;y<p0->h;++y)for(x=0;x<p0->w;++x)s+=p0->q[((size_t)y*p0->w+x)*p0->c+ch];
                a[i].q[ch]=ed_hw_sat16(hw?s/(ed_hw_acc)hw:0);
            }
            a[i].scale=p0->scale;
        }else if(n->op==ED_OP_RESIZE_NEAREST){
            uint32_t y,x,ch;
            for(y=0;y<h;++y)for(x=0;x<w;++x)for(ch=0;ch<c;++ch)
                a[i].q[((size_t)y*w+x)*c+ch]=p0->q[((size_t)(y/2u)*p0->w+x/2u)*c+ch];
            a[i].scale=p0->scale;
        }else if(n->op==ED_OP_CONCAT_CHANNEL){
            hw_act *p1=ref_act(image,a,n->input[1]);uint32_t y,x,ch;
            float s0=p0->scale,s1=p1->scale,so=s0>s1?s0:s1;if(so<=0)so=1.0f;
            for(y=0;y<h;++y)for(x=0;x<w;++x){
                ed_hw_q *d=a[i].q+((size_t)y*w+x)*c;
                for(ch=0;ch<p0->c;++ch){
                    float v=(float)p0->q[((size_t)y*w+x)*p0->c+ch]*s0;d[ch]=ed_hw_sat16((ed_hw_acc)lrintf(v/so));
                }
                for(ch=0;ch<p1->c;++ch){
                    float v=(float)p1->q[((size_t)y*w+x)*p1->c+ch]*s1;d[p0->c+ch]=ed_hw_sat16((ed_hw_acc)lrintf(v/so));
                }
            }
            a[i].scale=so;
        }else if(n->op==ED_OP_ADD||n->op==ED_OP_MUL||n->op==ED_OP_DIV){
            hw_act *p1=ref_act(image,a,n->input[1]);const float *c1=NULL;
            uint32_t h1=1,w1=1,c1n=1,y,x,ch;size_t N=elems(h,w,c),zi=0;float *tmp,mx=0.0f;
            if(p1){h1=p1->h;w1=p1->w;c1n=p1->c;}
            else{const ed_tensor *t=tensor_ref(m,n->input[1]);c1=tensor_f32(m,n->input[1]);
                if(t){h1=t->rank>2?t->dims[t->rank-3]:1;w1=t->rank>1?t->dims[t->rank-2]:1;c1n=t->dims[t->rank-1];}}
            tmp=(float*)malloc(N*sizeof(float));if(!tmp){free_acts(a);free(image->q);free(image);return ED_ERR_MEMORY;}
            for(y=0;y<h;++y)for(x=0;x<w;++x)for(ch=0;ch<c;++ch){
                uint32_t iy0=p0->h==1?0:y,ix0=p0->w==1?0:x,ic0=p0->c==1?0:ch;
                float u=(float)p0->q[((size_t)iy0*p0->w+ix0)*p0->c+ic0]*p0->scale,v,r,ar;
                if(p1){uint32_t iy=h1==1?0:y,ix=w1==1?0:x,ic=c1n==1?0:ch;v=(float)p1->q[((size_t)iy*w1+ix)*c1n+ic]*p1->scale;}
                else{uint32_t iy=h1==1?0:y,ix=w1==1?0:x,ic=c1n==1?0:ch;v=c1[((size_t)iy*w1+ix)*c1n+ic];}
                r=n->op==ED_OP_ADD?u+v:(n->op==ED_OP_MUL?u*v:(v!=0.0f?u/v:0.0f));
                tmp[zi++]=r;ar=r<0?-r:r;if(ar>mx)mx=ar;
            }
            a[i].scale=mx>0.0f?mx/(float)ED_HW_Q_MAX:1.0f;
            for(zi=0;zi<N;++zi)a[i].q[zi]=ed_hw_sat16((ed_hw_acc)lrintf(tmp[zi]/a[i].scale));
            free(tmp);
        }else if(n->op==ED_OP_CLIP){
            const float *lo=tensor_f32(m,n->input[1]),*hi=tensor_f32(m,n->input[2]);
            size_t z,N=elems(h,w,c);float l=lo?*lo:0.0f,hh=hi?*hi:6.0f;
            ed_hw_q lq=ed_hw_sat16((ed_hw_acc)lrintf(l/(p0->scale>0?p0->scale:1.0f)));
            ed_hw_q hq=ed_hw_sat16((ed_hw_acc)lrintf(hh/(p0->scale>0?p0->scale:1.0f)));
            for(z=0;z<N;++z)a[i].q[z]=ed_hw_clip(p0->q[z],lq,hq);
            a[i].scale=p0->scale;
        }else{free_acts(a);free(image->q);free(image);return ED_ERR_UNSUPPORTED;}
    }
    free(image->q);free(image);*out_acts=a;return ED_OK;
}

static void dfl_int(const hw_act *reg,float *dist){
    uint32_t loc,side,b,n=reg->h*reg->w;
    for(loc=0;loc<n;++loc)for(side=0;side<4u;++side){
        const ed_hw_q *q=reg->q+((size_t)loc*4u+side)*8u;
        ed_hw_q mx=q[0];int32_t sum=0,wsum=0;
        for(b=1;b<8u;++b)if(q[b]>mx)mx=q[b];
        for(b=0;b<8u;++b){
            ed_hw_q d=ed_hw_sat16((ed_hw_acc)q[b]-(ed_hw_acc)mx);
            /* Map difference through current scale into Q8 for the exp ROM. */
            ed_hw_q q8=ed_hw_sat16((ed_hw_acc)lrintf((float)d*reg->scale*256.0f));
            int32_t e=ed_hw_exp_lut(q8);sum+=e;wsum+=e*(int32_t)b;
        }
        dist[(size_t)loc*4u+side]=sum>0?(float)wsum/(float)sum:0.0f;
    }
}

ed_status ed_hw_predict(const ed_model *m,const ed_image *im,float st,float nt,ed_detection_list *out){
    hw_act *a=NULL;ed_detection *cand=NULL;size_t cn=0,cap=4096;uint32_t l,y,x,c;ed_status s;
    if(!m||!im||!out||!out->items||!im->rgb)return ED_ERR_ARGUMENT;
    cand=(ed_detection*)malloc(cap*sizeof(*cand));if(!cand)return ED_ERR_MEMORY;
    s=graph_hw(m,im,&a);if(s!=ED_OK){free(cand);return s;}
    {uint32_t dbg,ln;ed_hw_q amx=0;float smx=0.0f;
     for(dbg=0;dbg<elems(a[0].h,a[0].w,a[0].c);++dbg){ed_hw_q v=a[0].q[dbg];if(v<0)v=(ed_hw_q)-v;if(v>amx)amx=v;}
     fprintf(stderr,"fpga-model node0 max|q|=%d scale=%.6g %ux%ux%u\n",(int)amx,a[0].scale,a[0].h,a[0].w,a[0].c);
     for(ln=0;ln<ED_PICODET_LEVELS;++ln){
         hw_act *sc=&a[ed_picodet_cls_nodes[ln]],*rw=&a[ed_picodet_raw_cls_nodes[ln]],*ft=&a[ed_picodet_feature_nodes[ln]];
         size_t z,N;float local=0,rmax=0,fmax=0;
         N=elems(sc->h,sc->w,sc->c);for(z=0;z<N;++z){float sv=(float)sc->q[z]*sc->scale;if(sv<0)sv=-sv;if(sv>local)local=sv;}
         N=elems(rw->h,rw->w,rw->c);{float rmin=1e9f,rhi=-1e9f;for(z=0;z<N;++z){float sv=(float)rw->q[z]*rw->scale;if(sv<rmin)rmin=sv;if(sv>rhi)rhi=sv;if(sv<0)sv=-sv;if(sv>rmax)rmax=sv;}
         fprintf(stderr,"fpga-model L%u raw_range=[%.3g,%.3g]\n",ln,rmin,rhi);}
         N=elems(ft->h,ft->w,ft->c);for(z=0;z<N;++z){float sv=(float)ft->q[z]*ft->scale;if(sv<0)sv=-sv;if(sv>fmax)fmax=sv;}
         if(local>smx)smx=local;
         fprintf(stderr,"fpga-model L%u feat=%.4g raw=%.4g score=%.4g sc_scale=%.4g q0=%d %ux%ux%u\n",
                 ln,fmax,rmax,local,sc->scale,(int)(sc->q?sc->q[0]:0),sc->h,sc->w,sc->c);
     }
     fprintf(stderr,"fpga-model max_abs_score=%.6g\n",smx);}
    for(l=0;l<ED_PICODET_LEVELS;++l){
        hw_act *sc=&a[ed_picodet_cls_nodes[l]],*rg=&a[ed_picodet_reg_nodes[l]];
        float *dist=(float*)malloc((size_t)rg->h*rg->w*4u*sizeof(float));
        if(!dist){s=ED_ERR_MEMORY;break;}
        dfl_int(rg,dist);
        for(y=0;y<rg->h;++y)for(x=0;x<rg->w;++x)for(c=0;c<m->class_count&&c<sc->c;++c){
            float score=(float)sc->q[((size_t)y*rg->w+x)*sc->c+c]*sc->scale;
            if(score<0.0f)score=0.0f;if(score>1.0f)score=1.0f;
            if(score>=st&&cn<cap){
                const float *d=dist+((size_t)y*rg->w+x)*4u;float stride=(float)ed_picodet_strides[l];
                float cx=((float)x+0.5f)*stride,cy=((float)y+0.5f)*stride;ed_detection *z=&cand[cn++];
                z->x1=(cx-d[0]*stride)*(float)im->width/320.0f;
                z->y1=(cy-d[1]*stride)*(float)im->height/320.0f;
                z->x2=(cx+d[2]*stride)*(float)im->width/320.0f;
                z->y2=(cy+d[3]*stride)*(float)im->height/320.0f;
                z->score=score;z->class_id=c;
            }
        }
        free(dist);
    }
    if(s==ED_OK){
        size_t i;cn=ed_nms(cand,cn,nt);
        for(i=0;i<cn;++i){
            if(cand[i].x1<0)cand[i].x1=0;if(cand[i].y1<0)cand[i].y1=0;
            if(cand[i].x2>(float)im->width)cand[i].x2=(float)im->width;
            if(cand[i].y2>(float)im->height)cand[i].y2=(float)im->height;
        }
        out->count=cn<out->capacity?cn:out->capacity;memcpy(out->items,cand,out->count*sizeof(*out->items));
    }
    free_acts(a);free(cand);return s;
}
