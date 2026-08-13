#include "ed_internal.h"
#include <math.h>
#include <stdlib.h>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif
typedef struct{const float*in,*w,*b;float*out;uint32_t ih,iw,ic,oc,k,s,p;int depthwise,use_avx,use_neon;}conv_job;
static uint32_t out_dim(uint32_t n,uint32_t k,uint32_t s,uint32_t p){return(n+2u*p-k)/s+1u;}
int ed_cpu_has_avx2_fma(void){
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
int r[4];unsigned __int64 x;__cpuidex(r,1,0);if(!(r[2]&(1<<27))||!(r[2]&(1<<28))||!(r[2]&(1<<12)))return 0;x=_xgetbv(0);if((x&6)!=6)return 0;__cpuidex(r,7,0);return(r[1]&(1<<5))!=0;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
unsigned a,b,c,d;uint32_t lo,hi;if(!__get_cpuid(1,&a,&b,&c,&d)||!(c&bit_OSXSAVE)||!(c&bit_AVX)||!(c&bit_FMA))return 0;__asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));if((lo&6u)!=6u)return 0;return __get_cpuid_count(7,0,&a,&b,&c,&d)&&(b&bit_AVX2);
#else
return 0;
#endif
}
int ed_cpu_has_neon(void){
#if defined(ED_HAVE_NEON_IMPL) && (defined(__aarch64__) || defined(_M_ARM64))
return 1;
#elif defined(ED_HAVE_NEON_IMPL) && defined(__ARM_NEON)
return 1;
#else
return 0;
#endif
}
static void conv_range(void*context,size_t begin,size_t end){conv_job*j=(conv_job*)context;
#if defined(ED_HAVE_AVX2_IMPL)
if(j->use_avx){if(j->depthwise)ed_depthwise_conv2d_f32_avx2_range(j->in,j->ih,j->iw,j->ic,j->w,j->b,j->k,j->s,j->p,j->out,begin,end);else ed_conv2d_f32_avx2_range(j->in,j->ih,j->iw,j->ic,j->w,j->b,j->oc,j->k,j->s,j->p,j->out,begin,end);return;}
#endif
#if defined(ED_HAVE_NEON_IMPL)
if(j->use_neon){if(j->depthwise)ed_depthwise_conv2d_f32_neon_range(j->in,j->ih,j->iw,j->ic,j->w,j->b,j->k,j->s,j->p,j->out,begin,end);else ed_conv2d_f32_neon_range(j->in,j->ih,j->iw,j->ic,j->w,j->b,j->oc,j->k,j->s,j->p,j->out,begin,end);return;}
#endif
{uint32_t ow=out_dim(j->iw,j->k,j->s,j->p);size_t site;for(site=begin;site<end;++site){uint32_t y=(uint32_t)(site/ow),x=(uint32_t)(site%ow),o,ky,kx;if(j->depthwise){for(o=0;o<j->ic;++o){float sum=j->b?j->b[o]:0.0f;for(ky=0;ky<j->k;++ky){int iy=(int)(y*j->s+ky)-(int)j->p;if(iy<0||iy>=(int)j->ih)continue;for(kx=0;kx<j->k;++kx){int ix=(int)(x*j->s+kx)-(int)j->p;if(ix>=0&&ix<(int)j->iw)sum+=j->in[((size_t)(uint32_t)iy*j->iw+(uint32_t)ix)*j->ic+o]*j->w[((size_t)ky*j->k+kx)*j->ic+o];}}j->out[site*j->ic+o]=sum;}}else for(o=0;o<j->oc;++o){float sum=j->b?j->b[o]:0.0f;for(ky=0;ky<j->k;++ky){int iy=(int)(y*j->s+ky)-(int)j->p;if(iy<0||iy>=(int)j->ih)continue;for(kx=0;kx<j->k;++kx){int ix=(int)(x*j->s+kx)-(int)j->p;uint32_t c;if(ix<0||ix>=(int)j->iw)continue;for(c=0;c<j->ic;++c)sum+=j->in[((size_t)(uint32_t)iy*j->iw+(uint32_t)ix)*j->ic+c]*j->w[(((size_t)o*j->k+ky)*j->k+kx)*j->ic+c];}}j->out[site*j->oc+o]=sum;}}}}
void ed_conv2d_f32(const float*in,uint32_t ih,uint32_t iw,uint32_t ic,const float*w,const float*b,uint32_t oc,uint32_t k,uint32_t s,uint32_t p,float*out){size_t sites=(size_t)out_dim(ih,k,s,p)*out_dim(iw,k,s,p),work=sites*oc*k*k*ic;conv_job j={in,w,b,out,ih,iw,ic,oc,k,s,p,0,ed_cpu_has_avx2_fma(),ed_cpu_has_neon()};if(ed_runtime_threads()==1u||work<200000u)conv_range(&j,0,sites);else(void)ed_parallel_for(sites,conv_range,&j);}
void ed_depthwise_conv2d_f32(const float*in,uint32_t ih,uint32_t iw,uint32_t c,const float*w,const float*b,uint32_t k,uint32_t s,uint32_t p,float*out){size_t sites=(size_t)out_dim(ih,k,s,p)*out_dim(iw,k,s,p),work=sites*c*k*k;conv_job j={in,w,b,out,ih,iw,c,c,k,s,p,1,ed_cpu_has_avx2_fma(),ed_cpu_has_neon()};if(ed_runtime_threads()==1u||work<200000u)conv_range(&j,0,sites);else(void)ed_parallel_for(sites,conv_range,&j);}
void ed_hardswish_f32(float*v,size_t n){size_t i;for(i=0;i<n;++i){float t=v[i]+3.0f;if(t<0)t=0;if(t>6)t=6;v[i]=v[i]*t*(1.0f/6.0f);}}
void ed_dfl_decode_f32(const float*l,uint32_t n,uint32_t rm,float*d){uint32_t p,s,b,bins=rm+1u;for(p=0;p<n;++p)for(s=0;s<4u;++s){const float*q=l+((size_t)p*4u+s)*bins;float m=q[0],sum=0,weighted=0;for(b=1;b<bins;++b)if(q[b]>m)m=q[b];for(b=0;b<bins;++b){float e=expf(q[b]-m);sum+=e;weighted+=e*b;}d[(size_t)p*4u+s]=weighted/sum;}}
static float iou(const ed_detection*a,const ed_detection*b){float x1=a->x1>b->x1?a->x1:b->x1,y1=a->y1>b->y1?a->y1:b->y1,x2=a->x2<b->x2?a->x2:b->x2,y2=a->y2<b->y2?a->y2:b->y2,w=x2-x1,h=y2-y1,inter,aa,bb,den;if(w<=0||h<=0)return 0;inter=w*h;aa=(a->x2-a->x1)*(a->y2-a->y1);bb=(b->x2-b->x1)*(b->y2-b->y1);den=aa+bb-inter;return den>0?inter/den:0;}
static int desc(const void*x,const void*y){const ed_detection*a=x,*b=y;return a->score<b->score?1:(a->score>b->score?-1:0);}
size_t ed_nms(ed_detection*a,size_t n,float t){size_t i,j,o=0;uint8_t*dead;if(!a||!n)return 0;qsort(a,n,sizeof(*a),desc);dead=(uint8_t*)calloc(n,1);if(!dead)return 0;for(i=0;i<n;++i)if(!dead[i]){a[o++]=a[i];for(j=i+1;j<n;++j)if(!dead[j]&&a[i].class_id==a[j].class_id&&iou(&a[i],&a[j])>t)dead[j]=1;}free(dead);return o;}
uint64_t ed_picodet_forward_flops(void){
    /* PP-PicoDet-S 320 official FLOP count (multiply-adds counted as 2). */
    return 730000000ull;
}
float ed_estimate_energy_mj(uint32_t forwards,float watts,float latency_ms){
    if(forwards==0u||watts<=0.0f||latency_ms<=0.0f)return 0.0f;
    return watts*(latency_ms/1000.0f);
}

size_t ed_nms_soft(ed_detection*a,size_t n,float sigma,float score_min){
    size_t i,j,o=0;if(!a||!n)return 0;if(!(sigma>0.0f))return ed_nms(a,n,0.5f);
    if(score_min<0.0f)score_min=0.0f;
    qsort(a,n,sizeof(*a),desc);
    for(i=0;i<n;++i){
        if(a[i].score<score_min)continue;
        for(j=i+1;j<n;++j){
            float u;if(a[j].class_id!=a[i].class_id||a[j].score<score_min)continue;
            u=iou(&a[i],&a[j]);
            if(u>0.0f){
                if(sigma>=9.0f){if(u>0.30f)a[j].score*=(1.0f-u);}
                else a[j].score*=expf(-u*u/sigma);
            }
        }
    }
    for(i=0;i<n;++i)if(a[i].score>=score_min)a[o++]=a[i];
    if(o>1u)qsort(a,o,sizeof(*a),desc);
    return o;
}
