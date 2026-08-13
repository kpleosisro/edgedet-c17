#include "ed_internal.h"
#include <arm_neon.h>

static uint32_t out_dim(uint32_t n,uint32_t k,uint32_t s,uint32_t p){return(n+2u*p-k)/s+1u;}
static float horizontal_sum(float32x4_t v){float32x2_t x=vadd_f32(vget_low_f32(v),vget_high_f32(v));x=vpadd_f32(x,x);return vget_lane_f32(x,0);}

void ed_conv2d_f32_neon_range(const float*in,uint32_t ih,uint32_t iw,uint32_t ic,const float*w,const float*b,uint32_t oc,uint32_t k,uint32_t s,uint32_t p,float*out,size_t begin,size_t end){
    uint32_t ow=out_dim(iw,k,s,p);size_t site;
    for(site=begin;site<end;++site){uint32_t y=(uint32_t)(site/ow),x=(uint32_t)(site%ow),o,ky,kx;
        for(o=0;o<oc;++o){float32x4_t sum=vdupq_n_f32(0.0f);float tail=b?b[o]:0.0f;
            for(ky=0;ky<k;++ky){int iy=(int)(y*s+ky)-(int)p;if(iy<0||iy>=(int)ih)continue;
                for(kx=0;kx<k;++kx){int ix=(int)(x*s+kx)-(int)p;uint32_t c=0;if(ix<0||ix>=(int)iw)continue;
                    {const float*q=in+((size_t)(uint32_t)iy*iw+(uint32_t)ix)*ic,*z=w+(((size_t)o*k+ky)*k+kx)*ic;
                        for(;c+4u<=ic;c+=4u)sum=vmlaq_f32(sum,vld1q_f32(q+c),vld1q_f32(z+c));for(;c<ic;++c)tail+=q[c]*z[c];}
                }
            }
            out[site*oc+o]=tail+horizontal_sum(sum);
        }
    }
}

void ed_depthwise_conv2d_f32_neon_range(const float*in,uint32_t ih,uint32_t iw,uint32_t c,const float*w,const float*b,uint32_t k,uint32_t s,uint32_t p,float*out,size_t begin,size_t end){
    uint32_t ow=out_dim(iw,k,s,p);size_t site;
    for(site=begin;site<end;++site){uint32_t y=(uint32_t)(site/ow),x=(uint32_t)(site%ow),ch,ky,kx;float*q=out+site*c;
        for(ch=0;ch+4u<=c;ch+=4u){float32x4_t sum=b?vld1q_f32(b+ch):vdupq_n_f32(0.0f);
            for(ky=0;ky<k;++ky){int iy=(int)(y*s+ky)-(int)p;if(iy<0||iy>=(int)ih)continue;
                for(kx=0;kx<k;++kx){int ix=(int)(x*s+kx)-(int)p;if(ix<0||ix>=(int)iw)continue;
                    sum=vmlaq_f32(sum,vld1q_f32(in+((size_t)(uint32_t)iy*iw+(uint32_t)ix)*c+ch),vld1q_f32(w+((size_t)ky*k+kx)*c+ch));
                }
            }
            vst1q_f32(q+ch,sum);
        }
        for(;ch<c;++ch){float sum=b?b[ch]:0.0f;
            for(ky=0;ky<k;++ky){int iy=(int)(y*s+ky)-(int)p;if(iy<0||iy>=(int)ih)continue;
                for(kx=0;kx<k;++kx){int ix=(int)(x*s+kx)-(int)p;if(ix>=0&&ix<(int)iw)sum+=in[((size_t)(uint32_t)iy*iw+(uint32_t)ix)*c+ch]*w[((size_t)ky*k+kx)*c+ch];}
            }
            q[ch]=sum;
        }
    }
}
