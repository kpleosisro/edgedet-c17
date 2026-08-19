#include "ed_hw.h"
#include <stdint.h>

static uint32_t out_dim(uint32_t n,uint32_t k,uint32_t s,uint32_t p){
    return(n+2u*p-k)/s+1u;
}

void ed_hw_conv2d(const ed_hw_q *in,uint32_t ih,uint32_t iw,uint32_t ic,
                  const ed_hw_w *w,const int32_t *b,uint32_t oc,
                  uint32_t k,uint32_t s,uint32_t p,ed_hw_acc *out){
    uint32_t oh=out_dim(ih,k,s,p),ow=out_dim(iw,k,s,p),y,x,o,ky,kx,c;
    for(y=0;y<oh;++y)for(x=0;x<ow;++x)for(o=0;o<oc;++o){
        ed_hw_acc acc=b?b[o]:0;
        for(ky=0;ky<k;++ky){
            int iy=(int)(y*s+ky)-(int)p;if(iy<0||iy>=(int)ih)continue;
            for(kx=0;kx<k;++kx){
                int ix=(int)(x*s+kx)-(int)p;if(ix<0||ix>=(int)iw)continue;
                const ed_hw_q *ip=in+((size_t)(uint32_t)iy*iw+(uint32_t)ix)*ic;
                const ed_hw_w *wp=w+(((size_t)o*k+ky)*k+kx)*ic;
                for(c=0;c<ic;++c)acc=ed_hw_mac(acc,ip[c],wp[c]);
            }
        }
        out[((size_t)y*ow+x)*oc+o]=acc;
    }
}

void ed_hw_depthwise_conv2d(const ed_hw_q *in,uint32_t ih,uint32_t iw,uint32_t ch,
                            const ed_hw_w *w,const int32_t *b,uint32_t k,uint32_t s,
                            uint32_t p,ed_hw_acc *out){
    uint32_t oh=out_dim(ih,k,s,p),ow=out_dim(iw,k,s,p),y,x,c,ky,kx;
    for(y=0;y<oh;++y)for(x=0;x<ow;++x)for(c=0;c<ch;++c){
        ed_hw_acc acc=b?b[c]:0;
        for(ky=0;ky<k;++ky){
            int iy=(int)(y*s+ky)-(int)p;if(iy<0||iy>=(int)ih)continue;
            for(kx=0;kx<k;++kx){
                int ix=(int)(x*s+kx)-(int)p;if(ix<0||ix>=(int)iw)continue;
                acc=ed_hw_mac(acc,in[((size_t)(uint32_t)iy*iw+(uint32_t)ix)*ch+c],
                              w[((size_t)ky*k+kx)*ch+c]);
            }
        }
        out[((size_t)y*ow+x)*ch+c]=acc;
    }
}
