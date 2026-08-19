#include "ed_hw.h"

static ed_compute g_compute = ED_COMPUTE_HOST;

ed_status ed_runtime_set_compute(ed_compute compute){
    if(compute!=ED_COMPUTE_HOST&&compute!=ED_COMPUTE_FPGA_MODEL)return ED_ERR_ARGUMENT;
    g_compute=compute;return ED_OK;
}
ed_compute ed_runtime_compute(void){return g_compute;}

/* acc := acc + a * b  (signed 16 x signed 8 -> 64-bit lane) */
ed_hw_acc ed_hw_mac(ed_hw_acc acc,ed_hw_q a,ed_hw_w b){
    return acc+(ed_hw_acc)a*(ed_hw_acc)b;
}
int32_t ed_hw_add32(int32_t a,int32_t b){
    int64_t s=(int64_t)a+(int64_t)b;
    if(s>2147483647ll)return 2147483647;if(s<-2147483647ll-1)return (int32_t)(-2147483647ll-1);
    return (int32_t)s;
}
int32_t ed_hw_sub32(int32_t a,int32_t b){
    int64_t s=(int64_t)a-(int64_t)b;
    if(s>2147483647ll)return 2147483647;if(s<-2147483647ll-1)return (int32_t)(-2147483647ll-1);
    return (int32_t)s;
}
ed_hw_q ed_hw_sat16(ed_hw_acc value){
    if(value>ED_HW_Q_MAX)return (ed_hw_q)ED_HW_Q_MAX;
    if(value<ED_HW_Q_MIN)return (ed_hw_q)ED_HW_Q_MIN;
    return (ed_hw_q)value;
}
ed_hw_q ed_hw_requant(ed_hw_acc acc,ed_hw_scale scale){
    ed_hw_acc y;
    if(scale.shift<0){
        if(scale.shift<-30)return ed_hw_sat16(acc>=0?ED_HW_Q_MAX:ED_HW_Q_MIN);
        y=acc*(ed_hw_acc)scale.mult<<(-scale.shift);
    }else{
        y=acc*(ed_hw_acc)scale.mult;
        if(scale.shift>0){
            ed_hw_acc rnd=(ed_hw_acc)1<<(scale.shift-1);
            if(y>=0)y=(y+rnd)>>scale.shift;else y=-((-y+rnd)>>scale.shift);
        }
    }
    return ed_hw_sat16(y);
}
ed_hw_q ed_hw_relu(ed_hw_q x){return x<0?(ed_hw_q)0:x;}
ed_hw_q ed_hw_clip(ed_hw_q x,ed_hw_q lo,ed_hw_q hi){
    if(x<lo)return lo;if(x>hi)return hi;return x;
}
/* x_q8 / 6 + 0.5, clip to [0,1], as Q15.  1/6 ≈ 43/256. */
ed_hw_q ed_hw_hard_sigmoid_q8(ed_hw_q x_q8){
    int32_t t=((int32_t)x_q8*43>>8)+128; /* Q8, 0.5 = 128 */
    if(t<0)t=0;if(t>256)t=256;
    return (ed_hw_q)((t*32767)>>8);
}
ed_hw_scale ed_hw_scale_from_max(ed_hw_acc max_abs,int32_t target){
    ed_hw_scale s;int32_t shift=0;ed_hw_acc m=max_abs<0?-max_abs:max_abs;
    if(target<=0)target=ED_HW_Q_MAX;
    if(m==0){s.mult=1;s.shift=0;return s;}
    while(m>target&&shift<60){m>>=1;++shift;}
    s.mult=1;s.shift=shift;return s;
}
void ed_hw_requant_buffer(const ed_hw_acc *acc,size_t n,ed_hw_scale scale,ed_hw_q *out){
    size_t i;for(i=0;i<n;++i)out[i]=ed_hw_requant(acc[i],scale);
}
