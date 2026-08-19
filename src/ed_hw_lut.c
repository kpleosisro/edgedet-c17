#include "ed_hw.h"
#include <math.h>

static int16_t g_sig[ED_HW_LUT];
static int32_t g_exp[ED_HW_LUT];
static int16_t g_sqrt[ED_HW_LUT];
static int g_ready;

void ed_hw_lut_init(void){
    int i;if(g_ready)return;
    /* Domain x in [-8, 8], Q8 index. Output sigmoid Q15, exp Q10. */
    for(i=0;i<ED_HW_LUT;++i){
        float x=-8.0f+16.0f*(float)i/(float)(ED_HW_LUT-1);
        float s=1.0f/(1.0f+expf(-x));
        float e=expf(x>0.0f?0.0f:x);
        g_sig[i]=(int16_t)(s*32767.0f+0.5f);
        g_exp[i]=(int32_t)(e*1024.0f+0.5f);
        {float u=(float)i/(float)(ED_HW_LUT-1);g_sqrt[i]=(int16_t)(sqrtf(u)*32767.0f+0.5f);}
    }
    g_ready=1;
}
const int16_t *ed_hw_sigmoid_rom(void){ed_hw_lut_init();return g_sig;}
const int32_t *ed_hw_exp_rom(void){ed_hw_lut_init();return g_exp;}
const int16_t *ed_hw_sqrt_rom(void){ed_hw_lut_init();return g_sqrt;}

static int lut_index_q8(ed_hw_q x_q8){
    /* Map Q8 real=x/256 in [-8,8] onto 0..255. */
    int32_t t=((int32_t)x_q8+8*256)*(ED_HW_LUT-1)/(16*256);
    if(t<0)t=0;if(t>ED_HW_LUT-1)t=ED_HW_LUT-1;return (int)t;
}
ed_hw_q ed_hw_sigmoid_lut(ed_hw_q x_q8){
    ed_hw_lut_init();return g_sig[lut_index_q8(x_q8)];
}
int32_t ed_hw_exp_lut(ed_hw_q x_q8){
    ed_hw_lut_init();return g_exp[lut_index_q8(x_q8)];
}
ed_hw_q ed_hw_sqrt_lut(ed_hw_q x_q15){
    int32_t t;ed_hw_lut_init();
    if(x_q15<0)x_q15=0;
    t=((int32_t)x_q15*(ED_HW_LUT-1))/32767;
    if(t<0)t=0;if(t>ED_HW_LUT-1)t=ED_HW_LUT-1;
    return g_sqrt[t];
}
