#include "ed_hw.h"
#include "ed_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ED_TEST_MODEL_PATH
#define ED_TEST_MODEL_PATH "../models/picodet_s_coco80_fp16.edm"
#endif

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);return 1;}}while(0)

int main(void){
    ed_hw_acc acc=0;int i;
    int16_t in[9]={1,2,3,4,5,6,7,8,9};int8_t w[9]={1,1,1,1,1,1,1,1,1};
    ed_hw_q relu,clip;
    CHECK(ed_runtime_compute()==ED_COMPUTE_HOST);
    CHECK(ed_runtime_set_compute((ed_compute)9)==ED_ERR_ARGUMENT);
    CHECK(ed_runtime_set_compute(ED_COMPUTE_FPGA_MODEL)==ED_OK);
    CHECK(ed_runtime_compute()==ED_COMPUTE_FPGA_MODEL);
    CHECK(ed_runtime_set_compute(ED_COMPUTE_HOST)==ED_OK);

    for(i=0;i<9;++i)acc=ed_hw_mac(acc,in[i],w[i]);
    CHECK(acc==45);
    CHECK(ed_hw_add32(100,-40)==60);
    CHECK(ed_hw_sub32(10,3)==7);
    CHECK(ed_hw_sat16(40000)==32767);
    CHECK(ed_hw_sat16(-40000)==-32768);
    relu=ed_hw_relu(-3);CHECK(relu==0);
    clip=ed_hw_clip(50,-10,10);CHECK(clip==10);

    ed_hw_lut_init();
    {ed_hw_q s0=ed_hw_sigmoid_lut((ed_hw_q)(-8*256));
     ed_hw_q s1=ed_hw_sigmoid_lut(0);
     ed_hw_q s2=ed_hw_sigmoid_lut((ed_hw_q)(8*256));
     CHECK(s0<s1&&s1<s2);CHECK(s1>14000&&s1<19000);}
    {ed_hw_q hs=ed_hw_hard_sigmoid_q8(0);CHECK(hs>15000&&hs<18000);}
    {ed_hw_q sq=ed_hw_sqrt_lut(32767);CHECK(sq>32000);}

    {
        ed_hw_q xin[4]={256,256,256,256}; /* 1.0 in Q8 */
        ed_hw_w ww[1]={64}; /* ~0.5 if scale 1/127 later */
        int32_t b[1]={0};ed_hw_acc out[4];
        ed_hw_conv2d(xin,2,2,1,ww,b,1,1,1,0,out);
        CHECK(out[0]==256*64);
    }

    {
        ed_model *m=NULL;uint8_t pix[32*32*3];ed_image im={pix,32,32,32*3u};
        ed_detection det[ED_MAX_DETECTIONS];ed_detection_list list={det,ED_MAX_DETECTIONS,0};
        memset(pix,128,sizeof(pix));
        CHECK(ed_model_load(ED_TEST_MODEL_PATH,&m)==ED_OK);
        CHECK(ed_runtime_set_compute(ED_COMPUTE_FPGA_MODEL)==ED_OK);
        CHECK(ed_predict(m,&im,0.99f,0.4f,&list)==ED_OK);
        CHECK(ed_runtime_set_compute(ED_COMPUTE_HOST)==ED_OK);
        CHECK(ed_predict(m,&im,0.99f,0.4f,&list)==ED_OK);
        ed_model_free(m);
    }
    return 0;
}
