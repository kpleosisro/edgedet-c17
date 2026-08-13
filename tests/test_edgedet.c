#include "edgedet.h"
#include "ed_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ED_TEST_MODEL_PATH
#define ED_TEST_MODEL_PATH "../models/picodet_s_coco80_fp16.edm"
#endif

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);return 1;}}while(0)
static int near(float a,float b){return fabsf(a-b)<1e-5f;}
static int has_disk_dtype(const char *path,uint32_t dtype){FILE*f=fopen(path,"rb");ed_edm_header h;uint32_t i;if(!f)return 0;if(fread(&h,1,sizeof(h),f)!=sizeof(h)||fseek(f,(long)h.tensor_table_offset,SEEK_SET)!=0){fclose(f);return 0;}for(i=0;i<h.tensor_count;++i){ed_edm_tensor_disk d;if(fread(&d,1,sizeof(d),f)!=sizeof(d))break;if(d.dtype==dtype){fclose(f);return 1;}}fclose(f);return 0;}
int main(void){
    static const uint8_t abc[]={'a','b','c'};static const uint8_t sha[32]={0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};uint8_t got[32];
    float in[4]={1,2,3,4},w[1]={2},b[1]={1},out[4],dlog[8]={0,0,0,0,0,0,0,0},dist[4];
    ed_detection ds[3]={{0,0,10,10,0.9f,0},{1,1,9,9,0.8f,0},{20,20,30,30,0.7f,0}};
    ed_model m={0},*loaded=NULL;ed_dataset *bad_dataset=NULL;ed_status s;char path[]="test_roundtrip.edm",half_path[]="test_roundtrip_fp16.edm",int8_path[]="test_roundtrip_int8.edm",int4_path[]="test_roundtrip_int4.edm",bad_path[]="test_corrupt.edb";
    ed_sha256(abc,3,got);CHECK(ed_digest_equal(got,sha));CHECK(ed_crc32("123456789",9)==0xcbf43926u);
    ed_conv2d_f32(in,2,2,1,w,b,1,1,1,0,out);CHECK(near(out[0],3)&&near(out[3],9));
    ed_dfl_decode_f32(dlog,1,1,dist);CHECK(near(dist[0],0.5f)&&near(dist[3],0.5f));CHECK(ed_nms(ds,3,0.5f)==2);
    {
        ed_detection soft[3]={{0,0,10,10,0.9f,0},{1,1,9,9,0.8f,0},{20,20,30,30,0.7f,0}};
        size_t kept=ed_nms_soft(soft,3,0.5f,0.05f);
        CHECK(kept==3);CHECK(soft[0].score>0.89f);CHECK(soft[1].score>0.69f);CHECK(soft[2].score<0.8f);
    }
    m.architecture=ED_ARCH_PICODET_S_320;m.precision=ED_PRECISION_FP32;m.flags=ED_MODEL_FLAG_FROZEN_BACKBONE;m.class_count=3;m.class_names=calloc(3,ED_CLASS_NAME_BYTES);m.metadata=malloc(3);memcpy(m.metadata,"{}",3);m.metadata_bytes=2;strcpy(m.class_names[0],"car");strcpy(m.class_names[1],"cat");strcpy(m.class_names[2],"dog");
    CHECK(ed_model_save(&m,path)==ED_OK);CHECK(ed_model_load(path,&loaded)==ED_OK);CHECK(ed_model_class_count(loaded)==3);CHECK(strcmp(ed_model_class_name(loaded,2),"dog")==0);ed_model_free(loaded);
    {
        FILE *f=fopen(path,"r+b");int byte;
        CHECK(f!=NULL);CHECK(fseek(f,20,SEEK_SET)==0);byte=fgetc(f);CHECK(byte!=EOF);
        CHECK(fseek(f,20,SEEK_SET)==0);CHECK(fputc(byte^1,f)!=EOF);CHECK(fclose(f)==0);
        CHECK(ed_model_load(path,&loaded)!=ED_OK);
    }
    remove(path);free(m.class_names);free(m.metadata);
    {
        static const uint8_t invalid[64]={0};FILE *f=fopen(bad_path,"wb");
        CHECK(f!=NULL);CHECK(fwrite(invalid,1,sizeof(invalid),f)==sizeof(invalid));CHECK(fclose(f)==0);
        CHECK(ed_dataset_load(bad_path,&bad_dataset)==ED_ERR_FORMAT);remove(bad_path);
    }
    s=ed_model_load(ED_TEST_MODEL_PATH,&loaded);CHECK(s==ED_OK);CHECK(ed_model_class_count(loaded)==80);CHECK(ed_model_save_fp16(loaded,half_path)==ED_OK);CHECK(ed_model_save_int8(loaded,int8_path)==ED_OK);CHECK(ed_model_save_int4(loaded,int4_path)==ED_OK);CHECK(has_disk_dtype(int4_path,ED_DTYPE_INT4_STORAGE));CHECK(has_disk_dtype(int4_path,ED_DTYPE_INT8_STORAGE));ed_model_free(loaded);loaded=NULL;CHECK(ed_model_load(half_path,&loaded)==ED_OK);remove(half_path);{ed_model*quantized=NULL;FILE*f;int byte;CHECK(ed_model_load(int8_path,&quantized)==ED_OK);CHECK(ed_model_class_count(quantized)==80);ed_model_free(quantized);remove(int8_path);quantized=NULL;CHECK(ed_model_load(int4_path,&quantized)==ED_OK);CHECK(ed_model_class_count(quantized)==80);ed_model_free(quantized);f=fopen(int4_path,"r+b");CHECK(f!=NULL);CHECK(fseek(f,-1,SEEK_END)==0);byte=fgetc(f);CHECK(byte!=EOF);CHECK(fseek(f,-1,SEEK_END)==0);CHECK(fputc(byte^1,f)!=EOF);CHECK(fclose(f)==0);CHECK(ed_model_load(int4_path,&quantized)==ED_ERR_CHECKSUM);remove(int4_path);}
    {
        ed_model *lower=NULL,*upper=NULL,*amb=NULL;const char *lo[]={"car"},*up[]={"Car"},*am[]={"Ambulance"};
        const ed_tensor *wa,*wb;
        CHECK(ed_model_load(ED_TEST_MODEL_PATH,&lower)==ED_OK);CHECK(ed_model_load(ED_TEST_MODEL_PATH,&upper)==ED_OK);
        CHECK(ed_model_remap_classes(lower,lo,1)==ED_OK);CHECK(ed_model_remap_classes(upper,up,1)==ED_OK);
        wa=ed_find_tensor_const(lower,"conv2d_84.w_0");wb=ed_find_tensor_const(upper,"conv2d_84.w_0");
        CHECK(wa&&wb&&wa->data_bytes==wb->data_bytes);CHECK(memcmp(wa->data,wb->data,(size_t)wa->data_bytes)==0);
        CHECK(ed_model_load(ED_TEST_MODEL_PATH,&amb)==ED_OK);
        CHECK(ed_find_source_class(amb,"Ambulance")==ed_find_source_class(amb,"truck"));CHECK(ed_find_source_class(amb,"Ambulance")>=0);
        CHECK(ed_model_remap_classes(amb,am,1)==ED_OK);CHECK(strcmp(ed_model_class_name(amb,0),"Ambulance")==0);
        ed_model_free(lower);ed_model_free(upper);ed_model_free(amb);
    }
    {
        uint8_t pixels[16u*8u*3u]={0};ed_image image={pixels,16u,8u,16u*3u};ed_detection detections[ED_MAX_DETECTIONS];ed_detection_list list={detections,ED_MAX_DETECTIONS,0};
        CHECK(ed_predict_tiled(loaded,&image,0.99f,0.2f,2u,1u,0.06f,0,&list)==ED_OK);
        CHECK(ed_predict_tiled(loaded,&image,0.99f,0.2f,1u,1u,0.0f,0,&list)==ED_ERR_ARGUMENT);
        CHECK(ed_predict_auto(loaded,&image,0.99f,0.2f,&list)==ED_OK);
        CHECK(ed_picodet_forward_flops()>0);CHECK(ed_estimate_energy_mj(2u,1.0f,116.0f)>0.0f);
        CHECK(ed_runtime_set_busy_core_watts(8.0f)==ED_OK);CHECK(ed_runtime_set_power_limit_w(1.0f)==ED_OK);
        {uint64_t t0=ed_monotonic_ns();volatile float acc=0.0f;size_t k;ed_runtime_power_mark_begin();
            while(ed_monotonic_ns()-t0<8000000ull)acc+=1.0f;ed_runtime_power_mark_end();
            CHECK(ed_runtime_last_average_w()>=0.0f);CHECK(ed_runtime_last_average_w()<1.6f);CHECK(ed_runtime_last_sleep_ms()>=0.0f);(void)acc;}
        CHECK(ed_runtime_set_power_limit_w(0.0f)==ED_OK);
    }
    {const char*classes[]={"feline","pet"};const uint32_t sources[8]={1,0,0,0,1,2,0,0},counts[2]={1,2};
        float heads_w[4u*2u*96u],heads_b[4u*2u];uint32_t z;
        CHECK(ed_model_remap_class_blends(loaded,classes,sources,counts,2)==ED_OK);CHECK(ed_model_class_count(loaded)==2);CHECK(strcmp(ed_model_class_name(loaded,1),"pet")==0);
        for(z=0;z<4u*2u*96u;++z)heads_w[z]=0.01f;for(z=0;z<4u*2u;++z)heads_b[z]=-1.0f;
        CHECK(ed_model_remap_class_heads(loaded,classes,2,heads_w,heads_b)==ED_OK);CHECK(ed_model_class_count(loaded)==2);}
    {const char*classes[]={"car","rocket"};CHECK(ed_model_remap_classes(loaded,classes,2)==ED_OK);CHECK(ed_model_class_count(loaded)==2);CHECK(strcmp(ed_model_class_name(loaded,1),"rocket")==0);}
    {
        ed_train_config bad={0};ed_train_report report;ed_dataset dummy;
        memset(&dummy,0,sizeof(dummy));
        CHECK(ed_train(loaded,NULL,&bad,&report)==ED_ERR_ARGUMENT);
        bad.budget_ms=1000;bad.threads=1;bad.mosaic_size=3;bad.accumulation=1;bad.scope=ED_TRAIN_OUTPUTS;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.mosaic_size=0;bad.head_adapter=(ed_head_adapter)2;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.head_adapter=ED_HEAD_ADAPTER_INDEPENDENT;bad.nesterov=2;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.nesterov=0;bad.roi_head=2;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.roi_head=0;bad.picofeat_adapter=2;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.picofeat_adapter=0;bad.sample_mode=(ed_sample_mode)9;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.sample_mode=ED_SAMPLE_AUTO;bad.max_optimizer_steps=10;bad.restart_after_steps=10;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.restart_after_steps=0;bad.max_optimizer_steps=0;bad.quality_adapter=2;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.quality_adapter=1;bad.aligned_loss=0;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
        bad.aligned_loss=1;bad.quality_adapter=0;bad.scope=ED_TRAIN_QUALITY;
        CHECK(ed_train(loaded,&dummy,&bad,&report)==ED_ERR_ARGUMENT);
    }
    ed_model_free(loaded);
    puts("all edgedet tests passed");return 0;
}
