#include "edgedet.h"
#include "ed_internal.h"
#include "ed_hw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char*arg(int n,char**v,const char*k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
static int has_flag(int n,char**v,const char*k){int i;for(i=1;i<n;++i)if(strcmp(v[i],k)==0)return 1;return 0;}
static ed_status dump_detections(const char *path,const ed_model*m,const ed_dataset*d,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float overlap,int include_full,float soft_nms){
    FILE *f=fopen(path,"wb");uint32_t image_index;ed_status status=ED_OK;
    if(!f)return ED_ERR_IO;
    fputs("image_id,class_id,score,x1,y1,x2,y2\n",f);
    for(image_index=0;image_index<d->header.record_count;++image_index){
        const uint8_t *encoded;const ed_edb_annotation_disk *annotations;uint32_t encoded_n,annotation_n,w,h,j;uint8_t *rgb=NULL;ed_image image;ed_detection found[ED_MAX_DETECTIONS];ed_detection_list list={found,ED_MAX_DETECTIONS,0};
        status=ed_dataset_record(d,image_index,&encoded,&encoded_n,&annotations,&annotation_n,&w,&h);(void)annotations;(void)annotation_n;if(status!=ED_OK)break;
        status=ed_decode_image(encoded,encoded_n,&rgb,&image.width,&image.height);if(status!=ED_OK)break;
        if(image.width!=w||image.height!=h){free(rgb);status=ED_ERR_FORMAT;break;}image.rgb=rgb;image.stride_bytes=image.width*3u;
        {float predict_nt=soft_nms>0.0f?1.0f:nt;
            if(tiles_x==0u&&tiles_y==0u)status=ed_predict_auto(m,&image,st,predict_nt,&list);
            else if(tiles_x*tiles_y>1u)status=ed_predict_tiled(m,&image,st,predict_nt,tiles_x,tiles_y,overlap,include_full,&list);
            else status=ed_predict(m,&image,st,predict_nt,&list);}
        if(status==ED_OK&&soft_nms>0.0f)list.count=ed_nms_soft(found,list.count,soft_nms,st);free(rgb);if(status!=ED_OK)break;
        for(j=0;j<list.count;++j)fprintf(f,"%u,%u,%.9g,%.9g,%.9g,%.9g,%.9g\n",image_index,found[j].class_id,found[j].score,found[j].x1,found[j].y1,found[j].x2,found[j].y2);
    }
    if(fclose(f)!=0&&status==ED_OK)status=ED_ERR_IO;return status;
}
int main(int argc,char**argv){
    const char*dp=arg(argc,argv,"--dataset"),*mp=arg(argc,argv,"--model"),*tp=arg(argc,argv,"--threads"),*sp=arg(argc,argv,"--score-threshold"),*np=arg(argc,argv,"--nms-threshold"),*xp=arg(argc,argv,"--tiles-x"),*yp=arg(argc,argv,"--tiles-y"),*op=arg(argc,argv,"--tile-overlap"),*softp=arg(argc,argv,"--soft-nms"),*dump=arg(argc,argv,"--dump-detections"),*rt=arg(argc,argv,"--runtime");
    float threshold=sp?strtof(sp,NULL):0.025f,nms=np?strtof(np,NULL):0.6f,tile_overlap=op?strtof(op,NULL):0.0f,soft_nms=0.0f;
    uint32_t tiles_x=1u,tiles_y=1u;int include_full=has_flag(argc,argv,"--include-full"),tta_flip=has_flag(argc,argv,"--tta-flip");
    ed_model*m=NULL;ed_dataset*d=NULL;ed_map_report r;ed_status s;uint32_t i;
    if(xp&&strcmp(xp,"auto")==0){tiles_x=0u;tiles_y=0u;}else{if(xp)tiles_x=(uint32_t)strtoul(xp,NULL,10);if(yp)tiles_y=(uint32_t)strtoul(yp,NULL,10);}
    if(softp){if(strcmp(softp,"linear")==0)soft_nms=9.0f;else soft_nms=strtof(softp,NULL);}
    else if(has_flag(argc,argv,"--soft-nms"))soft_nms=0.5f;
    if(!dp||!mp){fprintf(stderr,"usage: edeval --dataset D.edb --model M.edm [--runtime host|fpga-model] [--threads N] [--score-threshold 0.025] [--nms-threshold 0.6] [--tiles-x auto|N --tiles-y N --tile-overlap F --include-full] [--soft-nms SIGMA|linear] [--tta-flip] [--dump-detections FILE.csv]\n");return 2;}
    if(rt){if(strcmp(rt,"fpga-model")==0){if(ed_runtime_set_compute(ED_COMPUTE_FPGA_MODEL)!=ED_OK)return 2;}
        else if(strcmp(rt,"host")!=0){fprintf(stderr,"invalid --runtime\n");return 2;}}
    if(tp&&ed_runtime_set_threads((uint32_t)strtoul(tp,NULL,10))!=ED_OK)return 2;
    if((s=ed_dataset_load(dp,&d))!=ED_OK||(s=ed_model_load(mp,&m))!=ED_OK){fprintf(stderr,"eval: %s\n",ed_status_string(s));ed_dataset_free(d);ed_model_free(m);return 1;}
    s=ed_evaluate_map50_tta(m,d,threshold,nms,tiles_x,tiles_y,tile_overlap,include_full,soft_nms,tta_flip,&r);
    if(s!=ED_OK){fprintf(stderr,"eval: %s\n",ed_status_string(s));ed_dataset_free(d);ed_model_free(m);return 1;}
    for(i=0;i<r.class_count;++i)printf("%s AP50=%.6f max_recall50=%.6f\n",ed_model_class_name(m,i),r.per_class_ap[i],r.per_class_recall[i]);
    printf("continuous_map50=%.6f voc07_map50=%.6f\n",r.map50,r.map50_11point);
    if(dump&&(s=dump_detections(dump,m,d,threshold,nms,tiles_x,tiles_y,tile_overlap,include_full,soft_nms))!=ED_OK){fprintf(stderr,"dump: %s\n",ed_status_string(s));ed_dataset_free(d);ed_model_free(m);return 1;}
    ed_dataset_free(d);ed_model_free(m);return 0;
}
