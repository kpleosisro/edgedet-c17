#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char*arg(int n,char**v,const char*k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
int main(int argc,char**argv){
    const char*mp=arg(argc,argv,"--model"),*ip=arg(argc,argv,"--input"),*tp=arg(argc,argv,"--threads"),*xp=arg(argc,argv,"--tiles-x"),*yp=arg(argc,argv,"--tiles-y"),*op=arg(argc,argv,"--tile-overlap"),*np=arg(argc,argv,"--nms-threshold"),*sp=arg(argc,argv,"--score-threshold"),*pl=arg(argc,argv,"--power-limit"),*cw=arg(argc,argv,"--core-watts");ed_model*m=NULL;ed_detection det[ED_MAX_DETECTIONS];
    ed_detection_list list={det,ED_MAX_DETECTIONS,0};ed_image im;uint8_t *encoded=NULL,*rgb=NULL;size_t encoded_n=0;ed_status s;size_t i;
    uint32_t tiles_x=0u,tiles_y=0u;float overlap=op?strtof(op,NULL):0.06f,nms=np?strtof(np,NULL):0.6f,score=sp?strtof(sp,NULL):0.25f;
    if(!mp||!ip){fprintf(stderr,"usage: edinfer --model M.edm --input IMAGE [--threads N] [--score-threshold F] [--nms-threshold F] [--tiles-x auto|N --tiles-y N --tile-overlap F] [--power-limit W] [--core-watts W]\n");return 2;}
    if(xp&&strcmp(xp,"auto")!=0)tiles_x=(uint32_t)strtoul(xp,NULL,10);if(yp)tiles_y=(uint32_t)strtoul(yp,NULL,10);
    if(tp&&ed_runtime_set_threads((uint32_t)strtoul(tp,NULL,10))!=ED_OK)return 2;
    if(cw&&ed_runtime_set_busy_core_watts(strtof(cw,NULL))!=ED_OK)return 2;
    if(pl){if(!tp)(void)ed_runtime_set_threads(1u);if(ed_runtime_set_power_limit_w(strtof(pl,NULL))!=ED_OK)return 2;}
    if((s=ed_read_entire_file(ip,&encoded,&encoded_n))!=ED_OK||(s=ed_decode_image(encoded,encoded_n,&rgb,&im.width,&im.height))!=ED_OK){fprintf(stderr,"image: %s\n",ed_status_string(s));free(encoded);return 1;}free(encoded);
    im.rgb=rgb;im.stride_bytes=im.width*3u;
    if((s=ed_model_load(mp,&m))!=ED_OK){fprintf(stderr,"infer: %s\n",ed_status_string(s));free(rgb);return 1;}
    if(tiles_x==0u&&tiles_y==0u)s=ed_predict_auto(m,&im,score,nms,&list);
    else if(tiles_x*tiles_y>1u)s=ed_predict_tiled(m,&im,score,nms,tiles_x,tiles_y,overlap,0,&list);
    else s=ed_predict(m,&im,score,nms,&list);
    if(s!=ED_OK){fprintf(stderr,"infer: %s\n",ed_status_string(s));free(rgb);ed_model_free(m);return 1;}
    for(i=0;i<list.count;++i)printf("%s %.6f %.2f %.2f %.2f %.2f\n",ed_model_class_name(m,det[i].class_id),det[i].score,det[i].x1,det[i].y1,det[i].x2,det[i].y2);
    free(rgb);ed_model_free(m);return 0;
}
