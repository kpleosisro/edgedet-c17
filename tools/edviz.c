#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg(int n,char **v,const char *k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
static void put_pixel(uint8_t *rgb,uint32_t w,uint32_t h,int x,int y,uint8_t r,uint8_t g,uint8_t b){
    if(x<0||y<0||x>=(int)w||y>=(int)h)return;{uint8_t *p=rgb+((size_t)(uint32_t)y*w+(uint32_t)x)*3u;p[0]=r;p[1]=g;p[2]=b;}
}
static void draw_rect(uint8_t *rgb,uint32_t w,uint32_t h,int x1,int y1,int x2,int y2,uint8_t r,uint8_t g,uint8_t b){
    int x,y;if(x1>x2){int t=x1;x1=x2;x2=t;}if(y1>y2){int t=y1;y1=y2;y2=t;}
    for(x=x1;x<=x2;++x){put_pixel(rgb,w,h,x,y1,r,g,b);put_pixel(rgb,w,h,x,y2,r,g,b);put_pixel(rgb,w,h,x,y1+1,r,g,b);put_pixel(rgb,w,h,x,y2-1,r,g,b);}
    for(y=y1;y<=y2;++y){put_pixel(rgb,w,h,x1,y,r,g,b);put_pixel(rgb,w,h,x2,y,r,g,b);put_pixel(rgb,w,h,x1+1,y,r,g,b);put_pixel(rgb,w,h,x2-1,y,r,g,b);}
}
static int write_ppm(const char *path,const uint8_t *rgb,uint32_t w,uint32_t h){
    FILE *f=fopen(path,"wb");if(!f)return 0;fprintf(f,"P6\n%u %u\n255\n",w,h);
    if(fwrite(rgb,1,(size_t)w*h*3u,f)!=(size_t)w*h*3u){fclose(f);return 0;}return fclose(f)==0;
}
int main(int argc,char **argv){
    const char *dp=arg(argc,argv,"--dataset"),*mp=arg(argc,argv,"--model"),*od=arg(argc,argv,"--output-dir"),*max_text=arg(argc,argv,"--max"),*sp=arg(argc,argv,"--score-threshold"),*np=arg(argc,argv,"--nms-threshold");
    uint32_t max_images=24u,i;float score=sp?strtof(sp,NULL):0.25f,nms=np?strtof(np,NULL):0.42f;
    ed_model *m=NULL;ed_dataset *d=NULL;ed_status s;
    if(!dp||!mp||!od){fprintf(stderr,"usage: edviz --dataset D.edb --model M.edm --output-dir DIR [--max N] [--score-threshold F] [--nms-threshold F]\n");return 2;}
    if(max_text)max_images=(uint32_t)strtoul(max_text,NULL,10);
    if((s=ed_dataset_load(dp,&d))!=ED_OK||(s=ed_model_load(mp,&m))!=ED_OK){fprintf(stderr,"viz: %s\n",ed_status_string(s));ed_dataset_free(d);ed_model_free(m);return 1;}
    if(max_images>d->header.record_count)max_images=d->header.record_count;
    for(i=0;i<max_images;++i){
        const uint8_t *enc;const ed_edb_annotation_disk *ann;uint32_t enc_n,ann_n,w,h,j;uint8_t *rgb=NULL;ed_image image;ed_detection det[ED_MAX_DETECTIONS];ed_detection_list list={det,ED_MAX_DETECTIONS,0};char path[768];
        if(ed_dataset_record(d,i,&enc,&enc_n,&ann,&ann_n,&w,&h)!=ED_OK||ed_decode_image(enc,enc_n,&rgb,&w,&h)!=ED_OK)continue;
        image.rgb=rgb;image.width=w;image.height=h;image.stride_bytes=w*3u;
        if(ed_predict(m,&image,score,nms,&list)!=ED_OK){free(rgb);continue;}
        for(j=0;j<ann_n;++j)if(!(ann[j].flags&ED_ANN_FLAG_IGNORE))draw_rect(rgb,w,h,(int)ann[j].x1,(int)ann[j].y1,(int)ann[j].x2,(int)ann[j].y2,0,220,40);
        for(j=0;j<list.count;++j)draw_rect(rgb,w,h,(int)det[j].x1,(int)det[j].y1,(int)det[j].x2,(int)det[j].y2,220,30,30);
        snprintf(path,sizeof(path),"%s/%04u.ppm",od,i);
        if(!write_ppm(path,rgb,w,h))fprintf(stderr,"write failed %s\n",path);
        free(rgb);
    }
    ed_dataset_free(d);ed_model_free(m);return 0;
}
