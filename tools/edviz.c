#include "ed_internal.h"
#include "ed_hw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg(int n,char **v,const char *k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
static void put_pixel(uint8_t *rgb,uint32_t w,uint32_t h,int x,int y,uint8_t r,uint8_t g,uint8_t b){
    if(x<0||y<0||x>=(int)w||y>=(int)h)return;{uint8_t *p=rgb+((size_t)(uint32_t)y*w+(uint32_t)x)*3u;p[0]=r;p[1]=g;p[2]=b;}
}
static void fill_rect(uint8_t *rgb,uint32_t w,uint32_t h,int x1,int y1,int x2,int y2,uint8_t r,uint8_t g,uint8_t b){
    int x,y;if(x1>x2){int t=x1;x1=x2;x2=t;}if(y1>y2){int t=y1;y1=y2;y2=t;}
    for(y=y1;y<=y2;++y)for(x=x1;x<=x2;++x)put_pixel(rgb,w,h,x,y,r,g,b);
}
static void draw_rect(uint8_t *rgb,uint32_t w,uint32_t h,int x1,int y1,int x2,int y2,uint8_t r,uint8_t g,uint8_t b){
    int x,y;if(x1>x2){int t=x1;x1=x2;x2=t;}if(y1>y2){int t=y1;y1=y2;y2=t;}
    for(x=x1;x<=x2;++x){put_pixel(rgb,w,h,x,y1,r,g,b);put_pixel(rgb,w,h,x,y2,r,g,b);put_pixel(rgb,w,h,x,y1+1,r,g,b);put_pixel(rgb,w,h,x,y2-1,r,g,b);}
    for(y=y1;y<=y2;++y){put_pixel(rgb,w,h,x1,y,r,g,b);put_pixel(rgb,w,h,x2,y,r,g,b);put_pixel(rgb,w,h,x1+1,y,r,g,b);put_pixel(rgb,w,h,x2-1,y,r,g,b);}
}

/* 5x7 glyphs, bits 4..0 = left..right. Index 0 is space; then '!'..'Z' via ASCII-32. */
static const uint8_t font5x7[][7]={
    {0,0,0,0,0,0,0},/*space*/
    {4,4,4,4,0,4,0},{10,10,0,0,0,0,0},{10,10,31,10,31,10,10},{4,15,20,14,5,30,4},
    {24,25,2,4,8,19,3},{8,20,8,21,18,13,0},{4,4,0,0,0,0,0},{2,4,8,8,8,4,2},{8,4,2,2,2,4,8},
    {0,4,21,14,21,4,0},{0,4,4,31,4,4,0},{0,0,0,0,0,4,8},{0,0,0,31,0,0,0},{0,0,0,0,0,4,0},
    {1,1,2,4,8,16,16},/* / */
    {14,17,19,21,25,17,14},{4,12,4,4,4,4,14},{14,17,1,6,8,16,31},{14,17,1,6,1,17,14},
    {2,6,10,18,31,2,2},{31,16,30,1,1,17,14},{6,8,16,30,17,17,14},{31,1,2,4,8,8,8},
    {14,17,17,14,17,17,14},{14,17,17,15,1,2,12},
    {0,4,0,0,0,4,0},{0,4,0,0,0,4,8},{2,4,8,16,8,4,2},{0,0,31,0,31,0,0},{8,4,2,1,2,4,8},
    {14,17,1,2,4,0,4},
    {14,17,23,21,23,16,14},/* @ unused */
    {14,17,17,31,17,17,17},{30,17,17,30,17,17,30},{14,17,16,16,16,17,14},{30,17,17,17,17,17,30},
    {31,16,16,30,16,16,31},{31,16,16,30,16,16,16},{14,17,16,23,17,17,14},{17,17,17,31,17,17,17},
    {14,4,4,4,4,4,14},{1,1,1,1,1,17,14},{17,18,20,24,20,18,17},{16,16,16,16,16,16,31},
    {17,27,21,21,17,17,17},{17,25,21,19,17,17,17},{14,17,17,17,17,17,14},{30,17,17,30,16,16,16},
    {14,17,17,17,21,18,13},{30,17,17,30,20,18,17},{14,17,16,14,1,17,14},{31,4,4,4,4,4,4},
    {17,17,17,17,17,17,14},{17,17,17,17,17,10,4},{17,17,17,21,21,21,10},{17,17,10,4,10,17,17},
    {17,17,10,4,4,4,4},{31,1,2,4,8,16,31}
};
static int glyph_index(int ch){
    if(ch>='a'&&ch<='z')ch=ch-'a'+'A';
    if(ch>=32&&ch<=90)return ch-32;
    return '?'-32;
}
static int text_scale(uint32_t w,uint32_t h){uint32_t m=w<h?w:h;return m>=900?3:(m>=420?2:1);}
static void draw_char(uint8_t *rgb,uint32_t w,uint32_t h,int x,int y,int ch,int scale,uint8_t r,uint8_t g,uint8_t b){
    const uint8_t *glyph=font5x7[glyph_index(ch)];int row,col,sy,sx;
    for(row=0;row<7;++row)for(col=0;col<5;++col)if(glyph[row]&(16u>>col))
        for(sy=0;sy<scale;++sy)for(sx=0;sx<scale;++sx)put_pixel(rgb,w,h,x+col*scale+sx,y+row*scale+sy,r,g,b);
}
static void draw_label(uint8_t *rgb,uint32_t w,uint32_t h,int x,int y,const char *text,int scale,uint8_t r,uint8_t g,uint8_t b){
    int len=(int)strlen(text),tw=len*(5*scale+scale)+scale,th=7*scale+2*scale,tx,ty,i;
    if(x<0)x=0;if(y<0)y=0;
    if(x+tw>=(int)w)x=(int)w-tw-1;if(x<0)x=0;
    if(y+th>=(int)h)y=(int)h-th-1;if(y<0)y=0;
    fill_rect(rgb,w,h,x,y,x+tw,y+th,r,g,b);
    tx=x+scale;ty=y+scale;
    for(i=0;text[i];++i){draw_char(rgb,w,h,tx,ty,(unsigned char)text[i],scale,255,255,255);tx+=5*scale+scale;}
}
static float box_iou(const ed_detection *a,const ed_detection *b){
    float x1=a->x1>b->x1?a->x1:b->x1,y1=a->y1>b->y1?a->y1:b->y1,x2=a->x2<b->x2?a->x2:b->x2,y2=a->y2<b->y2?a->y2:b->y2;
    float iw=x2-x1,ih=y2-y1,inter,den;if(iw<=0.0f||ih<=0.0f)return 0.0f;inter=iw*ih;
    den=(a->x2-a->x1)*(a->y2-a->y1)+(b->x2-b->x1)*(b->y2-b->y1)-inter;return den>0.0f?inter/den:0.0f;
}
static size_t nms_display(ed_detection *items,size_t n,float iou_threshold){
    size_t i,j,kept=0;uint8_t *dead;
    if(!items||!n)return 0;
    dead=(uint8_t*)calloc(n,1);if(!dead)return n;
    for(i=0;i<n;++i){
        size_t best=i;if(dead[i])continue;
        for(j=i+1;j<n;++j)if(!dead[j]&&items[j].score>items[best].score)best=j;
        {ed_detection tmp=items[i];items[i]=items[best];items[best]=tmp;}
        dead[i]=1;items[kept++]=items[i];
        for(j=i+1;j<n;++j)if(!dead[j]&&box_iou(&items[i],&items[j])>iou_threshold)dead[j]=1;
    }
    free(dead);return kept;
}
static int write_ppm(const char *path,const uint8_t *rgb,uint32_t w,uint32_t h){
    FILE *f=fopen(path,"wb");if(!f)return 0;fprintf(f,"P6\n%u %u\n255\n",w,h);
    if(fwrite(rgb,1,(size_t)w*h*3u,f)!=(size_t)w*h*3u){fclose(f);return 0;}return fclose(f)==0;
}
int main(int argc,char **argv){
    const char *dp=arg(argc,argv,"--dataset"),*mp=arg(argc,argv,"--model"),*od=arg(argc,argv,"--output-dir"),*max_text=arg(argc,argv,"--max"),*sp=arg(argc,argv,"--score-threshold"),*np=arg(argc,argv,"--nms-threshold"),*xp=arg(argc,argv,"--tiles-x"),*yp=arg(argc,argv,"--tiles-y"),*rt=arg(argc,argv,"--runtime");
    uint32_t max_images=24u,i,tiles_x=0u,tiles_y=0u;float score=sp?strtof(sp,NULL):0.25f,nms=np?strtof(np,NULL):0.42f;
    ed_model *m=NULL;ed_dataset *d=NULL;ed_status s;
    if(!dp||!mp||!od){fprintf(stderr,"usage: edviz --dataset D.edb --model M.edm --output-dir DIR [--runtime host|fpga-model] [--max N] [--score-threshold F] [--nms-threshold F] [--tiles-x auto|N]\n");return 2;}
    if(rt){if(strcmp(rt,"fpga-model")==0){if(ed_runtime_set_compute(ED_COMPUTE_FPGA_MODEL)!=ED_OK)return 2;}
        else if(strcmp(rt,"host")!=0){fprintf(stderr,"invalid --runtime\n");return 2;}}
    if(max_text)max_images=(uint32_t)strtoul(max_text,NULL,10);
    if(xp&&strcmp(xp,"auto")!=0)tiles_x=(uint32_t)strtoul(xp,NULL,10);if(yp)tiles_y=(uint32_t)strtoul(yp,NULL,10);
    if((s=ed_dataset_load(dp,&d))!=ED_OK||(s=ed_model_load(mp,&m))!=ED_OK){fprintf(stderr,"viz: %s\n",ed_status_string(s));ed_dataset_free(d);ed_model_free(m);return 1;}
    if(max_images>d->header.record_count)max_images=d->header.record_count;
    for(i=0;i<max_images;++i){
        const uint8_t *enc;const ed_edb_annotation_disk *ann;uint32_t enc_n,ann_n,w,h,j;uint8_t *rgb=NULL;ed_image image;ed_detection det[ED_MAX_DETECTIONS];ed_detection_list list={det,ED_MAX_DETECTIONS,0};char path[768];int scale;
        if(ed_dataset_record(d,i,&enc,&enc_n,&ann,&ann_n,&w,&h)!=ED_OK||ed_decode_image(enc,enc_n,&rgb,&w,&h)!=ED_OK)continue;
        (void)ann;(void)ann_n;
        image.rgb=rgb;image.width=w;image.height=h;image.stride_bytes=w*3u;
        if(tiles_x==0u&&tiles_y==0u){if(xp&&strcmp(xp,"auto")==0)s=ed_predict_auto(m,&image,score,nms,&list);else s=ed_predict(m,&image,score,nms,&list);}
        else if(tiles_x*tiles_y>1u)s=ed_predict_tiled(m,&image,score,nms,tiles_x,tiles_y,0.06f,0,&list);
        else s=ed_predict(m,&image,score,nms,&list);
        if(s!=ED_OK){free(rgb);continue;}
        list.count=nms_display(det,list.count,0.50f);
        scale=text_scale(w,h);
        for(j=0;j<list.count;++j){
            static const uint8_t pal[][3]={{220,30,30},{30,90,220},{230,160,20},{160,40,220},{20,180,180},{220,80,20},{80,180,40},{180,40,90}};
            const uint8_t *c=pal[det[j].class_id%(sizeof(pal)/sizeof(pal[0]))];
            const char *name=ed_model_class_name(m,det[j].class_id);char label[64];int x1=(int)det[j].x1,y1=(int)det[j].y1,ly;
            if(!name)name="?";
            snprintf(label,sizeof(label),"%s %.2f",name,det[j].score);
            draw_rect(rgb,w,h,x1,y1,(int)det[j].x2,(int)det[j].y2,c[0],c[1],c[2]);
            ly=y1-(7*scale+3*scale);if(ly<0)ly=y1+3;
            draw_label(rgb,w,h,x1,ly,label,scale,c[0],c[1],c[2]);
        }
        snprintf(path,sizeof(path),"%s/%04u.ppm",od,i);
        if(!write_ppm(path,rgb,w,h))fprintf(stderr,"write failed %s\n",path);
        free(rgb);
    }
    ed_dataset_free(d);ed_model_free(m);return 0;
}
