#include "ed_space.h"
#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg(int n,char **v,const char *k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
static int has_flag(int n,char **v,const char *k){int i;for(i=1;i<n;++i)if(strcmp(v[i],k)==0)return 1;return 0;}

static float iou(const ed_space_box *a,const ed_edb_annotation_disk *b){
    float x1=a->x1>b->x1?a->x1:b->x1,y1=a->y1>b->y1?a->y1:b->y1;
    float x2=a->x2<b->x2?a->x2:b->x2,y2=a->y2<b->y2?a->y2:b->y2;
    float w=x2-x1,h=y2-y1,inter,aa,bb;
    if(w<=0||h<=0)return 0;inter=w*h;
    aa=(a->x2-a->x1)*(a->y2-a->y1);bb=(b->x2-b->x1)*(b->y2-b->y1);
    return inter/(aa+bb-inter+1e-6f);
}

int main(int argc,char **argv){
    const char *dp=arg(argc,argv,"--dataset"),*mp=arg(argc,argv,"--model"),*maxs=arg(argc,argv,"--max");
    int do_fwd=has_flag(argc,argv,"--forward"),do_adapt=has_flag(argc,argv,"--adapt");
    ed_dataset *d=NULL;ed_model *m=NULL;ed_space_frame *frame;uint32_t i,use,hits=0,gts=0,teachers=0;
    if(!dp){fprintf(stderr,"usage: edspace --dataset D.edb [--model M.edm] [--max N] [--forward] [--adapt]\n");return 2;}
    if(ed_dataset_load(dp,&d)!=ED_OK){fprintf(stderr,"edspace: dataset\n");return 1;}
    if(mp&&ed_model_load(mp,&m)!=ED_OK){fprintf(stderr,"edspace: model\n");ed_dataset_free(d);return 1;}
    frame=(ed_space_frame*)calloc(1,sizeof(*frame));if(!frame)return 1;
    use=d->header.record_count;if(maxs){uint32_t cap=(uint32_t)strtoul(maxs,NULL,10);if(cap<use)use=cap;}
    for(i=0;i<use;++i){
        const uint8_t *enc;const ed_edb_annotation_disk *ann;uint32_t enc_n,ann_n,w,h,j,t;
        uint8_t *rgb=NULL;uint32_t iw,ih;
        if(ed_dataset_record(d,i,&enc,&enc_n,&ann,&ann_n,&w,&h)!=ED_OK)continue;
        if(ed_decode_image(enc,enc_n,&rgb,&iw,&ih)!=ED_OK)continue;
        if(ed_space_ingest_rgb(frame,rgb,iw,ih,iw*3u)!=ED_OK){free(rgb);continue;}
        free(rgb);
        (void)ed_space_step(frame,m,do_fwd&&m!=NULL,do_adapt&&m!=NULL);
        teachers+=frame->teacher_count;gts+=ann_n;
        for(j=0;j<ann_n;++j){
            float sx=320.0f/(float)w,sy=320.0f/(float)h;ed_edb_annotation_disk g=ann[j];
            g.x1*=sx;g.x2*=sx;g.y1*=sy;g.y2*=sy;
            for(t=0;t<frame->teacher_count;++t)if(iou(&frame->teacher[t],&g)>=0.3f){++hits;break;}
        }
        printf("frame %u teacher=%u det=%u gt=%u\n",i,frame->teacher_count,frame->det_count,ann_n);
    }
    printf("teacher_boxes=%u gt_boxes=%u teacher_hits@0.3=%u\n",teachers,gts,hits);
    free(frame);ed_dataset_free(d);ed_model_free(m);return 0;
}
