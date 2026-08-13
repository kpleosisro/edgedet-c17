#include "ed_internal.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    ed_detection detection;
    uint32_t image_index;
} scored_detection;

static float box_iou(float ax1,float ay1,float ax2,float ay2,
                     float bx1,float by1,float bx2,float by2){
    float x1=ax1>bx1?ax1:bx1,y1=ay1>by1?ay1:by1,x2=ax2<bx2?ax2:bx2,y2=ay2<by2?ay2:by2;
    float w=x2-x1,h=y2-y1,inter,area_a,area_b;
    if(w<=0.0f||h<=0.0f)return 0.0f;inter=w*h;
    area_a=(ax2-ax1)*(ay2-ay1);area_b=(bx2-bx1)*(by2-by1);
    return inter/(area_a+area_b-inter+1e-12f);
}
static int score_desc(const void *a,const void *b){
    float x=((const scored_detection*)a)->detection.score;
    float y=((const scored_detection*)b)->detection.score;
    return x<y?1:(x>y?-1:0);
}
static int append_detection(scored_detection **items,size_t *count,size_t *capacity,
                            const ed_detection *d,uint32_t image_index){
    if(*count==*capacity){size_t next=*capacity?*capacity*2u:4096u;void *p=realloc(*items,next*sizeof(**items));if(!p)return 0;*items=(scored_detection*)p;*capacity=next;}
    (*items)[*count].detection=*d;(*items)[*count].image_index=image_index;++*count;return 1;
}
static ed_status run_predict(const ed_model*m,const ed_image*image,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float tile_overlap,int include_full,ed_detection_list*list){
    if(tiles_x==0u&&tiles_y==0u)return ed_predict_auto(m,image,st,nt,list);
    if(tiles_x*tiles_y>1u)return ed_predict_tiled(m,image,st,nt,tiles_x,tiles_y,tile_overlap,include_full,list);
    return ed_predict(m,image,st,nt,list);
}
static ed_status predict_hflip_tta(const ed_model*m,const ed_image*image,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float tile_overlap,int include_full,ed_detection_list*out){
    ed_detection orig[ED_MAX_DETECTIONS],flipped[ED_MAX_DETECTIONS],merged[ED_MAX_DETECTIONS*2u];
    ed_detection_list a={orig,ED_MAX_DETECTIONS,0},b={flipped,ED_MAX_DETECTIONS,0};
    uint8_t *rgb=NULL;ed_image mirror;uint32_t y,x;size_t i,n;ed_status status;
    status=run_predict(m,image,st,nt,tiles_x,tiles_y,tile_overlap,include_full,&a);if(status!=ED_OK)return status;
    rgb=(uint8_t*)malloc((size_t)image->height*image->width*3u);if(!rgb)return ED_ERR_MEMORY;
    for(y=0;y<image->height;++y)for(x=0;x<image->width;++x){
        const uint8_t *src=image->rgb+(size_t)y*image->stride_bytes+(size_t)(image->width-1u-x)*3u;
        uint8_t *dst=rgb+((size_t)y*image->width+x)*3u;dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];
    }
    mirror.rgb=rgb;mirror.width=image->width;mirror.height=image->height;mirror.stride_bytes=image->width*3u;
    status=run_predict(m,&mirror,st,nt,tiles_x,tiles_y,tile_overlap,include_full,&b);free(rgb);if(status!=ED_OK)return status;
    for(i=0;i<b.count;++i){float x1=b.items[i].x1,x2=b.items[i].x2;b.items[i].x1=(float)image->width-x2;b.items[i].x2=(float)image->width-x1;}
    n=0;for(i=0;i<a.count&&n<sizeof(merged)/sizeof(merged[0]);++i)merged[n++]=a.items[i];
    for(i=0;i<b.count&&n<sizeof(merged)/sizeof(merged[0]);++i)merged[n++]=b.items[i];
    n=ed_nms(merged,n,nt);
    out->count=n<out->capacity?n:out->capacity;memcpy(out->items,merged,out->count*sizeof(*out->items));return ED_OK;
}
static ed_status evaluate_map50_impl(const ed_model*m,const ed_dataset*d,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float tile_overlap,int include_full,float soft_nms_sigma,int hflip,ed_map_report*r){
    scored_detection *predictions=NULL;size_t prediction_count=0,prediction_capacity=0;uint32_t *gt_count=NULL;
    uint8_t **matched=NULL;uint32_t image_index,c;ed_status status=ED_OK;
    if(!m||!d||!r||st<0.0f||st>1.0f||nt<0.0f||nt>1.0f)return ED_ERR_ARGUMENT;
    if(m->class_count!=d->header.class_count)return ED_ERR_FORMAT;
    for(c=0;c<m->class_count;++c)if(strncmp(m->class_names[c],d->class_names[c],ED_CLASS_NAME_BYTES)!=0)return ED_ERR_FORMAT;
    memset(r,0,sizeof(*r));r->class_count=m->class_count;
    gt_count=(uint32_t*)calloc(m->class_count,sizeof(*gt_count));matched=(uint8_t**)calloc(d->header.record_count,sizeof(*matched));
    if(!gt_count||!matched){status=ED_ERR_MEMORY;goto done;}
    for(image_index=0;image_index<d->header.record_count;++image_index){
        const uint8_t *encoded;const ed_edb_annotation_disk *annotations;uint32_t encoded_n,annotation_n,w,h,j;
        uint8_t *rgb=NULL;ed_image image;ed_detection found[ED_MAX_DETECTIONS];ed_detection_list list={found,ED_MAX_DETECTIONS,0};
        status=ed_dataset_record(d,image_index,&encoded,&encoded_n,&annotations,&annotation_n,&w,&h);if(status!=ED_OK)goto done;
        matched[image_index]=(uint8_t*)calloc(annotation_n?annotation_n:1u,1);if(!matched[image_index]){status=ED_ERR_MEMORY;goto done;}
        for(j=0;j<annotation_n;++j)if(!(annotations[j].flags&ED_ANN_FLAG_IGNORE)&&annotations[j].class_id<m->class_count)++gt_count[annotations[j].class_id];
        status=ed_decode_image(encoded,encoded_n,&rgb,&image.width,&image.height);if(status!=ED_OK)goto done;
        if(image.width!=w||image.height!=h){free(rgb);status=ED_ERR_FORMAT;goto done;}
        image.rgb=rgb;image.stride_bytes=image.width*3u;
        {float predict_nt=soft_nms_sigma>0.0f?1.0f:nt;
            if(hflip)status=predict_hflip_tta(m,&image,st,predict_nt,tiles_x,tiles_y,tile_overlap,include_full,&list);
            else status=run_predict(m,&image,st,predict_nt,tiles_x,tiles_y,tile_overlap,include_full,&list);}
        if(status==ED_OK&&soft_nms_sigma>0.0f)list.count=ed_nms_soft(found,list.count,soft_nms_sigma,st);
        free(rgb);if(status!=ED_OK)goto done;
        for(j=0;j<list.count;++j)if(!append_detection(&predictions,&prediction_count,&prediction_capacity,&found[j],image_index)){status=ED_ERR_MEMORY;goto done;}
    }
    qsort(predictions,prediction_count,sizeof(*predictions),score_desc);
    for(c=0;c<m->class_count;++c){
        uint32_t tp=0,fp=0;size_t k,nc=0;float prev_recall=0.0f,ap=0.0f,ap11=0.0f;
        float *recall=NULL,*precision=NULL;
        for(k=0;k<prediction_count;++k)if(predictions[k].detection.class_id==c)++nc;
        recall=(float*)malloc((nc?nc:1u)*sizeof(float));precision=(float*)malloc((nc?nc:1u)*sizeof(float));
        if(!recall||!precision){free(recall);free(precision);status=ED_ERR_MEMORY;goto done;}nc=0;
        for(k=0;k<prediction_count;++k){
            const scored_detection *p=&predictions[k];const ed_edb_record_disk *record;const ed_edb_annotation_disk *anns;
            uint32_t j,best=UINT32_MAX;float best_iou=0.0f;int ignored=0;
            if(p->detection.class_id!=c)continue;record=&d->records[p->image_index];anns=(const ed_edb_annotation_disk*)(d->file_data+record->annotation_offset);
            for(j=0;j<record->annotation_count;++j){float iou=box_iou(p->detection.x1,p->detection.y1,p->detection.x2,p->detection.y2,anns[j].x1,anns[j].y1,anns[j].x2,anns[j].y2);
                if((anns[j].flags&ED_ANN_FLAG_IGNORE)&&iou>=0.5f)ignored=1;
                if(!(anns[j].flags&ED_ANN_FLAG_IGNORE)&&anns[j].class_id==c&&!matched[p->image_index][j]&&iou>best_iou){best_iou=iou;best=j;}
            }
            if(best_iou>=0.5f&&best!=UINT32_MAX){matched[p->image_index][best]=1u;++tp;}
            else if(!ignored)++fp;else continue;
            recall[nc]=gt_count[c]?(float)tp/(float)gt_count[c]:0.0f;precision[nc]=(float)tp/(float)(tp+fp);++nc;
        }
        if(nc){size_t j;for(j=nc-1;j>0;--j)if(precision[j-1]<precision[j])precision[j-1]=precision[j];for(j=0;j<nc;++j){float delta=recall[j]-prev_recall;if(delta>0.0f)ap+=delta*precision[j];prev_recall=recall[j];}
            for(j=0;j<=10u;++j){float threshold=(float)j/10.0f,best=0.0f;size_t z;for(z=0;z<nc;++z)if(recall[z]>=threshold&&precision[z]>best)best=precision[z];ap11+=best/11.0f;}
        }
        r->per_class_ap[c]=ap;r->per_class_recall[c]=gt_count[c]?(float)tp/(float)gt_count[c]:0.0f;r->map50+=ap/(float)m->class_count;r->map50_11point+=ap11/(float)m->class_count;free(recall);free(precision);
    }
done:
    if(matched){for(image_index=0;image_index<d->header.record_count;++image_index)free(matched[image_index]);}
    free(matched);free(gt_count);free(predictions);return status;
}
ed_status ed_evaluate_map50_ex(const ed_model*m,const ed_dataset*d,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float tile_overlap,int include_full,float soft_nms_sigma,ed_map_report*r){return evaluate_map50_impl(m,d,st,nt,tiles_x,tiles_y,tile_overlap,include_full,soft_nms_sigma,0,r);}
ed_status ed_evaluate_map50_tta(const ed_model*m,const ed_dataset*d,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float tile_overlap,int include_full,float soft_nms_sigma,int hflip,ed_map_report*r){return evaluate_map50_impl(m,d,st,nt,tiles_x,tiles_y,tile_overlap,include_full,soft_nms_sigma,hflip,r);}
ed_status ed_evaluate_map50_tiled(const ed_model*m,const ed_dataset*d,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float tile_overlap,int include_full,ed_map_report*r){return evaluate_map50_impl(m,d,st,nt,tiles_x,tiles_y,tile_overlap,include_full,0.0f,0,r);}
ed_status ed_evaluate_map50(const ed_model*m,const ed_dataset*d,float st,float nt,ed_map_report*r){return evaluate_map50_impl(m,d,st,nt,1u,1u,0.0f,0,0.0f,0,r);}
