#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char stem[256];
    uint8_t *image;
    size_t image_bytes;
    uint32_t width,height;
    ed_edb_annotation_disk *annotations;
    uint32_t annotation_count;
} yolo_record;

static const char *arg(int argc,char **argv,const char *key){int i;for(i=1;i+1<argc;++i)if(strcmp(argv[i],key)==0)return argv[i+1];return NULL;}
static int has_flag(int argc,char **argv,const char *key){int i;for(i=1;i<argc;++i)if(strcmp(argv[i],key)==0)return 1;return 0;}
static int split_classes(const char *text,char (**out)[ED_CLASS_NAME_BYTES],uint32_t *count){
    char *copy,*token;uint32_t n=1,i=0;char(*classes)[ED_CLASS_NAME_BYTES];
    if(!text||!*text)return 0;for(token=(char*)text;*token;++token)if(*token==',')++n;
    classes=(char(*)[ED_CLASS_NAME_BYTES])calloc(n,ED_CLASS_NAME_BYTES);copy=(char*)malloc(strlen(text)+1u);if(!classes||!copy){free(classes);free(copy);return 0;}
    strcpy(copy,text);token=strtok(copy,",");while(token&&i<n){if(!*token||strlen(token)>=ED_CLASS_NAME_BYTES){free(classes);free(copy);return 0;}strcpy(classes[i++],token);token=strtok(NULL,",");}
    free(copy);*out=classes;*count=i;return i>0;
}
static int load_record(const char *images,const char *labels,const char *stem,uint32_t class_count,int filter_class,uint32_t keep_class,int keep_empty,yolo_record *r){
    char path[1400],line[512];FILE *f=NULL;uint8_t *rgb=NULL;uint32_t cap=0;
    snprintf(path,sizeof(path),"%s/%s.jpg",images,stem);if(ed_read_entire_file(path,&r->image,&r->image_bytes)!=ED_OK)return 0;
    if(ed_decode_image(r->image,r->image_bytes,&rgb,&r->width,&r->height)!=ED_OK)goto fail;free(rgb);rgb=NULL;
    snprintf(path,sizeof(path),"%s/%s.txt",labels,stem);f=fopen(path,"rb");if(!f)goto fail;
    while(fgets(line,sizeof(line),f)){unsigned cls;float cx,cy,w,h;ed_edb_annotation_disk a;
        if(sscanf(line,"%u %f %f %f %f",&cls,&cx,&cy,&w,&h)!=5||cls>=class_count||w<=0.0f||h<=0.0f)goto fail;
        if(filter_class&&cls!=keep_class)continue;
        memset(&a,0,sizeof(a));a.x1=(cx-w*0.5f)*(float)r->width;a.y1=(cy-h*0.5f)*(float)r->height;a.x2=(cx+w*0.5f)*(float)r->width;a.y2=(cy+h*0.5f)*(float)r->height;
        if(a.x1<0.0f)a.x1=0.0f;if(a.y1<0.0f)a.y1=0.0f;if(a.x2>(float)r->width)a.x2=(float)r->width;if(a.y2>(float)r->height)a.y2=(float)r->height;
        if(a.x2<=a.x1||a.y2<=a.y1)goto fail;a.class_id=(uint16_t)(filter_class?0u:cls);
        if(r->annotation_count==cap){uint32_t next=cap?cap*2u:16u;void *p=realloc(r->annotations,(size_t)next*sizeof(*r->annotations));if(!p)goto fail;r->annotations=(ed_edb_annotation_disk*)p;cap=next;}
        r->annotations[r->annotation_count++]=a;
    }
    fclose(f);f=NULL;if(r->annotation_count==0u&&!keep_empty)goto fail;return 1;
fail:
    if(f)fclose(f);free(rgb);free(r->image);free(r->annotations);memset(r,0,sizeof(*r));return 0;
}
static int records_equal(const yolo_record *a,const yolo_record *b){
    size_t annotation_bytes;
    if(a->image_bytes!=b->image_bytes||a->width!=b->width||a->height!=b->height||a->annotation_count!=b->annotation_count)return 0;
    if(memcmp(a->image,b->image,a->image_bytes)!=0)return 0;
    annotation_bytes=(size_t)a->annotation_count*sizeof(*a->annotations);
    return annotation_bytes==0u||memcmp(a->annotations,b->annotations,annotation_bytes)==0;
}
int main(int argc,char **argv){
    const char *images=arg(argc,argv,"--images"),*labels=arg(argc,argv,"--labels"),*list_path=arg(argc,argv,"--list"),*class_text=arg(argc,argv,"--classes"),*output=arg(argc,argv,"--output"),*keep_text=arg(argc,argv,"--keep-class"),*class_name=arg(argc,argv,"--class-name");
    char(*classes)[ED_CLASS_NAME_BYTES]=NULL,line[512];yolo_record *records=NULL;ed_edb_record_disk *disk=NULL;ed_edb_header h;FILE *list=NULL,*out=NULL;uint32_t class_count=0,count=0,cap=0,i;uint64_t off;
    uint32_t input_class_count=0,keep_class=0,duplicates=0;int filter_class=keep_text!=NULL,deduplicate=has_flag(argc,argv,"--deduplicate"),keep_empty=has_flag(argc,argv,"--keep-empty");
    if(!images||!labels||!list_path||!class_text||!output){fprintf(stderr,"usage: edpack-yolo --images DIR --labels DIR --list stems.txt --classes car --output data.edb [--keep-class ID --class-name NAME] [--keep-empty] [--deduplicate]\n");return 2;}
    if(!split_classes(class_text,&classes,&class_count))goto fail;input_class_count=class_count;
    if(filter_class){char *end=NULL;unsigned long parsed=strtoul(keep_text,&end,10);if(!*keep_text||!end||*end||parsed>=input_class_count){fprintf(stderr,"invalid --keep-class\n");goto fail;}keep_class=(uint32_t)parsed;if(class_name&&(!*class_name||strlen(class_name)>=ED_CLASS_NAME_BYTES)){fprintf(stderr,"invalid --class-name\n");goto fail;}}
    list=fopen(list_path,"rb");if(!list){fprintf(stderr,"cannot open %s\n",list_path);goto fail;}
    while(fgets(line,sizeof(line),list)){size_t n=strcspn(line," \t\r\n");if(!n)continue;line[n]=0;if(n>=sizeof(records[0].stem))goto fail;
        if(count==cap){uint32_t next=cap?cap*2u:256u;void *p=realloc(records,(size_t)next*sizeof(*records));if(!p)goto fail;records=(yolo_record*)p;memset(records+cap,0,(size_t)(next-cap)*sizeof(*records));cap=next;}
        strcpy(records[count].stem,line);if(!load_record(images,labels,line,input_class_count,filter_class,keep_class,keep_empty,&records[count])){fprintf(stderr,"skip invalid record %s\n",line);continue;}
        if(deduplicate){uint32_t existing;for(existing=0;existing<count;++existing)if(records_equal(&records[existing],&records[count]))break;if(existing<count){free(records[count].image);free(records[count].annotations);memset(&records[count],0,sizeof(records[count]));++duplicates;continue;}}
        ++count;
    }
    fclose(list);list=NULL;if(!count)goto fail;if(filter_class){char selected[ED_CLASS_NAME_BYTES];strcpy(selected,class_name?class_name:classes[keep_class]);memset(classes[0],0,ED_CLASS_NAME_BYTES);strcpy(classes[0],selected);class_count=1u;}disk=(ed_edb_record_disk*)calloc(count,sizeof(*disk));if(!disk)goto fail;memset(&h,0,sizeof(h));h.magic=ED_EDB_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.record_count=count;h.class_count=class_count;
    h.record_table_offset=sizeof(h);h.class_table_offset=h.record_table_offset+(uint64_t)count*sizeof(*disk);h.data_offset=h.class_table_offset+(uint64_t)class_count*ED_CLASS_NAME_BYTES;off=h.data_offset;
    for(i=0;i<count;++i){disk[i].image_offset=off;disk[i].image_bytes=(uint32_t)records[i].image_bytes;disk[i].image_crc32=ed_crc32(records[i].image,records[i].image_bytes);disk[i].width=records[i].width;disk[i].height=records[i].height;off+=records[i].image_bytes;disk[i].annotation_offset=off;disk[i].annotation_count=records[i].annotation_count;disk[i].record_crc32=ed_crc32(records[i].annotations,(size_t)records[i].annotation_count*sizeof(*records[i].annotations));off+=(uint64_t)records[i].annotation_count*sizeof(*records[i].annotations);}
    h.file_bytes=off;h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));out=fopen(output,"wb");if(!out)goto fail;
    if(fwrite(&h,1,sizeof(h),out)!=sizeof(h)||fwrite(disk,sizeof(*disk),count,out)!=count||fwrite(classes,ED_CLASS_NAME_BYTES,class_count,out)!=class_count)goto fail;
    for(i=0;i<count;++i)if(fwrite(records[i].image,1,records[i].image_bytes,out)!=records[i].image_bytes||(records[i].annotation_count&&fwrite(records[i].annotations,sizeof(*records[i].annotations),records[i].annotation_count,out)!=records[i].annotation_count))goto fail;
    if(fclose(out)!=0){out=NULL;goto fail;}out=NULL;printf("packed %u records, %u classes, %llu bytes, skipped_duplicates=%u\n",count,class_count,(unsigned long long)h.file_bytes,duplicates);
    for(i=0;i<count;++i){free(records[i].image);free(records[i].annotations);}free(records);free(disk);free(classes);return 0;
fail:
    if(list)fclose(list);if(out)fclose(out);for(i=0;i<count;++i){free(records[i].image);free(records[i].annotations);}free(records);free(disk);free(classes);return 1;
}
