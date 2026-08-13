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
static int split_classes(const char *text,char (**out)[ED_CLASS_NAME_BYTES],uint32_t *count){
    char *copy,*token;uint32_t n=1,i=0;char(*classes)[ED_CLASS_NAME_BYTES];
    if(!text||!*text)return 0;for(token=(char*)text;*token;++token)if(*token==',')++n;
    classes=(char(*)[ED_CLASS_NAME_BYTES])calloc(n,ED_CLASS_NAME_BYTES);copy=(char*)malloc(strlen(text)+1u);if(!classes||!copy){free(classes);free(copy);return 0;}
    strcpy(copy,text);token=strtok(copy,",");while(token&&i<n){if(!*token||strlen(token)>=ED_CLASS_NAME_BYTES){free(classes);free(copy);return 0;}strcpy(classes[i++],token);token=strtok(NULL,",");}
    free(copy);*out=classes;*count=i;return i>0;
}
static int load_record(const char *images,const char *labels,const char *stem,uint32_t class_count,yolo_record *r){
    char path[1400],line[512];FILE *f=NULL;uint8_t *rgb=NULL;uint32_t cap=0;
    snprintf(path,sizeof(path),"%s/%s.jpg",images,stem);if(ed_read_entire_file(path,&r->image,&r->image_bytes)!=ED_OK)return 0;
    if(ed_decode_image(r->image,r->image_bytes,&rgb,&r->width,&r->height)!=ED_OK)goto fail;free(rgb);rgb=NULL;
    snprintf(path,sizeof(path),"%s/%s.txt",labels,stem);f=fopen(path,"rb");if(!f)goto fail;
    while(fgets(line,sizeof(line),f)){unsigned cls;float cx,cy,w,h;ed_edb_annotation_disk a;
        if(sscanf(line,"%u %f %f %f %f",&cls,&cx,&cy,&w,&h)!=5||cls>=class_count||w<=0.0f||h<=0.0f)goto fail;
        memset(&a,0,sizeof(a));a.x1=(cx-w*0.5f)*(float)r->width;a.y1=(cy-h*0.5f)*(float)r->height;a.x2=(cx+w*0.5f)*(float)r->width;a.y2=(cy+h*0.5f)*(float)r->height;
        if(a.x1<0.0f)a.x1=0.0f;if(a.y1<0.0f)a.y1=0.0f;if(a.x2>(float)r->width)a.x2=(float)r->width;if(a.y2>(float)r->height)a.y2=(float)r->height;
        if(a.x2<=a.x1||a.y2<=a.y1)goto fail;a.class_id=(uint16_t)cls;
        if(r->annotation_count==cap){uint32_t next=cap?cap*2u:16u;void *p=realloc(r->annotations,(size_t)next*sizeof(*r->annotations));if(!p)goto fail;r->annotations=(ed_edb_annotation_disk*)p;cap=next;}
        r->annotations[r->annotation_count++]=a;
    }
    fclose(f);f=NULL;if(r->annotation_count==0u)goto fail;return 1;
fail:
    if(f)fclose(f);free(rgb);free(r->image);free(r->annotations);memset(r,0,sizeof(*r));return 0;
}
int main(int argc,char **argv){
    const char *images=arg(argc,argv,"--images"),*labels=arg(argc,argv,"--labels"),*list_path=arg(argc,argv,"--list"),*class_text=arg(argc,argv,"--classes"),*output=arg(argc,argv,"--output");
    char(*classes)[ED_CLASS_NAME_BYTES]=NULL,line[512];yolo_record *records=NULL;ed_edb_record_disk *disk=NULL;ed_edb_header h;FILE *list=NULL,*out=NULL;uint32_t class_count=0,count=0,cap=0,i;uint64_t off;
    if(!images||!labels||!list_path||!class_text||!output){fprintf(stderr,"usage: edpack-yolo --images DIR --labels DIR --list stems.txt --classes car --output data.edb\n");return 2;}
    if(!split_classes(class_text,&classes,&class_count))goto fail;list=fopen(list_path,"rb");if(!list){fprintf(stderr,"cannot open %s\n",list_path);goto fail;}
    while(fgets(line,sizeof(line),list)){size_t n=strcspn(line," \t\r\n");if(!n)continue;line[n]=0;if(n>=sizeof(records[0].stem))goto fail;
        if(count==cap){uint32_t next=cap?cap*2u:256u;void *p=realloc(records,(size_t)next*sizeof(*records));if(!p)goto fail;records=(yolo_record*)p;memset(records+cap,0,(size_t)(next-cap)*sizeof(*records));cap=next;}
        strcpy(records[count].stem,line);if(!load_record(images,labels,line,class_count,&records[count])){fprintf(stderr,"invalid record %s\n",line);goto fail;}++count;
    }
    fclose(list);list=NULL;if(!count)goto fail;disk=(ed_edb_record_disk*)calloc(count,sizeof(*disk));if(!disk)goto fail;memset(&h,0,sizeof(h));h.magic=ED_EDB_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.record_count=count;h.class_count=class_count;
    h.record_table_offset=sizeof(h);h.class_table_offset=h.record_table_offset+(uint64_t)count*sizeof(*disk);h.data_offset=h.class_table_offset+(uint64_t)class_count*ED_CLASS_NAME_BYTES;off=h.data_offset;
    for(i=0;i<count;++i){disk[i].image_offset=off;disk[i].image_bytes=(uint32_t)records[i].image_bytes;disk[i].image_crc32=ed_crc32(records[i].image,records[i].image_bytes);disk[i].width=records[i].width;disk[i].height=records[i].height;off+=records[i].image_bytes;disk[i].annotation_offset=off;disk[i].annotation_count=records[i].annotation_count;disk[i].record_crc32=ed_crc32(records[i].annotations,(size_t)records[i].annotation_count*sizeof(*records[i].annotations));off+=(uint64_t)records[i].annotation_count*sizeof(*records[i].annotations);}
    h.file_bytes=off;h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));out=fopen(output,"wb");if(!out)goto fail;
    if(fwrite(&h,1,sizeof(h),out)!=sizeof(h)||fwrite(disk,sizeof(*disk),count,out)!=count||fwrite(classes,ED_CLASS_NAME_BYTES,class_count,out)!=class_count)goto fail;
    for(i=0;i<count;++i)if(fwrite(records[i].image,1,records[i].image_bytes,out)!=records[i].image_bytes||fwrite(records[i].annotations,sizeof(*records[i].annotations),records[i].annotation_count,out)!=records[i].annotation_count)goto fail;
    if(fclose(out)!=0){out=NULL;goto fail;}out=NULL;printf("packed %u records, %u classes, %llu bytes\n",count,class_count,(unsigned long long)h.file_bytes);
    for(i=0;i<count;++i){free(records[i].image);free(records[i].annotations);}free(records);free(disk);free(classes);return 0;
fail:
    if(list)fclose(list);if(out)fclose(out);for(i=0;i<count;++i){free(records[i].image);free(records[i].annotations);}free(records);free(disk);free(classes);return 1;
}
