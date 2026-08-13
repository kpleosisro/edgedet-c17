#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *image;
    size_t image_bytes;
    uint32_t width,height;
    ed_edb_annotation_disk *annotations;
    uint32_t annotation_count;
} manifest_record;

static const char *arg(int argc,char **argv,const char *key){int i;for(i=1;i+1<argc;++i)if(strcmp(argv[i],key)==0)return argv[i+1];return NULL;}
static int load_classes(const char *path,char (**out)[ED_CLASS_NAME_BYTES],uint32_t *out_count){
    FILE *f=fopen(path,"rb");char line[256];char(*classes)[ED_CLASS_NAME_BYTES]=NULL;uint32_t count=0,cap=0;
    if(!f)return 0;while(fgets(line,sizeof(line),f)){size_t n=strcspn(line,"\r\n");void *p;line[n]=0;if(!n||n>=ED_CLASS_NAME_BYTES)goto fail;
        if(count==cap){uint32_t next=cap?cap*2u:8u;p=realloc(classes,(size_t)next*ED_CLASS_NAME_BYTES);if(!p)goto fail;classes=(char(*)[ED_CLASS_NAME_BYTES])p;memset(classes+cap,0,(size_t)(next-cap)*ED_CLASS_NAME_BYTES);cap=next;}
        strcpy(classes[count++],line);
    }fclose(f);if(!count){free(classes);return 0;}*out=classes;*out_count=count;return 1;
fail:fclose(f);free(classes);return 0;
}
static int load_record(const char *root,char *line,uint32_t class_count,manifest_record *r){
    char path[1600],*token=strtok(line," \t\r\n");uint8_t *rgb=NULL;uint32_t cap=0;
    if(!token)return 0;snprintf(path,sizeof(path),"%s/%s",root,token);
    if(ed_read_entire_file(path,&r->image,&r->image_bytes)!=ED_OK||r->image_bytes>UINT32_MAX)return 0;
    if(ed_decode_image(r->image,r->image_bytes,&rgb,&r->width,&r->height)!=ED_OK)goto fail;free(rgb);rgb=NULL;
    while((token=strtok(NULL," \t\r\n"))!=NULL){unsigned cls;float x1,y1,x2,y2;ed_edb_annotation_disk a;void *p;
        if(sscanf(token,"%f,%f,%f,%f,%u",&x1,&y1,&x2,&y2,&cls)!=5||cls>=class_count)goto fail;
        memset(&a,0,sizeof(a));a.x1=x1;a.y1=y1;a.x2=x2;a.y2=y2;a.class_id=(uint16_t)cls;
        if(a.x1<0.0f)a.x1=0.0f;if(a.y1<0.0f)a.y1=0.0f;if(a.x2>(float)r->width)a.x2=(float)r->width;if(a.y2>(float)r->height)a.y2=(float)r->height;
        if(a.x2<=a.x1||a.y2<=a.y1)goto fail;
        if(r->annotation_count==cap){uint32_t next=cap?cap*2u:8u;p=realloc(r->annotations,(size_t)next*sizeof(*r->annotations));if(!p)goto fail;r->annotations=(ed_edb_annotation_disk*)p;cap=next;}
        r->annotations[r->annotation_count++]=a;
    }
    return 1;
fail:free(rgb);free(r->image);free(r->annotations);memset(r,0,sizeof(*r));return 0;
}
int main(int argc,char **argv){
    const char *root=arg(argc,argv,"--root"),*manifest_path=arg(argc,argv,"--manifest"),*classes_path=arg(argc,argv,"--classes-file"),*output=arg(argc,argv,"--output");
    char(*classes)[ED_CLASS_NAME_BYTES]=NULL,line[4096];manifest_record *records=NULL;ed_edb_record_disk *disk=NULL;ed_edb_header h;FILE *manifest=NULL,*out=NULL;uint32_t class_count=0,count=0,cap=0,i;uint64_t off;
    if(!root||!manifest_path||!classes_path||!output){fprintf(stderr,"usage: edpack-manifest --root DIR --manifest manifest.txt --classes-file classes.txt --output data.edb\n");return 2;}
    if(!load_classes(classes_path,&classes,&class_count))goto fail;manifest=fopen(manifest_path,"rb");if(!manifest)goto fail;
    while(fgets(line,sizeof(line),manifest)){void *p;if(line[0]=='#'||strspn(line," \t\r\n")==strlen(line))continue;
        if(count==cap){uint32_t next=cap?cap*2u:256u;p=realloc(records,(size_t)next*sizeof(*records));if(!p)goto fail;records=(manifest_record*)p;memset(records+cap,0,(size_t)(next-cap)*sizeof(*records));cap=next;}
        if(!load_record(root,line,class_count,&records[count])){fprintf(stderr,"invalid manifest record %u\n",count);goto fail;}++count;
    }
    fclose(manifest);manifest=NULL;if(!count)goto fail;disk=(ed_edb_record_disk*)calloc(count,sizeof(*disk));if(!disk)goto fail;
    memset(&h,0,sizeof(h));h.magic=ED_EDB_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.record_count=count;h.class_count=class_count;
    h.record_table_offset=sizeof(h);h.class_table_offset=h.record_table_offset+(uint64_t)count*sizeof(*disk);h.data_offset=h.class_table_offset+(uint64_t)class_count*ED_CLASS_NAME_BYTES;off=h.data_offset;
    for(i=0;i<count;++i){disk[i].image_offset=off;disk[i].image_bytes=(uint32_t)records[i].image_bytes;disk[i].image_crc32=ed_crc32(records[i].image,records[i].image_bytes);disk[i].width=records[i].width;disk[i].height=records[i].height;off+=records[i].image_bytes;disk[i].annotation_offset=off;disk[i].annotation_count=records[i].annotation_count;disk[i].record_crc32=ed_crc32(records[i].annotations,(size_t)records[i].annotation_count*sizeof(*records[i].annotations));off+=(uint64_t)records[i].annotation_count*sizeof(*records[i].annotations);}
    h.file_bytes=off;h.header_crc32=ed_crc32(&h,sizeof(h));out=fopen(output,"wb");if(!out)goto fail;
    if(fwrite(&h,1,sizeof(h),out)!=sizeof(h)||fwrite(disk,sizeof(*disk),count,out)!=count||fwrite(classes,ED_CLASS_NAME_BYTES,class_count,out)!=class_count)goto fail;
    for(i=0;i<count;++i)if(fwrite(records[i].image,1,records[i].image_bytes,out)!=records[i].image_bytes||fwrite(records[i].annotations,sizeof(*records[i].annotations),records[i].annotation_count,out)!=records[i].annotation_count)goto fail;
    if(fclose(out)!=0){out=NULL;goto fail;}out=NULL;printf("packed %u records, %u classes, %llu bytes\n",count,class_count,(unsigned long long)h.file_bytes);
    for(i=0;i<count;++i){free(records[i].image);free(records[i].annotations);}free(records);free(disk);free(classes);return 0;
fail:
    if(manifest)fclose(manifest);if(out)fclose(out);for(i=0;i<count;++i){free(records[i].image);free(records[i].annotations);}free(records);free(disk);free(classes);return 1;
}
