#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char id[64];
    char image_path[1024];
    ed_edb_annotation_disk *annotations;
    uint32_t annotation_count, width, height, image_bytes;
} pack_record;

static const char *arg(int argc,char **argv,const char *key){
    int i;for(i=1;i+1<argc;++i)if(strcmp(argv[i],key)==0)return argv[i+1];return NULL;
}
static char *read_text(const char *path,size_t *size){
    uint8_t *p=NULL;size_t n=0;if(ed_read_entire_file(path,&p,&n)!=ED_OK)return NULL;
    p=(uint8_t*)realloc(p,n+1u);if(!p)return NULL;p[n]=0;if(size)*size=n;return (char*)p;
}
static const char *bounded_find(const char *p,const char *end,const char *needle){
    size_t n=strlen(needle);for(;p+n<=end;++p)if(memcmp(p,needle,n)==0)return p;return NULL;
}
static int text_between(const char *begin,const char *end,const char *open,const char *close,
                        char *out,size_t cap){
    const char *a=bounded_find(begin,end,open),*b;size_t n;if(!a)return 0;a+=strlen(open);
    b=bounded_find(a,end,close);if(!b)return 0;while(a<b&&(*a==' '||*a=='\t'||*a=='\r'||*a=='\n'))++a;
    while(b>a&&(b[-1]==' '||b[-1]=='\t'||b[-1]=='\r'||b[-1]=='\n'))--b;
    n=(size_t)(b-a);if(n>=cap)n=cap-1u;memcpy(out,a,n);out[n]=0;return 1;
}
static int class_id(const char *name,char (*classes)[ED_CLASS_NAME_BYTES],uint32_t count){
    uint32_t i;for(i=0;i<count;++i)if(strcmp(name,classes[i])==0)return (int)i;return -1;
}
static int parse_xml(const char *path,char (*classes)[ED_CLASS_NAME_BYTES],uint32_t class_count,
                     pack_record *record){
    char *xml=read_text(path,NULL),value[64];const char *p,*end,*obj,*obj_end;
    ed_edb_annotation_disk *anns=NULL;uint32_t count=0,cap=0;
    if(!xml)return 0;end=xml+strlen(xml);
    if(!text_between(xml,end,"<width>","</width>",value,sizeof(value)))goto fail;
    record->width=(uint32_t)strtoul(value,NULL,10);
    if(!text_between(xml,end,"<height>","</height>",value,sizeof(value)))goto fail;
    record->height=(uint32_t)strtoul(value,NULL,10);p=xml;
    while((obj=bounded_find(p,end,"<object>"))!=NULL){
        char name[ED_CLASS_NAME_BYTES],box[64];int cid,difficult=0;ed_edb_annotation_disk a;
        obj_end=bounded_find(obj,end,"</object>");if(!obj_end)goto fail;
        if(!text_between(obj,obj_end,"<name>","</name>",name,sizeof(name)))goto fail;
        if(text_between(obj,obj_end,"<difficult>","</difficult>",value,sizeof(value)))difficult=atoi(value)!=0;
        cid=class_id(name,classes,class_count);memset(&a,0,sizeof(a));
        if(!text_between(obj,obj_end,"<xmin>","</xmin>",box,sizeof(box)))goto fail;a.x1=(float)atof(box)-1.0f;
        if(!text_between(obj,obj_end,"<ymin>","</ymin>",box,sizeof(box)))goto fail;a.y1=(float)atof(box)-1.0f;
        if(!text_between(obj,obj_end,"<xmax>","</xmax>",box,sizeof(box)))goto fail;a.x2=(float)atof(box);
        if(!text_between(obj,obj_end,"<ymax>","</ymax>",box,sizeof(box)))goto fail;a.y2=(float)atof(box);
        a.class_id=(uint16_t)(cid>=0?cid:0);a.flags=(uint16_t)((cid<0||difficult)?ED_ANN_FLAG_IGNORE:0u);
        if(count==cap){uint32_t next=cap?cap*2u:16u;void*q=realloc(anns,(size_t)next*sizeof(*anns));if(!q)goto fail;anns=(ed_edb_annotation_disk*)q;cap=next;}
        anns[count++]=a;p=obj_end+9;
    }
    free(xml);record->annotations=anns;record->annotation_count=count;return record->width&&record->height;
fail:free(anns);free(xml);return 0;
}
static int split_classes(const char *text,char (**out)[ED_CLASS_NAME_BYTES],uint32_t *count){
    char *copy,*tok;uint32_t n=1,i=0;char(*classes)[ED_CLASS_NAME_BYTES];
    if(!text||!*text)return 0;for(tok=(char*)text;*tok;++tok)if(*tok==',')++n;
    classes=(char(*)[ED_CLASS_NAME_BYTES])calloc(n,ED_CLASS_NAME_BYTES);copy=(char*)malloc(strlen(text)+1u);
    if(!classes||!copy){free(classes);free(copy);return 0;}strcpy(copy,text);tok=strtok(copy,",");
    while(tok&&i<n){if(!*tok||strlen(tok)>=ED_CLASS_NAME_BYTES){free(classes);free(copy);return 0;}strcpy(classes[i++],tok);tok=strtok(NULL,",");}
    free(copy);*out=classes;*count=i;return i>0;
}
int main(int argc,char **argv){
    const char *root=arg(argc,argv,"--voc-root"),*split=arg(argc,argv,"--split"),
               *class_text=arg(argc,argv,"--classes"),*output=arg(argc,argv,"--output");
    char path[1200],line[128];char(*classes)[ED_CLASS_NAME_BYTES]=NULL;uint32_t class_count=0,count=0,cap=0,i;
    pack_record *records=NULL;ed_edb_record_disk *disk=NULL;ed_edb_header h;FILE *list=NULL,*out=NULL;uint64_t off;
    if(!root||!split||!class_text||!output){fprintf(stderr,"usage: edpack-voc --voc-root VOC2007 --split trainval --classes car,cat,dog --output data.edb\n");return 2;}
    if(!split_classes(class_text,&classes,&class_count)){fprintf(stderr,"invalid class list\n");goto fail;}
    snprintf(path,sizeof(path),"%s/ImageSets/Main/%s.txt",root,split);list=fopen(path,"rb");if(!list){fprintf(stderr,"cannot open %s\n",path);goto fail;}
    while(fgets(line,sizeof(line),list)){
        char xml_path[1200];size_t len=strcspn(line," \t\r\n");FILE *image;
        if(!len||len>=sizeof(records[0].id))continue;line[len]=0;
        if(count==cap){uint32_t next=cap?cap*2u:1024u;void*q=realloc(records,(size_t)next*sizeof(*records));if(!q)goto fail;records=(pack_record*)q;memset(records+cap,0,(size_t)(next-cap)*sizeof(*records));cap=next;}
        strcpy(records[count].id,line);snprintf(records[count].image_path,sizeof(records[count].image_path),"%s/JPEGImages/%s.jpg",root,line);
        snprintf(xml_path,sizeof(xml_path),"%s/Annotations/%s.xml",root,line);
        if(!parse_xml(xml_path,classes,class_count,&records[count])){fprintf(stderr,"invalid XML: %s\n",xml_path);goto fail;}
        image=fopen(records[count].image_path,"rb");if(!image||fseek(image,0,SEEK_END)!=0){if(image)fclose(image);fprintf(stderr,"cannot read %s\n",records[count].image_path);goto fail;}
        {long bytes=ftell(image);if(bytes<=0){fclose(image);goto fail;}records[count].image_bytes=(uint32_t)bytes;}fclose(image);++count;
    }
    fclose(list);list=NULL;if(!count){fprintf(stderr,"empty split\n");goto fail;}
    disk=(ed_edb_record_disk*)calloc(count,sizeof(*disk));if(!disk)goto fail;
    memset(&h,0,sizeof(h));h.magic=ED_EDB_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.record_count=count;h.class_count=class_count;
    h.record_table_offset=sizeof(h);h.class_table_offset=h.record_table_offset+(uint64_t)count*sizeof(*disk);
    h.data_offset=h.class_table_offset+(uint64_t)class_count*ED_CLASS_NAME_BYTES;off=h.data_offset;
    for(i=0;i<count;++i){disk[i].image_offset=off;disk[i].image_bytes=records[i].image_bytes;disk[i].width=records[i].width;disk[i].height=records[i].height;off+=records[i].image_bytes;disk[i].annotation_offset=off;disk[i].annotation_count=records[i].annotation_count;off+=(uint64_t)records[i].annotation_count*sizeof(ed_edb_annotation_disk);}
    h.file_bytes=off;h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));out=fopen(output,"wb+");if(!out){fprintf(stderr,"cannot create %s\n",output);goto fail;}
    if(fwrite(&h,1,sizeof(h),out)!=sizeof(h)||fwrite(disk,sizeof(*disk),count,out)!=count||fwrite(classes,ED_CLASS_NAME_BYTES,class_count,out)!=class_count)goto fail;
    for(i=0;i<count;++i){uint8_t buffer[65536];size_t n;FILE *image=fopen(records[i].image_path,"rb");uint32_t crc=0xffffffffu;
        if(!image)goto fail;while((n=fread(buffer,1,sizeof(buffer),image))!=0){if(fwrite(buffer,1,n,out)!=n){fclose(image);goto fail;}}
        fclose(image);{uint8_t *bytes=NULL;size_t bytes_n=0;if(ed_read_entire_file(records[i].image_path,&bytes,&bytes_n)!=ED_OK)goto fail;crc=ed_crc32(bytes,bytes_n);free(bytes);}disk[i].image_crc32=crc;
        if(records[i].annotation_count&&fwrite(records[i].annotations,sizeof(ed_edb_annotation_disk),records[i].annotation_count,out)!=records[i].annotation_count)goto fail;
        disk[i].record_crc32=ed_crc32(records[i].annotations,(size_t)records[i].annotation_count*sizeof(ed_edb_annotation_disk));
    }
    if(fseek(out,(long)h.record_table_offset,SEEK_SET)!=0||fwrite(disk,sizeof(*disk),count,out)!=count||fclose(out)!=0){out=NULL;goto fail;}out=NULL;
    printf("packed %u records, %u classes, %llu bytes\n",count,class_count,(unsigned long long)h.file_bytes);
    for(i=0;i<count;++i)free(records[i].annotations);free(records);free(disk);free(classes);return 0;
fail:
    if(list)fclose(list);if(out)fclose(out);for(i=0;i<count;++i)free(records[i].annotations);free(records);free(disk);free(classes);return 1;
}
