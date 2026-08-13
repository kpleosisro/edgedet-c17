#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "ed_internal.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <stdio.h>
#endif

size_t ed_process_peak_rss(void){
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;memset(&info,0,sizeof(info));info.cb=(DWORD)sizeof(info);
    if(!GetProcessMemoryInfo(GetCurrentProcess(),&info,info.cb))return 0;
    return (size_t)info.PeakWorkingSetSize;
#else
    FILE *f=fopen("/proc/self/status","r");char line[256];size_t kb=0;
    if(!f)return 0;
    while(fgets(line,sizeof(line),f))if(sscanf(line,"VmHWM: %zu",&kb)==1)break;
    fclose(f);return kb*1024u;
#endif
}

const char *ed_status_string(ed_status status) {
    static const char *names[] = {"ok","invalid argument","I/O error","invalid format",
        "checksum mismatch","out of memory","unsupported","deadline reached","internal error"};
    return (unsigned)status < sizeof(names)/sizeof(names[0]) ? names[status] : "unknown";
}

uint64_t ed_monotonic_ms(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (uint64_t)((now.QuadPart * 1000) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (uint64_t)ts.tv_sec*1000u + (uint64_t)ts.tv_nsec/1000000u;
#endif
}
uint64_t ed_monotonic_ns(void){
#if defined(_WIN32)
    static LARGE_INTEGER freq;LARGE_INTEGER now;if(freq.QuadPart==0)QueryPerformanceFrequency(&freq);QueryPerformanceCounter(&now);return (uint64_t)((now.QuadPart*UINT64_C(1000000000))/freq.QuadPart);
#else
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return(uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
#endif
}

ed_status ed_read_entire_file(const char *path, uint8_t **data, size_t *size) {
    FILE *f; long n; uint8_t *p;
    if (!path || !data || !size) return ED_ERR_ARGUMENT;
    f=fopen(path,"rb"); if(!f)return ED_ERR_IO;
    if(fseek(f,0,SEEK_END)!=0 || (n=ftell(f))<0 || fseek(f,0,SEEK_SET)!=0){fclose(f);return ED_ERR_IO;}
    p=(uint8_t*)malloc((size_t)n ? (size_t)n : 1u); if(!p){fclose(f);return ED_ERR_MEMORY;}
    if((size_t)n && fread(p,1,(size_t)n,f)!=(size_t)n){free(p);fclose(f);return ED_ERR_IO;}
    fclose(f); *data=p; *size=(size_t)n; return ED_OK;
}

ed_status ed_write_entire_file(const char *path, const void *data, size_t size) {
    FILE *f; if(!path || (!data && size))return ED_ERR_ARGUMENT;
    f=fopen(path,"wb"); if(!f)return ED_ERR_IO;
    if(size && fwrite(data,1,size,f)!=size){fclose(f);return ED_ERR_IO;}
    if(fclose(f)!=0)return ED_ERR_IO; return ED_OK;
}

ed_status ed_dataset_load(const char *path, ed_dataset **out_dataset) {
    ed_dataset *d; const uint8_t *bytes; size_t n; ed_edb_header h, hc; ed_status s;ed_mapped_file map;
    if(!out_dataset)return ED_ERR_ARGUMENT; *out_dataset=NULL;
    s=ed_mapped_file_open(path,&map); if(s!=ED_OK)return s;bytes=map.data;n=map.size;
    if(n<sizeof(h)){ed_mapped_file_close(&map);return ED_ERR_FORMAT;} memcpy(&h,bytes,sizeof(h));
    if(h.magic!=ED_EDB_MAGIC || h.major!=1u || h.endian_tag!=ED_ENDIAN_TAG || h.header_bytes!=sizeof(h) || h.file_bytes!=n){ed_mapped_file_close(&map);return ED_ERR_FORMAT;}
    if(h.record_count==0u || h.class_count==0u || h.class_count>ED_MAX_CLASSES){ed_mapped_file_close(&map);return ED_ERR_FORMAT;}
    if(h.record_table_offset + (uint64_t)h.record_count*sizeof(ed_edb_record_disk)>n ||
       h.class_table_offset + (uint64_t)h.class_count*ED_CLASS_NAME_BYTES>n || h.data_offset>n){ed_mapped_file_close(&map);return ED_ERR_FORMAT;}
    hc=h; hc.header_crc32=0u; if(ed_crc32(&hc,sizeof(hc))!=h.header_crc32){ed_mapped_file_close(&map);return ED_ERR_CHECKSUM;}
    d=(ed_dataset*)calloc(1,sizeof(*d)); if(!d){ed_mapped_file_close(&map);return ED_ERR_MEMORY;}
    d->file_data=bytes; d->file_bytes=n;d->file_handle=map.file_handle;d->mapping_handle=map.mapping_handle; d->header=h;
    d->records=(const ed_edb_record_disk*)(bytes+h.record_table_offset);
    d->class_names=(const char(*)[ED_CLASS_NAME_BYTES])(bytes+h.class_table_offset);
    {
        uint32_t i;
        for(i=0;i<h.record_count;++i){
            const ed_edb_record_disk *r=&d->records[i];
            uint64_t ann_bytes=(uint64_t)r->annotation_count*sizeof(ed_edb_annotation_disk);
            uint32_t j;if(r->width==0u||r->height==0u||r->image_bytes==0u||
               r->image_offset>n||r->image_bytes>n-r->image_offset||
               r->annotation_offset>n||ann_bytes>n-r->annotation_offset){
                ed_dataset_free(d);return ED_ERR_FORMAT;
            }
            for(j=0;j<r->annotation_count;++j){const ed_edb_annotation_disk*a=(const ed_edb_annotation_disk*)(bytes+r->annotation_offset)+j;
                if(!isfinite(a->x1)||!isfinite(a->y1)||!isfinite(a->x2)||!isfinite(a->y2)||a->x1<0.0f||a->y1<0.0f||a->x2<=a->x1||a->y2<=a->y1||a->x2>(float)r->width||a->y2>(float)r->height||(!(a->flags&ED_ANN_FLAG_IGNORE)&&a->class_id>=h.class_count)){ed_dataset_free(d);return ED_ERR_FORMAT;}
            }
        }
    }
    *out_dataset=d; return ED_OK;
}

void ed_dataset_free(ed_dataset *d){if(d){ed_mapped_file map;memset(&map,0,sizeof(map));map.data=d->file_data;map.size=d->file_bytes;map.file_handle=d->file_handle;map.mapping_handle=d->mapping_handle;ed_mapped_file_close(&map);free(d);}}
uint64_t ed_dataset_count(const ed_dataset *d){return d?d->header.record_count:0u;}

ed_tensor *ed_find_tensor(ed_model *m,const char *name){uint32_t i;if(!m||!name)return NULL;for(i=0;i<m->tensor_count;++i)if(strcmp(m->tensors[i].name,name)==0)return &m->tensors[i];return NULL;}
const ed_tensor *ed_find_tensor_const(const ed_model *m,const char *name){return ed_find_tensor((ed_model*)m,name);}

ed_status ed_dataset_record(const ed_dataset*d,uint32_t index,const uint8_t**image_bytes,
                            uint32_t*image_size,const ed_edb_annotation_disk**annotations,
                            uint32_t*annotation_count,uint32_t*width,uint32_t*height){
    const ed_edb_record_disk*r;
    if(!d||index>=d->header.record_count||!image_bytes||!image_size||!annotations||
       !annotation_count||!width||!height)return ED_ERR_ARGUMENT;
    r=&d->records[index];*image_bytes=d->file_data+r->image_offset;*image_size=r->image_bytes;
    *annotations=(const ed_edb_annotation_disk*)(d->file_data+r->annotation_offset);
    *annotation_count=r->annotation_count;*width=r->width;*height=r->height;
    if(ed_crc32(*image_bytes,*image_size)!=r->image_crc32||ed_crc32(*annotations,(size_t)*annotation_count*sizeof(ed_edb_annotation_disk))!=r->record_crc32)return ED_ERR_CHECKSUM;return ED_OK;
}
