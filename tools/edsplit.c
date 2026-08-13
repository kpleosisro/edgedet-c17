#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg(int argc,char **argv,const char *key){int i;for(i=1;i+1<argc;++i)if(strcmp(argv[i],key)==0)return argv[i+1];return NULL;}
static int has_flag(int argc,char **argv,const char *key){int i;for(i=1;i<argc;++i)if(strcmp(argv[i],key)==0)return 1;return 0;}

typedef struct {
    uint8_t *classes;
    uint32_t *indices;
    uint32_t count,capacity,select_count,remainder;
} split_stratum;

static uint64_t split_random(uint64_t *state){
    uint64_t x=*state;x^=x>>12;x^=x<<25;x^=x>>27;*state=x;return x*UINT64_C(2685821657736338717);
}

static void free_strata(split_stratum *strata,uint32_t count){
    uint32_t i;for(i=0;i<count;++i){free(strata[i].classes);free(strata[i].indices);}free(strata);
}

static ed_status stratified_indices(const ed_dataset *d,uint32_t mod,uint64_t seed,
                                    uint32_t *fit_idx,uint32_t *sel_idx,uint32_t *nf,uint32_t *ns){
    const uint32_t signature_bytes=(d->header.class_count+7u)/8u,n=d->header.record_count;
    split_stratum *strata=NULL;uint8_t *signature=NULL,*selected=NULL;uint32_t strata_n=0,strata_capacity=0,i,j,target,base=0;
    ed_status status=ED_OK;
    signature=(uint8_t*)calloc(signature_bytes?signature_bytes:1u,1);selected=(uint8_t*)calloc(n,1);
    if(!signature||!selected){status=ED_ERR_MEMORY;goto done;}
    for(i=0;i<n;++i){
        const uint8_t *image;const ed_edb_annotation_disk *annotations;uint32_t image_n,annotation_n,w,h,g=UINT32_MAX;
        memset(signature,0,signature_bytes);
        status=ed_dataset_record(d,i,&image,&image_n,&annotations,&annotation_n,&w,&h);if(status!=ED_OK)goto done;
        for(j=0;j<annotation_n;++j)if(!(annotations[j].flags&ED_ANN_FLAG_IGNORE))signature[annotations[j].class_id>>3]|=(uint8_t)(1u<<(annotations[j].class_id&7u));
        for(j=0;j<strata_n;++j)if(memcmp(strata[j].classes,signature,signature_bytes)==0){g=j;break;}
        if(g==UINT32_MAX){
            split_stratum *p;if(strata_n==strata_capacity){uint32_t next=strata_capacity?strata_capacity*2u:8u;p=(split_stratum*)realloc(strata,(size_t)next*sizeof(*strata));if(!p){status=ED_ERR_MEMORY;goto done;}strata=p;memset(strata+strata_capacity,0,(size_t)(next-strata_capacity)*sizeof(*strata));strata_capacity=next;}
            g=strata_n++;strata[g].classes=(uint8_t*)malloc(signature_bytes?signature_bytes:1u);if(!strata[g].classes){status=ED_ERR_MEMORY;goto done;}memcpy(strata[g].classes,signature,signature_bytes);
        }
        if(strata[g].count==strata[g].capacity){uint32_t next=strata[g].capacity?strata[g].capacity*2u:16u;uint32_t *p=(uint32_t*)realloc(strata[g].indices,(size_t)next*sizeof(uint32_t));if(!p){status=ED_ERR_MEMORY;goto done;}strata[g].indices=p;strata[g].capacity=next;}
        strata[g].indices[strata[g].count++]=i;
    }
    target=(n+mod/2u)/mod;if(target==0u)target=1u;if(target>=n)target=n-1u;
    for(i=0;i<strata_n;++i){strata[i].select_count=strata[i].count/mod;strata[i].remainder=strata[i].count%mod;base+=strata[i].select_count;}
    while(base<target){
        uint32_t best=UINT32_MAX,best_remainder=0;
        for(i=0;i<strata_n;++i)if(strata[i].select_count<strata[i].count&&(best==UINT32_MAX||strata[i].remainder>best_remainder)){best=i;best_remainder=strata[i].remainder;}
        if(best==UINT32_MAX)break;strata[best].select_count++;strata[best].remainder=0;base++;
    }
    for(i=0;i<strata_n;++i){
        uint64_t rng=seed^UINT64_C(0x9e3779b97f4a7c15);for(j=0;j<signature_bytes;++j)rng=(rng^strata[i].classes[j])*UINT64_C(1099511628211);
        for(j=strata[i].count;j>1u;--j){uint32_t k=(uint32_t)(split_random(&rng)%j),tmp=strata[i].indices[j-1u];strata[i].indices[j-1u]=strata[i].indices[k];strata[i].indices[k]=tmp;}
        for(j=0;j<strata[i].select_count;++j)selected[strata[i].indices[j]]=1u;
    }
    *nf=0u;*ns=0u;for(i=0;i<n;++i){if(selected[i])sel_idx[(*ns)++]=i;else fit_idx[(*nf)++]=i;}
    printf("strata=%u ",strata_n);
done:
    free(signature);free(selected);free_strata(strata,strata_n);return status;
}

static ed_status write_subset(const ed_dataset *d,const uint32_t *keep,uint32_t keep_n,const char *path){
    ed_edb_header h;ed_edb_record_disk *disk;uint8_t *file;uint64_t off;uint32_t i;ed_status status;
    if(keep_n==0u)return ED_ERR_ARGUMENT;
    memset(&h,0,sizeof(h));h.magic=ED_EDB_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);
    h.record_count=keep_n;h.class_count=d->header.class_count;
    h.record_table_offset=sizeof(h);h.class_table_offset=h.record_table_offset+(uint64_t)keep_n*sizeof(ed_edb_record_disk);
    h.data_offset=h.class_table_offset+(uint64_t)h.class_count*ED_CLASS_NAME_BYTES;off=h.data_offset;
    disk=(ed_edb_record_disk*)calloc(keep_n,sizeof(*disk));if(!disk)return ED_ERR_MEMORY;
    for(i=0;i<keep_n;++i){
        const ed_edb_record_disk *r=&d->records[keep[i]];
        disk[i]=*r;disk[i].image_offset=off;off+=r->image_bytes;disk[i].annotation_offset=off;
        off+=(uint64_t)r->annotation_count*sizeof(ed_edb_annotation_disk);
    }
    h.file_bytes=off;h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));
    file=(uint8_t*)calloc((size_t)off,1);if(!file){free(disk);return ED_ERR_MEMORY;}
    memcpy(file,&h,sizeof(h));memcpy(file+h.record_table_offset,disk,(size_t)keep_n*sizeof(*disk));
    memcpy(file+h.class_table_offset,d->class_names,(size_t)h.class_count*ED_CLASS_NAME_BYTES);
    for(i=0;i<keep_n;++i){
        const uint8_t *img;const ed_edb_annotation_disk *ann;uint32_t img_n,ann_n,w,hh;
        if(ed_dataset_record(d,keep[i],&img,&img_n,&ann,&ann_n,&w,&hh)!=ED_OK){free(file);free(disk);return ED_ERR_FORMAT;}
        memcpy(file+disk[i].image_offset,img,img_n);
        memcpy(file+disk[i].annotation_offset,ann,(size_t)ann_n*sizeof(*ann));
    }
    status=ed_write_entire_file(path,file,(size_t)off);free(file);free(disk);return status;
}

int main(int argc,char **argv){
    const char *input=arg(argc,argv,"--input"),*fit=arg(argc,argv,"--fit"),*select=arg(argc,argv,"--select"),*mod_text=arg(argc,argv,"--modulo"),*seed_text=arg(argc,argv,"--seed");
    ed_dataset *d=NULL;uint32_t *fit_idx=NULL,*sel_idx=NULL,mod=5u,i,nf=0,ns=0;uint64_t seed=7u;ed_status status;
    int stratify=has_flag(argc,argv,"--stratify-combinations");
    if(!input||!fit||!select){fprintf(stderr,"usage: edsplit --input data.edb --fit fit.edb --select select.edb [--modulo 5] [--stratify-combinations --seed 7]\n");return 2;}
    if(mod_text){mod=(uint32_t)strtoul(mod_text,NULL,10);if(mod<2u)return 2;}
    if(seed_text)seed=strtoull(seed_text,NULL,10);
    if((status=ed_dataset_load(input,&d))!=ED_OK){fprintf(stderr,"load: %s\n",ed_status_string(status));return 1;}
    fit_idx=(uint32_t*)malloc((size_t)d->header.record_count*sizeof(uint32_t));
    sel_idx=(uint32_t*)malloc((size_t)d->header.record_count*sizeof(uint32_t));
    if(!fit_idx||!sel_idx){ed_dataset_free(d);free(fit_idx);free(sel_idx);return 1;}
    if(stratify)status=stratified_indices(d,mod,seed,fit_idx,sel_idx,&nf,&ns);
    else for(i=0;i<d->header.record_count;++i){if((i%mod)==0u)sel_idx[ns++]=i;else fit_idx[nf++]=i;}
    if(status!=ED_OK){fprintf(stderr,"split: %s\n",ed_status_string(status));free(fit_idx);free(sel_idx);ed_dataset_free(d);return 1;}
    status=write_subset(d,fit_idx,nf,fit);if(status==ED_OK)status=write_subset(d,sel_idx,ns,select);
    if(status!=ED_OK)fprintf(stderr,"split: %s\n",ed_status_string(status));
    else printf("fit=%u select=%u modulo=%u mode=%s seed=%llu\n",nf,ns,mod,stratify?"class-combinations":"index",(unsigned long long)seed);
    free(fit_idx);free(sel_idx);ed_dataset_free(d);return status==ED_OK?0:1;
}
