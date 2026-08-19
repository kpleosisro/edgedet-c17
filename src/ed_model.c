#include "ed_internal.h"
#include "ed_hw.h"
#include "ed_graph.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float ed_clampf(float value,float minimum,float maximum){
    return value<minimum?minimum:(value>maximum?maximum:value);
}

static int range_ok(uint64_t off,uint64_t bytes,size_t total){return off<=total&&bytes<=total-off;}
static int tensor_elements(const ed_edm_tensor_disk*d,size_t*out){size_t n=1;uint32_t i;if(!d||!out||d->rank==0u||d->rank>ED_MAX_TENSOR_RANK)return 0;for(i=0;i<ED_MAX_TENSOR_RANK;++i){if(d->dims[i]==0u||n>SIZE_MAX/d->dims[i])return 0;n*=d->dims[i];}*out=n;return 1;}
#define ED_INT4_MIN_ELEMENTS 131072u
static uint16_t float_to_half(float value){uint32_t x,sign,exp,mant;memcpy(&x,&value,4);sign=(x>>16)&0x8000u;exp=(x>>23)&255u;mant=x&0x7fffffu;if(exp==255u)return(uint16_t)(sign|(mant?0x7e00u:0x7c00u));if(exp>142u)return(uint16_t)(sign|0x7c00u);if(exp<113u){uint32_t shift;if(exp<103u)return(uint16_t)sign;mant|=0x800000u;shift=125u-exp;mant=(mant+(1u<<(shift-1u))-1u+((mant>>shift)&1u))>>shift;return(uint16_t)(sign|mant);}exp-=112u;mant=mant+0xfffu+((mant>>13)&1u);if(mant&0x800000u){mant=0;++exp;if(exp>=31u)return(uint16_t)(sign|0x7c00u);}return(uint16_t)(sign|(exp<<10)|(mant>>13));}
static float half_to_float(uint16_t h){uint32_t sign=((uint32_t)h&0x8000u)<<16,exp=((uint32_t)h>>10)&31u,mant=(uint32_t)h&1023u,x;if(exp==0u){if(mant==0u)x=sign;else{exp=113u;while((mant&1024u)==0u){mant<<=1;--exp;}mant&=1023u;x=sign|(exp<<23)|(mant<<13);}}else if(exp==31u)x=sign|0x7f800000u|(mant<<13);else x=sign|((exp+112u)<<23)|(mant<<13);{float value;memcpy(&value,&x,4);return value;}}

ed_status ed_model_load(const char *path,ed_model **out){
    uint8_t *file=NULL,digest[32];size_t n=0;ed_edm_header h,hc;ed_model*m=NULL;ed_status s;uint32_t i;
    if(!out)return ED_ERR_ARGUMENT;*out=NULL;s=ed_read_entire_file(path,&file,&n);if(s!=ED_OK)return s;
    if(n<sizeof(h)){s=ED_ERR_FORMAT;goto fail;}memcpy(&h,file,sizeof(h));
    if(h.magic!=ED_EDM_MAGIC||h.major!=1u||h.endian_tag!=ED_ENDIAN_TAG||h.header_bytes!=sizeof(h)||h.file_bytes!=n||
       h.architecture!=ED_ARCH_PICODET_S_320||h.class_count==0u||h.class_count>ED_MAX_CLASSES){s=ED_ERR_FORMAT;goto fail;}
    if(!range_ok(h.tensor_table_offset,(uint64_t)h.tensor_count*sizeof(ed_edm_tensor_disk),n)||
       !range_ok(h.class_table_offset,(uint64_t)h.class_count*ED_CLASS_NAME_BYTES,n)||
       !range_ok(h.metadata_offset,h.metadata_bytes,n)||h.payload_offset>n){s=ED_ERR_FORMAT;goto fail;}
    hc=h;hc.header_crc32=0;if(ed_crc32(&hc,sizeof(hc))!=h.header_crc32){s=ED_ERR_CHECKSUM;goto fail;}
    ed_sha256(file+h.payload_offset,n-(size_t)h.payload_offset,digest);if(!ed_digest_equal(digest,h.payload_sha256)){s=ED_ERR_CHECKSUM;goto fail;}
    m=(ed_model*)calloc(1,sizeof(*m));if(!m){s=ED_ERR_MEMORY;goto fail;}
    m->architecture=h.architecture;m->precision=h.precision;m->flags=h.flags;m->class_count=h.class_count;m->tensor_count=h.tensor_count;m->metadata_bytes=h.metadata_bytes;
    memcpy(m->source_sha256,h.source_sha256,32);
    m->class_names=(char(*)[ED_CLASS_NAME_BYTES])malloc((size_t)m->class_count*ED_CLASS_NAME_BYTES);
    m->tensors=(ed_tensor*)calloc(m->tensor_count,sizeof(ed_tensor));m->metadata=(char*)malloc((size_t)m->metadata_bytes+1u);
    if(!m->class_names||(!m->tensors&&m->tensor_count)||!m->metadata){s=ED_ERR_MEMORY;goto fail;}
    memcpy(m->class_names,file+h.class_table_offset,(size_t)m->class_count*ED_CLASS_NAME_BYTES);
    memcpy(m->metadata,file+h.metadata_offset,m->metadata_bytes);m->metadata[m->metadata_bytes]=0;
    for(i=0;i<m->tensor_count;++i){
        ed_edm_tensor_disk d;ed_tensor*t=&m->tensors[i];memcpy(&d,file+h.tensor_table_offset+(uint64_t)i*sizeof(d),sizeof(d));
        if(d.rank>ED_MAX_TENSOR_RANK||!range_ok(d.data_offset,d.data_bytes,n)){s=ED_ERR_FORMAT;goto fail;}
        if(ed_crc32(file+d.data_offset,(size_t)d.data_bytes)!=d.data_crc32){s=ED_ERR_CHECKSUM;goto fail;}
        memcpy(t->name,d.name,sizeof(t->name));t->name[sizeof(t->name)-1]=0;t->rank=d.rank;memcpy(t->dims,d.dims,sizeof(t->dims));t->flags=d.flags;
        if(d.dtype==ED_PRECISION_FP32){t->dtype=d.dtype;t->data_bytes=d.data_bytes;t->data=malloc((size_t)t->data_bytes?t->data_bytes:1u);if(!t->data){s=ED_ERR_MEMORY;goto fail;}memcpy(t->data,file+d.data_offset,(size_t)t->data_bytes);}
        else if(d.dtype==ED_DTYPE_FP16){size_t j,elements=(size_t)d.data_bytes/2u;const uint16_t*half=(const uint16_t*)(file+d.data_offset);float*values;if((d.data_bytes&1u)!=0u||elements>SIZE_MAX/sizeof(float)){s=ED_ERR_FORMAT;goto fail;}values=(float*)malloc(elements*sizeof(float));if(!values){s=ED_ERR_MEMORY;goto fail;}for(j=0;j<elements;++j)values[j]=half_to_float(half[j]);t->dtype=ED_PRECISION_FP32;t->data_bytes=(uint64_t)elements*sizeof(float);t->data=values;}
        else if(d.dtype==ED_DTYPE_INT8_STORAGE){size_t j,elements,group_size;uint32_t groups,g;float*values;const float*scales;const int8_t*quantized;if(d.data_bytes<8u||!tensor_elements(&d,&elements)||elements>SIZE_MAX/sizeof(float)){s=ED_ERR_FORMAT;goto fail;}memcpy(&groups,file+d.data_offset,4u);if(groups==0u||elements%groups!=0u||d.data_bytes!=4u+(uint64_t)groups*4u+(uint64_t)elements){s=ED_ERR_FORMAT;goto fail;}scales=(const float*)(file+d.data_offset+4u);quantized=(const int8_t*)(file+d.data_offset+4u+(size_t)groups*4u);values=(float*)malloc(elements*sizeof(float));if(!values){s=ED_ERR_MEMORY;goto fail;}group_size=elements/groups;for(g=0;g<groups;++g){float scale;memcpy(&scale,scales+g,4u);if(!isfinite(scale)||scale<0.0f){free(values);s=ED_ERR_FORMAT;goto fail;}for(j=0;j<group_size;++j)values[(size_t)g*group_size+j]=(float)quantized[(size_t)g*group_size+j]*scale;}t->dtype=ED_PRECISION_FP32;t->data_bytes=(uint64_t)elements*sizeof(float);t->data=values;}
        else if(d.dtype==ED_DTYPE_INT4_STORAGE){size_t j,elements,group_size,packed_bytes;uint32_t groups,g;float*values;const float*scales;const uint8_t*packed;if(d.data_bytes<8u||!tensor_elements(&d,&elements)||elements>SIZE_MAX/sizeof(float)){s=ED_ERR_FORMAT;goto fail;}memcpy(&groups,file+d.data_offset,4u);packed_bytes=(elements+1u)/2u;if(groups==0u||elements%groups!=0u||d.data_bytes!=4u+(uint64_t)groups*4u+(uint64_t)packed_bytes){s=ED_ERR_FORMAT;goto fail;}scales=(const float*)(file+d.data_offset+4u);packed=file+d.data_offset+4u+(size_t)groups*4u;values=(float*)malloc(elements*sizeof(float));if(!values){s=ED_ERR_MEMORY;goto fail;}group_size=elements/groups;for(g=0;g<groups;++g){float scale;memcpy(&scale,scales+g,4u);if(!isfinite(scale)||scale<0.0f){free(values);s=ED_ERR_FORMAT;goto fail;}for(j=0;j<group_size;++j){size_t index=(size_t)g*group_size+j;uint8_t nibble=(uint8_t)((packed[index/2u]>>((index&1u)*4u))&15u);int8_t q=nibble>=8u?(int8_t)((int)nibble-16):(int8_t)nibble;values[index]=(float)q*scale;}}t->dtype=ED_PRECISION_FP32;t->data_bytes=(uint64_t)elements*sizeof(float);t->data=values;}
        else{s=ED_ERR_UNSUPPORTED;goto fail;}
    }
    free(file);*out=m;return ED_OK;
fail:free(file);ed_model_free(m);return s;
}

ed_status ed_model_save(const ed_model*m,const char*path){
    ed_edm_header h;ed_edm_tensor_disk*d=NULL;uint8_t*file=NULL;uint64_t off,total;uint32_t i;ed_status s;
    if(!m||!path||!m->class_names||(!m->tensors&&m->tensor_count))return ED_ERR_ARGUMENT;
    memset(&h,0,sizeof(h));h.magic=ED_EDM_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.architecture=m->architecture;h.precision=m->precision;h.flags=m->flags;h.tensor_count=m->tensor_count;h.class_count=m->class_count;h.metadata_bytes=m->metadata_bytes;
    h.tensor_table_offset=sizeof(h);h.class_table_offset=h.tensor_table_offset+(uint64_t)m->tensor_count*sizeof(ed_edm_tensor_disk);h.metadata_offset=h.class_table_offset+(uint64_t)m->class_count*ED_CLASS_NAME_BYTES;h.payload_offset=h.metadata_offset+m->metadata_bytes;
    off=h.payload_offset;for(i=0;i<m->tensor_count;++i){off=(off+63u)&~63u;off+=m->tensors[i].data_bytes;}total=off;h.file_bytes=total;if(total>SIZE_MAX)return ED_ERR_MEMORY;
    file=(uint8_t*)calloc((size_t)total,1);if(!file)return ED_ERR_MEMORY;d=(ed_edm_tensor_disk*)(file+h.tensor_table_offset);
    memcpy(file+h.class_table_offset,m->class_names,(size_t)m->class_count*ED_CLASS_NAME_BYTES);if(m->metadata_bytes)memcpy(file+h.metadata_offset,m->metadata,m->metadata_bytes);
    off=h.payload_offset;for(i=0;i<m->tensor_count;++i){const ed_tensor*t=&m->tensors[i];off=(off+63u)&~63u;memcpy(d[i].name,t->name,sizeof(d[i].name));d[i].dtype=t->dtype;d[i].rank=t->rank;memcpy(d[i].dims,t->dims,sizeof(d[i].dims));d[i].data_offset=off;d[i].data_bytes=t->data_bytes;d[i].data_crc32=ed_crc32(t->data,(size_t)t->data_bytes);d[i].flags=t->flags;memcpy(file+off,t->data,(size_t)t->data_bytes);off+=t->data_bytes;}
    memcpy(h.source_sha256,m->source_sha256,32);ed_sha256(file+h.payload_offset,(size_t)(total-h.payload_offset),h.payload_sha256);h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));memcpy(file,&h,sizeof(h));
    s=ed_write_entire_file(path,file,(size_t)total);free(file);return s;
}

ed_status ed_model_save_fp16(const ed_model*m,const char*path){
    ed_edm_header h;ed_edm_tensor_disk*d;uint8_t*file;uint64_t off,total;uint32_t i;ed_status status;
    if(!m||!path||!m->class_names||(!m->tensors&&m->tensor_count))return ED_ERR_ARGUMENT;memset(&h,0,sizeof(h));h.magic=ED_EDM_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.architecture=m->architecture;h.precision=m->precision;h.flags=m->flags;h.tensor_count=m->tensor_count;h.class_count=m->class_count;h.metadata_bytes=m->metadata_bytes;
    h.tensor_table_offset=sizeof(h);h.class_table_offset=h.tensor_table_offset+(uint64_t)m->tensor_count*sizeof(ed_edm_tensor_disk);h.metadata_offset=h.class_table_offset+(uint64_t)m->class_count*ED_CLASS_NAME_BYTES;h.payload_offset=h.metadata_offset+m->metadata_bytes;off=h.payload_offset;
    for(i=0;i<m->tensor_count;++i){const ed_tensor*t=&m->tensors[i];if(t->dtype!=ED_PRECISION_FP32||(t->data_bytes&3u)!=0u)return ED_ERR_UNSUPPORTED;off=(off+63u)&~63u;off+=(t->data_bytes/4u)*2u;}total=off;if(total>SIZE_MAX)return ED_ERR_MEMORY;h.file_bytes=total;file=(uint8_t*)calloc((size_t)total,1);if(!file)return ED_ERR_MEMORY;d=(ed_edm_tensor_disk*)(file+h.tensor_table_offset);
    memcpy(file+h.class_table_offset,m->class_names,(size_t)m->class_count*ED_CLASS_NAME_BYTES);if(m->metadata_bytes)memcpy(file+h.metadata_offset,m->metadata,m->metadata_bytes);off=h.payload_offset;
    for(i=0;i<m->tensor_count;++i){const ed_tensor*t=&m->tensors[i];const float*values=(const float*)t->data;size_t j,elements=(size_t)t->data_bytes/4u;uint16_t*half;off=(off+63u)&~63u;memcpy(d[i].name,t->name,sizeof(d[i].name));d[i].dtype=ED_DTYPE_FP16;d[i].rank=t->rank;memcpy(d[i].dims,t->dims,sizeof(d[i].dims));d[i].data_offset=off;d[i].data_bytes=(uint64_t)elements*2u;d[i].flags=t->flags;half=(uint16_t*)(file+off);for(j=0;j<elements;++j)half[j]=float_to_half(values[j]);d[i].data_crc32=ed_crc32(half,(size_t)d[i].data_bytes);off+=d[i].data_bytes;}
    memcpy(h.source_sha256,m->source_sha256,32);ed_sha256(file+h.payload_offset,(size_t)(total-h.payload_offset),h.payload_sha256);h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));memcpy(file,&h,sizeof(h));status=ed_write_entire_file(path,file,(size_t)total);free(file);return status;
}

ed_status ed_model_save_int8(const ed_model*m,const char*path){
    ed_edm_header h;ed_edm_tensor_disk*d;uint8_t*file;uint64_t off,total;uint32_t i;ed_status status;
    if(!m||!path||!m->class_names||(!m->tensors&&m->tensor_count))return ED_ERR_ARGUMENT;
    memset(&h,0,sizeof(h));h.magic=ED_EDM_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.architecture=m->architecture;h.precision=m->precision;h.flags=m->flags;h.tensor_count=m->tensor_count;h.class_count=m->class_count;h.metadata_bytes=m->metadata_bytes;
    h.tensor_table_offset=sizeof(h);h.class_table_offset=h.tensor_table_offset+(uint64_t)m->tensor_count*sizeof(ed_edm_tensor_disk);h.metadata_offset=h.class_table_offset+(uint64_t)m->class_count*ED_CLASS_NAME_BYTES;h.payload_offset=h.metadata_offset+m->metadata_bytes;off=h.payload_offset;
    for(i=0;i<m->tensor_count;++i){const ed_tensor*t=&m->tensors[i];if(t->dtype!=ED_PRECISION_FP32||(t->data_bytes&3u)!=0u)return ED_ERR_UNSUPPORTED;off=(off+63u)&~63u;if(t->rank==2u||t->rank==4u)off+=4u+(uint64_t)t->dims[0]*4u+t->data_bytes/4u;else if(t->rank==3u)off+=t->data_bytes/2u;else off+=t->data_bytes;}total=off;if(total>SIZE_MAX)return ED_ERR_MEMORY;h.file_bytes=total;file=(uint8_t*)calloc((size_t)total,1);if(!file)return ED_ERR_MEMORY;d=(ed_edm_tensor_disk*)(file+h.tensor_table_offset);
    memcpy(file+h.class_table_offset,m->class_names,(size_t)m->class_count*ED_CLASS_NAME_BYTES);if(m->metadata_bytes)memcpy(file+h.metadata_offset,m->metadata,m->metadata_bytes);off=h.payload_offset;
    for(i=0;i<m->tensor_count;++i){const ed_tensor*t=&m->tensors[i];const float*values=(const float*)t->data;size_t elements=(size_t)t->data_bytes/4u;off=(off+63u)&~63u;memcpy(d[i].name,t->name,sizeof(d[i].name));d[i].rank=t->rank;memcpy(d[i].dims,t->dims,sizeof(d[i].dims));d[i].data_offset=off;d[i].flags=t->flags;if(t->rank<=1u){d[i].dtype=ED_PRECISION_FP32;d[i].data_bytes=t->data_bytes;memcpy(file+off,t->data,(size_t)t->data_bytes);}else if(t->rank==3u){size_t j;uint16_t*half=(uint16_t*)(file+off);d[i].dtype=ED_DTYPE_FP16;d[i].data_bytes=(uint64_t)elements*2u;for(j=0;j<elements;++j)half[j]=float_to_half(values[j]);}else{uint32_t groups=t->dims[0],g;size_t group_size=elements/groups;float*scales=(float*)(file+off+4u);int8_t*quantized=(int8_t*)(file+off+4u+(size_t)groups*4u);d[i].dtype=ED_DTYPE_INT8_STORAGE;d[i].data_bytes=4u+(uint64_t)groups*4u+(uint64_t)elements;memcpy(file+off,&groups,4u);for(g=0;g<groups;++g){size_t j;float maximum=0.0f,scale;for(j=0;j<group_size;++j){float magnitude=fabsf(values[(size_t)g*group_size+j]);if(magnitude>maximum)maximum=magnitude;}scale=maximum>0.0f?maximum/127.0f:0.0f;memcpy(scales+g,&scale,4u);for(j=0;j<group_size;++j){float q=scale>0.0f?values[(size_t)g*group_size+j]/scale:0.0f;long rounded=lroundf(q);if(rounded>127)rounded=127;if(rounded<-127)rounded=-127;quantized[(size_t)g*group_size+j]=(int8_t)rounded;}}}d[i].data_crc32=ed_crc32(file+off,(size_t)d[i].data_bytes);off+=d[i].data_bytes;}
    memcpy(h.source_sha256,m->source_sha256,32);ed_sha256(file+h.payload_offset,(size_t)(total-h.payload_offset),h.payload_sha256);h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));memcpy(file,&h,sizeof(h));status=ed_write_entire_file(path,file,(size_t)total);free(file);return status;
}

ed_status ed_model_save_int4(const ed_model*m,const char*path){
    ed_edm_header h;ed_edm_tensor_disk*d;uint8_t*file;uint64_t off,total;uint32_t i;ed_status status;
    if(!m||!path||!m->class_names||(!m->tensors&&m->tensor_count))return ED_ERR_ARGUMENT;
    memset(&h,0,sizeof(h));h.magic=ED_EDM_MAGIC;h.major=1;h.endian_tag=ED_ENDIAN_TAG;h.header_bytes=sizeof(h);h.architecture=m->architecture;h.precision=m->precision;h.flags=m->flags;h.tensor_count=m->tensor_count;h.class_count=m->class_count;h.metadata_bytes=m->metadata_bytes;
    h.tensor_table_offset=sizeof(h);h.class_table_offset=h.tensor_table_offset+(uint64_t)m->tensor_count*sizeof(ed_edm_tensor_disk);h.metadata_offset=h.class_table_offset+(uint64_t)m->class_count*ED_CLASS_NAME_BYTES;h.payload_offset=h.metadata_offset+m->metadata_bytes;off=h.payload_offset;
    for(i=0;i<m->tensor_count;++i){const ed_tensor*t=&m->tensors[i];uint64_t elements;if(t->dtype!=ED_PRECISION_FP32||(t->data_bytes&3u)!=0u)return ED_ERR_UNSUPPORTED;elements=t->data_bytes/4u;off=(off+63u)&~63u;if((t->rank==2u||t->rank==4u)&&elements>=ED_INT4_MIN_ELEMENTS)off+=4u+(uint64_t)t->dims[0]*4u+(elements+1u)/2u;else if(t->rank==2u||t->rank==4u)off+=4u+(uint64_t)t->dims[0]*4u+elements;else if(t->rank==3u)off+=t->data_bytes/2u;else off+=t->data_bytes;}total=off;if(total>SIZE_MAX)return ED_ERR_MEMORY;h.file_bytes=total;file=(uint8_t*)calloc((size_t)total,1);if(!file)return ED_ERR_MEMORY;d=(ed_edm_tensor_disk*)(file+h.tensor_table_offset);
    memcpy(file+h.class_table_offset,m->class_names,(size_t)m->class_count*ED_CLASS_NAME_BYTES);if(m->metadata_bytes)memcpy(file+h.metadata_offset,m->metadata,m->metadata_bytes);off=h.payload_offset;
    for(i=0;i<m->tensor_count;++i){const ed_tensor*t=&m->tensors[i];const float*values=(const float*)t->data;size_t elements=(size_t)t->data_bytes/4u;off=(off+63u)&~63u;memcpy(d[i].name,t->name,sizeof(d[i].name));d[i].rank=t->rank;memcpy(d[i].dims,t->dims,sizeof(d[i].dims));d[i].data_offset=off;d[i].flags=t->flags;if(t->rank<=1u){d[i].dtype=ED_PRECISION_FP32;d[i].data_bytes=t->data_bytes;memcpy(file+off,t->data,(size_t)t->data_bytes);}else if(t->rank==3u){size_t j;uint16_t*half=(uint16_t*)(file+off);d[i].dtype=ED_DTYPE_FP16;d[i].data_bytes=(uint64_t)elements*2u;for(j=0;j<elements;++j)half[j]=float_to_half(values[j]);}else if(elements<ED_INT4_MIN_ELEMENTS){uint32_t groups=t->dims[0],g;size_t group_size=elements/groups;float*scales=(float*)(file+off+4u);int8_t*quantized=(int8_t*)(file+off+4u+(size_t)groups*4u);d[i].dtype=ED_DTYPE_INT8_STORAGE;d[i].data_bytes=4u+(uint64_t)groups*4u+(uint64_t)elements;memcpy(file+off,&groups,4u);for(g=0;g<groups;++g){size_t j;float maximum=0.0f,scale;for(j=0;j<group_size;++j){float magnitude=fabsf(values[(size_t)g*group_size+j]);if(magnitude>maximum)maximum=magnitude;}scale=maximum>0.0f?maximum/127.0f:0.0f;memcpy(scales+g,&scale,4u);for(j=0;j<group_size;++j){float q=scale>0.0f?values[(size_t)g*group_size+j]/scale:0.0f;long rounded=lroundf(q);if(rounded>127)rounded=127;if(rounded<-127)rounded=-127;quantized[(size_t)g*group_size+j]=(int8_t)rounded;}}}else{uint32_t groups=t->dims[0],g;size_t group_size=elements/groups;float*scales=(float*)(file+off+4u);uint8_t*packed=file+off+4u+(size_t)groups*4u;d[i].dtype=ED_DTYPE_INT4_STORAGE;d[i].data_bytes=4u+(uint64_t)groups*4u+((uint64_t)elements+1u)/2u;memcpy(file+off,&groups,4u);for(g=0;g<groups;++g){size_t j;float maximum=0.0f,scale;for(j=0;j<group_size;++j){float magnitude=fabsf(values[(size_t)g*group_size+j]);if(magnitude>maximum)maximum=magnitude;}scale=maximum>0.0f?maximum/7.0f:0.0f;memcpy(scales+g,&scale,4u);for(j=0;j<group_size;++j){size_t index=(size_t)g*group_size+j;float q=scale>0.0f?values[index]/scale:0.0f;long rounded=lroundf(q);uint8_t nibble;if(rounded>7)rounded=7;if(rounded<-7)rounded=-7;nibble=(uint8_t)((int8_t)rounded)&15u;packed[index/2u]|=(uint8_t)(nibble<<((index&1u)*4u));}}}d[i].data_crc32=ed_crc32(file+off,(size_t)d[i].data_bytes);off+=d[i].data_bytes;}
    memcpy(h.source_sha256,m->source_sha256,32);ed_sha256(file+h.payload_offset,(size_t)(total-h.payload_offset),h.payload_sha256);h.header_crc32=0;h.header_crc32=ed_crc32(&h,sizeof(h));memcpy(file,&h,sizeof(h));status=ed_write_entire_file(path,file,(size_t)total);free(file);return status;
}

ed_status ed_model_save_storage(const ed_model*m,const char*path,ed_storage storage){
    switch(storage){
        case ED_STORAGE_FP32:return ed_model_save(m,path);
        case ED_STORAGE_FP16:return ed_model_save_fp16(m,path);
        case ED_STORAGE_INT8:return ed_model_save_int8(m,path);
        case ED_STORAGE_INT4:return ed_model_save_int4(m,path);
        default:return ED_ERR_ARGUMENT;
    }
}

void ed_model_free(ed_model*m){uint32_t i;if(!m)return;for(i=0;i<m->tensor_count;++i)free(m->tensors[i].data);free(m->tensors);free(m->class_names);free(m->metadata);free(m);}
uint32_t ed_model_class_count(const ed_model*m){return m?m->class_count:0;}
const char*ed_model_class_name(const ed_model*m,uint32_t id){return m&&id<m->class_count?m->class_names[id]:NULL;}

static int ed_ascii_lower(int c){return (c>='A'&&c<='Z')?c-'A'+'a':c;}
static int ed_name_eq_ci(const char *a,const char *b){
    if(!a||!b)return 0;
    while(*a&&*b){if(ed_ascii_lower((unsigned char)*a)!=ed_ascii_lower((unsigned char)*b))return 0;++a;++b;}
    return *a==0&&*b==0;
}
static const char *ed_class_alias(const char *name){
    static const char *pairs[][2]={
        {"ambulance","truck"},{"van","truck"},{"pickup","truck"},{"lorry","truck"},{"semi","truck"},
        {"motorbike","motorcycle"},{"moto","motorcycle"},{"bike","motorcycle"},
        {"taxi","car"},{"sedan","car"},{"suv","car"},{"coupe","car"},{"vehicle","car"},{"auto","car"},
        {"minibus","bus"},{"coach","bus"},
        {NULL,NULL}
    };
    uint32_t i;for(i=0;pairs[i][0];++i)if(ed_name_eq_ci(name,pairs[i][0]))return pairs[i][1];
    return NULL;
}
int ed_find_source_class(const ed_model *m,const char *name){
    uint32_t i;const char *alias;
    if(!m||!name)return -1;
    for(i=0;i<m->class_count;++i)if(ed_name_eq_ci(name,m->class_names[i]))return (int)i;
    alias=ed_class_alias(name);
    if(alias)for(i=0;i<m->class_count;++i)if(ed_name_eq_ci(alias,m->class_names[i]))return (int)i;
    return -1;
}

ed_status ed_model_remap_classes(ed_model*m,const char*const*names,uint32_t count){
    char(*new_names)[ED_CLASS_NAME_BYTES];uint32_t level,c,old_count;if(!m||!names||count==0u||count>ED_MAX_CLASSES)return ED_ERR_ARGUMENT;
    for(c=0;c<count;++c)if(!names[c]||!*names[c]||strlen(names[c])>=ED_CLASS_NAME_BYTES)return ED_ERR_ARGUMENT;
    old_count=m->class_count;new_names=(char(*)[ED_CLASS_NAME_BYTES])calloc(count,ED_CLASS_NAME_BYTES);if(!new_names)return ED_ERR_MEMORY;for(c=0;c<count;++c)strcpy(new_names[c],names[c]);
    for(level=0;level<ED_PICODET_LEVELS;++level){const ed_graph_node*n=&ed_picodet_nodes[ed_picodet_raw_cls_nodes[level]];ed_tensor*wt,*bt;float*nw,*nb,*ow,*ob;uint32_t target,source,ic;
        if(n->op!=ED_OP_CONV||n->input_count<3||n->input[1]>-2||n->input[2]>-2){free(new_names);return ED_ERR_FORMAT;}wt=&m->tensors[-n->input[1]-2];bt=&m->tensors[-n->input[2]-2];
        if(wt->dims[0]!=old_count||wt->dims[3]!=96u||bt->dims[0]!=old_count){free(new_names);return ED_ERR_FORMAT;}nw=(float*)malloc((size_t)count*96u*sizeof(float));nb=(float*)malloc((size_t)count*sizeof(float));if(!nw||!nb){free(nw);free(nb);free(new_names);return ED_ERR_MEMORY;}ow=(float*)wt->data;ob=(float*)bt->data;
        for(target=0;target<count;++target){int match=ed_find_source_class(m,names[target]);if(match>=0){memcpy(nw+(size_t)target*96u,ow+(size_t)(uint32_t)match*96u,96u*sizeof(float));nb[target]=ob[match];}else{for(ic=0;ic<96u;++ic){float sum=0.0f;for(source=0;source<old_count;++source)sum+=ow[(size_t)source*96u+ic];nw[(size_t)target*96u+ic]=sum/(float)old_count;}nb[target]=-4.59511985f;}}
        free(wt->data);free(bt->data);wt->data=nw;wt->dims[0]=count;wt->data_bytes=(uint64_t)count*96u*sizeof(float);bt->data=nb;bt->dims[0]=count;bt->data_bytes=(uint64_t)count*sizeof(float);
    }
    free(m->class_names);m->class_names=new_names;m->class_count=count;return ED_OK;
}

ed_status ed_model_remap_class_indices(ed_model*m,const char*const*names,const uint32_t*source_class,uint32_t count){
    char(*new_names)[ED_CLASS_NAME_BYTES];uint32_t level,c,old_count;
    if(!m||!names||!source_class||count==0u||count>ED_MAX_CLASSES)return ED_ERR_ARGUMENT;
    old_count=m->class_count;for(c=0;c<count;++c)if(!names[c]||!*names[c]||strlen(names[c])>=ED_CLASS_NAME_BYTES||source_class[c]>=old_count)return ED_ERR_ARGUMENT;
    new_names=(char(*)[ED_CLASS_NAME_BYTES])calloc(count,ED_CLASS_NAME_BYTES);if(!new_names)return ED_ERR_MEMORY;for(c=0;c<count;++c)strcpy(new_names[c],names[c]);
    for(level=0;level<ED_PICODET_LEVELS;++level){const ed_graph_node*n=&ed_picodet_nodes[ed_picodet_raw_cls_nodes[level]];ed_tensor*wt,*bt;float*nw,*nb,*ow,*ob;uint32_t target;
        if(n->op!=ED_OP_CONV||n->input_count<3||n->input[1]>-2||n->input[2]>-2){free(new_names);return ED_ERR_FORMAT;}wt=&m->tensors[-n->input[1]-2];bt=&m->tensors[-n->input[2]-2];
        if(wt->dims[0]!=old_count||wt->dims[3]!=96u||bt->dims[0]!=old_count){free(new_names);return ED_ERR_FORMAT;}nw=(float*)malloc((size_t)count*96u*sizeof(float));nb=(float*)malloc((size_t)count*sizeof(float));if(!nw||!nb){free(nw);free(nb);free(new_names);return ED_ERR_MEMORY;}ow=(float*)wt->data;ob=(float*)bt->data;
        for(target=0;target<count;++target){uint32_t source=source_class[target];memcpy(nw+(size_t)target*96u,ow+(size_t)source*96u,96u*sizeof(float));nb[target]=ob[source];}
        free(wt->data);free(bt->data);wt->data=nw;wt->dims[0]=count;wt->data_bytes=(uint64_t)count*96u*sizeof(float);bt->data=nb;bt->dims[0]=count;bt->data_bytes=(uint64_t)count*sizeof(float);
    }
    free(m->class_names);m->class_names=new_names;m->class_count=count;return ED_OK;
}

ed_status ed_model_remap_class_blends(ed_model*m,const char*const*names,const uint32_t*source_class,const uint32_t*source_count,uint32_t count){
    char(*new_names)[ED_CLASS_NAME_BYTES];uint32_t level,c,old_count;
    if(!m||!names||!source_class||!source_count||count==0u||count>ED_MAX_CLASSES)return ED_ERR_ARGUMENT;
    old_count=m->class_count;
    for(c=0;c<count;++c){uint32_t k;if(!names[c]||!*names[c]||strlen(names[c])>=ED_CLASS_NAME_BYTES||source_count[c]==0u||source_count[c]>4u)return ED_ERR_ARGUMENT;for(k=0;k<source_count[c];++k)if(source_class[(size_t)c*4u+k]>=old_count)return ED_ERR_ARGUMENT;}
    new_names=(char(*)[ED_CLASS_NAME_BYTES])calloc(count,ED_CLASS_NAME_BYTES);if(!new_names)return ED_ERR_MEMORY;for(c=0;c<count;++c)strcpy(new_names[c],names[c]);
    for(level=0;level<ED_PICODET_LEVELS;++level){const ed_graph_node*n=&ed_picodet_nodes[ed_picodet_raw_cls_nodes[level]];ed_tensor*wt,*bt;float*nw,*nb,*ow,*ob;uint32_t target;
        if(n->op!=ED_OP_CONV||n->input_count<3||n->input[1]>-2||n->input[2]>-2){free(new_names);return ED_ERR_FORMAT;}wt=&m->tensors[-n->input[1]-2];bt=&m->tensors[-n->input[2]-2];
        if(wt->dims[0]!=old_count||wt->dims[3]!=96u||bt->dims[0]!=old_count){free(new_names);return ED_ERR_FORMAT;}nw=(float*)malloc((size_t)count*96u*sizeof(float));nb=(float*)malloc((size_t)count*sizeof(float));if(!nw||!nb){free(nw);free(nb);free(new_names);return ED_ERR_MEMORY;}ow=(float*)wt->data;ob=(float*)bt->data;
        for(target=0;target<count;++target){uint32_t ic,k;float scale=1.0f/(float)source_count[target];for(ic=0;ic<96u;++ic){float sum=0.0f;for(k=0;k<source_count[target];++k)sum+=ow[(size_t)source_class[(size_t)target*4u+k]*96u+ic];nw[(size_t)target*96u+ic]=sum*scale;}nb[target]=0.0f;for(k=0;k<source_count[target];++k)nb[target]+=ob[source_class[(size_t)target*4u+k]]*scale;}
        free(wt->data);free(bt->data);wt->data=nw;wt->dims[0]=count;wt->data_bytes=(uint64_t)count*96u*sizeof(float);bt->data=nb;bt->dims[0]=count;bt->data_bytes=(uint64_t)count*sizeof(float);
    }
    free(m->class_names);m->class_names=new_names;m->class_count=count;return ED_OK;
}

ed_status ed_model_remap_class_heads(ed_model*m,const char*const*names,uint32_t count,const float*weights,const float*biases){
    char(*new_names)[ED_CLASS_NAME_BYTES];uint32_t level,c,old_count;
    if(!m||!names||!weights||!biases||count==0u||count>ED_MAX_CLASSES)return ED_ERR_ARGUMENT;
    old_count=m->class_count;
    for(c=0;c<count;++c)if(!names[c]||!*names[c]||strlen(names[c])>=ED_CLASS_NAME_BYTES)return ED_ERR_ARGUMENT;
    new_names=(char(*)[ED_CLASS_NAME_BYTES])calloc(count,ED_CLASS_NAME_BYTES);if(!new_names)return ED_ERR_MEMORY;
    for(c=0;c<count;++c)strcpy(new_names[c],names[c]);
    for(level=0;level<ED_PICODET_LEVELS;++level){
        const ed_graph_node*n=&ed_picodet_nodes[ed_picodet_raw_cls_nodes[level]];ed_tensor*wt,*bt;float*nw,*nb;
        if(n->op!=ED_OP_CONV||n->input_count<3||n->input[1]>-2||n->input[2]>-2){free(new_names);return ED_ERR_FORMAT;}
        wt=&m->tensors[-n->input[1]-2];bt=&m->tensors[-n->input[2]-2];
        if(wt->dims[0]!=old_count||wt->dims[3]!=96u||bt->dims[0]!=old_count){free(new_names);return ED_ERR_FORMAT;}
        nw=(float*)malloc((size_t)count*96u*sizeof(float));nb=(float*)malloc((size_t)count*sizeof(float));
        if(!nw||!nb){free(nw);free(nb);free(new_names);return ED_ERR_MEMORY;}
        memcpy(nw,weights+(size_t)level*count*96u,(size_t)count*96u*sizeof(float));
        memcpy(nb,biases+(size_t)level*count,(size_t)count*sizeof(float));
        free(wt->data);free(bt->data);wt->data=nw;wt->dims[0]=count;wt->data_bytes=(uint64_t)count*96u*sizeof(float);
        bt->data=nb;bt->dims[0]=count;bt->data_bytes=(uint64_t)count*sizeof(float);
    }
    free(m->class_names);m->class_names=new_names;m->class_count=count;return ED_OK;
}

ed_status ed_predict(const ed_model*m,const ed_image*im,float st,float nt,ed_detection_list*out){
    float *input=NULL;ed_activation*a=NULL;ed_detection*candidates=NULL;size_t cand_n=0,cand_cap=4096;uint32_t y,x,c,l;ed_status s;
    if(!m||!im||!out||!out->items||!im->rgb||!im->width||!im->height||st<0||st>1||nt<0||nt>1)return ED_ERR_ARGUMENT;
    if(ed_runtime_compute()==ED_COMPUTE_FPGA_MODEL)return ed_hw_predict(m,im,st,nt,out);
    if(m->architecture!=ED_ARCH_PICODET_S_320||m->precision!=ED_PRECISION_FP32)return ED_ERR_UNSUPPORTED;
    candidates=(ed_detection*)malloc(cand_cap*sizeof(*candidates));if(!candidates)return ED_ERR_MEMORY;
    s=ed_prepare_input_320(im,&input);if(s!=ED_OK){free(candidates);return s;}
    s=ed_graph_execute(m,input,&a);if(s!=ED_OK)goto done;
    for(l=0;l<ED_PICODET_LEVELS;++l){const ed_activation*sc=&a[ed_picodet_cls_nodes[l]],*rg=&a[ed_picodet_reg_nodes[l]];float*dist=(float*)malloc((size_t)rg->h*rg->w*4u*sizeof(float));uint32_t locs=rg->h*rg->w;
        if(!dist){s=ED_ERR_MEMORY;goto done;}ed_dfl_decode_f32(rg->data,locs,7u,dist);
        for(y=0;y<rg->h;++y)for(x=0;x<rg->w;++x)for(c=0;c<m->class_count;++c){float score=sc->data[((size_t)y*rg->w+x)*m->class_count+c];if(score>=st&&cand_n<cand_cap){const float*d=dist+((size_t)y*rg->w+x)*4u;float stride=(float)ed_picodet_strides[l],cx=((float)x+0.5f)*stride,cy=((float)y+0.5f)*stride;ed_detection*z=&candidates[cand_n++];z->x1=(cx-d[0]*stride)*(float)im->width/320.0f;z->y1=(cy-d[1]*stride)*(float)im->height/320.0f;z->x2=(cx+d[2]*stride)*(float)im->width/320.0f;z->y2=(cy+d[3]*stride)*(float)im->height/320.0f;z->score=score;z->class_id=c;}}
        free(dist);
    }
    cand_n=ed_nms(candidates,cand_n,nt);
    /* Ground-truth boxes and the visible image domain are both bounded by the
       image.  Clipping after NMS leaves suppression and ranking unchanged, and
       can only remove out-of-frame area before IoU is measured. */
    {size_t i;for(i=0;i<cand_n;++i){
        candidates[i].x1=ed_clampf(candidates[i].x1,0.0f,(float)im->width);
        candidates[i].y1=ed_clampf(candidates[i].y1,0.0f,(float)im->height);
        candidates[i].x2=ed_clampf(candidates[i].x2,0.0f,(float)im->width);
        candidates[i].y2=ed_clampf(candidates[i].y2,0.0f,(float)im->height);
    }}
    out->count=cand_n<out->capacity?cand_n:out->capacity;memcpy(out->items,candidates,out->count*sizeof(*out->items));s=ED_OK;
done:ed_graph_activations_free(a);free(input);free(candidates);return s;
}
ed_status ed_predict_auto(const ed_model*m,const ed_image*im,float st,float nt,ed_detection_list*out){
    float ar;if(!m||!im||!im->width||!im->height)return ED_ERR_ARGUMENT;
    ar=(float)im->width/(float)im->height;
    if(ar>=1.6f)return ed_predict_tiled(m,im,st,nt,2u,1u,0.06f,0,out);
    if(ar*1.6f<=1.0f)return ed_predict_tiled(m,im,st,nt,1u,2u,0.06f,0,out);
    return ed_predict(m,im,st,nt,out);
}
ed_status ed_predict_tiled(const ed_model*m,const ed_image*im,float st,float nt,uint32_t tiles_x,uint32_t tiles_y,float overlap,int include_full,ed_detection_list*out){ed_detection*all,*temporary;size_t capacity,count=0;uint32_t tx,ty;ed_status status=ED_OK;if(!m||!im||!out||!out->items||!im->rgb||tiles_x==0u||tiles_y==0u||tiles_x>4u||tiles_y>4u||(tiles_x==1u&&tiles_y==1u)||overlap<0.0f||overlap>=0.5f)return ED_ERR_ARGUMENT;capacity=((size_t)tiles_x*tiles_y+(include_full?1u:0u))*ED_MAX_DETECTIONS;all=(ed_detection*)malloc(capacity*sizeof(*all));temporary=(ed_detection*)malloc(ED_MAX_DETECTIONS*sizeof(*temporary));if(!all||!temporary){free(all);free(temporary);return ED_ERR_MEMORY;}if(include_full){ed_detection_list list={temporary,ED_MAX_DETECTIONS,0};status=ed_predict(m,im,st,nt,&list);if(status!=ED_OK)goto done;memcpy(all+count,temporary,list.count*sizeof(*temporary));count+=list.count;}for(ty=0;ty<tiles_y;++ty)for(tx=0;tx<tiles_x;++tx){float xden=(float)tiles_x-(float)(tiles_x-1u)*overlap,yden=(float)tiles_y-(float)(tiles_y-1u)*overlap;uint32_t tile_w=(uint32_t)ceilf((float)im->width/xden),tile_h=(uint32_t)ceilf((float)im->height/yden),x0=tiles_x==1u?0u:(uint32_t)(((uint64_t)tx*(im->width-tile_w))/(tiles_x-1u)),y0=tiles_y==1u?0u:(uint32_t)(((uint64_t)ty*(im->height-tile_h))/(tiles_y-1u)),x1=x0+tile_w,y1=y0+tile_h,i;if(x1>im->width)x1=im->width;if(y1>im->height)y1=im->height;ed_image tile={im->rgb+(size_t)y0*im->stride_bytes+(size_t)x0*3u,x1-x0,y1-y0,im->stride_bytes};ed_detection_list list={temporary,ED_MAX_DETECTIONS,0};status=ed_predict(m,&tile,st,nt,&list);if(status!=ED_OK)goto done;for(i=0;i<list.count&&count<capacity;++i){temporary[i].x1+=(float)x0;temporary[i].x2+=(float)x0;temporary[i].y1+=(float)y0;temporary[i].y2+=(float)y0;all[count++]=temporary[i];}}count=ed_nms(all,count,nt);out->count=count<out->capacity?count:out->capacity;memcpy(out->items,all,out->count*sizeof(*out->items));
done:free(all);free(temporary);return status;}


ed_status ed_prepare_input_320(const ed_image*im,float**out){
    float *input;uint32_t y,x,c;static const float mean[3]={0.485f,0.456f,0.406f},std[3]={0.229f,0.224f,0.225f};
    if(!im||!out||!im->rgb||!im->width||!im->height)return ED_ERR_ARGUMENT;*out=NULL;
    input=(float*)malloc(320u*320u*3u*sizeof(float));if(!input)return ED_ERR_MEMORY;
    for(y=0;y<320u;++y){float sy=((float)y+0.5f)*(float)im->height/320.0f-0.5f;int y0=(int)floorf(sy);float fy=sy-(float)y0;int y1=y0+1;if(y0<0)y0=0;if(y1>=(int)im->height)y1=(int)im->height-1;
        for(x=0;x<320u;++x){float sx=((float)x+0.5f)*(float)im->width/320.0f-0.5f;int x0=(int)floorf(sx);float fx=sx-(float)x0;int x1=x0+1;if(x0<0)x0=0;if(x1>=(int)im->width)x1=(int)im->width-1;
            const uint8_t*p00=im->rgb+(size_t)y0*im->stride_bytes+(size_t)x0*3u,*p01=im->rgb+(size_t)y0*im->stride_bytes+(size_t)x1*3u,*p10=im->rgb+(size_t)y1*im->stride_bytes+(size_t)x0*3u,*p11=im->rgb+(size_t)y1*im->stride_bytes+(size_t)x1*3u;
            for(c=0;c<3u;++c){float v=(1-fy)*((1-fx)*p00[c]+fx*p01[c])+fy*((1-fx)*p10[c]+fx*p11[c]);input[((size_t)y*320u+x)*3u+c]=(v/255.0f-mean[c])/std[c];}
        }}*out=input;return ED_OK;
}
