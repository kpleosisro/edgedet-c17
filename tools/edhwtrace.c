#include "ed_hw.h"
#include "ed_internal.h"
#include "ed_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p,0755)
#endif

static const char *arg(int n,char **v,const char *k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
static const char *op_name(uint16_t op){
    switch(op){
    case ED_OP_CONV:return "CONV";case ED_OP_ADD:return "ADD";case ED_OP_CLIP:return "CLIP";
    case ED_OP_MUL:return "MUL";case ED_OP_DIV:return "DIV";case ED_OP_GLOBAL_AVG:return "GAP";
    case ED_OP_RELU:return "RELU";case ED_OP_HARD_SIGMOID:return "HSIG";
    case ED_OP_RESIZE_NEAREST:return "NEAR";case ED_OP_CONCAT_CHANNEL:return "CAT";
    case ED_OP_SIGMOID:return "SIGM";case ED_OP_SQRT:return "SQRT";default:return "OP";}
}

int main(int argc,char **argv){
    const char *mp=arg(argc,argv,"--model"),*ip=arg(argc,argv,"--input"),*od=arg(argc,argv,"--output-dir");
    ed_model *m=NULL;uint8_t *enc=NULL,*rgb=NULL;size_t enc_n=0;ed_image im;ed_detection det[ED_MAX_DETECTIONS];
    ed_detection_list list={det,ED_MAX_DETECTIONS,0};FILE *mf;uint32_t i;char path[512];
    if(!mp||!ip||!od){fprintf(stderr,"usage: edhwtrace --model M.edm --input IMAGE --output-dir DIR\n");return 2;}
    if(ed_read_entire_file(ip,&enc,&enc_n)!=ED_OK||ed_decode_image(enc,enc_n,&rgb,&im.width,&im.height)!=ED_OK){
        fprintf(stderr,"edhwtrace: image failed\n");free(enc);return 1;
    }
    free(enc);im.rgb=rgb;im.stride_bytes=im.width*3u;
    if(ed_model_load(mp,&m)!=ED_OK){free(rgb);return 1;}
    MKDIR(od);
    (void)ed_runtime_set_compute(ED_COMPUTE_FPGA_MODEL);
    if(ed_predict(m,&im,0.25f,0.42f,&list)!=ED_OK){fprintf(stderr,"edhwtrace: predict failed\n");free(rgb);ed_model_free(m);return 1;}
    snprintf(path,sizeof(path),"%s/manifest.json",od);
    mf=fopen(path,"wb");if(!mf){free(rgb);ed_model_free(m);return 1;}
    fprintf(mf,"{\n  \"arch\":\"picodet_s_320\",\n  \"nodes\":[\n");
    for(i=0;i<ed_picodet_node_count;++i){
        const ed_graph_node *n=&ed_picodet_nodes[i];
        fprintf(mf,"    {\"id\":%u,\"op\":\"%s\",\"inputs\":[%d,%d,%d],\"k\":%u,\"s\":%u,\"p\":%u,\"g\":%u}%s\n",
                i,op_name(n->op),n->input[0],n->input[1],n->input[2],n->kernel,n->stride,n->padding,n->groups,
                i+1u<ed_picodet_node_count?",":"");
    }
    fprintf(mf,"  ],\n  \"detections\":[\n");
    for(i=0;i<list.count;++i){
        fprintf(mf,"    {\"class\":\"%s\",\"score\":%.6f,\"box\":[%.2f,%.2f,%.2f,%.2f]}%s\n",
                ed_model_class_name(m,det[i].class_id),det[i].score,det[i].x1,det[i].y1,det[i].x2,det[i].y2,
                i+1<list.count?",":"");
    }
    fprintf(mf,"  ]\n}\n");fclose(mf);
    snprintf(path,sizeof(path),"%s/mac_tile_stimulus.bin",od);
    {FILE *sf=fopen(path,"wb");int16_t in[9]={1,2,3,4,5,6,7,8,9};int8_t w[9]={1,0,0,0,1,0,0,0,1};int32_t b=0;
        if(sf){fwrite(in,2,9,sf);fwrite(w,1,9,sf);fwrite(&b,4,1,sf);fclose(sf);}}
    printf("wrote %s (%u detections)\n",od,(unsigned)list.count);
    free(rgb);ed_model_free(m);return 0;
}
