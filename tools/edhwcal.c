#include "ed_internal.h"
#include "ed_graph.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg(int n,char **v,const char *k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}

int main(int argc,char **argv){
    const char *dp=arg(argc,argv,"--dataset"),*mp=arg(argc,argv,"--model"),*op=arg(argc,argv,"--output");
    const char *maxs=arg(argc,argv,"--max");
    ed_model *m=NULL;ed_dataset *d=NULL;uint32_t i,n,use,t;float *mx=NULL;FILE *f;
    if(!dp||!mp||!op){fprintf(stderr,"usage: edhwcal --dataset D.edb --model M.edm --output scales.edq [--max N]\n");return 2;}
    if(ed_dataset_load(dp,&d)!=ED_OK||ed_model_load(mp,&m)!=ED_OK){fprintf(stderr,"edhwcal: load failed\n");ed_dataset_free(d);ed_model_free(m);return 1;}
    mx=(float*)calloc(ed_picodet_node_count,sizeof(float));if(!mx)return 1;
    n=d->header.record_count;use=maxs?(uint32_t)strtoul(maxs,NULL,10):8u;if(use>n)use=n;
    for(i=0;i<use;++i){
        const uint8_t *enc;const ed_edb_annotation_disk *ann;uint32_t enc_n,ann_n,w,h;uint8_t *rgb=NULL;
        float *input=NULL;ed_activation *a=NULL;ed_image im;
        if(ed_dataset_record(d,i,&enc,&enc_n,&ann,&ann_n,&w,&h)!=ED_OK)continue;
        if(ed_decode_image(enc,enc_n,&rgb,&im.width,&im.height)!=ED_OK)continue;
        im.rgb=rgb;im.stride_bytes=im.width*3u;
        if(ed_prepare_input_320(&im,&input)==ED_OK&&ed_graph_execute(m,input,&a)==ED_OK){
            for(t=0;t<ed_picodet_node_count;++t){
                size_t z,N=(size_t)a[t].h*a[t].w*a[t].c;float local=0.0f;
                for(z=0;z<N;++z){float v=a[t].data[z];if(v<0)v=-v;if(v>local)local=v;}
                if(local>mx[t])mx[t]=local;
            }
        }
        ed_graph_activations_free(a);free(input);free(rgb);
    }
    f=fopen(op,"wb");if(!f){free(mx);ed_dataset_free(d);ed_model_free(m);return 1;}
    fwrite("EDQ1",1,4,f);fwrite(&ed_picodet_node_count,4,1,f);
    fwrite(mx,4,ed_picodet_node_count,f);fclose(f);
    printf("wrote %s nodes=%u images=%u\n",op,ed_picodet_node_count,use);
    free(mx);ed_dataset_free(d);ed_model_free(m);return 0;
}
