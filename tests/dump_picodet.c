#include "edgedet.h"
#include "ed_graph.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char**argv){ed_model*m=NULL;ed_activation*a=NULL;float*input;FILE*f;ed_status s;uint32_t i,c;const float mean[3]={.485f,.456f,.406f},std[3]={.229f,.224f,.225f};if(argc!=3)return 2;input=(float*)malloc(320u*320u*3u*sizeof(float));if(!input)return 1;for(i=0;i<320u*320u;++i)for(c=0;c<3u;++c)input[(size_t)i*3u+c]=(128.0f/255.0f-mean[c])/std[c];if((s=ed_model_load(argv[1],&m))!=ED_OK||(s=ed_graph_execute(m,input,&a))!=ED_OK){fprintf(stderr,"%s\n",ed_status_string(s));free(input);ed_model_free(m);return 1;}f=fopen(argv[2],"wb");if(!f)return 1;for(i=0;i<ED_PICODET_LEVELS;++i){ed_activation*x=&a[ed_picodet_cls_nodes[i]];fwrite(&x->h,4,1,f);fwrite(&x->w,4,1,f);fwrite(&x->c,4,1,f);fwrite(x->data,sizeof(float),(size_t)x->h*x->w*x->c,f);x=&a[ed_picodet_reg_nodes[i]];fwrite(&x->h,4,1,f);fwrite(&x->w,4,1,f);fwrite(&x->c,4,1,f);fwrite(x->data,sizeof(float),(size_t)x->h*x->w*x->c,f);}fclose(f);ed_graph_activations_free(a);ed_model_free(m);free(input);return 0;}
