#include "edgedet.h"
#include <stdio.h>
#include <string.h>
static const char*arg(int n,char**v,const char*k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
int main(int argc,char**argv){const char*input=arg(argc,argv,"--model"),*output=arg(argc,argv,"--output"),*precision=arg(argc,argv,"--precision");ed_model*m=NULL;ed_status s;if(!input||!output){fprintf(stderr,"usage: edcompact --model source.edm --output compact.edm [--precision fp16|int8|int4]\n");return 2;}s=ed_model_load(input,&m);if(s==ED_OK){if(!precision||strcmp(precision,"fp16")==0)s=ed_model_save_fp16(m,output);else if(strcmp(precision,"int8")==0)s=ed_model_save_int8(m,output);else if(strcmp(precision,"int4")==0)s=ed_model_save_int4(m,output);else s=ED_ERR_ARGUMENT;}if(s!=ED_OK)fprintf(stderr,"compact: %s\n",ed_status_string(s));ed_model_free(m);return s==ED_OK?0:1;}
