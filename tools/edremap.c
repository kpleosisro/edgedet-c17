#include "edgedet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg(int argc,char **argv,const char *key){int i;for(i=1;i+1<argc;++i)if(strcmp(argv[i],key)==0)return argv[i+1];return NULL;}
int main(int argc,char **argv){
    const char *input=arg(argc,argv,"--model"),*class_text=arg(argc,argv,"--classes"),*output=arg(argc,argv,"--output");ed_model *model=NULL;ed_status status;char *copy,*token;const char *names[ED_MAX_CLASSES];uint32_t count=0;
    if(!input||!class_text||!output){fprintf(stderr,"usage: edremap --model source.edm --classes car,bus --output target.edm\n");return 2;}
    copy=(char*)malloc(strlen(class_text)+1u);if(!copy)return 1;strcpy(copy,class_text);token=strtok(copy,",");while(token&&count<ED_MAX_CLASSES){names[count++]=token;token=strtok(NULL,",");}
    if(token||count==0u){free(copy);return 2;}status=ed_model_load(input,&model);if(status==ED_OK)status=ed_model_remap_classes(model,names,count);if(status==ED_OK)status=ed_model_save(model,output);
    if(status!=ED_OK)fprintf(stderr,"remap: %s\n",ed_status_string(status));ed_model_free(model);free(copy);return status==ED_OK?0:1;
}
