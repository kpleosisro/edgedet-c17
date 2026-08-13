#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char*arg(int n,char**v,const char*k){int i;for(i=1;i+1<n;++i)if(strcmp(v[i],k)==0)return v[i+1];return NULL;}
static int has_flag(int n,char**v,const char*k){int i;for(i=1;i<n;++i)if(strcmp(v[i],k)==0)return 1;return 0;}
static int cmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y;}
int main(int argc,char**argv){
    const char*mp=arg(argc,argv,"--model"),*ip=arg(argc,argv,"--input"),*dp=arg(argc,argv,"--dataset"),*tp=arg(argc,argv,"--threads"),*rp=arg(argc,argv,"--runs"),*xp=arg(argc,argv,"--tiles-x"),*yp=arg(argc,argv,"--tiles-y"),*np=arg(argc,argv,"--nms-threshold"),*op=arg(argc,argv,"--tile-overlap"),*wp=arg(argc,argv,"--watts"),*pl=arg(argc,argv,"--power-limit"),*cw=arg(argc,argv,"--core-watts");
    uint32_t runs=rp?(uint32_t)strtoul(rp,NULL,10):30u,tiles_x=1u,tiles_y=1u,i,forwards=1u;
    float nms=np?strtof(np,NULL):0.6f,overlap=op?strtof(op,NULL):0.06f,watts=wp?strtof(wp,NULL):1.0f;
    uint8_t*encoded=NULL,*rgb=NULL;size_t encoded_n=0;ed_image image;ed_model*m=NULL;ed_dataset*d=NULL;
    ed_detection found[ED_MAX_DETECTIONS];ed_detection_list list={found,ED_MAX_DETECTIONS,0};
    double*times,*avgp;ed_status s;int low_power=has_flag(argc,argv,"--low-power");float power_limit=pl?strtof(pl,NULL):0.0f;
    if(!mp||(!ip&&!dp)||!runs){fprintf(stderr,"usage: edbench --model M.edm (--input IMAGE|--dataset D.edb) [--runs N] [--threads N] [--tiles-x auto|N --tiles-y N] [--power-limit W] [--core-watts W] [--low-power]\n");return 2;}
    if(xp&&strcmp(xp,"auto")==0){tiles_x=0u;tiles_y=0u;}else{if(xp)tiles_x=(uint32_t)strtoul(xp,NULL,10);if(yp)tiles_y=(uint32_t)strtoul(yp,NULL,10);}
    if((low_power||power_limit>0.0f)&&!tp)tp="1";
    if(tp&&(s=ed_runtime_set_threads((uint32_t)strtoul(tp,NULL,10)))!=ED_OK)return 2;
    if(cw&&ed_runtime_set_busy_core_watts(strtof(cw,NULL))!=ED_OK)return 2;
    if(power_limit>0.0f&&ed_runtime_set_power_limit_w(power_limit)!=ED_OK)return 2;
    if(ip){if((s=ed_read_entire_file(ip,&encoded,&encoded_n))!=ED_OK||(s=ed_decode_image(encoded,encoded_n,&rgb,&image.width,&image.height))!=ED_OK){fprintf(stderr,"bench: %s\n",ed_status_string(s));return 1;}free(encoded);}
    else{
        const uint8_t*bytes;const ed_edb_annotation_disk*a;uint32_t bn,an,w,h;
        if((s=ed_dataset_load(dp,&d))!=ED_OK||(s=ed_dataset_record(d,0,&bytes,&bn,&a,&an,&w,&h))!=ED_OK||(s=ed_decode_image(bytes,bn,&rgb,&image.width,&image.height))!=ED_OK){fprintf(stderr,"bench: %s\n",ed_status_string(s));ed_dataset_free(d);return 1;}
        ed_dataset_free(d);
    }
    if((s=ed_model_load(mp,&m))!=ED_OK){fprintf(stderr,"bench: %s\n",ed_status_string(s));free(rgb);return 1;}
    image.rgb=rgb;image.stride_bytes=image.width*3u;
    times=(double*)malloc(runs*sizeof(*times));avgp=(double*)malloc(runs*sizeof(*avgp));if(!times||!avgp)return 1;
    for(i=0;i<runs+2u;++i){
        uint64_t begin=ed_monotonic_ns();list.count=0;
        if(tiles_x==0u&&tiles_y==0u)s=ed_predict_auto(m,&image,0.25f,nms,&list);
        else if(tiles_x*tiles_y>1u)s=ed_predict_tiled(m,&image,0.25f,nms,tiles_x,tiles_y,overlap,0,&list);
        else s=ed_predict(m,&image,0.25f,nms,&list);
        if(s!=ED_OK)return 1;
        if(i>=2u){times[i-2u]=(double)(ed_monotonic_ns()-begin)/1e6;avgp[i-2u]=ed_runtime_last_average_w();}
    }
    qsort(times,runs,sizeof(*times),cmp);qsort(avgp,runs,sizeof(*avgp),cmp);
    if(tiles_x==0u&&tiles_y==0u){float ar=(float)image.width/(float)image.height;forwards=(ar>=1.6f||ar*1.6f<=1.0f)?2u:1u;}
    else forwards=tiles_x*tiles_y?tiles_x*tiles_y:1u;
    printf("threads=%u runs=%u tiles=%ux%u power_limit_w=%.2f core_watts=%.2f median_ms=%.3f p95_ms=%.3f median_avg_w=%.3f last_sleep_ms=%.3f flops_per_fwd=%llu forwards=%u est_energy_at_%.2fW_mJ=%.3f\n",
        ed_runtime_threads(),runs,tiles_x,tiles_y,ed_runtime_power_limit_w(),ed_runtime_busy_core_watts(),
        times[runs/2u],times[(runs*95u-1u)/100u],avgp[runs/2u],ed_runtime_last_sleep_ms(),
        (unsigned long long)ed_picodet_forward_flops(),forwards,watts,ed_estimate_energy_mj(forwards,watts,(float)times[runs/2u]));
    free(times);free(avgp);free(rgb);ed_model_free(m);return 0;
}
