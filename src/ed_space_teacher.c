#include "ed_space.h"
#include <string.h>

#define ED_SPACE_WHITE 180u
#define ED_SPACE_MIN_AREA 40u
#define ED_SPACE_MAX_AREA 12000u

typedef struct {
    uint16_t x0,y0,x1,y1;
    uint32_t area;
} blob;

/* 4-connected flood fill with a static stack. FPGA: same compare/add walk. */
static uint32_t flood(uint8_t *seen,const uint8_t *gray,int sx,int sy,blob *b){
    static int16_t xs[ED_SPACE_SIZE*ED_SPACE_SIZE],ys[ED_SPACE_SIZE*ED_SPACE_SIZE];
    uint32_t sp=0,area=0;int minx=sx,miny=sy,maxx=sx,maxy=sy;
    xs[sp]=(int16_t)sx;ys[sp]=(int16_t)sy;++sp;
    while(sp){
        int x,y;--sp;x=xs[sp];y=ys[sp];
        if(x<0||y<0||x>=(int)ED_SPACE_SIZE||y>=(int)ED_SPACE_SIZE)continue;
        if(seen[(size_t)y*ED_SPACE_SIZE+(uint32_t)x])continue;
        if(gray[(size_t)y*ED_SPACE_SIZE+(uint32_t)x]<ED_SPACE_WHITE)continue;
        seen[(size_t)y*ED_SPACE_SIZE+(uint32_t)x]=1u;++area;
        if(x<minx)minx=x;if(y<miny)miny=y;if(x>maxx)maxx=x;if(y>maxy)maxy=y;
        if(sp+4u<ED_SPACE_SIZE*ED_SPACE_SIZE){
            xs[sp]=(int16_t)(x+1);ys[sp]=(int16_t)y;++sp;
            xs[sp]=(int16_t)(x-1);ys[sp]=(int16_t)y;++sp;
            xs[sp]=(int16_t)x;ys[sp]=(int16_t)(y+1);++sp;
            xs[sp]=(int16_t)x;ys[sp]=(int16_t)(y-1);++sp;
        }
    }
    b->x0=(uint16_t)minx;b->y0=(uint16_t)miny;b->x1=(uint16_t)maxx;b->y1=(uint16_t)maxy;b->area=area;
    return area;
}

ed_status ed_space_teacher(ed_space_frame *frame){
    static uint8_t seen[ED_SPACE_SIZE*ED_SPACE_SIZE];
    blob found[32];uint32_t n=0,x,y,i,j;
    if(!frame)return ED_ERR_ARGUMENT;
    memset(seen,0,sizeof(seen));
    frame->teacher_count=0;
    for(y=0;y<ED_SPACE_SIZE;++y)for(x=0;x<ED_SPACE_SIZE;++x){
        blob b;uint32_t w,h,area;
        if(seen[(size_t)y*ED_SPACE_SIZE+x])continue;
        if(frame->gray[(size_t)y*ED_SPACE_SIZE+x]<ED_SPACE_WHITE)continue;
        area=flood(seen,frame->gray,(int)x,(int)y,&b);
        w=(uint32_t)(b.x1>=b.x0?b.x1-b.x0+1u:0u);
        h=(uint32_t)(b.y1>=b.y0?b.y1-b.y0+1u:0u);
        if(area<ED_SPACE_MIN_AREA||area>ED_SPACE_MAX_AREA)continue;
        if(w<6u||h<6u||w>120u||h>120u)continue;
        /* Drop the giant ring: very large or very round-and-wide. */
        if(w>160u||h>160u)continue;
        if(n<32u)found[n++]=b;
    }
    /* Left-to-right. */
    for(i=0;i<n;++i)for(j=i+1;j<n;++j)if(found[j].x0<found[i].x0){blob t=found[i];found[i]=found[j];found[j]=t;}
    for(i=0;i<n&&frame->teacher_count<ED_SPACE_TEACHER_MAX;++i){
        ed_space_box *o=&frame->teacher[frame->teacher_count++];
        o->x1=(float)found[i].x0;o->y1=(float)found[i].y0;
        o->x2=(float)found[i].x1;o->y2=(float)found[i].y1;
        o->score=1.0f;o->class_id=i; /* A,B,C,D in x order */
    }
    return ED_OK;
}
