#include "ed_space.h"
#include <stdio.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);return 1;}}while(0)

int main(void){
    static uint8_t big[640*480*3];
    ed_space_frame frame;uint32_t x,y;
    memset(big,0,sizeof(big));
    /* Two white lobes at different x, like docking plates A/B. */
    for(y=200;y<240;++y)for(x=280;x<330;++x){uint8_t *p=big+((size_t)y*640+x)*3u;p[0]=p[1]=p[2]=255;}
    for(y=200;y<240;++y)for(x=340;x<395;++x){uint8_t *p=big+((size_t)y*640+x)*3u;p[0]=p[1]=p[2]=255;}
    CHECK(ed_space_frame_init(&frame)==ED_OK);
    CHECK(ed_space_ingest_rgb(&frame,big,640,480,640*3u)==ED_OK);
    CHECK(ed_space_teacher(&frame)==ED_OK);
    CHECK(frame.teacher_count>=2);
    CHECK(frame.teacher[0].x1<frame.teacher[1].x1);
    CHECK(frame.teacher[0].class_id==0);
    CHECK(frame.teacher[1].class_id==1);
    CHECK(ed_space_step(&frame,NULL,0,0)==ED_OK);
    return 0;
}
