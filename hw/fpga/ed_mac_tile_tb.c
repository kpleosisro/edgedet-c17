#include "ed_mac_tile.h"
#include "ed_hw.h"
#include <stdio.h>
#include <stdlib.h>

int main(void){
    int16_t in[9]={1,2,3,4,5,6,7,8,9};
    int8_t w[9]={1,0,-1,0,1,0,-1,0,1};
    int32_t bias=2,acc,ref=2;int i;int16_t q,qref;
    ed_hw_acc hw=2;
    acc=ed_mac_tile_acc(in,w,bias);
    for(i=0;i<9;++i){ref+=(int32_t)in[i]*(int32_t)w[i];hw=ed_hw_mac(hw,in[i],w[i]);}
    if(acc!=ref||acc!=(int32_t)hw){fprintf(stderr,"FAIL mac %d vs %d vs %d\n",acc,ref,(int)hw);return 1;}
    q=ed_mac_tile_sat16(acc);qref=ed_hw_sat16(acc);
    if(q!=qref){fprintf(stderr,"FAIL sat\n");return 1;}
    q=ed_mac_tile_sat16(100000);if(q!=32767){fprintf(stderr,"FAIL sat hi\n");return 1;}
    q=ed_mac_tile_sat16(-100000);if(q!=-32768){fprintf(stderr,"FAIL sat lo\n");return 1;}
    printf("ed_mac_tile ok acc=%d\n",acc);
    return 0;
}
