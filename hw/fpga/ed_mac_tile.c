#include "ed_mac_tile.h"

/* Verilog: acc <= bias; for i in 0..8 acc <= acc + $signed(in[i]) * $signed(w[i]); */
int32_t ed_mac_tile_acc(const int16_t in[ED_MAC_K*ED_MAC_K],
                        const int8_t w[ED_MAC_K*ED_MAC_K],
                        int32_t bias){
    int32_t acc=bias;int i;
    for(i=0;i<ED_MAC_K*ED_MAC_K;++i)acc+= (int32_t)in[i]*(int32_t)w[i];
    return acc;
}
/* Verilog: sat to signed 16-bit. */
int16_t ed_mac_tile_sat16(int32_t acc){
    if(acc>32767)return 32767;if(acc<-32768)return (int16_t)(-32768);
    return (int16_t)acc;
}
