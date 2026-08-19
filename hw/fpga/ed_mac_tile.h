#ifndef ED_MAC_TILE_H
#define ED_MAC_TILE_H
#include <stdint.h>
/* HLS/RTL-friendly 3x3, Cin=1, Cout=1 tile. No malloc. Verilog = MAC + sat. */
#define ED_MAC_K 3
int32_t ed_mac_tile_acc(const int16_t in[ED_MAC_K*ED_MAC_K],
                        const int8_t w[ED_MAC_K*ED_MAC_K],
                        int32_t bias);
int16_t ed_mac_tile_sat16(int32_t acc);
#endif
