# FPGA MAC tile

This is not the full detector. It is the smallest block a hardware engineer
can drop into Vivado or Quartus and check against software worksheets.

## Files

- `ed_mac_tile.h` / `ed_mac_tile.c` — 3x3, Cin=1, Cout=1 MAC + sat16
- `ed_mac_tile_tb.c` — bit-exact vs `ed_hw_mac`

## Suggested wrap

1. Import `ed_mac_tile.c` into Vitis HLS or rewrite the 9-term loop in Verilog.
2. Attach AXI-Stream for `in[9]` / `w[9]` or map the register file in
   `docs/fpga-datapath.md`.
3. Replay `edhwtrace` `mac_tile_stimulus.bin` and compare `ACC` to the C
   testbench.

Do not start with the 453-node PicoDet graph on the fabric. Port this tile,
then clone it across output channels.
