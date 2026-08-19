# FPGA datapath

Host FP32 (`--runtime host`, the default) is unchanged. `--runtime fpga-model`
is an opt-in integer path built from add, subtract, multiply, shift, compare,
and ROM lookups so a Verilog engineer can port it without `expf` / `sqrtf`.

## Number format

| Signal | Type | Role |
| --- | --- | --- |
| activation | int16 | `real = q * scale` |
| weight | int8 | one scale per output channel |
| accumulator | int64 | `acc += a * b` |
| requant | `{mult, shift}` | `y = sat16((acc * mult) >>> shift)` |

Sigmoid, exp (DFL), and sqrt are 256-entry ROMs (`ed_hw_lut.c`). Hard-sigmoid
is `clip(x/6+0.5, 0, 1)` using the integer multiply `43/256` for `1/6`.

## On-device training

FPGA-model training updates **only the Pico classification / DFL head**
(frozen backbone). Learning rate is a right-shift on an integer weight code.

```sh
build/edtrain --dataset train.edb --output adapted.edm \
  --runtime fpga-model --budget-ms 20000 --threads 2
```

Use `--selection-dataset` on the host path if you need the no-regression guard
for a shippable `.edm`. Default host `edtrain` is still the accuracy path.

## Worksheets

```sh
build/edhwcal --dataset val.edb --model model.edm --output scales.edq --max 8
build/edhwtrace --model model.edm --input photo.jpg --output-dir traces/
```

`traces/manifest.json` lists every PicoDet node (op, inputs, kernel, stride,
pad). `traces/mac_tile_stimulus.bin` feeds the 3x3 MAC tile.

## MAC tile (first thing to put on a board)

`hw/fpga/ed_mac_tile.c` is HLS-friendly: no malloc, static 3x3, one bias.
Register map for an AXI-Lite wrapper:

| Offset | Name | Meaning |
| ---: | --- | --- |
| 0x00 | START | write 1 to run |
| 0x04 | BIAS | int32 |
| 0x08 | ACC | int32 result (read) |
| 0x0C | STATUS | 1 = done |
| 0x10 | IN[0..8] | int16 pixels |
| 0x30 | W[0..8] | int8 weights |

`ed_mac_tile_tb` checks the tile against `ed_hw_mac` bit-exactly.

## Accuracy contract

- `--runtime host` must match today's FP32 AP.
- `--runtime fpga-model` must stay within **0.50 AP50** of host on fruit and
  car-RGB after scale calibration. Widen a layer to int16 if it misses.
