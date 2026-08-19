# FPGA-only space loop (no CPU)

On the spacecraft there is no Linux process and no `.edb` file. The camera
writes pixels into FPGA line buffers. The same C functions in `ed_space_*`
are the golden model of that loop.

## Flight

```
camera RGB/Bayer
    -> ed_space_ingest_rgb   (nearest 320x320, integer gray)
    -> ed_space_teacher      (white plate lobes, left-to-right = A,B,C,D)
    -> ed_space_forward      (integer PicoDet, optional)
    -> ed_space_adapt_head   (only if teacher found boxes)
    -> box bus to GNC
```

If the teacher finds nothing, **do not train**.

## Ground replay

`.edb` / JPEG exist only on Earth:

```sh
build/edspace --dataset build/docking-benchmark/val.edb --max 20
build/edspace --dataset build/docking-benchmark/val.edb \
  --model build/docking-benchmark/baseline-trained.edm --max 4 --forward --adapt
```

## What is not on the chip

JPEG decode, `.edb` packing, `malloc`, float `ed_train`, host GIoU assignment.

## Docking accuracy

PicoDet-S 320 alone is near 0% AP on these ~14 px markers. The teacher is the
label source (and can be the primary box source). Student AP cannot beat
teacher quality.
