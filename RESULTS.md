# Reproducible benchmark record

Reference machine: Intel Core i5-13500H (12 cores, 16 logical processors), WSL2
GCC release build, 12 requested runtime threads. Input is 320 x 320 RGB.

## Configuration selection

Official VOC2007 `train` (2,501 images) was used for adaptation and `val`
(2,510 images) for configuration selection. Only `car`, `cat`, and `dog` were
scored. All AP values are continuous AP at IoU 0.5.

The zero-update pretrained model scored:

```text
car=0.818948 cat=0.897624 dog=0.846032 mAP50=0.854202 VOC07=0.819474
```

A 50-second outer-budget run at learning rate 0.001 completed in 48,100 ms,
saw 1,628 source images through 407 mosaics, and performed 50 optimizer steps.
Its serialized checkpoints scored:

| Timed checkpoint | car | cat | dog | mAP50 | VOC07 11-point |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 s | 0.831394 | 0.897164 | 0.845018 | 0.857859 | 0.823911 |
| 20 s | 0.812619 | 0.888958 | 0.827213 | 0.842930 | 0.810796 |
| 30 s | 0.795852 | 0.871049 | 0.810642 | 0.825848 | 0.797936 |
| 40 s | 0.800661 | 0.873450 | 0.813173 | 0.829095 | 0.799787 |
| final | 0.805949 | 0.879495 | 0.815545 | 0.833663 | 0.803678 |

The evidence supports stopping near 10 seconds; longer output-head adaptation
overfits this small three-class target.

## Locked final run

The final model used all 5,011 VOC2007 `trainval` images as the sampler pool.
The unchanged 12-second outer budget completed in 10,140 ms including load,
initialization and final serialization. It saw 204 source images through 51
class-balanced mosaics, made six accumulated optimizer updates, and trained
13,580 of 1,146,835 parameters.

Evaluation was performed once on all 4,952 official VOC2007 test images:

| Model | car | cat | dog | continuous mAP50 | VOC07 11-point |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero update | 0.864136 | 0.893253 | 0.868158 | 0.875182 | 0.837702 |
| adapted | 0.866612 | 0.893303 | 0.869147 | 0.876354 | 0.838870 |

Final adapted artifact SHA-256:

```text
BCD997A6F71976F94603309F8C81434490F0110EB45507849A0841ABF0069E0D
```

## Inference and verification

FP32 inference over 30 quiet repetitions measured 61.782 ms median and 65.800
ms p95 at 12 threads, excluding model load and image-file I/O. Scalar-only and
runtime-dispatched AVX2 builds produce the same decoded text fixture. The
native graph's eight raw output heads were also compared during import against
the official ONNX model; cosine similarity was approximately 1.0 and the
largest absolute difference was 8.65e-5.

The INT8 15/20 ms gate is not met because an INT8 execution path is not yet
implemented. The NEON source is present but not hardware-verified. No native
Windows C17 compiler was installed in the development environment; the Win32
code path therefore remains compile-unverified.
