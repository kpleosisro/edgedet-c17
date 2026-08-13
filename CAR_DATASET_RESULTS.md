# Car Detection and Tracking Dataset benchmark

Source: <https://www.kaggle.com/datasets/amitkumargurjar/car-detection-and-tracking-dataset>

The CC0 dataset contains 499 consecutive video frames with one YOLO-format
`car` class. Its supplied chronological split is frames 0000-0399 for training
(400 images, 4,452 boxes) and 0400-0498 for validation (99 images, 644 boxes).
Accuracy below is continuous AP at IoU 0.5 on all 99 validation frames. The
post-processing parameters were selected on this validation set, so this is a
validation result rather than an untouched test-set estimate.

## Final result

Frozen-feature caching eliminates repeated backbone and neck execution during
the output-head optimizer steps. Twelve deterministic 2x2 mosaics are
forwarded once; only the four classification and four DFL output convolutions
are recomputed while training. The adapter now also records the dataset's
median box side in 320-space (13.56 px here), exposes zoom/mixed samplers,
optional EMA / WiSE-FT blending, and aspect-aware `ed_predict_auto` (2x1
tiles when width/height ≥ 1.6).

With a 10-second outer deadline, two clean runs finished in **3,131 ms** and
**3,678 ms**. Each performed 120 optimizer steps over 12 cached mosaics and
produced a byte-identical FP32 `.edm` file (SHA-256
`7405CC800240549A26960AD9032989286A09DB460D768E3609B8A3A083097096`).

The final model uses two horizontal inference tiles with 6% overlap and global
class-aware NMS at IoU 0.20. `edeval --tiles-x auto` selects this layout for
the 1920x1080 frames. This improves resolution for the small distant cars in
this traffic video.

| Model | Timed adaptation | Inference | AP50 | VOC07 11-point AP |
| --- | ---: | --- | ---: | ---: |
| compact COCO pretrained, zero update | 0 s | 2x1 tiles | 68.2434% | 67.8410% |
| compact adapted, deterministic schedule | **4.018-4.148 s** | 2x1 tiles | **73.0734%** | **69.0774%** |
| rebuilt adapter (GIoU, cache-16, 30 steps) | **4.013 s** | auto 2x1 tiles | **73.0696%** | **69.0747%** |
| Tiny-Training-inspired schedule (cache-12, 120 steps) | **3.131-3.678 s** | auto 2x1 tiles | **73.3386%** | **69.3097%** |
| Canonical compact COCO-80 + auto policy | **14.412 s** | auto 2x1 tiles | **73.4271%** | **69.4199%** |
| Locked FP32 COCO-80 binary, same auto policy | **14.295 s** | auto 2x1 tiles | **73.3757%** | **69.3490%** |
| hybrid-INT8 pretrained, zero update | 0 s | 2x1 tiles | 67.4800% | 67.3352% |

Training adds 5.10 AP points under the locked tiled inference setting and
clears the 60% benchmark gate. The 120 steps with accumulation four make 480
visits to the 12 cached mosaics and train 12,804 parameters. This is 40 cache
sweeps, not 40 full-dataset epochs. Model/dataset loading and final FP32
serialization are included in the reported outer time.

The locked-FP32 guardrail row uses `models/picodet_s_coco80.edm` with the
updated binary. Novel-class extras stay off because car is a single
high-affinity target. Tiled AP50 is unchanged at 73.3757%. Artifact SHA-256
`C3CBA8B9CC44D0325109EA30BEC4AF38449671E85210C26EF8BDD1D3C42CFF63`.

The canonical row starts from `picodet_s_coco80_fp16.edm`, not a pre-remapped
car artifact. Its timed calibration independently selects COCO channel 2 with
0.724512 training-subset AP50, after which the box-scale policy selects the
same mosaic/output-head recipe. The run trains and evaluates only the compact
source workflow; model/dataset loading and FP16 serialization are included.
The final artifact is `models/car_compact_fp16.edm` (2,317,442 bytes,
SHA-256 `B1C8C7ED4846836C7A6ACE5386404FD04BC81220A60EB4CB419A14A85DFCE85D`).

For comparison, the best adapted model with a single full-frame pass scores
59.7441% AP50. Tiling is therefore the main accuracy improvement; it is not a
free speedup.

## Tiny Training Engine research ablation

The sparse-update and memory-scheduling ideas were evaluated from
[On-Device Training Under 256KB Memory](https://arxiv.org/abs/2206.15472) and
the [authors' MIT-licensed implementation](https://github.com/mit-han-lab/tiny-training).
The detector already has a static native-C backward path with a frozen
backbone, so its equivalent of compile-time graph pruning is already present.

Magnitude-ranked 1/2 and 1/4 input-channel updates were implemented and
measured, then removed. At 480 steps, the 1/2-channel variant took 6.094 s
versus 6.217 s for dense output-head updates: only about 2% faster. Every
measured sparse setting also reduced AP50. Assignment, DFL/CIoU loss, and
forward decoding dominate this small trainable head, so the paper's sparse
update is a poor trade here.

The retained improvement is more low-rate optimizer steps over a smaller
feature cache:

| Cached mosaics | Steps | Initial LR | Timed adaptation | AP50 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 30 | 0.0005000 | 4.013 s | 73.0696% |
| 8 | 120 | 0.0001250 | 2.455 s | 73.2428% |
| 12 | 120 | 0.0001250 | 3.131-3.678 s | **73.3386%** |
| 16 | 120 | 0.0001250 | 3.789 s | 73.3721% |
| 16 | 240 | 0.0000625 | 5.113 s | 73.3793% |

The 12/120 schedule is locked because it is faster than the old run while
improving AP. The 240-step variant gains only 0.041 AP points for roughly two
extra seconds. Quantization-aware scaling from the paper is not applied yet:
the compact artifacts store quantized weights but expand them to FP32 before
training, so there are no live INT8 gradient scales to correct.

## Matched-scale tile training (negative result)

`--sample-mode tile` caches the same 2x1 / 1x2 crops that `ed_predict_auto`
uses at decode time, plus a 25% full-frame mix. The hypothesis was that
mosaic training shrinks the 13.6 px median cars and fights the tiled decoder.

| Cache | Steps | Time | Tiled AP50 |
| ---: | ---: | ---: | ---: |
| 12 mosaics (locked) | 120 | 3.131-3.678 s | **73.3386%** |
| 12 infer-tiles | 120 | 3.888 s | 68.1007% |
| 32 infer-tiles | 120 | 4.782 s | 68.9391% |

A 12-mosaic cache sees 48 source images; a 12-tile cache sees 12. The extra
images beat geometric scale matching on this frozen head. Auto sampling
therefore stays mosaic. Tile mode remains available for experiments.

## Adaptation lift vs pretrained

Zero-update tiled AP50 is 68.24%. The locked 12/120 mosaic adapter reaches
73.34% (**+5.10 AP**). Stronger adaptation attempts did not widen that gap:

| Recipe | Time | Tiled AP50 | Lift vs pretrain |
| --- | ---: | ---: | ---: |
| zero update | 0 s | 68.24% | — |
| mosaic 12 / 120 / LR 1.25e-4 (locked) | 3.13-3.68 s | **73.34%** | **+5.10** |
| adapt mix 16 / 120 / DFL×2 / LR 2e-4 | 4.95 s | 70.81% | +2.57 |
| mosaic 16 / 120 / DFL×2.5 / LR 2.5e-4 | 5.43 s | 71.85% | +3.61 |

The COCO car head is already strong on this camera. Extra learning rate and
mixed crops overfit the 16-sample cache and give up pretrained localization.
`--sample-mode adapt` and `--dfl-lr-scale` remain available; they are not the
default.

An 80% AP50 target on this split is not reached. Extra inference tiles
historically peaked near 73.7%. Head-only 5-second adaptation of PicoDet-S
320 cannot invent the missing localization for ~14 px cars. A custom
from-scratch PicoDet already failed. Reaching 80% would need a larger
pretrained extractor (PicoDet-M/L or 416 input) or unfreezing the backbone,
which breaks the sub-5 s / ~1 W budget.

`ed_picodet_forward_flops()` reports the official 0.73 GFLOP PicoDet-S 320
count. `--power-limit 1` pins the process to one CPU and duty-cycles so
`busy_core_watts * cpu_time / wall_time` stays at 1 W. WSL does not expose
Intel RAPL, so the busy-core default is 8 W (override with `--core-watts`).
Measured on this machine with the locked FP16 car model:

| Mode | Threads | Median wall | Enforced avg W | Sleep |
| --- | ---: | ---: | ---: | ---: |
| uncapped 320 single | 1 | 136.7 ms | n/a | 0 |
| `--power-limit 1` single | 1 | 1130 ms | **1.000 W** | 844 ms |
| `--power-limit 1` auto 2x1 | 1 | 2152 ms | **1.000 W** | 1003 ms/tile |

The detections are unchanged. The 1 W cap buys the envelope by waiting, not
by changing the network. A real 1 W SoC would run the same FLOPs without the
sleep if it can do ~0.73 GFLOP in about a second.

## Inference latency tradeoff

Measured over 100 quiet runs at 12 threads on the development PC, excluding
image-file I/O and model loading:

| Mode | Median | p95 |
| --- | ---: | ---: |
| one full 320x320 pass | 71.100 ms | 85.817 ms |
| two horizontal 320x320 passes, 6% overlap | 116.150 ms | 125.198 ms |

The high-accuracy mode is 1.63x slower at median latency. It does not meet the
original 15/20 ms INT8 latency gate; quantized runtime kernels remain future
work.

## Model-size optimization

The 80-class COCO model was remapped to one `car` classifier channel, then all
tensors were stored as FP16 in `.edm` and expanded to FP32 on load.

| Artifact | Bytes | Reduction | AP50 |
| --- | ---: | ---: | ---: |
| adapted FP32 | 4,608,196 | baseline | 73.3346% |
| adapted FP16 storage | **2,316,418** | **49.7%** | **73.3386%** |
| adapted hybrid INT8/FP16 storage | **1,350,596** | **70.7%** | **72.9380%** |

FP16 storage changes AP by +0.0038 percentage points here, within numerical
noise. Runtime computation remains FP32. The final artifact is
`models/car_video_fast_tiled_fp16.edm` with SHA-256
`D469796D7278DA1C229CA4BC2F4307DCD7BFB10647BACFCB5AFAB3D941E939B9`.

The hybrid artifact is 41.7% smaller than FP16 storage and loses 0.401 AP
points. It therefore passes the two-point compactness tolerance. Its path is
`models/car_video_fast_tiled_int8.edm` and SHA-256 is
`459C1C8C83417924DF471FD2AA35DA8A2F4DC8A51C352AA10F0A78B2ED9FF712`.
Training directly from the 1.351 MB pretrained artifact also works, reaching
72.5817% in 5.150 seconds; that pretrained file is
`models/picodet_s_coco_car_int8.edm`.

## Reproduction

```sh
build/edtrain --dataset build/car-video-train400.edb \
  --weights models/picodet_s_coco_car_fp16.edm \
  --budget-ms 10000 --threads 12 --learning-rate 0.000125 \
  --mosaic-size 4 --accumulation 4 \
  --feature-cache-samples 12 --max-steps 120 \
  --output build/car-video-fast.edm

build/edcompact --model build/car-video-fast.edm \
  --output models/car_video_fast_tiled_fp16.edm

build/edeval --dataset build/car-video-val.edb \
  --model models/car_video_fast_tiled_fp16.edm --threads 12 \
  --score-threshold 0.01 --nms-threshold 0.20 \
  --tiles-x auto
```

A previous leakage-resistance check trained only on frames 0000-0379, leaving
a 20-frame temporal gap before validation. It scored 58.4897% with the old
single-pass decoder. The final tiled configuration has not yet been repeated
on a separately held-out video sequence, which is the main remaining accuracy
qualification.
