# EdgeDet

EdgeDet is a dependency-light C17 implementation of the 320 x 320 PicoDet-S
detector (PPLCNet-0.75, LCPAN and PicoHeadV2). It loads external `.edm`
weights, packs VOC data into `.edb`, performs FP32 inference and AP50
evaluation, and adapts the final classification and DFL convolutions while the
pretrained feature extractor remains frozen.

The repository contains no Python source and does not require Python,
PyTorch, Paddle, TensorFlow, ONNX Runtime, BLAS or OpenMP. JPEG and PNG decode
uses the vendored `stb_image.h`. The scalar backend is portable; x86 builds add
a separately compiled AVX2/FMA convolution backend selected at runtime.

## Build

Linux or WSL with GCC:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Windows with a Visual Studio developer command prompt:

```bat
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Data and commands

Download and extract the official VOC2007 archive, then pack it entirely in C:

```sh
build/edpack-voc --voc-root /path/to/VOCdevkit/VOC2007 \
  --split trainval --classes car,cat,dog --output voc07-trainval.edb
build/edpack-voc --voc-root /path/to/VOCdevkit/VOC2007 \
  --split test --classes car,cat,dog --output voc07-test.edb
```

Adapt, evaluate and infer:

```sh
build/edtrain --dataset voc07-trainval.edb \
  --budget-ms 50000 --threads 12 --output adapted.edm
build/edeval --dataset voc07-test.edb --model adapted.edm --threads 12
build/edinfer --model adapted.edm --input image.jpg --threads 12
```

`edtrain` now defaults to the 2.381 MB
`models/picodet_s_coco80_fp16.edm` pretrain and FP16 output storage. Use
`--output-precision fp32|fp16|int8|int4` to choose the final model and timed
checkpoint storage. FP32 is an explicit opt-in, not the standard workflow.

For a frozen backbone, a short `--budget-ms` (15 s or less) automatically
forwards 12 augmented mosaics once and reuses those feature maps for 120
output-head updates at learning rate 0.000125. That is the locked
Tiny-Training-inspired schedule: more low-rate steps over a smaller cache,
not a full-dataset epoch. Override with `--feature-cache-samples`,
`--max-steps`, and `--learning-rate`. Sampling is
`--sample-mode auto|full|zoom|mixed|mosaic|tile`. Wide images (aspect ≥ 1.6) are
decoded with two overlapping tiles unless `--tiles-x` / `--tiles-y` override
that:

```sh
build/edtrain --dataset cars.edb \
  --budget-ms 10000 --threads 12 --learning-rate 0.000125 \
  --accumulation 4 --feature-cache-samples 12 --max-steps 120 \
  --output car-adapted.edm --output-precision fp16
build/edinfer --model car-adapted.edm --input image.jpg --threads 12 \
  --nms-threshold 0.20
build/edeval --dataset cars-val.edb --model car-adapted.edm --threads 12 \
  --score-threshold 0.01 --nms-threshold 0.20 --tiles-x auto
build/edbench --model car-adapted.edm --dataset cars-val.edb \
  --power-limit 1 --core-watts 8 --tiles-x auto --runs 8
```

YOLO text annotations can be packed from an explicit, deterministic stem list:

```sh
build/edpack-yolo --images path/images --labels path/labels \
  --list stems.txt --classes car --output cars.edb
build/edtrain --dataset cars.edb --output car-fp16.edm \
  --budget-ms 50000 --threads 12
build/edcompact --model car-fp16.edm --output car-hybrid-int8.edm \
  --precision int8
build/edcompact --model car-fp16.edm --output car-mixed-int4.edm \
  --precision int4
```

For development splits, `edsplit` retains the historical index-modulo mode and
adds class-combination stratification. Use the stratified mode for multi-class
data so single-class and mixed-class scenes occur on both sides of the split:

```sh
build/edsplit --input all.edb --fit fit.edb --select select.edb \
  --modulo 5 --stratify-combinations --seed 7
```

The split is deterministic for a fixed input order, modulo, and seed. A class
combination represented by only one image cannot occur in both subsets; that
image remains in `fit`. Evaluation can also export the exact post-NMS native
detections used by the scorer:

```sh
build/edeval --dataset select.edb --model adapted.edm --threads 12 \
  --score-threshold 0.01 --nms-threshold 0.42 \
  --dump-detections detections.csv
```

The compact artifact uses FP16 storage and expands to FP32 when loaded. This
reduces disk and transfer size, not runtime activation memory or FP32 latency.
The optional hybrid INT8 storage uses per-output-channel INT8 for pointwise
weights, FP16 for sensitive depthwise weights, and exact FP32 biases. The
mixed INT4 mode packs the largest pointwise tensor as signed W4 and keeps
smaller pointwise tensors at INT8 to avoid the measured collapse from naive
whole-network W4. Both modes expand to FP32 when loaded: these are compact
storage formats, not integer compute kernels.

The training timer starts before dataset and model loading in `edtrain`. The
command reserves the larger of one second or two percent of the requested
budget for final serialization. Dataset packing and later evaluation are not
inside this timer.

`edtrain` reads the target class table from the `.edb`, but it does not use
target names to choose pretrained channels when a larger source model is
available. It measures every source classifier against a balanced subset of
target boxes inside the timer. Strong targets retain the best source channel;
weak targets blend four high-affinity channels. Low-affinity classes default
to classification-only adaptation with a knowledge-preserving optimizer
restart, per-class classification-loss normalization, hard-negative boosting
on confused positives, and a larger learning rate than high-affinity classes.
Dataset box scale selects full-image or class-balanced mosaic samples.
Explicit CLI settings override this policy.

An opt-in research adapter, Cross-Level Gradient Consensus (CLGC), shares a
small fraction of the classification gradient across PicoDet's four pyramid
levels and then serializes ordinary head weights. It adds no inference-time
operator or parameter. The independent update remains the default; use
`--cross-level-mix 0.05` to reproduce the current candidate. The matched
ablation and limits of the novelty claim are documented in
[`docs/cross-level-gradient-consensus.md`](docs/cross-level-gradient-consensus.md).
The later held-out selection check did not confirm its validation gain, so it
remains experimental. `--nesterov` is also available as a reproducible optimizer
control; its measured gain is tiny and it is not enabled by default. The full
c-oloy KEEP/REJECT audit is in
[`docs/c-oloy-transfer-audit.md`](docs/c-oloy-transfer-audit.md).
The adapter and sparse-update novelty boundary is recorded in
[`docs/adapter-prior-art.md`](docs/adapter-prior-art.md).

## Verified status

On the official VOC2007 `train`/`val` split with the three target categories:

| Model | Timed adaptation | car AP50 | cat AP50 | dog AP50 | mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| COCO-pretrained, zero update | 0 s | 81.89% | 89.76% | 84.60% | 85.42% |
| Output-head adaptation, 0.001 LR | 10.14 s | 82.97% | 89.73% | 84.58% | 85.76% |

Both values are continuous AP at IoU 0.5. The low-rate output-head adaptation
is the locked validation winner. A single final run then used all 5,011
`trainval` images as the sampler pool and evaluated on all 4,952 held-out test
images:

| Model | Timed adaptation | car AP50 | cat AP50 | dog AP50 | mAP50 | VOC07 11-point |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO-pretrained, zero update | 0 s | 86.41% | 89.33% | 86.82% | 87.52% | 83.77% |
| Output-head adaptation, 0.001 LR | 10.14 s | 86.66% | 89.33% | 86.91% | 87.64% | 83.89% |

The final timed run saw 204 source images through 51 deterministic mosaics,
completed six accumulated optimizer steps, trained 13,580 of 1,146,835
parameters, and serialized within its 12-second outer deadline. The adapted
artifact SHA-256 is
`BCD997A6F71976F94603309F8C81434490F0110EB45507849A0841ABF0069E0D`.

Measured FP32 inference on the development PC was 61.78 ms median and 65.80 ms
p95 over 30 quiet runs at 12 threads, excluding image-file I/O and model
loading. INT8 and the
stated 15/20 ms latency gate remain open work. An ARM NEON convolution backend
is present, but an ARM cross-build/hardware run and an actual Windows compiler
run have not yet been verified.

See [pretrained-source.md](docs/pretrained-source.md),
[edm-format.md](docs/edm-format.md), and [edb-format.md](docs/edb-format.md).
The separate [car-video benchmark](CAR_DATASET_RESULTS.md) records the Kaggle
experiment: deterministic 3.13-3.68 second adaptation, 73.34% validation AP50,
compact-model measurements, and the tiled-inference latency tradeoff.
The [cat/dog/monkey benchmark](MULTICLASS_RESULTS.md) records both the legacy
greedy 416/104 split and the corrected class-combination-stratified split.
The current working native FP16 checkpoint reaches **70.3548%** continuous
mAP50 in **20.105 seconds** on the corrected 104-image selection set. The old
split placed almost every mixed monkey scene in validation and is retained
only as an overlap-stress result; it must not be described as an official
split. The current 90% target is not achieved. The
[car-video benchmark](CAR_DATASET_RESULTS.md) records 73.3757% AP50 from the
same FP32 source artifact and automatic policy; the compact-source car row
is 73.4271%.

## Scope and license

The verified accuracy is benchmark-specific, not a promise for every dataset
or CPU. Other processors honor the deadline by completing fewer updates.
PicoDet/PaddleDetection provenance and license information is retained in the
model and in `NOTICE`. No Ultralytics source or weights are used.
