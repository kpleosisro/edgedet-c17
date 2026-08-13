# Low-spec regression audit

## Scope

These runs use one worker thread on the development PC to exercise the path
intended for weak CPUs. They are not Intel Pentium hardware measurements.
Adaptation timing includes model and dataset loading, initialization, feature
cache construction, optimization, and final serialization. Selection and full
evaluation run after the timed pipeline.

The acceptance rule for this audit is stricter than merely reaching 80%:
adaptation must not reduce the incoming model on the declared held-out split.
Accuracy is dataset-specific; this repository does not promise 80% on an
arbitrary dataset.

## Verified results

All values below are continuous AP at IoU 0.5, score threshold 0.01, and NMS
threshold 0.42.

| Dataset / held-out split | Incoming mAP50 | Adapted mAP50 | Timed pipeline | Result |
| --- | ---: | ---: | ---: | --- |
| VOC2007 val, car/cat/dog | 84.6148% | **85.1563%** | **3.365 s** | KEEP; above 80%, +0.5415 AP |
| Kaggle RGB car validation | 67.5466% | **68.4037%** | **1.737 s** | KEEP; +0.8571 AP, below 80% |
| Cat/dog/monkey stratified selection | 50.5913% source mapping; 67.8692% calibrated | **70.3674%** | **24.381 s** | KEEP; below 80% |

VOC adapted per-class AP50 was car 82.7226%, cat 89.0916%, and dog 83.6546%.
The model trained 13,580 parameters for 80 optimizer steps and produced SHA-256
`C934976498474B25938D232DDF3159F4A9E7814685933811ACD3F45E24FA1AF8`.

The animal run identified the PR regression: its fixed record walk scored
68.4074% after full calibration. Restoring seeded random class-balanced cache
sampling produced 70.3674% and exactly reproduced the previously accepted
artifact SHA-256
`532677073F62AB632FAB463A0CA6EED35DAD983D9A8F796F9450EA8559C9EF40`.

The Kaggle car split has maximum observed recall near 86.6% at IoU 0.5. A
frozen 320-pixel PicoDet proposal path therefore has little margin for 80% AP;
the selection guard prevents regression but cannot manufacture missing boxes.
Reaching 80% there requires a stronger or higher-resolution pretrained
proposal extractor, not a low-spec scheduling claim.

## No-regression guard

With `--selection-dataset`, `edtrain` preserves the incoming source model before
calibration. It evaluates the source-mapped, calibrated, and serialized adapted
states with the same threshold. Calibrated or adapted states are rejected when
any class AP falls below the source; the highest-mAP eligible state is saved.
Use a representative disjoint selection split, because a very small guard set
can still overfit its decision.

## Reproduce

```sh
# Low-spec VOC adaptation (full validation is intentionally separate).
build/edtrain --dataset build/voc07-train.edb \
  --weights models/picodet_s_coco80_fp16.edm \
  --budget-ms 50000 --threads 1 --output-precision fp16 \
  --output build/low-spec-voc.edm
build/edeval --dataset build/voc07-val.edb \
  --model build/low-spec-voc.edm --threads 1 \
  --score-threshold 0.01 --nms-threshold 0.42

# Guarded Kaggle car adaptation.
build/edtrain --dataset build/car-rgb-train.edb \
  --selection-dataset build/car-rgb-valid.edb \
  --selection-score-threshold 0.01 \
  --weights models/picodet_s_coco80_fp16.edm \
  --budget-ms 50000 --threads 1 --output-precision fp16 \
  --output build/low-spec-car.edm

# Guarded novel-animal adaptation with full deterministic calibration.
build/edtrain --dataset build/monkey-cat-dog-strat-fit.edb \
  --selection-dataset build/monkey-cat-dog-strat-select.edb \
  --selection-score-threshold 0.01 --calibration-samples 64 \
  --weights models/picodet_s_coco80_fp16.edm \
  --budget-ms 50000 --threads 1 --nesterov \
  --output-precision fp16 --output build/low-spec-animal.edm
```
