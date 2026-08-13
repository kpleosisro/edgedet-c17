# Generalized cat, dog, and monkey benchmark

Source data: `c-oloy/datasets/prepared/kaggle_animals/monkey_cat_dog`.
The original deterministic split contains 416 training images and 104
validation images. This is a custom validation experiment, not an official
dataset split or an untouched external test.

## Current verified working result

The old Python splitter greedily filled per-class validation quotas. Mixed
images satisfied several quotas at once, so it moved nearly every monkey plus
cat/dog scene into validation and left none of those combinations in its 416
training images. A later 332/84 index-modulo fold cut from that easy training
partition therefore overstated generalization: a multi-view external teacher
reached 90.3407% there, but only 71.5834% on the old 104-image stress set.

`edsplit --stratify-combinations` fixes the development protocol in native C.
It groups records by their complete positive-class set and deterministically
allocates one fifth of every sufficiently populated group to selection. The
corrected 416/104 split is:

| Split | Images | Cat boxes | Dog boxes | Monkey boxes | cat+monkey | dog+monkey | cat+dog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fit | 416 | 382 | 351 | 446 | 26 | 20 | 47 |
| selection | 104 | 87 | 60 | 96 | 6 | 5 | 12 |

The single cat+dog+monkey image remains in fit because a one-image stratum
cannot be represented on both sides. Independently reconstructed manifests
repacked to byte-identical `.edb` files.

The current deployable result uses the compact FP16 COCO source, the existing
automatic novel-class policy, and opt-in Nesterov momentum. It is entirely
native C and stays inside the 50-second gate:

| Model | Timed pipeline | Cat AP50 | Dog AP50 | Monkey AP50 | mAP50 | VOC07 mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native compact checkpoint | **20.105 s** | 80.6893% | 75.2564% | 55.1188% | **70.3548%** | 68.7327% |

It trains 1,164 parameters for 360 optimizer steps, reports assignment recall
1.0000 and mean matched IoU 0.8912, and serializes a 2,319,042-byte artifact.
The exact artifact is `build/strat_nesterov_fp16.edm`.

Hashes:

- fit `.edb`: `34B761C835A17BF26007E2C810FA79C75124DB2518A5345D557487B386B429FA`
- selection `.edb`: `C154A3B4139D8A77143542F0D4ED488803B9C2486B1124B5DEB7BC93C7EBA3E3`
- adapted `.edm`: `532677073F62AB632FAB463A0CA6EED35DAD983D9A8F796F9450EA8559C9EF40`

Reproduce the working adaptation and evaluation:

```sh
build/edtrain --dataset build/monkey-cat-dog-strat-fit.edb \
  --weights models/picodet_s_coco80_fp16.edm \
  --budget-ms 50000 --threads 12 --nesterov \
  --output-precision fp16 --output build/strat_nesterov_fp16.edm

build/edeval --dataset build/monkey-cat-dog-strat-select.edb \
  --model build/strat_nesterov_fp16.edm --threads 12 \
  --score-threshold 0.01 --nms-threshold 0.42
```

The existing `adapt` mosaic/tile sampler was tested on the corrected split
and rejected: it scored 67.4729% mAP50, including only 45.5723% monkey AP.
The fixed NanoDet-plus-ShuffleNet external screen scored 82.3355% in 95.38
seconds; bounded original/flip/native fusion reached 83.6818%. These are
architecture diagnostics outside the repository, not shippable native or
sub-minute results. The augmentation experiment was stopped before scoring
at the user's request. **No verified experiment reaches 90%; that gate remains
open.**

| Split | Images | Cat boxes | Dog boxes | Monkey boxes |
| --- | ---: | ---: | ---: | ---: |
| train | 416 | 421 | 355 | 351 |
| validation | 104 | 48 | 56 | 191 |

A deterministic train-held-out fold was carved from the 416 training images
with `edsplit --modulo 5` (332 fit / 84 select). Every KEEP/REJECT decision
below used that fold after a full timed adaptation. Validation labels were
read only after the locked recipe was chosen.

## Generalized initialization

Training starts from the unmodified 80-class COCO artifact
`models/picodet_s_coco80.edm`. Target names are not used to choose source
channels. During the timed run a balanced calibration subset is forwarded
through all 80 COCO classifiers and each source channel is scored against
target boxes using continuous AP50.

Targets with calibration AP50 at least 0.60 retain one source channel. A weak
target blends the four highest-affinity channels. On the full 416-image
training set the calibrator selected:

- `cat`: COCO `cat`, affinity 0.785335
- `dog`: COCO `dog`, affinity 0.759331
- `monkey`: `bear`, `sheep`, `elephant`, and `teddy bear`, best affinity 0.385120

Low-affinity targets train classification outputs only. The novel-class
schedule is 240 output-head updates, a 50:50 mix with initialization, an
optimizer restart, then 120 more updates. The locked novel-class extras are
per-class classification-loss normalization, class-balanced hard-negative
boosting on confused positives, and a smaller learning rate on high-affinity
classes (0.35x) versus low-affinity classes (1.75x).

Assignment diagnostics on the locked run: recall 1.0000, mean matched IoU
0.8813. Localization is not the bottleneck, so DFL stays frozen.

## Legacy greedy-split result

Continuous AP at IoU 0.50, single-pass 320x320 inference, NMS IoU 0.42:

| Model | Timed pipeline | Cat AP50 | Dog AP50 | Monkey AP50 | mAP50 | VOC07 mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous equal-blend default (FP32) | 28.542 s | 64.3024% | 59.6557% | 56.3625% | 60.1069% | 59.6070% |
| Locked equal + class-loss-norm + hard-neg + class-LR | **25.515 s** | **64.3351%** | **59.5348%** | **57.7295%** | **60.5331%** | **59.9618%** |

The locked run trains 1,164 parameters for 360 optimizer steps over 64 cached
full-image feature maps. Loading, calibration, cache construction,
optimization, restart, and FP32 serialization are inside the 25.515-second
timer. Peak RSS was 270,135,296 bytes. Output size is 4,612,356 bytes.

Two clean locked runs produced byte-identical FP32 artifacts with SHA-256
`9D5CA2B74BA5FBD839B534B22DE0DC3EC4C45F87558A413360BA314D23534B97`.
Times were 25.515 s and 26.570 s.

On this legacy stress split, the 65% continuous mAP50 gate and the stretch
70% gate were not reached. The measured lift is +0.426 mAP50, almost all from monkey
(+1.367 AP) with cat unchanged and dog down 0.121 AP, inside the one-point
guardrail.

## Train-held-out KEEP / REJECT

All rows train on the 332-image fit split and evaluate the 84-image select
split. Baseline is equal-weight top-four blend, no extra loss knobs.

| Mechanism | Cat | Dog | Monkey | mAP50 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| equal top-four blend | 0.804 | 0.887 | 0.667 | 0.786 | KEEP (init) |
| best single COCO channel | 0.804 | 0.887 | 0.531 | 0.741 | REJECT |
| AP-weighted top-four | 0.804 | 0.887 | 0.666 | 0.786 | REJECT (tie / slightly worse) |
| 96-d ridge prototype | 0.783 | 0.887 | 0.376 | 0.682 | REJECT |
| 96-d prototype + residual | 0.804 | 0.887 | 0.627 | 0.773 | REJECT |
| source-span prototype | 0.814 | 0.887 | 0.677 | 0.792 | REJECT vs class-loss-norm |
| class-LR only | 0.809 | 0.875 | 0.702 | 0.796 | KEEP only with loss-norm |
| hard-neg only | 0.805 | 0.882 | 0.684 | 0.791 | KEEP only with loss-norm |
| class-loss-norm | 0.812 | 0.891 | 0.722 | 0.808 | KEEP |
| class-loss-norm + hard-neg | 0.816 | 0.889 | 0.730 | 0.812 | KEEP |
| **class-loss-norm + hard-neg + class-LR** | **0.825** | **0.880** | **0.738** | **0.814** | **KEEP (locked)** |
| class-wise WiSE mix | 0.811 | 0.878 | 0.711 | 0.800 | REJECT |
| extra 240+240 restart | 0.815 | 0.880 | 0.738 | 0.811 | REJECT |
| wise 0.65 | 0.814 | 0.881 | 0.735 | 0.810 | REJECT |
| mosaic sampling | 0.812 | 0.868 | 0.417 | 0.699 | REJECT |
| cache 96 | 0.820 | 0.884 | 0.720 | 0.808 | REJECT |
| cross-level tied head | 0.817 | 0.876 | 0.715 | 0.803 | REJECT |

Unconstrained PicoFeat prototypes are worse than the pretrained-channel
blend. The useful signal is loss accounting: frequent boxes no longer drown
the novel class, and high-confidence inter-class mistakes are up-weighted.

DFL / regression was not changed. A small focal/varifocal alpha-gamma grid
was not needed after class-loss-norm already occupies that loss family.
A full PicoFeat low-rank residual remains unmeasured; the existing
cross-level tied classifier was the available adapter and lost on the fold.

An affinity-sparse sub-tensor update inspired by *On-Device Training Under
256KB Memory* was also tested after the locked result. It froze the high-affinity
cat/dog classifier rows and updated only the four monkey rows: 388 effective
parameters, 25.268 s, assignment recall 1.0000, and mean matched IoU 0.8583.
The selection result was cat 0.8030, dog 0.8678, monkey 0.7379, mAP50 0.8029.
Monkey did not improve and the strong-class loss was material, so the mechanism
is **REJECTED** and its CLI prototype was removed. Artifact SHA-256:
`2B854FB64A5A66AFD36F4466117F88414A7D9F10B1D3EF6B1570CAD827D2BF2E`.

## Legacy 90% representation experiments

The target was raised to 90% continuous mAP50. The old custom 104-image
stress split was initially withheld from model selection. The following native-C
experiments were run only on the now-superseded 332/84 fit/select protocol, with the compact
`picodet_s_coco80_fp16.edm` source used for the final adapter and seed runs.

| Mechanism | Cat | Dog | Monkey | Select mAP50 | Time | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| locked output-head policy | 0.825 | 0.880 | 0.738 | about 0.814 | 25-27 s | reference |
| global-context residual | 0.815 | 0.879 | 0.754 | 0.8160 | 25.552 s | reject: +0.2 AP only |
| proposal-conditioned ROI residual | 0.815 | 0.880 | 0.754 | 0.8163 | 29.003 s | reject: tie |
| local 3x3 class residual | 0.796 | 0.889 | 0.677 | 0.7872 | 26.570 s | reject |
| compact cache seed 2 | 0.826 | 0.876 | 0.722 | 0.8076 | <50 s | reject |
| compact cache seed 3 | 0.822 | 0.882 | 0.719 | 0.8079 | <50 s | reject |
| rank-8 nonlinear PicoFeat adapter, 360 updates | 0.819 | 0.881 | 0.742 | 0.8138 | 31.229 s | reject: tie |
| rank-8 adapter, 4x adapter LR | 0.819 | 0.880 | 0.742 | 0.8137 | 31.387 s | reject |
| rank-8 adapter, 720 updates | 0.809 | 0.883 | 0.741 | 0.8109 | 38.308 s | reject |

The rank-8 adapter is a zero-output-initialized class-wise nonlinear residual
over each frozen 96-channel PicoFeat map. It adds 9,312 adapter parameters and
keeps DFL/regression unchanged. It is available as the opt-in
`--picofeat-adapter` research path, but is not the default because it did not
beat the locked fold score. The ROI experiment is likewise opt-in through
`--roi-head` and is not part of the locked recipe.

The reproducible compact rank-8 artifact is
`build/fit_picofeat_rank8_current.edm`, 2,338,672 bytes, SHA-256
`71ABC644C5AB0A4AFF91128CF93C0CE16BF05B3CB4AF937A13AA3B2ED7A2D4F7`.

The adapter's AP50 on its own 332-image fitting set was only 0.6661, so the
gap is underfitting/representation capacity rather than a held-out-only
generalization failure. Doubling updates lowered the held-out result. This
rules out more optimizer sweeps on PicoDet-S as a credible route to 90%.
A stronger pretrained extractor is required. Apache-2.0 NanoDet-Plus 1.5x at
416 is a compact-storage candidate (2.44M parameters, 34.1 COCO AP), but it is
a new native graph/weight port rather than a head-only change:
https://github.com/RangiLyu/nanodet.

## Experimental compact-source CLGC ablation

The compact canonical source was also tested with Cross-Level Gradient
Consensus (CLGC), which mixes a small mean-gradient term across matching
classification weights at PicoDet's four pyramid levels. It folds directly
into the ordinary head and adds no inference operator or parameter.

In a matched full-train/validation rerun, independent heads scored 60.2940%
mAP50 and `--cross-level-mix 0.05` scored 60.4171%. Both 5% runs produced
byte-identical artifacts. Larger mixes degraded accuracy; strict tying reduced
the effective classifier degrees of freedom from 1,164 to 300 but scored only
58.8209% and is rejected.

This 0.1231-point validation gain is promising but small. CLGC remains
experimental rather than replacing the locked recipe. The formulation, full
strength sweep, hashes, limitations, and publication gate are recorded in
[`docs/cross-level-gradient-consensus.md`](docs/cross-level-gradient-consensus.md).

The later held-out training selection check did not confirm this gain:
independent updates scored 81.3475% and CLGC 5% scored 81.2850%. Nesterov was
the best bounded c-oloy transfer control at 81.3885% selection and 60.3085%
full-validation mAP50, versus 60.2940% for momentum. Its 0.0145-point full
gain is reproducible but not material, so `--nesterov` remains opt-in. All
KEEP/REJECT evidence is in
[`docs/c-oloy-transfer-audit.md`](docs/c-oloy-transfer-audit.md).

## Traffic-car guardrail

The same default binary, same COCO-80 source, and automatic policy (no novel
class, so the extra loss knobs stay off) on the 400/99 car split:

| Model | Timed pipeline | tiled AP50 |
| --- | ---: | ---: |
| Previous generalized COCO-80 row | 12.912 s | 0.733757 |
| Locked binary, same auto policy | 14.295 s | **0.733757** |

SHA-256 `C3CBA8B9CC44D0325109EA30BEC4AF38449671E85210C26EF8BDD1D3C42CFF63`.
Assignment recall 0.764, mean IoU 0.535. The one-point car guardrail holds
exactly.

## Pig generalization

`c-oloy/datasets/prepared/kaggle_animals/pigs` is a valid 22 / 5 image split
of the `pig_face` class, but five validation images are too few for a
precise AP. The locked mechanism ran without any pig/pig_face/COCO-source
hardcoding. Calibration picked `teddy bear`, `baseball glove`, `cake`, and
`sandwich` (best affinity 0.088). Timed run 11.769 s, 388 trainable
parameters, validation AP50 0.328743. Treat this as a smoke check that the
policy executes on an unseen name, not as a 33% claim.

## External YOLOv8n comparison

Run entirely outside this repository (`/tmp/yolo_bench`) with Ultralytics
8.4.118, CPU `torch 2.6.0+cpu`, 12 threads, seed 1, 320x320, and the same
416 / 104 split exported from the prepared manifests. YOLOv8n remapped the
`cat` and `dog` heads by name; monkey was a new class. Metrics below are
Ultralytics box mAP50, not the native continuous AP50 scorer.

| Regime | Wall | Epochs | Cat | Dog | Monkey | mAP50 | Params | Infer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 s adaptation | 59.0 s | 2 | 0.216 | 0.262 | 0.366 | 0.282 | 3,006,233 | 12.5 ms |
| 50-epoch fine-tune | 1548 s | 50 | 0.511 | 0.683 | 0.734 | 0.643 | 3,006,233 | 18.3 ms |

In the 50-second envelope the native adapter (0.605 mAP50, 1,164 trainable
parameters) is far ahead of YOLOv8n. After a full 50-epoch CPU fine-tune
YOLOv8n reaches 0.643 Ultralytics mAP50 in 26 minutes and about 3 million
parameters. Those numbers are not interchangeable with the native continuous
AP50 column.

## Visualizations

`edviz` wrote `build/vis/*.ppm` (green = ground truth, red = prediction).
Separated dog/monkey frames such as `0003` are clean successes. Crowded
monkey-on-cat frames (`0000`, `0001`, `0004`, `0006`) still show overlapping
or merged class boxes. That matches the assignment diagnostic: boxes are
present, class identity is the remaining error.

## Reproduction

```sh
build/edtrain --dataset build/monkey-cat-dog-train.edb \
  --weights models/picodet_s_coco80.edm \
  --budget-ms 50000 --threads 12 \
  --output-precision fp32 \
  --output build/generalized_locked.edm

build/edeval --dataset build/monkey-cat-dog-val.edb \
  --model build/generalized_locked.edm --threads 12 \
  --score-threshold 0.01 --nms-threshold 0.42
```

Packed dataset hashes:

- train: `F81B2E08D168A77F53F5BD3F6196E8CF28430BDDB9F319890270104A1F368AEF`
- validation: `DE5276EA1C128FB6BEB7F5EDF8016A26ACF46E6E4B9536FE94FA4652D5CF24A9`

This verifies a reusable adaptation mechanism on the declared benchmark. It
does not claim 65% on this split or on every future dataset. A frozen COCO
backbone can still lack the representation needed for a sufficiently
different class.

## Superseded 90% representation and ranking screen

This work used only the 332-image fit / 84-image selection split until its
configuration was locked. That split was later shown to be structurally easy,
and the locked external teacher scored only 71.5834% on the old custom
104-image stress split. External Python probes live outside this repository
and are architecture screens only; they are not shipped implementations or
native timing claims.

The most important decomposition used NanoDet-Plus 1.5x proposals and an
ImageNet ShuffleNetV2-1.5x crop representation at 160 pixels:

| Diagnostic | Cat | Dog | Monkey | mAP50 |
| --- | ---: | ---: | ---: | ---: |
| Real semantics, oracle foreground/IoU rank | 0.9331 | 0.9262 | 0.8980 | **0.9191** |
| Oracle class and foreground ceiling | 0.9630 | 0.9997 | 0.9099 | **0.9575** |
| Best real learned rank (early-stopped linear quality) | 0.7747 | 0.9196 | 0.7155 | **0.8033** |

This established only a ceiling on the superseded selection split; it is not
evidence of 90% generalization. Real false-proposal ranking remained about
11.6 points below the semantic oracle. Proposal recall was
1.0000 / 1.0000 / 0.9265.

Measured rejections:

| Mechanism | Best selection mAP50 | Decision |
| --- | ---: | --- |
| Joint 3-class + background crop classifier | 0.7729 | REJECT: 99.3% fit accuracy, weak generalization |
| Separate nonlinear foreground quality head | 0.8000 | REJECT: fit IoU MAE 0.014, weak generalization |
| KNN/prototype foreground rank | 0.7930 | REJECT: extra storage, no gain |
| Proposal-consensus rank | 0.7812 | REJECT |
| RTMDet-tiny 416 zero-update | 0.6439 | REJECT |
| RTMDet-tiny adapted head with pretrained interpolation | 0.6926 | REJECT |
| Sparse ShuffleNet stage-4/BN ROI update | 0.7223 | REJECT: two epochs exceeded 50 seconds |
| MobileNetV3-320 class-agnostic RPN + crop semantics | 0.7549 | REJECT: monkey proposal recall 0.8676 |

The native locked model remains the canonical artifact. None of these
external screens satisfies the 90% accuracy, compactness, native-C, and
sub-minute gates together, so no replacement has been promoted.
