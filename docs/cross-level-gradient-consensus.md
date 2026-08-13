# Cross-Level Gradient Consensus

Status: experimental only; the validation gain did not survive the held-out
training selection check. Strict tying is REJECTED.

## Novelty hypothesis

PicoDet has four classification output convolutions, one for each feature
pyramid level. Standard output-head adaptation optimizes each level
independently. Cross-Level Gradient Consensus (CLGC) mixes a small mean-gradient
term into the update for matching class/channel weights across those levels:

```text
g_bar[c,k] = mean_l(g[l,c,k])
g_clgc[l,c,k] = (1 - rho) * g[l,c,k] + rho * g_bar[c,k]
```

The class biases remain level-specific. `rho = 0` is the standard independent
update. `rho = 1` gives a single shared residual update while retaining the
pretrained level-specific base weights.

CLGC changes training only. The updated weights are serialized into the normal
PicoDet classification tensors, so there is no adapter tensor, added operator,
parameter, FLOP, or latency at inference.

The research hypothesis is that a small consensus term regularizes extremely
short, low-data adaptation while preserving the scale specialization learned
by the pretrained pyramid. This appears distinct from the closest mechanisms
found in the scoped search: generic low-rank dense-prediction adapters add
trainable adapter modules, and classification-to-detection transfer initializes
detector weights but does not couple gradients across pyramid levels. This is
not a world-first claim; a broader literature and patent search is still
required before publication.

## Deterministic ablation

Dataset: 416 train / 104 validation cat-dog-monkey split. Source:
`models/picodet_s_coco80_fp16.edm`. All rows use the same calibration,
classification-only 360-step schedule, seed, 12 threads, and FP16 output.
Continuous AP is measured at IoU 0.50 with score threshold 0.01 and NMS 0.42.

| CLGC mix | Effective classifier parameters | Cat AP50 | Dog AP50 | Monkey AP50 | mAP50 | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 1,164 | 64.0188% | 59.3350% | 57.5284% | 60.2940% | control |
| **0.05** | **1,164** | **64.0706%** | **59.3803%** | **57.8003%** | **60.4171%** | **KEEP experimental** |
| 0.10 | 1,164 | 64.0319% | 59.3902% | 57.8234% | 60.4151% | no material benefit over 0.05 |
| 0.25 | 1,164 | 64.0991% | 59.2990% | 57.7581% | 60.3854% | weaker |
| 0.50 | 1,164 | 63.1017% | 59.5144% | 57.2925% | 59.9695% | REJECT |
| 1.00 | 300 | 59.4169% | 61.1452% | 55.9005% | 58.8209% | REJECT |

The 5% result is a 0.1231 AP-point gain over the matched control and improves
all three classes. Two runs produced byte-identical artifacts with SHA-256
`2E352C53D9654FBF366ACD39EE9C30171BC51FB68FD9933E7110E02448CCA9DE`.
The timed pipelines completed in 22.119 s and 24.012 s; timing varies with host
load, while weights and metrics are deterministic.

The matched car-video check was neutral: 73.4271% AP50 for independent updates
and 73.3942% at `rho = 0.25` (minus 0.0329 point). It rules out a large general
gain from aggressive sharing but does not yet validate `rho = 0.05` on that
dataset.

A subsequent held-out training selection check scored 81.3475% for the
independent control and 81.2850% for `rho = 0.05`. Therefore the 0.1231-point
full-validation improvement is not sufficient evidence to promote CLGC. It
remains available only to reproduce the research hypothesis.

## Use

The independent optimizer remains the default. Enable the research candidate
explicitly:

```sh
build/edtrain --dataset train.edb \
  --weights models/picodet_s_coco80_fp16.edm \
  --output adapted.edm --output-precision fp16 \
  --budget-ms 50000 --threads 12 --cross-level-mix 0.05
```

Use `--head-adapter cross-level` for the strict `rho = 1` ablation. That mode
is retained for reproducibility, not recommended for accuracy.

## Publication gate

CLGC is an implemented and repeatable hypothesis, not a demonstrated accuracy
improvement or publishable result. Before making a novelty or accuracy claim,
run at least three seeds on
multiple unrelated datasets, lock `rho` without selecting on each validation
set, evaluate held-out tests, compare against independent heads and a generic
low-rank adapter under equal time, and complete a broader prior-art search.
