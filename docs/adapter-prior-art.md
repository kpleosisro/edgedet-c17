# Adapter Prior-Art Boundary

This note limits the novelty claim before a PicoFeat adapter is implemented.
The search reviewed 60 results across six adapter/dense-prediction queries and
32 results across four on-device/sparse-update queries. Duplicate and
non-primary pages were discarded; the papers below are the closest mechanisms.

## Closest work

- **TinyTL** freezes feature-extractor weights, trains biases, and adds a small
  residual feature module to recover adaptation capacity. A generic lite
  residual feature adapter is therefore not new.
  <https://arxiv.org/abs/2007.11622>
- **On-Device Training Under 256KB Memory** introduces quantization-aware
  gradient scaling, contribution-selected sparse layer/sub-tensor updates, and
  compile-time pruning/reordering of the backward graph. Sparse update is prior
  art; applying it to calibrated detector class rows is an experiment, not a
  new primitive. <https://arxiv.org/abs/2206.15472>
- **LoRand** applies low-rank adapters to dense prediction while freezing the
  pretrained backbone. A generic low-rank object-detection adapter is not new.
  <https://openaccess.thecvf.com/content/CVPR2023/html/Yin_1_VS_100_Parameter-Efficient_Low_Rank_Adapter_for_Dense_Predictions_CVPR_2023_paper.html>
- **E3VA** separates a low-rank adapter highway from the frozen backbone and
  integrates it with an FPN to reduce backward memory and time. An adapter-only
  gradient highway around a feature pyramid is not new.
  <https://arxiv.org/abs/2306.09729>

## What transferred

The affinity-sparse classifier-row ablation from the sparse-update direction
trained 388 instead of 1,164 parameters and completed in 25.268 seconds, but
selection mAP50 fell from the locked 0.814 to 0.8029. It is rejected and the
prototype flag was removed.

Quantization-aware scaling is relevant only when gradients and updates are
actually quantized. The current timed adaptation is FP32 and quantizes the
serialized inference artifact afterward, so claiming QAS without a true INT8
training path would be incorrect.

Compile-time backward pruning is already compatible with this project's
explicit C kernels and frozen graph, but is a systems optimization rather than
an accuracy mechanism.

## Remaining claim space

A publishable contribution cannot simply be “low-rank PicoFeat adapter.” Any
future claim must rest on the complete combination and measured result: native
C implementation, cached frozen features, classification-only adaptation that
provably leaves DFL unchanged, deadline-aware training, compact quantized
serialization, and a reproducible accuracy/time advantage against ordinary
head tuning and representative adapter controls. A broader patent and paper
search is still required before using “first” or “novel” in a manuscript.
