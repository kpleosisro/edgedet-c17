# Pretrained model provenance

`models/picodet_s_coco80_fp16.edm` is the canonical training artifact. It is a
compact native-layout conversion of the official PaddleDetection PicoDet-S
320 postprocessed ONNX release. It is the only pretrained model shipped in
the repository. INT8, mixed-INT4, adapted, FP32, and VOC3 files are generated
or archival derivatives and are not part of the public source tree.

- Architecture: PicoDet-S, PPLCNet-0.75 + LCPAN + PicoHeadV2
- Pretraining dataset: Microsoft COCO
- Upstream project: PaddleDetection
- Source URL: <https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_lcnet_postprocessed.onnx>
- Project URL: <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/picodet>
- License: Apache License 2.0
- Input: RGB, 320 x 320, ImageNet mean and standard deviation
- Layout: activations NHWC; dense convolution weights OHWI; depthwise weights KKC
- All 80 COCO classifier channels are retained
- Compatibility mapping: COCO channel 2 -> `car`, 15 -> `cat`, 16 -> `dog`
- Regression head: retained unchanged and class agnostic

The one-time graph/weight extraction was performed outside this repository.
The repository and runtime do not contain or invoke Python or ONNX Runtime.
The `.edm` payload includes the source digest, individual tensor CRC32 values,
and its own SHA-256 payload digest.

Artifact SHA-256 values:

```text
picodet_s_coco80_fp16.edm  10B9D3086848109DFA8E1384952A637078FF639A111E53E9E97888ECFEC47A02
```

The artifact is externally loaded; it is not compiled into executable arrays.
