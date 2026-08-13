# Kaggle car RGB accuracy result

## Outcome

The best guarded PicoDet-S 320 FP16 model reached **70.1448% validation AP50**
and **70.7312% test AP50**. The unchanged COCO car baseline scored 67.5610%
and 69.5775%, respectively. This is a real improvement, but it is not the
requested 85%; the repository does not claim otherwise.

All reported values are continuous AP at IoU 0.5 with score threshold 0.001
and NMS threshold 0.42.

| Split / stage | AP50 | Maximum recall at IoU 0.5 | Decision |
| --- | ---: | ---: | --- |
| Grayscale validation baseline | 63.4282% | 85.7143% | Replaced the data path |
| RGB validation baseline | 67.5610% | 88.2353% | Incoming accuracy floor |
| RGB validation, aligned classification | 69.8511% | 88.2353% | Keep |
| RGB validation, adapter + aligned regression | **70.1448%** | **88.2353%** | **Final keep** |
| RGB test baseline | 69.5775% | 85.3333% | Test reference |
| RGB test, final model | **70.7312%** | **85.3333%** | **+1.1537 AP** |

The test split contains only 30 unique car-positive images, so the test number
is evidence for this split, not a broad production guarantee.

## Data correction

The earlier prepared database converted every JPEG to grayscale before feeding
the COCO-pretrained color backbone. The new databases use the original JPEGs
from `Cars Detection/{train,valid,test}/images` and keep YOLO class 2 (`Car`).
Every raw record occurred twice, so exact image-plus-annotation duplicates were
removed. The resulting positive-only splits contain 184 training, 45
validation, and 30 test records, with no exact image overlap between splits.

`edpack-yolo` now supports the required conversion directly:

```powershell
build\edpack_yolo.exe `
  --images <cars-root>\train\images `
  --labels <cars-root>\train\labels `
  --list build\rgb-train-stems.txt `
  --classes Ambulance,Bus,Car,Motorcycle,Truck `
  --keep-class 2 --class-name car --deduplicate `
  --output build\car-rgb-train.edb
```

Repeat that command with `valid` and `test` paths and their corresponding stem
lists. Add `--keep-empty` only when intentionally retaining non-car images as
hard negatives.

## Training changes

- `--aligned-loss` trains the score that is actually deployed,
  `sqrt(sigmoid(class) * alignment)`, instead of optimizing the raw class
  probability while ranking a different value.
- TaskAligned targets are normalized per ground-truth object; regression is
  weighted by the assigned score, with GIoU weight 2.5 and DFL weight 0.5.
- The bounded feature cache now covers distinct records deterministically
  instead of sampling the same small pool with replacement.
- Selection saves and reloads the requested FP16 artifact, then requires mAP50
  and every class AP50 to be non-decreasing before promotion.
- Predictions are clipped to the visible image after NMS, and evaluation now
  matches each prediction to the best still-unmatched ground truth.

The accepted sequence used a 64-record feature cache, full-image sampling, and
60-second adaptation budgets. Classification was corrected first, a small
PicoFeat residual was then added, and DFL regression was unlocked only after
the aligned regression loss was in place. Each stage used
`--selection-dataset build\car-rgb-valid.edb`; a regressing candidate was not
allowed to replace its input model.

## Rejected accuracy experiments

| Candidate evaluated on RGB validation | AP50 | Result |
| --- | ---: | --- |
| 2x2 tiles plus full frame | 44.6165% | Reject |
| Final model with NMS 0.60 | 69.5242% | Reject |
| Final model with Gaussian Soft-NMS | 69.5966% | Reject |
| Hard-negative continuation | 69.5764% | Guard restored incoming model |
| ROI residual continuation | 69.7320% | Guard restored incoming model |
| Quality-only ranking residual | 69.6984% | Guard restored incoming model |
| One-class external transfer | 63.1081% | Reject |
| Five-class external transfer remapped to car | 65.7012% | Reject |

These results are why the final artifact keeps NMS 0.42, full-frame inference,
and the guarded aligned model rather than selecting a more complicated but
less accurate path.

## Reproduce the reported evaluation

```powershell
build\edeval.exe --dataset build\car-rgb-valid.edb `
  --model build\car-rgb-aligned-reg.edm --threads 8 `
  --score-threshold 0.001 --nms-threshold 0.42

build\edeval.exe --dataset build\car-rgb-test.edb `
  --model build\car-rgb-aligned-reg.edm --threads 8 `
  --score-threshold 0.001 --nms-threshold 0.42
```

Verified SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `build/car-rgb-aligned-reg.edm` | `95AC32301117B93D085491A22B465FD6A76CE547DE7FF4CD5BACC0C76EB18C17` |
| `build/car-rgb-train.edb` | `8FABA3CF2935F8B77C7E29FAD34FDC733482CE0BF082304C7160F2B4724A3EE9` |
| `build/car-rgb-valid.edb` | `606C40EA0871157BD7A7179027A3F9D5B26A561285FD0CD3A5711E51A87BF206` |
| `build/car-rgb-test.edb` | `BC10C5F034FFC77EDB9DD0288FBA76E6A9D5405771C30FC5D525AFAF7086D439` |

## Why 85% was not claimed

Lowering the score floor raised recall but did not repair precision and box
ranking. Most of the 1.15 million-parameter extractor remains frozen; the
accepted adaptation changes only the output heads and small residual modules.
The next credible route to 85% is a validation-gated trainable feature tower or
a larger/416 pretrained extractor, followed by evaluation on a substantially
larger untouched test split. More learning-rate or NMS sweeps on this small
frozen representation are not an honest substitute.
