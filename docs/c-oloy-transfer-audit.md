# c-oloy Transfer Audit

Status: bounded audit complete. The source tree was read only; no c-oloy file
was modified.

The purpose of this audit was to reuse mechanisms, not names. Every candidate
had to fit the frozen-pretrained PicoDet pipeline, remain native C, and improve
a held-out multiclass result under the same time budget. A result that helped
only the evaluation split, or moved mAP by less than normal experiment-selection
noise, was not promoted to the default recipe.

## Source mechanisms

| c-oloy mechanism | Evidence in c-oloy | Transfer decision |
| --- | --- | --- |
| Gradient-objectness proposals and fixed templates | Cars R@1000 was 0.7313 at IoU 0.50 and 0.0661 at IoU 0.75; the template-lattice oracle also failed the tight-IoU gate. | **REJECT.** A classifier cannot recover boxes the proposal stage misses. PicoDet's pretrained dense regression is much stronger. |
| Analytic ridge head bootstrap | Small cars-only scratch gain, but VOC five-class AP50 fell from 0.02835 to 0.02242. | **REJECT.** It was not dataset-neutral, and this project already has stronger COCO-channel calibration. |
| Residual-correlation feature construction | Cars AP50 0.06441, but mAP50:95 0.01587 missed its own gate. | **REJECT.** Do not attach a failed scratch feature constructor to a stronger pretrained representation. |
| Quality-aware scoring, focal weighting, warmup/decay, multiscale heads | These were useful structural lessons in c-oloy. | **ALREADY PRESENT.** The PicoDet path uses quality-aware class scores, Varifocal-style weighting, a timed learning-rate schedule, and four pyramid outputs. |
| Nesterov momentum | Small, contained optimizer primitive with a tested native-C implementation. | **KEEP OPT-IN.** Ported as `--nesterov`; it is a control, not a novelty claim. |

The c-oloy measurements are recorded in
`C:/Users/sayal/OneDrive/Documents/c-oloy/EXPERIMENTS_TRIED.md` and its
`experiments/proposal-gradient-objectness-cars.md` report.

## Transfer ablations

All multiclass selection rows use the same 416-image adaptation partition and
held-out training subset. The full validation rows use the separate 104-image
cat/dog/monkey split. Values are continuous AP50.

| Candidate | Selection mAP50 | Full validation mAP50 | Decision |
| --- | ---: | ---: | --- |
| Independent momentum control | 81.3475% | 60.2940% | default |
| Pyramid gradient equalization, 5% | 81.1170% | 60.3450% | **REJECT**; validation-only gain did not survive selection |
| CLGC, 5% | 81.2850% | 60.4171% | **EXPERIMENTAL**; selection is 0.0625 point below control |
| Geometry-only assignment ramp, 30 steps | 81.3756% | not advanced | **REJECT**; 0.0281 point is immaterial |
| Pretrained-score assignment anchor, 30 steps | 81.3474% | not advanced | **REJECT**; neutral |
| Late pairwise ranking loss | 81.2966% | not advanced | **REJECT** |
| Nesterov momentum | **81.3885%** | **60.3085%** | **KEEP OPT-IN**; reproducible but tiny |

The Nesterov selection artifact was byte-identical on repeat
(`FBCBB4684DE083FDEE77A4FF4D39EDC94C167496886B655B1DDA0A838B2D8A64`).
The repeat timed adaptation completed in 44.564 seconds. The clean full
validation artifact hash is
`B66389ED99FA1B930530A770F79847BD7425CDE4CF819795B59DC7180CB53392`.

The car guardrail was effectively unchanged: 73.4271% for momentum and
73.4273% for Nesterov. Its artifact hash is
`81EC68DAF91218543BCB01E0E280BDC8CF505B44AC513C936832CBDECD5CAA17`.

## Conclusion

No c-oloy mechanism produced a material accuracy improvement after transfer.
The useful outcome is narrower: Nesterov is a safe optional control, and the
failed proposal, analytic, assignment, gradient-balancing, and ranking paths
are now ruled out with measured gates.

The remaining error is not an optimizer-coefficient problem. The locked
output-only recipe reaches 60.5331% on this split, with assignment recall
1.0000 and mean matched IoU 0.8813. Freezing the high-affinity class rows and
updating only the low-affinity rows was also tested: it reduced effective
parameters from 1,164 to 388 but scored only 80.2911% on the train-held-out
selection split, versus 81.4% for the locked recipe. It is rejected.

The next serious experiment should add representation capacity inside the
final PicoFeat block while keeping the backbone, neck, and regression path
frozen. A PicoFeat experiment requires a real backward path
through the selected block and a matched time/accuracy ablation; a feature-side
or low-rank adapter should not be claimed as novel until that experiment and a
broader prior-art search are complete.
