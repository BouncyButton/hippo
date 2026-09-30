# nnU-Net train/validation audit

This report uses full-volume inference with the stopped checkpoint on the archived 208/52 split.
The primary Dice is the unweighted mean of anterior and posterior Dice.

## Dice and generalization

| Metric | Training | Validation | Train - validation |
|---|---:|---:|---:|
| Pooled mean class Dice | 0.9207 | 0.8939 | +0.0268 |
| Macro mean class Dice | 0.9206 | 0.8934 | +0.0272 |
| Anterior Dice | 0.9279 | 0.9035 | +0.0244 |
| Posterior Dice | 0.9135 | 0.8843 | +0.0292 |
| Foreground-union Dice | 0.9238 | 0.9126 | +0.0113 |

## Assessment

The run shows **mild, measurable overfitting rather than a failure to generalize**. Mean class Dice drops by 0.0272 from training to validation, while foreground-union Dice drops by only 0.0113. The larger class-aware gap is explained mainly by anterior/posterior confusion: swaps are 1.86% of training errors but 9.43% of validation errors, and the 52 validation cases contain more swap voxels than all 208 training cases (3,185 versus 1,976).

Posterior segmentation is the weaker class on validation (Dice 0.8843 versus 0.9035 anterior). Its precision is 0.8772 and recall is 0.8915, so the dominant posterior tendency is mild over-segmentation. Overall foreground volume is not strongly biased: the mean validation bias is +23 voxels per case and the median is -27.

Most errors are contour-local. On validation, 93.33% lie within 1 mm of the ground-truth outer boundary and only 1.18% lie beyond 2 mm. The notable exception is class-boundary confusion: 87.44% of validation swaps are within 1 mm of the anterior/posterior interface, compared with 99.90% on training, and 4.62% of validation swaps extend beyond 2 mm from that interface.

The validation distribution has a longer lower tail: 8/52 cases have mean Dice below 0.85 and 21/52 below 0.90, versus 2/208 and 39/208 on training. The early stopper acted at the right time: after the best online validation epoch (25), training loss improved by 0.0067 while validation loss worsened by 0.0020 and online validation Dice fell by 0.0010; patience ended the run at epoch 35.

## Error composition

All entries below are voxel counts.

| Split | All errors | GT foreground | False additions | Misses | A/P swaps | ≤1 mm from outer boundary | >2 mm from outer boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 106,344 | 682,404 | 54,867 | 49,501 | 1,976 | 103,341 | 300 |
| validation | 33,778 | 174,350 | 15,897 | 14,696 | 3,185 | 31,525 | 399 |

### Anterior/posterior cut placement

| Split | Swap voxels | Mean swaps/case | Anterior → posterior | Posterior → anterior | ≤1 mm from correct interface | 1–2 mm from interface | >2 mm from interface |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,976 | 9.5 | 882 | 1,094 | 1,974 | 2 | 0 |
| validation | 3,185 | 61.3 | 1,575 | 1,610 | 2,785 | 253 | 147 |

The training cut is extremely accurate: effectively every swapped voxel is within 1 mm of the reference interface, with no swap farther than 2 mm. Validation is less precise but still usually close: 2,785 of 3,185 swaps are within 1 mm, and 3,038 are within 2 mm. The two swap directions are nearly balanced on validation (1,575 versus 1,610), so there is no evidence of a systematic shift of the cut toward either the anterior or posterior region. Instead, its position varies locally around the correct interface. The main generalization weakness is therefore increased cut variability, not a consistently misplaced cut.

### Directed errors

| Direction | Training count | Training share | Validation count | Validation share |
|---|---:|---:|---:|---:|
| Background → anterior | 24,886 | 23.40% | 7,155 | 21.18% |
| Background → posterior | 29,981 | 28.19% | 8,742 | 25.88% |
| Anterior → background | 24,298 | 22.85% | 7,343 | 21.74% |
| Posterior → background | 25,203 | 23.70% | 7,353 | 21.77% |
| Anterior → posterior | 882 | 0.83% | 1,575 | 4.66% |
| Posterior → anterior | 1,094 | 1.03% | 1,610 | 4.77% |

### Error position along the anterior–posterior axis

Each column is one normalized low-index-to-high-index decile of the stored anterior–posterior array axis.

| Split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 11.35% | 10.25% | 8.85% | 7.93% | 7.69% | 10.24% | 10.63% | 11.52% | 12.60% | 8.94% |
| validation | 10.99% | 10.20% | 8.00% | 7.02% | 7.09% | 15.03% | 11.18% | 11.16% | 11.33% | 8.02% |

## Validation Dice distribution

- Mean class: mean 0.8934, SD 0.0385, median 0.9059, IQR 0.8694–0.9212, range 0.7811–0.9443.
- Anterior: mean 0.9028, SD 0.0391, median 0.9141, IQR 0.8826–0.9328, range 0.8083–0.9511.
- Posterior: mean 0.8840, SD 0.0397, median 0.8992, IQR 0.8559–0.9126, range 0.7493–0.9459.

For comparison, training mean-class Dice is 0.9206 ± 0.0235, with median 0.9286, IQR 0.9055–0.9394, and range 0.8494–0.9570.

## Lowest validation Dice

| Case | Mean | Anterior | Posterior | Errors / GT foreground |
|---|---:|---:|---:|---:|
| hippocampus_330 | 0.7811 | 0.8129 | 0.7493 | 40.46% |
| hippocampus_164 | 0.8058 | 0.8083 | 0.8033 | 26.73% |
| hippocampus_358 | 0.8145 | 0.8165 | 0.8126 | 29.23% |
| hippocampus_349 | 0.8191 | 0.8306 | 0.8076 | 31.89% |
| hippocampus_305 | 0.8305 | 0.8450 | 0.8159 | 32.88% |
| hippocampus_280 | 0.8327 | 0.8270 | 0.8383 | 30.31% |
| hippocampus_317 | 0.8417 | 0.8456 | 0.8378 | 28.65% |
| hippocampus_350 | 0.8423 | 0.8336 | 0.8511 | 26.38% |
| hippocampus_244 | 0.8537 | 0.8601 | 0.8473 | 29.83% |
| hippocampus_037 | 0.8609 | 0.8658 | 0.8561 | 26.92% |

## Training dynamics

- Stopped after 35 epochs; the best monitored epoch was 25.
- Best online validation mean foreground Dice: 0.8921.
- At the stopping epoch, train loss was -0.8681 and validation loss was -0.8539.
- From the best monitored epoch to stopping, validation Dice changed by -0.0010 while training loss continued to improve by 0.0067.

## Interpretation guide

- A positive train–validation Dice gap measures generalization loss; use the per-case distribution to judge whether it is broad or driven by a few cases.
- False additions are background labeled as hippocampus; misses are hippocampus labeled as background; swaps confuse anterior and posterior while retaining foreground.
- Boundary-distance counts distinguish contour-local errors from larger misplaced regions. Spatial decile counts are in `summary.json` for sagittal, anterior–posterior, and axial axes.

The complete per-case results, pooled confusion matrices, directed errors, boundary distances, and spatial deciles are stored beside this report.
