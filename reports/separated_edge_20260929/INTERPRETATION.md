# Separated edge pilot: a small late Dice gain, no consistent coherence gain

All six models completed successfully in Slurm job 675451, exit 0, in 35m14s.
Completion was 29 September 2026 at 15:02 Europe/Rome. MedSAM3 job 675308 also
completed successfully. No additional folds were launched.

The six selected-checkpoint hashes were verified on the cluster, along with
selected per-case result hashes. All 562 downloaded report/metadata files matched
the download manifest. Checkpoint weights remain on the cluster.

## Comparability with the original 75-epoch experiment

The original pooled edge coefficients were preserved exactly. Both arms achieved
a calibration-time median weighted logit-gradient ratio of 0.1 in every seed,
using only the original ten training cases; their p95 ratios remained below 0.5.
The pooled rerun selected the same epochs as the original edge models: 73,63,74.
Its largest epoch-level validation-Dice difference from the historical trajectories
was 0.0002556 percentage points. This supports a tightly matched comparison.

The pilot retained the original 75-epoch cap and stopping policy: seeds 0/2 ran
75 epochs and seed 1 ran 71 epochs in both arms. The separated arm selected
73,67,73. All comparisons below are fold 0 only, averaged over three seeds and
52 validation case files; they are not the earlier three-fold aggregate.

## Selected-checkpoint results

| Measurement | Pooled edge | Separated edge | Change |
|---|---:|---:|---:|
| Macro Dice | 77.7751% | 77.8598% | +0.0847 percentage points |
| Whole-hippocampus Dice | 80.4164% | 80.4644% | +0.0480 percentage points |
| FP voxels per case | 1178.99 | 1176.17 | -2.81 |
| FN voxels per case | 302.72 | 301.53 | -1.19 |
| Inner hard neighbor disagreement | 7.2995% | 7.2652% | -0.0343 percentage points |
| Outer hard neighbor disagreement | 17.2367% | 17.3129% | +0.0762 percentage points |
| Correct true-boundary transitions | 24.8739% | 24.8623% | -0.0116 percentage points |
| Inner probability-contrast MSE | 0.019481 | 0.019733 | worse |
| Outer probability-contrast MSE | 0.043074 | 0.044352 | worse |
| All-face signed contrast MSE | 0.156322 | 0.157029 | worse |

Macro Dice improves in all three seeds (+0.0163,+0.0477,+0.1902 percentage points).
Its conditional paired-case 95% interval is [+0.0284,+0.1399] percentage points.
However, intervals for FN, FP and inner hard disagreement include zero; these
improve in only two of three seeds. Outer hard disagreement and both inner/outer
probability errors worsen in all three seeds. Crossing correctness is essentially
unchanged on average. Thus the small Dice gain is not evidence of broadly improved
constraint learning or voxel coherence.

At the common epoch-60 endpoint, separated macro Dice is LOWER in all three seeds:
77.2565% versus 77.3867% (-0.1302 percentage points). FP is higher and correct
boundary transitions are lower in all three seeds. At the final paired epochs
75/71/75, macro Dice is higher in all three seeds (mean +0.0979 percentage points).
The small late overlap gain is therefore not solely a selected-checkpoint artifact,
but this does not support faster learning.

For context, the original fold-0 selected means were baseline Dice 73.1461%,
bands 77.0807%, and bands+edge 77.7749%. The proposed change is small compared with
the original bands/edge improvement and leaves the earlier inner-band deterioration
relative to baseline largely unresolved.

## Gradient evidence relevant to PCGrad

These summaries cover epochs 5,15,30,45,60 and the final epoch: two fixed training
cases × three seeds × six measurement times = 36 probes per arm. They are repeated
measurements, not 36 independent cases. Gradients are over all trainable parameters
in FP32 eval mode, not actual mixed-precision AdamW updates.

| Gradient pair | Pooled: negative / 36; median cosine | Separated: negative / 36; median cosine |
|---|---:|---:|
| Inner edges vs crossing edges | 36/36; -0.947 | 36/36; -0.946 |
| Outer edges vs Dice | 36/36; -0.232 | 36/36; -0.225 |
| Inner edges vs outer edges | 26/36; -0.612 | 27/36; -0.612 |
| Combined edge vs Dice + bands | 8/36; +0.026 | 0/36; +0.052 |

There is strong, persistent component-level opposition. Equal rule means did not
remove it. Conversely, a two-objective PCGrad treatment of combined edge versus
Dice+bands would not activate at any of the sampled separated-arm points, because
that aggregate pair has nonnegative cosine throughout.

The component strengths also matter. Median weighted parameter-gradient norms as
fractions of Dice+bands were:

| Component | Pooled | Separated |
|---|---:|---:|
| Inner edge | 0.1014% | 0.0608% |
| Outer edge | 0.6718% | 0.2893% |
| Crossing edge | 2.1692% | 2.3936% |
| Combined edge | 2.4693% | 2.4000% |

Matching the total budget and assigning equal formula weights did not strengthen
the inner rule. After calibration it reduced the effective inner and outer
influence, while crossing remained much stronger. This helps explain why equal
rule weighting alone was not an effective remedy, although it is not a causal
proof of the validation changes.

The pilot provides a reason to investigate component-level gradient handling,
not evidence that PCGrad improves segmentation. Before a PCGrad training test,
confirm conflict patterns across all ten training cases and specify precisely
which loss components are projected. Symmetric projection could substantially
alter the useful crossing update when it opposes the much smaller inner update;
preserving boundary performance must remain an explicit evaluation requirement.
No PCGrad training or expansion to other folds has been started.

## Limits and retained evidence

This is an exploratory reused development fold. Bootstrap intervals average seeds
before resampling cases and condition on these fitted/selected models. They do
not capture new-training-set, new-fold or checkpoint-selection uncertainty;
patient independence of case files has not been established. Near-perfect training
inner-pair correctness (99.92–99.93%) versus about 85.5% validation correctness also
shows that training-gradient conflict is not the only plausible limitation.

Full tables and matched learning curves: results/REPORT.md and results/SUMMARY.json.
Detailed gradient counts, norms and reproduction checks: results/GRADIENT_ANALYSIS.json.
Per-case probes, epoch audits, calibrations and checkpoint provenance are retained
under results/fold0 and results/RESULT_MANIFEST.json.
