# CST teacher MSD experiment — 2026-09-21

This note records the completed cluster experiments for the coordinate-aware
2.5-D convolutional set transformer teacher and matched SwinUNETR audit.
Both MSD folds have been used for analysis and model selection; none of these
numbers is a prospective untouched-cohort estimate.

## Cluster runs

| Job | Purpose | Result directory | Status |
|---|---|---|---|
| `665270` | Three-seed base teacher, clustering, and Swin scoring | `/home/3160552/hippopotamus_runs/cst_teacher_fold0_multiseed_665270` | completed |
| `665307` | Conditional volume counterfactual repair sweep | `/home/3160552/hippopotamus_runs/cst_counterfactual_665307` | completed |
| `665313` | Nine-run profile-head ablation | `/home/3160552/hippopotamus_runs/cst_profile_ablation_665313` | completed |
| `665344` | Profile repair with p90 calibration | `/home/3160552/hippopotamus_runs/cst_profile_repair_665344` | stopped safely: no cases passed the conservative gate |
| `665349` | Profile repair with median training calibration | `/home/3160552/hippopotamus_runs/cst_profile_repair_665349` | completed |
| `665373` | Repeated nested patient-grouped risk probe | `/home/3160552/hippopotamus_runs/cst_risk_probe_665373` | completed |
| `665375` | Signed residual correction probe | `/home/3160552/hippopotamus_runs/cst_residual_correction_665375` | completed |
| `665384` | Full fold-1 Swin/CST replication | `/home/3160552/hippopotamus_runs/cst_fold1_replication_665384` | completed |
| `665389` | Frozen train-versus-validation Swin audit, folds 0 and 1 | `/home/3160552/hippopotamus_runs/swin_overfit_audit_665389` | completed |
| `665420` | Matched two-fold Swin early-stopping study | `/home/3160552/hippopotamus_runs/swin_early_stopping_665420` | completed |
| `665422` | Frozen-CST comparison on early-best Swin checkpoints | `/home/3160552/hippopotamus_runs/cst_early_stopped_reanalysis_665422` | completed |
| `665471` | Three-seed, patient-grouped trainable slice-QC head study | `/home/3160552/hippopotamus_runs/cst_slice_qc_665471` | completed |
| `665509` | GPU single-case feature/score parity and no-label CLI scoring | `/home/3160552/hippopotamus_runs/cst_qc_parity_665509` | completed |

## Finding 1: conditional anterior-volume imbalance is a useful alarm

The MRI-conditioned anterior-volume interval was the only base descriptor to
pass the three-seed discovery screen:

- mean held-out interval coverage: `0.872`;
- mean Swin violation rate: `0.090`;
- mean violation versus Dice-error correlation: `r = 0.419`;
- per-seed correlations: `0.465`, `0.330`, and `0.463`.

The 2-of-3 ensemble flagged four cases: `hippocampus_289`, `_349`, `_350`, and
`_358`.  Their mean Dice was `0.823`, versus `0.876` for the other 48 cases.
The three unanimous flags were `_349`, `_350`, and `_358`.  Every consensus
flag said that Swin assigned too little anterior tissue, and the label agreed
with that direction in all four cases.

Weak foreground-preserving A/P repair (`global_ap_bias`, gamma `0.001`)
changed only `0.007%` of voxels and improved mean hard Dice on the four flags
by `+0.00345` (bootstrap 95% interval `+0.00092` to `+0.00598`).  Three of four
cases improved; the only harmed case was the weak 2-of-3 flag `_289`.

No repair setting passed the full constraint gate.  Stronger enforcement
reduced profile/descriptor violations but made mean soft Dice negative and
harmed up to half of flagged cases.  This relationship should therefore be
used as a confidence alarm or gently gated auxiliary signal, not a hard bound.

## Finding 2: dense slice sequences expose implausible Swin profiles

The original 12-slice profile head failed.  The ablation isolated both the
required sampling density and loss:

| Variant | Teacher profile MAE | Profile disagreement vs Dice error |
|---|---:|---:|
| sparse 12, L1 | `0.1698 ± 0.0000` | `r = -0.067` |
| dense 32, L1 | `0.1780 ± 0.0000` | `r = -0.058` |
| dense 32, Smooth-L1 | `0.0446 ± 0.0017` | `r = +0.305` |

Dense Smooth-L1 reproduced in every seed:

- profile MAE: `0.0468`, `0.0442`, `0.0428`;
- disagreement/error correlation: `0.404`, `0.214`, `0.299`.

The median-training-error gate flagged ten cases.  Their mean Dice was
`0.841`, versus `0.880` for unflagged cases.  It captured 3 of the 5 and 6 of
the 10 worst-Dice validation cases.  Largest mismatches concentrated near
coronal positions 14, 33, and 35, suggesting anterior/posterior transition or
termination errors rather than random isolated noise.

Directly forcing Swin probabilities toward the teacher profile failed the
repair gate.  With weak per-slice A/P repair, hard Dice was effectively neutral
(`+0.00003`) and soft Dice decreased (`-0.00053`).  At strengths that repaired
90–100% of profile violations, mean hard Dice fell by `0.0056–0.0147` and
70–80% of cases were harmed.  The profile is consequently useful for ranking
failure risk, but it is not an accurate replacement target for Swin.

## Robust negative findings

- Union-volume intervals were vacuous: coverage `1.0`, violation rate `0`.
- Signed A/P centroid-gap violations did not track error: mean `r = 0.028`.
- Elongation was miscalibrated: every Swin prediction violated the interval.
- The synthetic corruption anomaly head did not transfer to real failures:
  mean `r = -0.071`.
- Morphology clusters were not stable across seeds.
- Sparse profiles and dense L1 profiles converged to constant-like solutions.
- Learned signed residual correction did not beat a zero correction target.  The
  best learned models had profile MAE `0.0262–0.0280`, versus `0.0248` for
  always predicting zero, with sign accuracy only about `57–60%`.

## Finding 3: CST helps localize error, but the useful signal is multivariate

On fold 0, repeated nested grouped cross-validation showed that the combined
CST embedding, relationship residual, and uncertainty features predicted
per-slice segmentation error substantially better than uncertainty alone:

- Pearson `r = 0.585 ± 0.003` versus `0.432`;
- `R² = 0.339 ± 0.003` versus `0.182`;
- worst-10% slice recall `0.428 ± 0.013` versus `0.257`;
- within-patient top-4 slices captured `47.4%` of error versus `36.7%`.

Fold 1 reproduced the combined selective-QC improvement even though raw profile
disagreement did not reproduce as a univariate case-level signal:

- combined slice-error `r = 0.633 ± 0.007` versus uncertainty `0.573`;
- top-20% slice error capture `61.0%` versus `54.0%`;
- within-patient top-4 capture `46.4%` versus `42.3%`.

The dense profile head remained accurate on fold 1 (teacher MAE `0.0460`), but
raw profile disagreement correlated negatively with case Dice error
(`r = -0.091`).  The robust conclusion is therefore narrower than the fold-0
discovery: CST features improve learned slice-level QC when combined with
uncertainty, but raw profile disagreement is not a portable standalone rule.

## Finding 4: the Swin checkpoints substantially overfit both folds

Frozen-checkpoint inference used identical deterministic preprocessing on all
208 training and 52 validation cases in each fold.  The splits had no patient
overlap.

| Fold | Train Dice | Validation Dice | Gap | Train HD95 | Validation HD95 | Peak epoch | Peak-to-final drop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.9719` | `0.8729` | `+0.0990` | `0.994` | `1.458` | 21 | `0.0038` |
| 1 | `0.9727` | `0.8627` | `+0.1101` | `0.995` | `1.617` | 21 | `0.0045` |

Training loss fell from about `0.735` to `0.026` in both folds, while
validation Dice stopped improving at epoch 21.  The small late validation drop
therefore understates the problem: the epoch-50 models memorize their training
sets roughly ten Dice points better than they generalize.  Best-checkpoint
saving and early stopping are required for subsequent Swin experiments.

## Finding 5: best-checkpoint selection reduces, but does not eliminate, overfit

A matched rerun used the original splits, preprocessing, seed, optimizer, and
50-epoch cap, adding only best-validation-Dice checkpointing and patience-8
stopping.  Both folds reproduced their original best logged Dice.

| Fold | Checkpoint | Epoch | Train Dice | Validation Dice | Gap | Validation HD95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | best | 21 | `0.9498` | `0.8768` | `0.0730` | `1.408` |
| 0 | patience stop | 29 | `0.9576` | `0.8741` | `0.0835` | `1.453` |
| 0 | original epoch 50 | 50 | `0.9719` | `0.8729` | `0.0990` | `1.458` |
| 1 | best | 22 | `0.9542` | `0.8672` | `0.0870` | `1.554` |
| 1 | patience stop | 30 | `0.9604` | `0.8648` | `0.0956` | `1.589` |
| 1 | original epoch 50 | 50 | `0.9727` | `0.8627` | `0.1101` | `1.617` |

Always deploy/evaluate the **best** checkpoint, not the stopping-point weights.
Patience 8 saves 40–42% of epochs and narrows the train/validation gap by
`0.023–0.026` Dice versus epoch 50, but the remaining `0.073–0.087` gap
calls for a separate regularization/data-diversity study.  The new flags are
opt-in; old Swin behavior remains the default.

## Finding 6: frozen CST clues survive the less-overfit Swin model

Job `665422` changed only Swin predictions.  It kept all six CST teacher
checkpoints frozen and repeated three-seed descriptor/profile diagnostics and
the same nested patient-grouped risk probe.

| Fold | Swin checkpoint | Anterior violation/error r, seeds 0–2 | Profile disagreement/error r, seeds 0–2 | Slice-risk r, uncertainty → combined | Top-20% error capture, uncertainty → combined |
|---:|---|---|---|---|---|
| 0 | epoch 50 | `0.347, 0.403, 0.378` | `0.404, 0.214, 0.299` | `0.432 → 0.593` | `43.7% → 57.7%` |
| 0 | early best | `0.396, 0.390, 0.368` | `0.376, 0.237, 0.282` | `0.339 → 0.534` | `42.6% → 57.6%` |
| 1 | epoch 50 | `0.298, 0.301, 0.280` | `-0.081, -0.064, -0.129` | `0.573 → 0.639` | `54.0% → 61.0%` |
| 1 | early best | `0.294, 0.234, 0.202` | `-0.061, -0.073, -0.146` | `0.532 → 0.633` | `51.9% → 61.8%` |

The dense-teacher 2-of-3 anterior-volume consensus flags stayed the same on
fold 0 (`_107`, `_169`, `_358`), with flagged mean Dice `0.856` versus `0.877`
unflagged after early stopping.  On fold 1, `_114` and `_177` remained while
`_228` changed to `_336`; flagged mean Dice `0.830` versus `0.869` unflagged.
These are different teachers from the four-case base-teacher consensus in
Finding 1, so their flag sets should not be conflated.

The anterior signal and **combined** slice-level error-ranking signal survive
both folds and checkpoint selection.  Raw profile disagreement does not
generalize across folds as a standalone case alarm.  Neither result establishes
a safe correction direction: signed-correction and strong-repair probes failed.
An untouched patient cohort is needed before claiming prospective performance.

## Finding 7: a trainable slice-QC head captures more error than uncertainty

Job `665471` trained small ridge and temporal-convolution heads on the six
frozen teacher feature banks from job `665422`. For each teacher seed, each
52-patient validation fold was split into five *patient-disjoint* outer folds.
Ridge regularization and temporal stopping were selected using only the outer
training patients. The target was mean anterior/posterior Dice error on each
of 32 coronal slices. The metric below is the fraction of total measured error
found by reviewing the 20% highest-scored slices. It describes ranking for
selective human QC, not improved segmentation Dice.

| Fold | Ridge uncertainty | Ridge portable relationship features | Ridge combined CST embedding + relationships | Temporal combined |
|---:|---:|---:|---:|---:|
| 0 | `42.7%` | `51.7%` | `58.8%` | `57.6%` |
| 1 | `52.1%` | `60.5%` | `62.6%` | `63.1%` |

These are matched three-seed averages of out-of-fold predictions, not averages
of the headline metrics. Patient-bootstrap gain of ridge combined versus ridge
uncertainty was `+16.1` percentage points on fold 0 (95% interval `+11.3` to
`+21.2`) and `+10.6` points on fold 1 (`+4.8` to `+15.2`). The portable
feature head also beat uncertainty on both folds: `+9.0` and `+8.4` points,
with positive bootstrap lower bounds. The small temporal network offered no
reliable fold-0 advantage over ridge, so ridge is the simpler default.

Final `.npz` ridge and `.pt` temporal artifacts were fitted per fold and seed.
Scoring uses no label, and the API refuses case names seen during head fitting.
The GPU extraction check on three cases found at most `0.00117` absolute
feature drift from the original batched feature bank, and at most `0.000257`
change in the saved ridge head's predicted slice error. This is expected
single-case versus 16-case floating-point kernel variation, not a different
feature definition. The label-free extraction CLI emitted 32 slices for a
held-out case, and the scoring CLI consumed that NPZ without a target mask.
The per-teacher 77-dimensional `combined` embeddings are not aligned across
independently trained CST seeds: a mismatched teacher/head transfer probe could
collapse even when within-teacher cross-validation was strong. Thus deploy a
*matched* teacher/head pair or three matched pairs averaged at the score level.
The 13-dimensional uncertainty-plus-relationship `portable` head is the safer
cross-teacher fallback. The cross-fold transfer probe is not a clean external
validation, since a teacher's own training set includes cases in the opposite
fold; the grouped within-fold estimates above are the defensible estimates.

This is a testable **error-ranking component**, not yet a learnable mask
constraint. It can prioritize suspect slices for review or serve as a cautious
gate for future weak auxiliary losses, but does not establish which voxels or
labels to change. Both folds were already involved in model selection, so the
frozen pipeline still needs a prospective cohort before claiming performance.

## Decision and next experiment

Do not add either discovered signal as a strong anatomy-matching loss.  Keep
two CST-derived features:

1. three-teacher consensus anterior-volume violation;
2. dense Smooth-L1 profile disagreement, preferably with per-slice residuals.

The cross-fitted error-risk head is useful for slice-level selective QC, and
the upstream Swin overfitting was reduced by best-validation checkpointing.
Keep this checkpoint selection for subsequent experiments.  Before adding
losses, address the residual train/validation gap with a controlled
regularization or augmentation study and reserve untouched patients for
testing the frozen selection procedure.

After selecting that training regime, use the resulting CST risk only to gate
or weight a weak auxiliary loss, and compare:

- baseline Swin fine-tuning;
- weak anterior-volume signal with unanimous-teacher gating;
- learned CST error-risk gating;
- both signals together.

Freeze this design before a prospective test.  Folds 0 and 1 have both been
used for model selection and are not untouched validation cohorts.
