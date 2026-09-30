# Local image-and-mask correction attempt — 2026-09-22

This is the follow-up to the [error-type/action gate](EXPERIMENT_20260922_ERROR_TYPES.md).
It tests an actual spatial correction proposal, rather than a scalar
foreground or A/P bias. The result is **negative**: the proposal degrades
held-out Swin segmentation, and the acceptance rule correctly abstains.

## Runs

| Job | Purpose | Output | Status |
|---|---|---|---|
| `665558` | Eight-case, one-epoch GPU integration smoke | `/home/3160552/hippopotamus_runs/cst_local_correction_smoke_665558` | completed |
| `665563` | Two-fold, five-way patient-held-out local-correction study | `/home/3160552/hippopotamus_runs/cst_local_correction_full_665563` | completed |
| `665575` | Voxel-edit accounting | `/home/3160552/hippopotamus_runs/cst_local_edit_audit_665575` | completed |
| `665577` | Confidence-stratified edit accounting | `/home/3160552/hippopotamus_runs/cst_local_edit_audit_665577` | completed |

## Proposal and leakage controls

The small 2.5-D residual CNN sees five adjacent coronal MRI planes, five
corresponding three-class Swin softmax planes, and coronal position. The CST
variant additionally sees the frozen teacher's expected anterior/posterior
cross-sectional profiles, interpolated from 32 sampled positions. Its
three-channel spatial residual is zero-initialized and bounded to `±2` logits,
then added to `log(Swin probability)`. The image/mask ablation zeros only the
two CST channels. Both variants have the same capacity and training schedule.

For each MSD fold, the 52 patients on which the original Swin and teacher were
not trained were partitioned into five outer patient-held-out groups. From
each outer training portion, a separate inner patient validation portion
selected the model epoch and a policy. The QC risk head used for candidate
slice policies was fitted only on inner-training patients. Policy candidates
were no edit, or applying the proposal to the top 20%, top 40%, or all
coronal slices by that independently fitted QC head. The policy could be
accepted only if inner-validation mean Dice increased and no more than 25% of
inner patients were harmed. Outer-test labels were read only after model,
epoch, and policy selection. No patient crossed these roles within a split.

The model loss was weighted cross-entropy (training-only Swin mistakes weighted
4×), a foreground soft-Dice term, and a small residual penalty. It was capped
at 12 epochs with inner-validation stopping. This is one specified local
architecture/loss attempt, not a search over corrections.

## Held-out result

Every one of the 20 outer models selected epoch 1. Training loss continued to
fall afterward while inner-validation hard Dice worsened. Every inner-policy
screen selected **no edit**.

| Fold | Variant | Raw mean case Dice change | Patient-bootstrap 95% interval | Patients harmed by raw proposal | Accepted change |
|---:|---|---:|---:|---:|---:|
| 0 | MRI + Swin | `−0.00615` | `[−0.00771, −0.00465]` | `48/52` | `0`, all abstained |
| 0 | MRI + Swin + CST | `−0.00593` | `[−0.00756, −0.00439]` | `44/52` | `0`, all abstained |
| 1 | MRI + Swin | `−0.00538` | `[−0.00668, −0.00408]` | `44/52` | `0`, all abstained |
| 1 | MRI + Swin + CST | `−0.00525` | `[−0.00670, −0.00386]` | `45/52` | `0`, all abstained |

The CST channels slightly reduced the average damage but did not produce a
useful edit. The accepted pipeline exactly preserves Swin predictions here;
its zero harm is abstention, **not** improved segmentation.

## Why it fails

On fold 0, the CST variant changed `13,013` voxels: `5,370` existing Swin
errors were fixed, but `7,463` previously correct voxels became wrong (plus
`180` wrong-to-wrong changes). On fold 1 it fixed `5,970` and introduced
`7,451` errors among `13,697` changed voxels. This is the decisive imbalance.

Restricting edits to low-confidence Swin voxels is not an obvious rescue. With
baseline maximum class probability below `0.6`, the CST proposal fixed versus
introduced `920` versus `1,048` voxels on fold 0 and `965` versus `957` on fold
1. The latter is only eight net voxels across 52 cases. At confidence ≥`0.8`,
the corresponding counts were `2,475` versus `3,902` and `2,859` versus
`4,111`. Confidence gating reduces exposure to harm, but no threshold has
demonstrated a reliable patient-level Dice benefit. No threshold was tuned on
outer-test cases or deployed.

## Untouched-cohort status and decision

The local proposal failed the inner acceptance criterion on every split and
has a clearly negative patient-held-out effect in both studied folds. A further
untouched-cohort segmentation test of this *abstaining* pipeline would only
measure the unchanged Swin baseline, so it was not run or represented as a
correction result.

MSD has five disjoint validation partitions, so folds 2–4 can in principle be
reserved for a new, frozen protocol. However, the existing fold-0/1 Swin and
CST checkpoints were trained on those patients. A clean untouched test would
require new base models and teacher models trained without the reserved
patients, plus out-of-fold base predictions on the corrector-training cases;
reusing the existing checkpoints would leak test-patient labels through their
weights. The available ADNI labels are binary hippocampus, not the MSD
anterior/posterior target, and the official MSD challenge test labels are not
publicly released. Do not claim prospective performance from this study.

**Decision:** retain CST for selective slice QC; reject this local residual
corrector and its proposed edits. A future correction study should first show
positive *inner* patient validation with a more constrained boundary-focused
objective before committing to the substantial clean retraining required for
an untouched test.
