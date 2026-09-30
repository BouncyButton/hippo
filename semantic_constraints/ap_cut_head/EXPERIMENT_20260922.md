# A/P cut supervision experiment, 2026-09-22

## Frozen probability head: completed

Cluster job `665709` used the early-stopped Swin checkpoints and their
matching validation probabilities. It generated training-fold probabilities
from each same checkpoint, then fitted two otherwise identical cut heads:
MRI+Swin probabilities and Swin probabilities with the MRI channel zeroed.
Each head was selected on an inner training-fold holdout and evaluated on the
52 patient-disjoint cases in each development validation fold. The output is a
categorical distribution over between-slice coronal cuts. No segmentation
labels were changed during inference.

| Fold | Arm | Swin cut MAE | Head cut MAE | Better / worse patients |
| --- | --- | ---: | ---: | ---: |
| 0 | MRI+Swin | 0.885 | 0.846 | 2 / 0 |
| 0 | Swin only | 0.885 | 0.846 | 2 / 0 |
| 1 | MRI+Swin | 1.058 | 1.038 | 1 / 0 |
| 1 | Swin only | 1.058 | 1.038 | 1 / 0 |

In fold 0 the two arms made identical hard cut predictions for all 52 cases.
In fold 1 they disagreed in two cases, but neither arm had a net advantage in
cut MAE. The modest improvements therefore do not establish an image-derived
patient-specific cue; they are consistent with sharpening or correcting a few
existing Swin transitions. In particular, a supervised cut head alone is not
yet a trustworthy source for an LTN-like ordering constraint.

This is a *frozen-output probe*, not joint Swin training. Training-fold Swin
probabilities are in-sample, while outer-fold probabilities are held out.
Folds 0 and 1 have also been repeatedly examined during method development,
so these are not untouched confirmatory estimates.

## Cut-voxel decoder-feature probe: completed

Cluster job `665712` tests the literal proposed target: mark foreground
voxels in the two slices adjacent to the best-fitting A/P cut as a cut band,
train a dense head on Swin's **frozen final-decoder features**, and pool the
head's output to a cut distribution. This probes whether the representation
before Swin's final class logits contains localization information the first
head missed. It does not edit segmentation or train the LTN loss.

| Fold | Swin cut MAE | Cut-voxel head MAE | Better / worse patients | Inner validation head MAE at selected epoch |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.885 | 0.923 | 1 / 3 | 0.000 |
| 1 | 1.058 | 1.154 | 1 / 6 | 0.048 |

The apparent inner-validation success is **not** independent evidence: the
frozen Swin backbone had itself been trained on those patients, although the
cut head had not. The patient-held-out outer folds reveal a large gap, and
the head is worse than simply keeping Swin's cut in both folds. This does not
rule out a jointly trained auxiliary head or out-of-fold feature generation,
but it rejects this frozen-decoder cut-band head as a cut source for an LTN
constraint or a segmentation correction.

## Conditional LTN candidate

`logic.py` encodes a soft implication based on a distribution over cuts:
for foreground at Y=y, anterior probability should match `P(c < y)` and
posterior probability its complement. The cut distribution is detached in the
consistency term and must be anchored by independent cut supervision. This
constraint is deliberately **not** active in any segmentation training run;
the two tested cut heads have not earned that trust and could otherwise
enforce a plausible plane at the wrong patient-specific location.
