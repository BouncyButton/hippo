# Auxiliary A/P cut head

This experiment does **not** modify Swin's segmentation or apply a cut at inference.
It asks whether a small head, conditioned on the 1 mm MRI crop and frozen
early-stopped Swin probabilities, can locate the per-case coronal A/P interface
more accurately than Swin's own class transition. The head outputs 63 scores
for the boundaries between adjacent Y slices in a 64³ volume. Training uses
the best-fit cut of the training label with a narrow soft categorical target.
An otherwise identical `swin_only` arm zeros the MRI to test whether the image
adds information beyond the frozen probabilities and coordinate prior.

The 58/260 masks with mixed A/P slices remain unchanged; the auxiliary target
is their unique best-fit cut. The benchmark reports cut MAE, exact/within-one
rate, patients helped/harmed relative to Swin, and an atypical-cut subgroup.
No label from the outer validation set is used for training or checkpoint
selection. However, folds 0 and 1 have already been repeatedly studied in
this project and are development evaluations, not a fresh confirmatory test.
Frozen Swin outputs on the training fold are in-sample, which may cause a
train/test probability-domain shift; failure here does not prove the idea
impossible. A successful result needs replication with out-of-fold training
probabilities or joint Swin training before any deployment claim.

Run local unit tests:

```bash
python -m pytest semantic_constraints/ap_cut_head/test_cut_head.py -q
```

Cluster launcher (generates the frozen *train* probabilities once per fold):

```bash
sbatch semantic_constraints/ap_cut_head/run_cluster_20260922.sh
```

`logic.py` contains a candidate LTN-like ordering loss. For a learned cut
distribution q(c), its soft truth at coronal slice y is `P(c < y)`. Within
foreground support, that should agree with conditional A/P probabilities.
The cut distribution is detached by default in this loss so the segmentation
cannot move the cut merely to rationalize its current errors. **Do not add
this loss to Swin training yet**: the cut head must first beat Swin's own cut
and the coordinate prior on held-out patients. A generic plane-existence loss
does not provide patient-specific cut location.

## Literal cut-voxel probe

`voxel_train.py` is a separate, stronger test of the user's proposed target:
it takes **frozen final-decoder Swin features**, labels foreground voxels in
the two adjacent coronal slices as a cut band, and learns a dense cut-voxel
heatmap. A differentiable two-slice pooling converts the heatmap into a
distribution over 63 cut boundaries. It reports localization only; the Swin
segmentation remains unchanged. The checkpoint-selected head uses an inner
training-fold split, and the outer fold is never used for checkpoint selection.

```bash
python -m pytest semantic_constraints/ap_cut_head/test_voxel_model.py -q
sbatch semantic_constraints/ap_cut_head/run_voxel_cluster_20260922.sh
```

This probe reuses the frozen train probabilities from the first job and the
matching early-stopped checkpoint. Because the backbone was itself trained on
the head's training patients *including its inner-validation patients*, the
inner selection metric is not an independent end-to-end validation of the
representation. The two patient-held-out outer folds both perform worse than
Swin's own cut; see `EXPERIMENT_20260922.md`. A different, jointly trained
auxiliary head or out-of-fold feature protocol would be a separate experiment.
