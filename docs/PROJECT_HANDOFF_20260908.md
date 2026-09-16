# Hippocampus segmentation with learnable constraints — project handoff, 2026-09-08

## Research goal and user preferences

The project trains a three-class SwinUNETR on the MSD Task04 hippocampus data:
background, anterior hippocampus, and posterior hippocampus. The practical goal
is to improve segmentation by adding constraints that the network learns during
training. The user does not accept a claim that the model is simply at capacity:
there are many structured errors left. They also do not want an auxiliary
landmark/cut prediction head. Proposed next steps should remain differentiable
constraints on the segmentation network, or training-only mechanisms that do
not change inference architecture.

The current development cohort is fold 0: 208 training cases and 52 validation
cases. No official test labels are used. The validation fold has now been
examined repeatedly and must be treated as development data; a frozen candidate
eventually needs confirmation on other folds.

Repository: `/Users/filippofocaccia/Desktop/hippo`.

Cluster SSH alias: `bocconi-cluster`; user ID `3160552`. The user has authorized
cluster inspection, job submission, and copying checkpoints locally. Local
shell commands must be prefixed with `rtk`; use `rtk proxy` when raw output is
needed. Do not spawn subagents unless the user explicitly asks.

## Official Family-B Dice and translation-equivariance cohort

These four runs are the clean matched cohort. They share fold 0, the 208/52
split, batch size 1, AMP, 50 epochs, 64-cubed inputs, AdamW/StepLR, Dice
including background, and source digest
`cf7d77bc714d99d96d9975eb84bfb0d38644cc6b7305e3defb017371953628bf`.
The treatment is the original translation-equivariance loss at weight 0.1.

| Seed | Arm/job | Best Dice | Mean 21–30 | Primary mean 41–50 | Epoch 50 |
|---:|---|---:|---:|---:|---:|
| 0 | Dice control, 646983 | 0.878599 | 0.876964 | 0.875515 | 0.874762 |
| 0 | Equivariance, 650078 | 0.884449 | 0.882148 | 0.881572 | 0.881510 |
| 1 | Dice control, 648443 | 0.877637 | 0.876434 | 0.874308 | 0.873975 |
| 1 | Equivariance, 650080 | 0.883125 | 0.881273 | 0.879501 | 0.878796 |

The mean paired equivariance gain is +0.005625 on the primary epochs-41–50
endpoint. It beats the same-seed control at every epoch 6–50 for seed 0 and at
44/45 epochs for seed 1. This is the strongest replicated positive result so
far.

Authoritative report:
`experiments/equivariance_family_b_20260905/TWO_SEED_RESULTS_20260907.md`.

### What equivariance changes

A shared local-CPU evaluator was run over all 52 cases for the four final
checkpoints. The absolute CPU Dice differs from cluster CUDA/AMP, but paired
voxel deltas are valid because each comparison uses the same runtime.

| Seed | Arm | Total errors | FP | FN | A/P swaps |
|---:|---|---:|---:|---:|---:|
| 0 | Dice | 41,382 | 17,234 | 20,155 | 3,993 |
| 0 | Equivariance | 38,605 | 17,641 | 16,963 | 4,001 |
| 1 | Dice | 42,893 | 17,517 | 21,161 | 4,215 |
| 1 | Equivariance | 39,600 | 16,560 | 18,863 | 4,177 |

Across seeds, equivariance removes 6,070 errors (−7.20%), dominated by 5,490
fewer false negatives. A/P swaps change by only −30 pooled. The supported claim
is therefore outer-boundary/foreground-recall regularization, not improved A/P
semantic separation.

### Confidence effect of equivariance

The four-way local confidence audit is in
`experiments/equivariance_family_b_20260905/error_confidence_fourway_20260907/`.
Maximum softmax probability is the confidence definition.

| Model | Errors | Errors at confidence ≥0.99 | Fraction ≥0.99 | Mean error confidence |
|---|---:|---:|---:|---:|
| Dice seed 0 | 41,382 | 32,731 | 79.09% | 96.70% |
| Equivariance seed 0 | 38,605 | 28,694 | 74.33% | 95.96% |
| Dice seed 1 | 42,893 | 34,378 | 80.15% | 96.90% |
| Equivariance seed 1 | 39,600 | 30,291 | 76.49% | 96.29% |

Equivariance lowers the confidently-wrong count by about 4,000 voxels in each
seed and modestly softens persistent errors. It does not solve saturation:
74–76% of its remaining errors are still at ≥0.99 confidence, and median wrong
confidence remains above 0.9998. Roughly 90–91% of all errors are within one
voxel of the ground-truth outer boundary. More distant errors become fewer but
remain a hard, often confident tail.

The separate official seed-0 CUDA audit found 39,242 errors, of which 31,027
(79.07%) were at ≥0.99 confidence. Do not mix those CUDA absolute counts with
the local CPU table.

## Translation augmentation-only control

Job 650680 (`aug-paired-s0`) completed successfully on 2026-09-07. It used the
same fold/seed/training schedule and an on-the-fly, compute-matched translation
control: one identity image/label view plus one aligned image/label translation
sampled from ±2 voxels along one axis, with the two supervised losses averaged.
There is no equivariance penalty.

Cluster output:
`/mnt/beegfsstudents/home/3160552/translation_followup_matched_20260906_02/augmentation_seed0`.

| Arm | Best Dice | Primary mean 41–50 | Epoch 50 |
|---|---:|---:|---:|
| Official Dice seed 0 | 0.878599 | 0.875515 | 0.874762 |
| Equivariance seed 0 | 0.884449 | 0.881572 | 0.881510 |
| Augmentation-only seed 0 | **0.885196** | **0.882849** | **0.883037** |

Augmentation-only beats the official seed-0 baseline by +0.007334 on the
primary endpoint and equivariance by +0.001277. This indicates that translated
supervised exposure explains at least the dominant part of the seed-0
equivariance gain. It is only one augmentation seed, so it is not yet a
replicated claim.

### Equivariance versus augmentation voxel audit

The final epoch-50 checkpoints were compared against the identical seed-0 Dice
baseline on local CPU. Full report:
`experiments/augmentation_family_b_20260907/SEED0_EQUIVARIANCE_VS_AUGMENTATION_PIXEL_AUDIT_20260908.md`.

| Model | Baseline errors fixed | New errors introduced | Final errors | Net reduction |
|---|---:|---:|---:|---:|
| Equivariance | 10,482 | 7,705 | 38,605 | 2,777 |
| Augmentation-only | **12,396** | 9,090 | **38,076** | **3,306** |

Of the 41,382 baseline errors, 7,541 are fixed by both methods, 2,941 only by
equivariance, 4,855 only by augmentation, and 26,045 by neither. Augmentation
therefore fixes 1,914 additional baseline errors but also introduces 1,385
additional errors, yielding 529 fewer final errors than equivariance.

Relative to equivariance, augmentation has 1,017 fewer FP, 771 more FN, 283
fewer A/P swaps, 123 fewer outer errors beyond two voxels, and 127 fewer deep
A/P swaps. Equivariance has a stronger recall bias; augmentation is more
balanced. Augmentation wins total error count in 28/52 cases versus 22/52 for
equivariance, with two ties. The paired interval includes zero, so the direct
one-seed advantage is modest.

Local checkpoints and regenerated error maps are under
`experiments/augmentation_family_b_20260907/`.

## Dice+cross-entropy experiment and scheduled audit

Job 651089 (`dicece-s0`) is running on the cluster. It is Dice + ordinary mean
multiclass CE over the entire crop, CE weight 1.0, no constraint, no
augmentation, seed 0, 50 epochs, AMP, with calibration telemetry. At the last
check on 2026-09-08 around 15:30 CEST it was at epoch 44/50 after 16:48 runtime.
Latest hard Dice was 0.871093; best was 0.874332 at epoch 21. It is provisionally
about 0.004 below Dice-only and does not look favorable by segmentation Dice.

Cluster root:
`/mnt/beegfsstudents/home/3160552/loss_followup_20260907_01`.

Source root:
`/mnt/beegfsstudents/home/3160552/loss_followup_20260907_01/source`.

The source digest is
`f168bb27b7e61f6d0dcf03479407f795e8878f7730b17520c8f194699f3c5c50`,
which differs from the official Family-B cohort. A strict loss-attribution claim
therefore still needs a Dice-only source bridge through this new code path.

Post-training CUDA audit job 652136 (`dicece-audit`) has been submitted with
dependency `afterok:651089`. It requests one GPU for up to two hours and will
audit both the final and best Dice+CE checkpoints against the official seed-0
Dice final checkpoint on the same 52 cases. It was pending on the dependency at
the last check.

The audit tests the user's hypothesis: Dice+CE may score worse because it is
less decisive on difficult voxels, potentially leaving useful uncertainty for
later constraints. It measures:

- total errors, FP, FN, and A/P swaps;
- maximum-softmax confidence, normalized entropy, and top-two class margin;
- fixed, introduced, and coordinate-persistent errors;
- outer-boundary distance, A/P-interface distance, and normalized coronal
  location;
- full confidence histograms in 0.10 ranges from 0.00–0.10 through 0.90–1.00,
  separately for all errors, FP, FN, swaps, persistent errors, fixed baseline
  errors, and newly introduced Dice+CE errors;
- the older ≥0.99 saturation statistic, retained separately.

Because this is a three-class model, max-softmax confidence cannot be below
one third, so the first three 0.10 histogram bins should be empty. The job writes
`audit.json`, `REPORT.md`, CSV tables, `error_confidence_and_location.png`, and
`confidence_histograms_0p10.png` under:

`/mnt/beegfsstudents/home/3160552/loss_followup_20260907_01/dice_ce_error_confidence_audit_20260908/`.

Local audit code and submission evidence:

- `evaluation/audit_pair_error_confidence.py`
- `experiments/loss_constraint_followup_20260905/DICE_CE_ERROR_AUDIT.sbatch`
- `experiments/loss_constraint_followup_20260905/DICE_CE_ERROR_AUDIT_SUBMISSION_20260908.json`

The automatic verdict is supported only when Dice+CE has more hard errors and
all uncertainty indicators move in the hypothesized direction: lower error
confidence, higher entropy, smaller top-two margin, more errors below 0.99, and
lower confidence on persistent-error coordinates. Association does not prove
that uncertainty caused the Dice loss.

## Elongation constraint

An early run was misfiled as a control. Job 611097 (`swin-elong-f0`,
2026-07-30) used elongation weight 0.05 and threshold 3.8819. The model's
elongation statistic rose from about 1.05 to 4.36 and the constraint loss
collapsed, so the constraint was active. Its implementation is no longer in
the repository or cluster source.

The original rule is `sqrt(lambda_max/lambda_min)` on the union foreground,
with a minimum threshold. The frozen-logit audit rejects it:

- a distal spur, missing midbody, and excessive thickness all improve the
  statistic while worsening Dice in all 52 cases;
- an exact anterior/posterior channel swap is invisible because the union is
  unchanged;
- 71.85% of gradient energy falls on already-correct background;
- only 8.996% and 9.174% fall on FP and FN respectively.

Do not revive the original scalar elongation rule. Report:
`evaluation/elongation_frozen_audit_20260906/README.md`.

The earlier Family-A comparison that treated job 611097 as a clean control must
be withdrawn. The surviving clean Family-A pair is control 616958 versus
equivariance 616615, approximately +0.0080 best and +0.009078 final.

## A/P constraint state

The original A/P cut and tolerance-aware aggregate family has been investigated
and closed as a negative branch on this dataset. Mixed A/P slices occur in
47/208 training and 11/52 validation cases. The existing formulations do not
contain enough cut-localization information and can sharpen the wrong answer.
Do not submit the current `ap_cut` implementation. Keep its code and analyses as
negative-result artifacts.

The user remains interested in a learnable semantic constraint, but explicitly
does not want a separate prediction head. Any future A/P mechanism should act
directly on segmentation probabilities/logits and must pass frozen-prediction
falsifiers before training.

## Recommended next experimental sequence

1. Let job 651089 finish, then inspect job 652136 and read both final- and
   best-checkpoint confidence reports. If Dice+CE is worse and remains just as
   confidently wrong, reject CE as the next supervised foundation. If its
   errors are materially softer and concentrated at difficult boundaries,
   treat that as a mechanism lead, not proof of downstream constraint benefit.
2. Submit the prepared augmentation + original equivariance seed-0 arm. This is
   now the decisive factorial test: augmentation-only already explains most of
   the gain, while the voxel audit shows 2,941 corrections unique to
   equivariance. The combination tests whether those corrections can be added
   without inheriting both sets of regressions.
3. Replicate the promising augmentation result on seed 1 before claiming that
   augmentation is reliably superior to equivariance.
4. For thesis-grade attribution across code eras, run the new-source Dice-only
   bridge. Do not attribute old-control versus new-source differences entirely
   to CE or augmentation without it.
5. If augmentation + equivariance improves over augmentation-only, replicate it
   on seed 1. If it does not, conclude that the original equivariance loss adds
   no value beyond translated supervised exposure in this formulation.
6. Only after completing this matched core should broader augmentation be
   tested. The best next candidate from the literature review is mild smooth
   local elastic deformation, followed by small rotations/scales. MRI intensity
   gamma/contrast/noise/bias-field augmentation is more likely to help scanner
   robustness and calibration than the current geometric boundary failure.

Prepared launcher:
`experiments/equivariance_family_b_20260905/next_training.sbatch`.
Its `augmentation_equivariance` mode uses the original equivariance weight 0.1
plus translation augmentation. Always launch from a fresh frozen source copy
and a new output directory.

## Important interpretation rules

- Primary training endpoint: mean hard Dice across epochs 41–50. Report epoch
  50 separately; treat best epoch as exploratory.
- Compare every treatment with its same-seed, same-code, same-exposure control.
- Two seeds establish directional replication, not precise population
  uncertainty.
- Local CPU hard maps are appropriate for paired error decomposition. Cluster
  CUDA/AMP metrics remain authoritative for absolute Dice and official
  confidence counts.
- `confidence < 0.99` means less saturated, not necessarily uncertain. Use the
  full 0.10 histogram, entropy, and top-two margin before claiming indecision.
- The current evidence supports translation consistency as a boundary/recall
  mechanism. It does not support an A/P semantic-repair claim.
- Successful frozen-logit repair is a mechanism check, not a guarantee that
  end-to-end training will realize the same improvement.

## Useful local commands and files

Interactive MRI, prediction, and directed-error viewer for a labeled case:

```bash
rtk proxy .venv/bin/python utils/view_model_predictions.py \
  --patient 017 \
  --split train \
  --weights experiments/augmentation_family_b_20260907/checkpoints/baseline_seed0_checkpoint_latest.pt \
  --device cpu \
  --show-errors
```

Change `--weights` to the equivariance or augmentation checkpoint in the same
directory. The viewer shows ground truth, prediction, six directed error types,
and synchronized sagittal/coronal/axial sliders.

Key code:

- `thesis/new_constraints/train_swinunetr_constraints.py`
- `thesis/new_constraints/translation_augmentation.py`
- `evaluation/fold0_voxel_errors.py`
- `evaluation/compare_fold0_voxel_errors.py`
- `evaluation/audit_equivariance_error_confidence.py`
- `evaluation/audit_pair_error_confidence.py`
- `utils/view_model_predictions.py`
