# Translation experiments, 2026-09-05

## Submitted replication

Seed 0: Slurm **650078**. Seed 1: **650080**. Cluster root:
`/mnt/beegfsstudents/home/3160552/equivariance_family_b_20260905_01`.
The account currently allows two submitted jobs and one running job. Seed 1
therefore waits for seed 0. Pending continuation 650079 was cancelled to make
room for seed 1; there are no automatic continuations reserved.

The immutable source copy matches **every file** in both official control
manifests, aggregate digest
`cf7d77bc714d99d96d9975eb84bfb0d38644cc6b7305e3defb017371953628bf`.
No new CE, teacher, augmentation or diagnostic code enters these jobs.
The treatment changes only the active constraint: legacy foreground squared
soft-Dice equivariance, lambda 0.1, one extra translated sample per batch,
uniform six axis shifts +/-2, five-epoch linear warmup.

Both use fold 0, seeds 0/1, batch size 1, AMP, Dice including background,
64-cubed inputs, no augmentation, AdamW lr 1e-4/wd 1e-5, StepLR20/gamma0.5,
50 epochs, final-only constraint evaluation, A100 MIG 4g.40gb.

Primary endpoint: paired mean hard-Dice difference at epochs 41–50 against
the same-seed official controls. Also report epoch 50 and mean 21–30. Best epoch
is exploratory. An epoch-5 score cannot accept/reject the quality hypothesis.
Two seeds check replication; they do not estimate a reliable significance
threshold. Fold 0 has been used repeatedly and remains development data.

If a run reaches the 24-hour allocation limit, `run.sbatch SEED PREVIOUS_JOB`
resumes the exact run from its atomic latest checkpoint. It restores model,
optimizer, scheduler, AMP scaler and RNG. It only retries TIMEOUT, PREEMPTED or
NODE_FAIL and skips a completed run; numerical/software failure needs inspection.
Submit from the protocol root with explicit log paths once a queue slot is free.

## What would make equivariance valuable?

The claim to test is: **translation consistency improves a single unaugmented
prediction beyond the gain from supervised translation augmentation.** Lower
disagreement alone is insufficient: an equally wrong pair also agrees.

The official GPU audit gives a boundary-localization hypothesis: the full
translated ensemble improved Dice by 0.006946 and reduced 1,418 FP and 919 FN,
while A/P swaps increased by 8. Its extra frozen-gradient repair over Dice alone
was much smaller, about 0.000341 for the largest tested step. These are exploratory
inference/logit results, not measured training improvements or promised gains.

Priorities:

1. Finish the untouched Family-B replication.
2. Run the ordinary augmentation control. Then compare augmentation plus the
   **same original equivariance loss** against augmentation alone. This tests
   whether the relationship adds value after exposure to shifts is controlled.
3. Evaluate the common-support teacher as a separate formulation, keeping Dice.
   Calibrate on training cases first; measure sampling noise and where gradient
   energy goes. Do not simultaneously add CE, confidence thresholds, EMA,
   stronger transforms or a new schedule.
4. For a winning fixed formulation, confirm on untouched folds. A prespecified
   low-label study is a later scientific extension, not a rescue for a null result.

Transformation consistency is established work; novelty must come from the
specific failure analysis and convincing controlled evidence, not the mere use
of a teacher. Relevant primary sources: [Mean Teacher](https://arxiv.org/abs/1703.01780)
and [transformation-consistent medical segmentation](https://arxiv.org/abs/1903.00348).

## Implemented next experiments

`--translation-augmentation` applies identity with probability 1/2, otherwise
one of six +/-2 axis translations uniformly. Images and labels move together
with zero padding; spatial cropping is the ordinary translation transform.
Validation remains unaugmented. Sampling is a stateless function of
seed/epoch/batch, separate from both data-order and constraint RNG. Enabling a
constraint therefore cannot change the augmentation sequence. This is a
one-forward augmentation control; it matches optimizer steps, **not** GPU work
or the two-view input exposure of equivariance. A later compute-matched paired
supervision control is needed for a claim about compute efficiency.

`--teacher-support common` intersects valid overlap across the entire configured
12-shift set (+/-1, +/-2 per axis). For 64-cubed inputs this is the central
60-cubed region: 82.397% of crop voxels. The surrounding rim still receives
ordinary supervised Dice; it receives no teacher KL. Common overlap guarantees
valid coordinate correspondence, **not identical receptive-field context**.
The older union-support behavior remains the default for reproducibility.

For detached teacher q and student p, the KL student-logit gradient at T=1 is
`(p-q)/N`. On fixed support N, uniform distinct two-view sampling gives the
same expected gradient as the full 12-view probability ensemble. The exact
66-pair unit test verifies this. KL values differ because target entropy changes,
and sampled gradients can have high variance. There is no guarantee of an
unbiased optimizer trajectory or improved segmentation. Consistently wrong,
translation-invariant predictions remain fixed points of the auxiliary term.

The CUDA-only calibrator uses 32 deterministic **training** cases from the
verified Family-B epoch-5 Dice control at
`onecut_protocol_20260902/msd_fold0_none_calibration_seed0`. It checks checkpoint
weights against the final model hash in its completion manifest, configuration,
data/split hashes and runtime. It records eight two-view draws per case, gradient
ratios/cosines, relative sampling noise and the full teacher's gradient energy
on FP/FN/A-P-swap/correct regions. Lambda is fixed by
`min(0.10/median(ratio), 0.50/p95(ratio))`; these are frozen **logit-gradient**
scales, not network-parameter calibration or evidence of efficacy. This report
must be inspected before teacher submission, particularly negative cosines or
high sampling noise. A new shape/temperature/view policy requires recalibration.

## Prepared launch commands — not yet submitted

The development source is staged separately at
`/mnt/beegfsstudents/home/3160552/translation_followup_20260905_01/source`.
Run from that directory. The running replication source is never edited.
The account's queue is currently full; these commands need a free slot.

```bash
# Next recommended training: augmentation-only, seed 0 (then seed 1).
sbatch experiments/equivariance_family_b_20260905/next_training.sbatch \
  augmentation 0 /mnt/beegfsstudents/home/3160552/translation_followup_20260905_01/augmentation_seed0

# Same augmentation + original equivariance, only as a matched comparison.
sbatch experiments/equivariance_family_b_20260905/next_training.sbatch \
  augmentation_equivariance 0 /mnt/beegfsstudents/home/3160552/translation_followup_20260905_01/augmentation_equivariance_seed0

# Bounded GPU calibration, no optimizer and no validation cases.
sbatch experiments/equivariance_family_b_20260905/calibrate_teacher.sbatch \
  /mnt/beegfsstudents/home/3160552/translation_followup_20260905_01/teacher_train32.json

# After inspecting that report; this launcher binds its source/runtime/settings.
sbatch experiments/equivariance_family_b_20260905/next_training.sbatch \
  teacher 0 /mnt/beegfsstudents/home/3160552/translation_followup_20260905_01/teacher_seed0 \
  /mnt/beegfsstudents/home/3160552/translation_followup_20260905_01/teacher_train32.json
```

New source preserves the default Dice path in local tests, but its real GPU
training/calibration has not yet run. Reuse of archived no-constraint trajectories
for this new source needs an explicit bridge; augmentation versus augmentation
plus equivariance uses the same new source for both arms. New training scripts
accept `RESUME_RUN=1` for an interrupted run, with the same immutable arguments.
Prepared commands are not a submitted sweep or automatic training chain.
