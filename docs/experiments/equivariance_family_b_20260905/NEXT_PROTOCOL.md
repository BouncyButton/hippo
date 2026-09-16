# Translation experiments, 2026-09-05

## Active queue

Seed 0: Slurm **650078**. Seed 1: **650080**. Cluster root:
`/mnt/beegfsstudents/home/3160552/equivariance_family_b_20260905_01`.
Seed 0 completed successfully. At the 2026-09-06 12:16 CEST check, seed 1 was
healthy at epoch 9 and running on gnode02. The augmentation-only seed-0 control
was initially submitted as Slurm **650672**, then cancelled while still pending:
its identity-with-probability-one-half schedule did not exactly match Family B's
shifted-view exposure. The final compute-matched replacement is Slurm **650680**,
pending behind seed 1 from a new frozen source snapshot.

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

`--translation-augmentation` now implements the compute-matched paired control.
Every batch retains the identity image/label and adds one uniformly sampled
translation from the exact Family-B set: +/-2 voxels along one of the three
axes. Images and labels receive the same zero-padded transform. The two
supervised Dice losses are averaged. The sampler consumes the same seed+1
translation-generator stream as Family B, so matched seeds receive the same
shift sequence. The augmentation-plus-equivariance arm reuses the shifted
logits for consistency and therefore also takes exactly two forwards, rather
than adding a third. Validation remains unaugmented.

`--teacher-support common` intersects valid overlap across the entire configured
12-shift set (+/-1, +/-2 per axis). For 64-cubed inputs this is the central
60-cubed region: 82.397% of crop voxels. The surrounding rim still receives
ordinary supervised Dice; it receives no teacher KL. Common overlap guarantees
valid coordinate correspondence, **not identical receptive-field context**.
The older union-support behavior remains the default for reproducibility.

The A/P-adjacent loss branch is now closed by F1–F7. Do not add cut, slab-focal,
or cut-distribution terms to this translation study. If teacher KL is evaluated,
its claim remains translation consistency and outer-boundary correction; the
official full teacher increased A/P swaps by 8 and supplies no positive A/P claim.

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

## Follow-up launches

The corrected paired-augmentation source is staged separately at
`/mnt/beegfsstudents/home/3160552/translation_followup_matched_20260906_02/source`.
Its source digest is
`f168bb27b7e61f6d0dcf03479407f795e8878f7730b17520c8f194699f3c5c50`.
The running replication source was never edited. The account's queue is full.

```bash
# Submitted as Slurm 650680. Earlier mismatched jobs were cancelled before start.
sbatch experiments/equivariance_family_b_20260905/next_training.sbatch \
  augmentation 0 /mnt/beegfsstudents/home/3160552/translation_followup_matched_20260906_02/augmentation_seed0

# Same augmentation + original equivariance, only as a matched comparison.
sbatch experiments/equivariance_family_b_20260905/next_training.sbatch \
  augmentation_equivariance 0 /mnt/beegfsstudents/home/3160552/translation_followup_matched_20260906_02/augmentation_equivariance_seed0

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

## Evidence-based decision sequence

The current Dice-only runs form a small factorial block rather than a model
search:

| Arm | Translation augmentation | Equivariance | Status |
|---|---:|---:|---|
| Official control | no | no | complete, seeds 0/1 |
| Family B | no | yes | seed 0 complete; seed 1 running |
| Augmentation control | yes | no | seed 0 queued as 650680 |
| Matched combined arm | yes | yes | submit next if Family B replicates |

The combined arm is necessary even if augmentation-only matches the Family-B
gain. Its paired comparison against augmentation-only measures whether the
consistency loss adds value after translated-label exposure is controlled. Use
the observed 0.001207 matched-control seed gap as a materiality reference, not
as a formal significance threshold. A combined-arm gain clearly above that
scale would support an independent constraint effect; a change near zero would
show that the two mechanisms are largely redundant.

Dice+CE is the next supervised objective, but do not launch Dice+CE plus
equivariance as an isolated arm. First train Dice+CE with no constraint under
the selected augmentation exposure. CE changes the supervised gradient scale,
so calibrate the equivariance coefficient on deterministic training cases from
that checkpoint, freeze it, and then train the matched Dice+CE plus
equivariance arm. This produces two interpretable deltas: Dice+CE versus Dice,
and equivariance versus its own Dice+CE control.

The seed-0 error audit does not supply a reason to reject equivariance on A/P
safety grounds. Its eight additional swaps are 0.20% of 3,993 baseline swaps,
with a paired interval spanning improvements and deteriorations. Treat swaps as
a monitored secondary endpoint and inspect their directional and case-tail
distribution; the positive constraint claim is outer-boundary recovery, not
A/P repair.
