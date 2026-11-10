# Three baseline architectures under one training-compute allowance

Submit one Slurm job from the cluster repository root:

```bash
sbatch scripts/run_compute_matched_baselines.sbatch
```

An isolated copy of the required sources is also staged on the cluster. Submit
that copy with:

```bash
cd /mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261003/compute_matched_baselines
sbatch scripts/run_compute_matched_baselines.sbatch
```

This is **one job**, using one `4g.40gb` GPU in the `stud` partition, four CPUs,
24 GB host memory and a maximum wall time of 23 hours 50 minutes. It trains
U-Net, SwinUNETR and the nnU-Net-derived topology, each on folds 0/1/2 × seeds
0/1/2, producing **27 sequential runs**. There are no array tasks or concurrent
training processes. This respects the user's requirement of one running job
and one queued job. The wall-time request fits the verified 24-hour `stud`
QOS limit; actual training stops when the runs exhaust their compute allowances.
The default interpreter is the existing cluster `hippocampus` environment;
there are no environment installations or training jobs launched by this change.

## Frozen data and maximum training schedules

All runs use batch one, 64³ inputs and three-class segmentation. A pass visits
each of the ten training cases once. Seeds change initialization, order and
augmentation; they **do not change the ten selected cases**. The three fold
subsets were recovered from the original cluster archive, checked against the
full dataset folds, and frozen in `splits_10patients.json`. Every model uses the
same ten training cases and 52 validation cases within a fold. Source archive
members and hashes are recorded in `split_provenance.json`.

| Baseline | Estimated GFLOPs/training update | Maximum updates | Ten-case pass equivalent | Stop position |
| --- | ---: | ---: | ---: | --- |
| MONAI U-Net | 9.580978301 | **12,134** | **1,213.4** | 1,213 passes + 4 updates |
| SwinUNETR | 155.012136022 | **749** | **74.9** | 74 passes + 9 updates |
| nnU-Net-derived 64³ topology | 355.293896301 | **327** | **32.7** | 32 passes + 7 updates |

The maximum is the literal **116,259 GFLOPs per run**. SwinUNETR therefore gets
749 updates: its original 750-update reference consumed 116,259.1020165 GFLOPs,
slightly exceeding the rounded maximum. No run completes an extra final epoch.
These epoch counts describe ten-case passes, whereas the native nnU-Net trainer
defines an epoch as 250 random patch iterations. This custom comparison loop
uses the ten-case pass convention for all models.

The allowance is estimated arithmetic, including model forward, loss, backward,
gradient clipping when applicable, and optimizer update. A multiply-add counts
as two FLOPs. Validation, shared CPU preprocessing/augmentation, checkpoint I/O,
initialization and precision-management operations are excluded consistently.
The complete profiling evidence and assumptions are in
[`TRAINING_SCHEDULES.md`](../../reports/compute_budget_5pct_fold0_seed0_20261003/TRAINING_SCHEDULES.md).

The small U-Net downsamples immediately and uses narrow features. nnU-Net
processes wider features at full 64³ resolution, and SwinUNETR has a substantial
convolutional decoder plus attention. U-Net consequently has 16.18× cheaper
updates than SwinUNETR and 37.08× cheaper updates than this nnU-Net configuration.
The additional updates reuse the same ten cases; they do not increase the
number of independent training examples.

## Declared training recipes

- Shared historical per-case normalization, padding and center crop; shared
  `mild_v1` augmentation, applied on the CPU before transferring to the GPU.
- U-Net: channels 16/32/64/128/256, two residual units, softmax Dice and AdamW.
- SwinUNETR: feature size 24, no activation checkpointing, softmax Dice and AdamW.
- U-Net and SwinUNETR: initial LR 1e-4, weight decay 1e-5, 5% linear warmup,
  then cosine decay to 1% of the initial LR over their allotted updates.
- nnU-Net-derived baseline: explicit five-stage PlainConvUNet with features
  32/64/128/256/320, He initialization, deep supervision, foreground Dice+CE,
  Nesterov SGD (LR .01, momentum .99, weight decay 3e-5), gradient clipping 12,
  and polynomial LR decay (exponent .9) over its allotted updates.
- CUDA bfloat16 autocast with FP32 parameters and optimizer state. No gradient
  accumulation or GradScaler-skipped updates. CPU smoke checks use FP32.
- Twenty validation checkpoints at equally spaced fractions of each model's
  allowed update count; select highest mean anterior/posterior hard Dice.
  **No early stopping.**

The original dataset-specific nnU-Net plan was not retained. This explicitly
saved topology follows the repository default planner rules for isotropic 64³
inputs. Shared input size, batch size and preprocessing intentionally replace
its unrestricted automatic pipeline. In a paper, call it a **matched-input
nnU-Net-derived baseline**, and list its recipe. This protocol compares complete
declared recipes; shared losses/optimizers would be needed to isolate architecture
alone. `protocol.json` contains all hyperparameters and accounting values.

MONAI must be 1.6.0. Profiling used dynamic-network-architectures 0.4.1; the
cluster has 0.4.4. The launcher permits compatible 0.4.x versions, logs versions,
and verifies exact model parameter counts. FLOPs remain the declared arithmetic
estimate, rather than a count of hardware-specific GPU instructions.

## Source preservation, outputs and interrupted jobs

The job freezes the required sources and protocol under the campaign's
`source/` folder. A lock makes all 27 runs use that same snapshot. The launcher
checks a source hash manifest on every start; training checks the dataset,
split and topology hashes. Run configurations record the runtime versions,
patient IDs, source hash, recipe, cost and allowance.

Default campaign directory:
`/mnt/beegfsstudents/home/3160552/compute_matched_baselines_<job_id>/`.
Each run writes to `<model>/fold<fold>/seed<seed>/run/`:

- `checkpoint_best.pt`, `checkpoint_latest.pt`, `validation_best.json`,
  `validation_latest.json`, and validation curves in `metrics.csv`.
- `config.json`, a durable `budget_ledger.json`, and a hash-bound
  `completion.json`. Failures leave `failure.json`.

A full training attempt is reserved before forward. After interruption,
uncheckpointed attempts remain charged, even if their updates must be discarded.
Resuming thus stays under the cap, but may finish with fewer retained optimizer
updates. `discarded_attempts` identifies such runs. Checkpoints preserve the
optimizer, training order, RNG state and selected model. Completed runs are
verified and skipped. Before paper comparisons, inspect interrupted runs and
decide whether to repeat the affected runs as explicitly separate experiments.

To resume, submit the same job file with the original campaign directory:

```bash
HIPPO_COMPUTE_RUN_ROOT=/mnt/beegfsstudents/home/3160552/compute_matched_baselines_<original_job_id> \
  sbatch scripts/run_compute_matched_baselines.sbatch
```

Optional environment overrides: `HIPPO_PYTHON`, `HIPPO_DATA_PKL` and
`HIPPO_COMPUTE_RUN_ROOT`. The default input is
`datasets/Dataset101_MSD/msd_hippocampus_full.pkl`, bound to the original data hash.

After the final experiment, the job verifies all 27 completion manifests and
writes `summary.json`, `summary.csv`, and `summary.md`. Re-run manually with:

```bash
/home/3160552/.conda/envs/hippocampus/bin/python \
  <campaign_directory>/source/scripts/summarize_compute_matched_baselines.py \
  --root <campaign_directory>
```

Results describe **selected validation performance**. Report per-fold seed
variation and descriptive aggregate mean/SD. The nine runs per model are not
nine independent datasets; folds overlap in training data. An independent test
set is required for final held-out performance claims. These runs establish
baseline performance within the allowance; equal compute does not guarantee
equal convergence or that every recipe is optimal. Reprofile constraint losses
and account for calibration costs before fitting constrained models to this cap.

## Verification

Six automated checks pass, including real 64³ forward/backward/optimizer and
checkpoint selection for all three models, interrupted-run accounting, immutable
completed-run verification, split provenance, all 27 experiment identities, arithmetic
caps, validation schedules, aggregate completion, interrupted checkpoint
publication recovery and tamper detection.
The Slurm script also passes `bash -n`.

```bash
python -m unittest scripts.test_compute_matched_baselines -v
bash -n scripts/run_compute_matched_baselines.sbatch
```

CPU synthetic smoke checks validate the training code, not segmentation quality
or CUDA execution. No full baseline run has been performed by these checks.
