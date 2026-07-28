# Paper reproduction: SwinUNETR + LTN

This directory reproduces the protocol in
`notebooks/segmentation-ltn-camera-ready.ipynb` for the IJCNN 2025 paper.
It uses only MONAI's `Task04_Hippocampus` Decathlon dataset, not the
repository's pickled `Dataset101_MSD` copy.

## Protocol

- 5-fold `KFold(shuffle=True, random_state=42)`;
- the notebook's first `1.00`, `0.25`, or `0.05` fraction of each fold's
  training indices;
- `Spacingd((1.5, 1.5, 1.5))`, then resize to `64 x 64 x 64`;
- SwinUNETR defaults, AdamW (`1e-4`, `1e-5`), batch size 4, 100 epochs;
- warmup-cosine scheduler with 10 warmup and 100 total epoch steps;
- paper baseline or the notebook's LTN knowledge base.

Each output directory persists the source-data manifest, exact index arrays,
configuration, per-epoch metrics, final metrics, and final weights.

## Important reproducibility note

The official Task04 `dataset.json` describes 260 labelled training cases. The
notebook calls `DecathlonDataset(section="training")` with its default
`val_frac=0.2`, leaving an **effective 208-case dataset** before the notebook's
own KFold. The runner intentionally asserts 208 by default and records both
counts. This differs from the paper's claim of 394 samples, so do not claim an
exact numerical reproduction until the authors' source dataset/version is
identified.

The camera-ready notebook's geometry constraints are computed on `argmax`
masks. That is retained here for protocol fidelity, but it makes those terms
non-differentiable with respect to the segmentation logits. A soft-mask LTN
variant is a separate, corrected extension—not a direct paper replication.

## Cluster use

First prepare the official Decathlon dataset once (do not do this in an array):

```bash
sbatch thesis/paper_reproduction/run_paper_ltn_cluster.sh \
  --prepare-data
```

Then submit the 30 paper runs:

```bash
sbatch --array=0-29%2 thesis/paper_reproduction/run_paper_ltn_cluster.sh --matrix
```

To log every array task and its final checkpoint to W&B, first authenticate in
the `hippocampus` environment with `wandb login`, then append `--wandb`:

```bash
sbatch --array=0-29%2 thesis/paper_reproduction/run_paper_ltn_cluster.sh --matrix --wandb
```

The array mapping is: each fraction in `1.0`, `0.25`, `0.05`, then baseline
folds 1–5 followed by LTN folds 1–5.

After all jobs complete, generate the paper-style aggregate tables:

```bash
python thesis/paper_reproduction/aggregate_paper_results.py \
  --runs-root thesis/runs/paper_reproduction \
  --output-json thesis/runs/paper_reproduction/aggregate.json \
  --output-md thesis/runs/paper_reproduction/aggregate.md
```

`dice_all_classes` is the primary metric because it implements Equation 16 in
the paper using proper one-hot MONAI inputs. `dice_foreground` is reported as a
secondary medical-segmentation metric. The notebook's raw-label `DiceMetric`
call is not valid under current MONAI and is deliberately not used as a target
metric.

If a job is interrupted, resubmit its same configuration with `--resume`; the
stable output directory includes model, optimizer, scheduler, and RNG state.
