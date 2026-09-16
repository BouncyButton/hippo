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

Per-epoch CSV and W&B metrics include the aggregate training loss, Dice-formula
truth, each LTN constraint truth, aggregate constraint satisfaction, hard and
soft anterior/posterior volume gaps, and validation Dice for background,
anterior, and posterior separately. Baseline runs leave LTN-only fields empty
because those constraints are not part of their optimizer objective.

## What is and is not trainable

The Dice formula is differentiable and trains SwinUNETR. In the released
camera-ready notebook, the connectedness, nesting, and volume formulas are
computed after `argmax`. Those three formula values therefore have no gradient
path to the segmentation logits. They still change the nonlinear `SatAgg`
value and can indirectly rescale the Dice gradient, but they cannot teach the
model which voxels should change to satisfy a geometric rule.

The runner exposes two explicitly named volume groundings:

- `paper-hard` (default) reproduces the notebook's `argmax` volume grounding;
- `soft-probability` sums the anterior and posterior softmax probabilities,
  producing expected class volumes and a differentiable Equation 7.

Connectedness and nesting remain paper-hard in both modes. This distinction is
recorded in every run configuration and output path used by epsilon sweeps.

## Important reproducibility note

The official Task04 `dataset.json` describes 260 labelled training cases. The
notebook calls `DecathlonDataset(section="training")` with its default
`val_frac=0.2`, leaving an **effective 208-case dataset** before the notebook's
own KFold. The runner intentionally asserts 208 by default and records both
counts. This differs from the paper's claim of 394 samples, so do not claim an
exact numerical reproduction until the authors' source dataset/version is
identified.

The paper also lists both directions of the nesting rule, while the released
notebook and Equation 11 implement one `Nested` formula. This runner follows
the notebook for reproduction fidelity.

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
the `hippocampus` environment with `wandb login`, then append `--wandb`. The
cluster launcher defaults to the verified `focacciafilippo-bocconi-university`
entity and
`hippopotamus-project` project:

```bash
sbatch --array=0-29%2 thesis/paper_reproduction/run_paper_ltn_cluster.sh --matrix --wandb
```

The array mapping is: each fraction in `1.0`, `0.25`, `0.05`, then baseline
folds 1–5 followed by LTN folds 1–5.

## Epsilon experiments

The paper fixes `gamma_v = 0.0001` and `epsilon = 5000` voxels. Start with the
5% training regime, where the paper reports the largest LTN effect, and sweep
five stricter tolerances:

```bash
sbatch --array=0-24%2 thesis/paper_reproduction/run_paper_ltn_cluster.sh \
  --epsilon-matrix \
  --epsilon-values 0,500,1000,2500,5000 \
  --train-fraction 0.05 \
  --volume-grounding paper-hard
```

Repeat the same matrix with a genuinely differentiable volume term:

```bash
sbatch --array=0-24%2 thesis/paper_reproduction/run_paper_ltn_cluster.sh \
  --epsilon-matrix \
  --epsilon-values 0,500,1000,2500,5000 \
  --train-fraction 0.05 \
  --volume-grounding soft-probability
```

For a custom list containing `N` epsilon values, submit array indices
`0..(5N-1)`. The paper value `5000` acts as the control. Run the hard and soft
matrices in the same output root: their stable directory names include both
epsilon and grounding.

Aggregate completed sweep runs without comparing satisfaction scores computed
at different thresholds:

```bash
python thesis/paper_reproduction/aggregate_epsilon_results.py \
  --runs-root thesis/runs/paper_reproduction \
  --output-json thesis/runs/paper_reproduction/epsilon_aggregate.json \
  --output-md thesis/runs/paper_reproduction/epsilon_aggregate.md
```

The aggregate reports a threshold-independent hard volume gap in voxels,
paper-reference satisfaction at epsilon 5000, configured-epsilon satisfaction,
and the configured violation rate. Dice remains the primary outcome. It also
creates three PNG figures beside the Markdown report:

- all-class and foreground Dice with fold-level standard-deviation bars;
- predicted hard-volume gap, configured satisfaction, and violation rate;
- the Equation 7 satisfaction curve for every tested epsilon.

Use `--plots-dir PATH` to place the figures elsewhere. The Markdown report
embeds all three figures at the end.

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
