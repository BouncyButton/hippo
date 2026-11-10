# Edge-constraint screen: ordinary SwinUNETR, 5%, fold 0, seed 0

Twenty matched training runs compare new edge objectives with a fresh **Dice + bands** baseline. This tests whether boundary supervision can improve the ordinary network; it does not establish that a loss reproduces the auxiliary MRI encoder's mechanism.

## Fixed experiment

- Dataset101 MSD hippocampus; the existing fold-0 split: the same 10 training cases from the 208-case training partition, and 52 validation cases. No additional training cases or seeds.
- Ordinary one-input, three-output SwinUNETR, feature size 24, 15,703,029 trainable parameters. Entire model initialized from seed 0. No auxiliary branch, guide, extra prediction head, plane views or unwrapping.
- Native 1 mm geometry, centered padding to 64³, batch size 1, no augmentation, no pretrained weights. Labels are used to construct training losses only.
- AdamW, learning rate 1e-4, weight decay 1e-5; StepLR every 20 epochs with factor 0.5; CUDA bfloat16 forward and float32 loss calculation.
- Five common Dice-only warmup epochs. Calibrate from these same training cases, ramp constraints over epochs 6–10, then keep coefficients fixed.
- At most 75 epochs; early stopping after at least 25 epochs, patience 8, improvement threshold 0.0005. Save the checkpoint with the highest mean per-case anterior/posterior Dice, even if an improvement is below the early-stopping threshold.

## Twenty runs

Four controls, all starting from the identical initial weights and case order:

| ID | Loss added to Dice + bands |
|---|---|
| `bands` | None: the primary baseline |
| `legacy_edge` | Unmodified original edge constraint |
| `same_only` | Only original edges whose endpoints have the same foreground/background label |
| `split_equal` | Separate original crossing and same-label means, combined equally |

Sixteen further runs cross every combination of these four factors:

| Factor | First setting | Second setting |
|---|---|---|
| Comparison | Difference of probabilities | Difference of scores, using softplus |
| Distance | 1 mm | 1 mm and 3 mm |
| Weighting | Equal pair weights within each distance | Equal weights per boundary voxel, then available distances, then directions |
| Interface | Hippocampus/background | Hippocampus/background plus anterior/posterior |

These sixteen use crossing pairs only. `pairs` gives each available distance half the total weight in the 1+3 mm setting. `anchors` first gives each inside boundary voxel equal total weight, even when a thin structure has no valid longer pair. It splits that weight between available distances and then outward directions. It consequently balances physical support per boundary voxel, not anatomical regions such as head/body/tail.

## Exact loss definitions

Let `z0,zA,zP` be the three segmentation scores. The outer score is `s=logaddexp(zA,zP)-z0`; its sigmoid is exactly total foreground probability. The internal A/P score is `s=zA-zP`; its sigmoid is anterior probability conditional on foreground. For each oriented ground-truth crossing, `i` is inside/anterior and `j` is outside/posterior:

- Probability comparison: `[1 - (sigmoid(s_i)-sigmoid(s_j))]^2`.
- Score comparison: `softplus(s_j-s_i)`, a zero-margin ranking penalty. It retains a gradient for confidently reversed orderings, but cannot alone set absolute probabilities. Dice and bands supply that supervision. This score objective has no finite minimizer by itself and is not a calibrated probability loss.
- Outer + A/P: equal raw mean of the two interface losses, calibrated **jointly to one edge-gradient budget**, not one full budget each. If A/P support is absent the implementation falls back to outer only; the preflight requires both interfaces in all ten actual training cases.
- `legacy_edge` keeps the existing mean squared difference of foreground residuals on all adjacent faces inside the two-step band. `same_only` and `split_equal` keep that original padded-grid support to isolate its composition. The new crossing variants exclude artificial padding; this difference is explicitly included in exported pair counts.
- Dice is the existing MONAI three-class soft Dice including background. Bands are the existing two-step balanced foreground/background BCE without focal or degree weighting.

The wider pair extends an actual 1 mm crossing symmetrically: `i→j` becomes `i-u→j+u`, giving endpoints 3 mm apart. Every intervening voxel must be in the original image and stay in the prescribed class on its side. A pair cannot jump across a second boundary, background gap, or padding. Missing wider pairs are omitted rather than invented by label interpolation. Fully background or single-class regions have no crossing loss; Dice and bands continue to supervise the image where their respective supports exist.

## Calibration and interpretation

Use training-only final-logit gradient RMS to select coefficients. First calibrate bands relative to Dice, targeting a median ratio of 0.1 with a 95th-percentile cap of 0.5. Then calibrate the complete edge loss relative to Dice + weighted bands with the same rule. This is a one-time calibration after epoch 5, not an adaptive gradient limiter or a guarantee of equal parameter updates. All runs must reproduce the warmup and band coefficient. No validation result tunes these coefficients.

Reports include per-case and aggregate A/P Dice, foreground Dice, extra/missed voxels, A/P swaps, foreground mean surface distance and HD95 in millimetres. Save paired differences from Dice + bands and the 32 matched single-factor contrasts inside the sixteen-run matrix. Surface metrics with an empty prediction are undefined and reported as such. Checkpoints are validation selected; this single-fold, single-seed screen ranks candidates for follow-up and provides no independent test-set claim or reliable significance estimate.

## Execution and provenance

Run the twenty experiments sequentially in one `stud` allocation: one `4g.40gb` GPU, four CPUs, 24 GB host RAM, at most 12 hours. The worker first runs a real-data CUDA forward/backward preflight and checks all twenty objectives and ten training geometries. Each run has an atomic resumable checkpoint; completed runs are verified and skipped on a manual resubmission. A failed run stops the worker. No automatic resubmission or new seed is authorized by this script.

The payload freezes source hashes, protocol, local test evidence and dataset hash. Submission checks quota (at least 4 GiB free), refuses ambiguous prior submission attempts, and records the scheduler job ID. Preserve twenty best checkpoints, validation predictions, metrics, loss calibration records, pair counts and reports. Only the active run keeps optimizer state, removed after its completion record is durable. No dataset is copied into this payload and no existing results are deleted.

Local commands from the repository root:

```bash
.venv/bin/python -m scripts.edge_ablation.tests --mode unit --output evaluation_output/edge_ablation_20261008/unit_checks.json
.venv/bin/python -m scripts.edge_ablation.prepare
.venv/bin/python -m scripts.edge_ablation.tests --mode model --device cpu --root docs/experiments/edge_ablation_20261008 --pkl datasets/Dataset101_MSD/msd_hippocampus_full_cluster_20260906.pkl --output evaluation_output/edge_ablation_20261008/model_smoke.json
.venv/bin/python -m scripts.edge_ablation.package
```

The local MONAI 1.4 CPU check is supplemented by the required MONAI 1.6 CUDA check inside the allocation before training starts. The production worker validates the frozen runtime version.
