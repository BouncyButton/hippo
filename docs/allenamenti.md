# Training runs

## SwinUNETR — MSD, fold 0

**Completed:** 2026-07-22  
**Training job:** Slurm `600169` (replacement for `600166`, which failed before training because Slurm staged the batch script under `/var/spool`).  
**Inference job:** Slurm `600236`.

### Data split

- Dataset: `datasets/Dataset101_MSD/msd_hippocampus_full.pkl`
- CV split: `datasets/Dataset101_MSD/splits_final.json`
- Fold: `0`
- Training cases: `208`
- Held-out validation/inference cases: `52`
- Classes: background plus two foreground hippocampus labels (`3` total)

There is no separate external test split in this setup. The reported inference results use fold 0's 52 held-out validation cases.

### Training configuration

- Model: MONAI `SwinUNETR`, one input channel, three output classes, gradient checkpointing enabled
- Epochs: `50`
- Batch size: `2`
- Input spatial size: `64 × 64 × 64`
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Scheduler: StepLR, halve learning rate every `20` epochs (`gamma=0.5`)
- GPU: NVIDIA A100 80 GB PCIe, 40 GB MIG slice
- W&B: disabled for this run

The batch launcher is `scripts/run_swinunetr_cluster.sh`. It must resolve the repository through `SLURM_SUBMIT_DIR`; resolving it from the batch script path fails because Slurm executes a staged copy.

### Checkpoint

```text
/mnt/beegfsstudents/home/3160552/hippo/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0/model.pt
```

### Held-out inference results

Inference reused `evaluation/evaluate.py`'s `evaluate_swinunetr` function, with the same preprocessing and fold selection used during training.

| Metric | Value |
| --- | ---: |
| Hard Dice | 0.8729 |
| Soft Dice | 0.8653 |
| IoU / Jaccard | 0.7754 |
| Recall | 0.8681 |
| Precision | 0.8798 |
| Accuracy | 0.9970 |
| HD95 | 1.4577 |

The final training-log hard Dice was `0.8737`, consistent with inference.

Outputs from inference:

```text
/mnt/beegfsstudents/home/3160552/hippo/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0/inference_fold0_val_600236/
```

This directory contains `metrics_summary.json`, `prediction_index.csv`, 52 predicted masks (`*_pred.npy`), and 52 probability tensors (`*_prob.npy`).

## SwinUNETR + translation equivariance — MSD, fold 0

**Completed:** 2026-07-30  
**Training job:** Slurm `611071` (`swin-trans-f0`)  
**State:** `COMPLETED`, exit code `0:0`, elapsed time `00:27:54`

### Question and protocol

This was the first isolated experiment with the differentiable translation-equivariance constraint. It used the same MSD fold 0 split as the unconstrained SwinUNETR baseline above: 208 training cases and 52 held-out validation cases. The model was trained from scratch; it was not initialized from the baseline checkpoint.

Submission command:

```bash
sbatch --job-name=swin-trans-f0 \
  thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set translation \
  --equivariance-weight 0.10 \
  --fold 0 \
  --epochs 50 \
  --batch-size 2 \
  --no-amp \
  --seed 0 \
  --constraint-warmup-epochs 5 \
  --constraint-eval-every 5
```

The remaining training settings matched the baseline: `64 × 64 × 64` inputs, AdamW with learning rate `1e-4` and weight decay `1e-5`, and StepLR with period `20` and `gamma=0.5`. For each training batch, one of the six axis-aligned translations of two voxels was sampled. Because `equivariance_max_samples=1`, the additional translated forward pass was applied to the first sample of each batch of two.

### Results

| Metric | Value |
| --- | ---: |
| Final hard Dice, epoch 50 | `0.8793` |
| Best validation hard Dice, epoch 21 | `0.8824` |
| Final optimization satisfaction, all six translations | `0.9622` |
| Final legacy linear satisfaction | `0.9295` |
| Final legacy adherence at threshold `0.90` | `0.9391` (`293/312`) |

The final hard Dice improved by approximately `+0.0064` over the baseline held-out inference Dice of `0.8729`, or about `+0.64` percentage points. The best validation epoch improved by approximately `+0.0095`, but this is a validation-selected value and should not be treated as an independent test result. The final epoch comparison is the more conservative preliminary result.

The primary equivariance measurement is the quadratic-denominator satisfaction used by the differentiable loss (`0.9622`). The legacy linear score and its thresholded adherence also depend on probability confidence: identical but uncertain soft maps can score below one. Furthermore, `293/312` refers to passing case-translation pairs (`52` cases × `6` directions), not to patients satisfying all six translations.

The logged `val_dice_soft=0.01270685` is invalid and must not be compared with the baseline soft Dice. The inherited MONAI evaluator converts positive softmax probabilities to Boolean masks. This does not affect the supervised training loss or the reported hard Dice.

### Artifacts

Run directory:

```text
/mnt/beegfsstudents/home/3160552/hippo/models/swin_unetr_new_constraints/msd_fold0_translation_20260730_121523_611071/
```

Important files are `config.json`, `metrics.csv`, `final_metrics.json`, `checkpoint_best.pt`, `checkpoint_latest.pt`, and `MSD_fold0/model.pt`.

This is encouraging but not yet a controlled causal result. The next required comparison is `constraint-set none` through the same runner, followed by repetitions on additional seeds or folds.

## SwinUNETR + LTN — full-data five-fold epsilon sweep

**Completed:** 2026-07-30

**Sweep root:** `thesis/runs/paper_reproduction/full_fraction_epsilon_pilot_20260729`

**Analysis:** `thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729`

### Question and protocol

This experiment tested how the volume-tolerance parameter `ε` affects segmentation and constraint adherence in the paper-reproduction LTN objective. The grid was `ε ∈ {0, 500, 1000, 2500, 5000}` and every value was run on the same five folds, producing 25 completed runs.

- Method: `ltn`
- Model: MONAI `SwinUNETR`, one input channel and three output classes
- Training fraction: `1.0`
- Volume grounding: `paper-hard`
- Folds: `5`, shuffled with split seed `42`
- Model/data-loader seed: `42`
- Epochs: `100`
- Batch size: `4`
- Input spatial size: `64 × 64 × 64`
- Resampling spacing: `1.5 × 1.5 × 1.5`
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-5`
- Scheduler: 10-step warm-up followed by cosine decay
- Volume sharpness: `γ=0.0001`

All values below are the mean ± sample standard deviation across the five folds (`n=5`). Foreground Dice is the mean of anterior and posterior Dice; all-class Dice also includes background.

### Segmentation results

| ε | All-class Dice | Foreground Dice | Background Dice | Anterior Dice | Posterior Dice |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8624 ± 0.0086 | 0.7982 ± 0.0127 | 0.9906 ± 0.0006 | 0.8114 ± 0.0122 | 0.7850 ± 0.0133 |
| 500 | 0.8637 ± 0.0085 | 0.8002 ± 0.0124 | 0.9907 ± 0.0006 | 0.8130 ± 0.0115 | 0.7873 ± 0.0136 |
| 1000 | 0.8661 ± 0.0086 | 0.8038 ± 0.0126 | 0.9909 ± 0.0006 | 0.8153 ± 0.0121 | 0.7922 ± 0.0132 |
| **2500** | **0.8673 ± 0.0085** | **0.8054 ± 0.0125** | **0.9911 ± 0.0006** | **0.8179 ± 0.0113** | **0.7930 ± 0.0137** |
| 5000 | 0.8663 ± 0.0082 | 0.8040 ± 0.0120 | 0.9910 ± 0.0005 | 0.8170 ± 0.0106 | 0.7909 ± 0.0136 |

`ε=2500` gave the best mean all-class and foreground Dice. Relative to the paper setting `ε=5000`, its paired foreground improvement was `+0.00147` (`+0.147` percentage points, paired SD `0.00067`) and all five folds improved. Its paired all-class improvement was `+0.00099` (`+0.099` percentage points), again with five improvements out of five. These are directionally consistent but very small absolute gains; with only five folds they should not be presented as definitive significance evidence.

The foreground difference was mainly posterior: relative to `ε=5000`, `ε=2500` changed background Dice by only `+0.00003`, anterior Dice by `+0.00081`, and posterior Dice by `+0.00212`.

### Held-out constraint adherence

The most interpretable threshold-independent volume diagnostic is the absolute anterior/posterior hard-volume gap. Configured volume truth and violation rate additionally depend on the chosen `ε`, so their absolute values are not directly comparable between rows.

| ε | Hard volume gap (voxels) ↓ | Configured volume truth ↑ | Volume violation rate ↓ | Connectedness score ↑ | Nesting-event rate ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1100.2 ± 70.5 | 0.0339 ± 0.0326 | 1.0000 ± 0.0000 | 0.5249 ± 0.0083 | 0.0620 ± 0.0763 |
| 500 | 1102.1 ± 86.5 | 0.2978 ± 0.0437 | 0.7354 ± 0.0308 | 0.5252 ± 0.0080 | 0.0628 ± 0.0374 |
| 1000 | 1059.7 ± 99.7 | 0.6011 ± 0.0532 | 0.4184 ± 0.0527 | 0.5238 ± 0.0081 | **0.0386 ± 0.0219** |
| **2500** | **1036.2 ± 126.7** | 0.9694 ± 0.0147 | 0.0386 ± 0.0219 | 0.5227 ± 0.0112 | 0.0580 ± 0.0442 |
| 5000 | 1121.7 ± 101.2 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5227 ± 0.0078 | 0.0770 ± 0.0356 |

The main adherence findings were:

- `ε=0` was too strict for the observed predictions: every validation case violated the configured volume tolerance and mean configured truth was only `0.0339`.
- Increasing `ε` necessarily raised configured volume truth and lowered the configured violation rate. This is partly a change in the scoring rule, not necessarily a change in the predictions.
- `ε=2500` had the smallest mean hard volume gap (`1036` voxels), compared with `1122` at `ε=5000`. However, the fold standard deviations were large relative to this difference, so the evidence for a genuine volume improvement is weak.
- Connectedness was effectively unchanged across the sweep (`0.5227–0.5252`). There is no evidence that changing `ε` improved connectedness.
- Nesting events were uncommon but non-monotonic. `ε=1000` had the lowest mean event rate, while the Dice-optimal `ε=2500` was intermediate. There is no consistent nesting improvement as `ε` increases.

### Training-time constraint truths

The following are mean values at epoch 100 across the five folds. They describe the terms supplied to SatAgg during training and are not numerically identical to the held-out structural diagnostics above because their transformations and reductions differ.

| ε | Total satisfaction ↑ | Dice truth ↑ | Connectedness truth ↑ | Volume truth ↑ | Nesting truth ↑ | Training loss ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5096 | 0.9470 | 0.9739 | 0.0239 | 0.9880 | 0.4904 |
| 500 | 0.5843 | 0.9450 | 0.9739 | 0.1744 | 0.9880 | 0.4157 |
| 1000 | 0.6513 | 0.9423 | 0.9739 | 0.3126 | 0.9856 | 0.3487 |
| 2500 | 0.8834 | 0.9693 | 0.9739 | 0.8039 | 0.9856 | 0.1166 |
| 5000 | 0.9720 | 0.9806 | 0.9739 | 0.9999 | 0.9751 | 0.0280 |

The total satisfaction and training loss are dominated by the configured volume truth. Here, training loss is exactly `1 - SatAgg`, not a conventional loss that can be compared independently of the selected `ε`. Consequently, the very low loss at `ε=5000` does not mean that it learned a better segmentation or better anatomy.

### Equation 7 dead zone and gradient interpretation

The configured volume truth is

```text
truth = exp[-γ × max(volume_gap - ε, 0)²]
```

This creates an exact dead zone: whenever the predicted gap is at or below `ε`, volume truth is `1`. Observed validation fold-mean gaps were approximately `850–1200` voxels, so `ε=5000` placed all of them deep inside the dead zone. Its perfect configured validation truth and zero violation rate therefore show that the threshold was permissive, not that the model achieved the smallest volume discrepancy.

Under `paper-hard`, volume, connectedness, and nesting are computed from `argmax` masks. These operations are non-differentiable. The hard truth values enter SatAgg, but they do not supply an anatomical correction direction to the network; instead, they mainly alter the scalar applied to the differentiable Dice gradient through SatAgg. This explains why satisfaction and loss separate strongly across `ε`, while Dice, connectedness, and nesting remain comparatively close.

### Relation to the unconstrained baseline

The earlier held-out SwinUNETR baseline above reported hard foreground Dice `0.8729`, but it used a different split and training/evaluation protocol (`208/52` cases, 50 epochs and a different metric aggregation), so it cannot be directly compared with this sweep.

Within the paper-reproduction pipeline, the only completed matched no-LTN baseline is fold 1: foreground Dice was `0.8045`, versus `0.8056` for `ε=2500` on the same fold, a gain of only `0.0011`. A five-fold no-LTN control is required before claiming that LTN improves over an unconstrained model.

### Conclusion

`ε=2500` is the preferred setting for the current `paper-hard` reproduction because it achieved the highest Dice, improved over `ε=5000` in every paired fold, and had the lowest mean hard volume gap. The evidence does **not** show that progressively increasing `ε` improves connectedness or nesting, and the near-perfect configured adherence at large `ε` is largely caused by the Equation 7 dead zone. The most defensible conclusion is that `ε=2500` is a small optimization sweet spot within this LTN formulation, not evidence that the non-differentiable hard constraints taught the network better anatomy.

A targeted experiment using soft/differentiable volume grounding is justified if the next question is whether the constraint can directly shape the predicted anatomy rather than merely rescale the Dice gradient.

### Analysis artifacts

- [Full written analysis](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/thesis_progress_update.md)
- [Dice versus epsilon](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/01_dice_vs_epsilon.png)
- [Paired Dice changes versus ε=5000](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/02_paired_dice_change_vs_5000.png)
- [Classwise Dice](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/03_classwise_dice_vs_epsilon.png)
- [Volume and constraint behavior](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/04_volume_constraint_behavior.png)
- [Equation 7 dead zones](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/05_equation7_dead_zones.png)
- [Learning curves and truth trajectories](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/06_learning_curves.png)
- [Dice/constraint selection trade-off](../thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/07_selection_tradeoff.png)
