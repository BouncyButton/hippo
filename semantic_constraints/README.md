# Semantic Constraints

This folder contains experiments for adding semantic constraint losses to a
segmentation model.

The idea is simple:

1. Run a trained model on validation data.
2. Measure useful properties of the prediction and the ground truth.
3. Search for rules that are usually true in the ground truth.
4. Test whether those rules can repair bad predictions.
5. Train again with the selected rules as an extra loss.
6. Compare against a matched baseline without the extra loss.

For more detail, read `ARCHITECTURE.md`.

## Main Files

| File | Purpose |
| --- | --- |
| `primitives.py` | Differentiable measurements like volume, distance, overlap, compactness, and connectedness. |
| `registry.py` | Names and metadata for the measurements that can be searched. |
| `probe_model.py` | Runs a frozen model and writes prediction-vs-ground-truth measurements. |
| `evaluate_candidates.py` | Ranks possible constraint rules from the probe output. |
| `counterfactual_repair.py` | Tests whether a rule can locally repair frozen predictions. |
| `automatic_constraint_discovery.py` | Cross-fits candidates and selects rules with aligned, statistically useful repairs. |
| `select_constraints.py` | Writes the final selected constraints to JSON. |
| `compiled_losses.py` | Turns selected JSON constraints into PyTorch losses. |
| `train_with_constraints.py` | Fine-tunes a model with or without constraint losses. |
| `final_evaluate.py` | Compares trained checkpoints on the reserved holdout fold. |
| `on_grokking_behavior.py` | Separate toy experiment. It is not part of the segmentation pipeline. |

## Expected Inputs

The scripts need:

- a dataset available through MONAI Decathlon style loading;
- a model checkpoint for `probe_model.py` or `train_with_constraints.py`;
- Python packages from the root `requirements.txt`;
- usually a GPU for real training.

Checkpoints and generated probe outputs are not included in this repository.

## Quick Pipeline

Run these commands from the repository root:

```bash
python semantic_constraints/probe_model.py
python semantic_constraints/evaluate_candidates.py --top-k 15
python semantic_constraints/counterfactual_repair.py --top-k 5
python semantic_constraints/select_constraints.py
python semantic_constraints/train_with_constraints.py
```

The stricter automatic procedure combines induction, gradient-alignment
screening, counterfactual repair, bootstrap confidence intervals, and final
selection in one cross-fitted run:

```bash
python semantic_constraints/automatic_constraint_discovery.py \
  --load-weights semantic_constraints/runs/baseline/best_model_weights.pth \
  --root-dir tmp \
  --train-fraction 0.1
```

Its `selected_constraints.json` can be passed directly to
`train_with_constraints.py --constraints`.

The default output folder is:

```text
semantic_constraints/probe_outputs/
```

Training outputs are written under:

```text
semantic_constraints/runs/
```

Those folders are ignored by git.

## Smoke Tests

Use small limits when checking that the code runs:

```bash
python semantic_constraints/probe_model.py --max-batches 2 --device cpu
python semantic_constraints/evaluate_candidates.py --top-k 5
python semantic_constraints/counterfactual_repair.py \
  --candidate-ranks 1 \
  --gammas 0.1 \
  --steps 2 \
  --max-batches 2 \
  --device cpu
```

CPU smoke tests are useful for checking the pipeline. They are not meaningful
experiments.

## Train A Matched Baseline

Always compare a constrained run to a baseline run with the same data,
checkpoint, optimizer settings, and number of epochs.

Baseline fine-tuning:

```bash
python semantic_constraints/train_with_constraints.py \
  --no-constraints \
  --holdout-fold 2 \
  --init-weights path/to/initial_checkpoint.pth \
  --output-dir semantic_constraints/runs/baseline_finetune
```

Constrained fine-tuning:

```bash
python semantic_constraints/train_with_constraints.py \
  --holdout-fold 2 \
  --init-weights path/to/initial_checkpoint.pth \
  --constraints semantic_constraints/probe_outputs/selected_constraints.json \
  --output-dir semantic_constraints/runs/constrained_finetune
```

Final comparison:

```bash
python semantic_constraints/final_evaluate.py \
  --holdout-fold 2 \
  --model baseline semantic_constraints/runs/baseline_finetune/best_model_weights.pth \
  --model constrained semantic_constraints/runs/constrained_finetune/best_model_weights.pth
```

Do not use the final holdout fold while choosing constraints. Use it only after
the rules and training settings are fixed.

## Primitive Example

```python
import torch
from semantic_constraints import distance, volume

loss_distance = torch.relu(distance(mask_a, mask_b, spacing=(1.5, 0.5, 1.5)) - 12.0)
loss_ratio = torch.relu(volume(mask_a) / volume(mask_b).clamp_min(1e-8) - 0.4)
```

Masks should be soft class masks shaped `(B, *spatial)` or `(B, 1, *spatial)`.
