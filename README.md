# Hippo project

This repository contains the code needed to run segmentation baselines, build
datasets, and try the semantic constraint experiments.

The repository does not include raw medical images, checkpoints, W&B runs,
virtual environments, or generated outputs. Those files are large and should be
created or downloaded on the machine where the experiments run.

## Documentation

Project guides, research notes, and experiment protocols are collected in
[`docs/`](docs/README.md). Start with the [repository guide](docs/REPO_GUIDE.md).

## Folder Guide

| Folder | What it is for |
| --- | --- |
| `docs/` | Project documentation, research notes, and experiment protocols. |
| `baselines/` | Model code and helper scripts for nnUNet, UNETR++, SwinUNETR, 3D U-Net, and NCA baselines. |
| `datasets/` | Scripts and metadata for building dataset folders in nnUNet-style format. |
| `semantic_constraints/` | Experiments for discovering, selecting, and training with semantic constraints. |
| `scripts/` | End-to-end launch scripts for baseline training and evaluation. |
| `evaluation/` | Evaluation scripts used after baseline training. |

## First Setup

Create a Python environment before running anything:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For UNETR++, use the separate conda setup script:

```bash
scripts/setup_unetrpp_env.sh
```

If you use W&B artifacts, log in before running dataset or model scripts:

```bash
wandb login
```

## Common Workflows

Build or sync a dataset:

```bash
python datasets/Dataset102_MNI/create_mni_dataset.py --target datasets/Dataset102_MNI
```

Run a short nnUNet sanity run:

```bash
scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

Run a short UNETR++ sanity run:

```bash
scripts/run_unetrpp_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

Run the semantic constraint pipeline:

```bash
python semantic_constraints/probe_model.py
python semantic_constraints/evaluate_candidates.py --top-k 15
python semantic_constraints/counterfactual_repair.py --top-k 5
python semantic_constraints/select_constraints.py
python semantic_constraints/train_with_constraints.py
```

Read the corresponding guide in [the documentation index](docs/README.md) before
running a component. Those guides explain expected inputs and output locations.

## 3D validation viewer

View all fold 0 validation patients with native MRI slices, 3D masks, predictions,
and separate foreground/A-P errors:

```bash
python utils/view_fold0_3d.py
```

The default uses the saved `baseline_seed0/error_maps` predictions and opens an
offline HTML viewer. Use `--prediction-dir PATH` for another audited prediction
folder or native-space NIfTI predictions, and `--ground-truth-only` to inspect
annotations without predictions. See [viewer instructions](docs/utils/VIEW_FOLD0_3D.md)
for input formats, controls, and portability to another PC.
