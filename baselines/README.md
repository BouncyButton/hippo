# Baselines

This folder contains model code and helpers for baseline segmentation
experiments.

## Main Baselines

| Folder or file | Purpose |
| --- | --- |
| `nnUNet/` | Local nnUNet v2 package used by `scripts/run_nnunet_pipeline.sh`. |
| `unetr_plus_plus/` | UNETR++ package used by `scripts/run_unetrpp_pipeline.sh`. |
| `swin_unetr/` | SwinUNETR model code. |
| `3dunet/` | Simple 3D U-Net code. |
| `M3D_NCA/` | Medical 3D neural cellular automata code. |
| `NCAdapt/` | Continual-learning and neural cellular automata baseline code. |
| `save_nnunet_run.py` | Uploads a trained nnUNet fold to W&B. |
| `save_unetrpp_run.py` | Uploads a trained UNETR++ fold to W&B. |

## Recommended Launch Commands

Use the scripts from the repository root. They prepare splits, build train-only
copies, train the model, upload artifacts, and run evaluation.

Short nnUNet sanity run:

```bash
scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

Full nnUNet run:

```bash
scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode full --folds "0 1 2 3 4"
```

Short UNETR++ sanity run:

```bash
scripts/run_unetrpp_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

Full UNETR++ run:

```bash
scripts/run_unetrpp_pipeline.sh --dataset Dataset102_MNI --mode full --folds "0 1 2 3 4"
```

## Notes

The baseline folders do not include trained weights, virtual environments,
preprocessed data, or result folders. Those are created when the launch scripts
run.

UNETR++ expects a Python 3.8 conda environment. Create it with:

```bash
scripts/setup_unetrpp_env.sh
```

nnUNet is installed automatically by `scripts/run_nnunet_pipeline.sh` if the
`nnUNetv2_train` command is not already available.

