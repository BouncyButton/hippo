# Scripts

This folder contains end-to-end launch scripts.

## nnUNet Pipeline

```bash
scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

What it does:

1. Finds or builds the dataset.
2. Creates deterministic train/test and cross-validation splits.
3. Builds a train-only nnUNet raw dataset copy.
4. Runs nnUNet preprocessing and training.
5. Uploads fold artifacts to W&B.
6. Runs evaluation on the test split.

Useful options:

```bash
--dataset Dataset102_MNI
--mode sanity
--mode full
--folds "0 1 2 3 4"
--skip-train
--skip-eval
--max-cases 2
```

## UNETR++ Pipeline

Create the conda environment first:

```bash
scripts/setup_unetrpp_env.sh
```

Then run:

```bash
scripts/run_unetrpp_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

What it does:

1. Activates the UNETR++ conda environment.
2. Finds or builds the dataset.
3. Creates deterministic train/test and cross-validation splits.
4. Builds a train-only UNETR++ task dataset.
5. Runs UNETR++ preprocessing and training.
6. Uploads fold artifacts to W&B.
7. Runs evaluation on the test split.

## W&B Helpers

`download_wandb_pth.py` downloads model weights from W&B.

`upload_wandb_pth.py` uploads model weights to W&B.

