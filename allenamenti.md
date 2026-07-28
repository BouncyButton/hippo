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
