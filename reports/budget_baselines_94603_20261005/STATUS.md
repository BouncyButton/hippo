# Baselines at the 94,603.78 GFLOP training cap

Submitted as one sequential allocation, job **684354**, fold0 seed0, on 2026-10-05. All models use the same ten training volumes and 52 validation volumes, padded 64³ inputs, and batch size1. Original MRI patient grouping remains unverified. No constraints or deliberate learning-rate slowdown are applied.

| Model | Update cap | Volume-pass epochs | Charged GFLOPs at cap |
|---|---:|---:|---:|
| Corrected residual U-Net, width48 |1200|120|94603.7770092|
| SwinUNETR, feature24 |610|61|94557.40297342|
| Planned nnU-Net PlainConvUNet |266|26.6|94508.176416066|

Epoch equivalents count ten batch-one optimizer updates per full training-volume pass. The U-Net and SwinUNETR use Dice; nnU-Net retains its native Dice+CE and deep supervision, with SGD momentum0.95. The U-Net uses AdamW LR1e-4 with100 absolute warmup updates followed by a constant rate; SwinUNETR uses AdamW with warmup and cosine decay; nnU-Net uses polynomial SGD decay. Exact settings are frozen in the model protocols.

Eight targeted local checks passed. Actual cluster model initialization verified parameter counts, losses, optimizers, source hashes, dataset hash, and update caps before submission. Submission and verification evidence are saved in `submission.json` and `preflight.json`.

Final status check: allocation **COMPLETED**, exit code0:0, elapsed10min23sec. All three model completion receipts passed the controller's artifact, source, dataset, split, runtime-recipe, metric, and compute-ledger verification. No training attempts were discarded. Final results are recorded in `summary_final.json`; `summary_snapshot.json` preserves the earlier partial status.

| Model | Best validation macro hard Dice | Final validation macro hard Dice | Final anterior Dice | Final posterior Dice |
|---|---:|---:|---:|---:|
| U-Net |84.13358%|84.13358%|84.85728%|83.40988%|
| SwinUNETR |67.34851%|67.26844%|74.04804%|60.48884%|
| nnU-Net |84.94933%|84.94933%|85.57301%|84.32565%|

U-Net and nnU-Net selected their endpoint checkpoints; SwinUNETR selected update508. These are one-fold, one-seed development results on the52 validation volumes. Successful execution and compute accounting do not establish that each model's hyperparameters are optimal.

The authorized retirement of15 older U-Net fix/pilot run directories completed, deleting12,973,310,984 bytes. Locally verified metadata, sources, metrics, and checkpoint hashes are preserved under `../unet48_stability_20261005/fix_retirement_metadata/`; deletion receipt is `../unet48_stability_20261005/fix_retirement_deletion_receipt.json`. Checkpoint binaries in those retired run directories are no longer available. New baseline runs are separate from the retired directories.

Launcher: `../../scripts/run_budget_baselines.sbatch`. Cluster run root: `/mnt/beegfsstudents/home/3160552/budget_baselines_94603_684354`.
