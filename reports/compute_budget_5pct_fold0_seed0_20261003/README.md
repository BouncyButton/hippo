# Reference training compute: 5% MSD, fold 0, seed 0

The reconstructed reference allowance is **approximately 1.16 × 10^14 FLOPs
(116.26 TFLOPs)** for the unconstrained SwinUNETR run
`low_data_noaug_seed0_20260917_01/runs/baseline_5pct_seed0_noaug`.
This is a total operation count, not FLOPs per second or GPU capacity.

## Run identity and update count

- Dataset: MSD hippocampus; fold 0; seed 0; 10 training cases (5% subset).
- Input: `[1, 1, 64, 64, 64]`; three output classes.
- Architecture: historical `build_swinunetr`, MONAI 1.6.0 defaults, feature size 24,
  depths `(2, 2, 2, 2)`, heads `(3, 6, 12, 24)`, 15,703,029 parameters.
- No activation checkpointing, dropout, augmentation, pretrained initialization,
  or auxiliary constraint.
- Dice loss; AdamW, learning rate 0.0001, weight decay 0.00001.
- 75 epochs × 10 batches = **750 training updates**.
- The historical audit confirms stopping at epoch 75 and selection at epoch 73.
  The reference uses all 750 updates, rather than only the 730 updates preceding
  the selected checkpoint.
- Saved configuration SHA-256:
  `fe46958445bc29406c0bc1983e4174c4641f2dc84d36daddb5a0000bbfd7d134`.

Sources are the saved configuration under
`reports/separated_edge_generalization_20260929/historical_baseline_configs/`,
`reports/separated_edge_generalization_20260929/historical_baseline_trainer.py`,
and run R01 in
`reports/hippocampus_research_review_20260924/evidence/normalized.json`.
The saved constraint-zero source matches its historical SHA-256.

## Arithmetic accounting

A multiply and an addition count as two FLOPs. Basic scalar arithmetic and
elementary functions each count as one operation; memory operations, integer
indexing and comparisons are excluded. Dense convolution padding follows the
usual full-kernel convention, rather than counting device-specific skipped
boundary multiplies.

| Stage | Dense kernel FLOPs/step | Supplementary estimated FLOPs/step | Combined GFLOPs/step |
| --- | ---: | ---: | ---: |
| Model forward | 50,859,802,176 | 1,214,965,317 | 52.0748 |
| Dice loss and baseline zero term | 0 | 6,029,345 | 0.0060 |
| Backward | 101,354,699,904 | 1,356,796,723 | 102.7115 |
| AdamW update | 0 | 219,842,557 | 0.2198 |
| Total | 152,214,502,080 | 2,797,633,942 | **155.0121** |

Thus the working estimate is

`B_ref = 750 × 155,012,136,022 = 116,259,102,016,500 FLOPs`.

For comparison, the reproducible **dense convolution/matrix-only** count is
`114,160,876,560,000 FLOPs`. The supplementary estimate contributes 1.80% of
the combined total. Do not mix these accounting definitions across methods.

Supplementary formulas are explicit in `supplementary_flops` in the profiler.
They cover scalar arithmetic, reductions, softmax, bias operations, Dice and
AdamW; fused normalization and activation kernels use declared arithmetic
estimates. GELU uses 5 operations/element forward and 10 backward; normalization
uses mean/variance/normalization and optional affine-operation estimates.
The results are arithmetic estimates, not exact historical hardware instruction
counts. Many displayed integer digits reflect reproducible bookkeeping rather
than physical measurement precision; cite approximately **1.16 × 10^14 FLOPs**.

## Verification and limitations

The same accounting was run on shape-only meta tensors and a real CPU training
step, including forward, loss, backward and AdamW. All stage totals agree
exactly; see `verification.json`, `profile_meta.json` and `profile_cpu.json`.
Synthetic inputs are sufficient because the model has no value-dependent
compute branches under this fixed-shape configuration; historical weights and
MRI data are unnecessary for this reconstruction.

The original runtime was PyTorch 2.12.1+cu130 / MONAI 1.6.0 on an A100 GPU.
Profiling used local PyTorch 2.2.2 / the exact MONAI 1.6.0 package. An incorrect
convolution weight-gradient counting formula in the older PyTorch counter was
replaced with conventional dense MAC accounting. Neither library installations
in the repository environment nor historical experiment files were changed.

The CPU reconstruction uses float32 and explicit scalar AdamW (`foreach=False`),
while the historical GPU run used AMP and the default optimizer backend.
Operation counts follow declared mathematical formulas rather than GPU kernel
implementations. CUDA GradScaler overhead is excluded; its gradient unscale is
approximately another 15.7 million multiplications per step, about 0.01% of
this estimate. Tiny scalar scheduler/optimizer bias-correction calculations
outside tensor dispatch are also excluded.

Validation (52 cases per epoch), final evaluations, data preprocessing/loading,
checkpoint I/O and historical diagnostic audits are excluded. Therefore this
budget is a **training allowance**, not the complete experiment's cost. No
constraint calibration is charged to this unconstrained baseline.

## Reproduction

Install the historical MONAI into a temporary directory without changing the
project environment, then run the saved profiler:

```bash
rtk proxy .venv/bin/python -m pip install --no-deps --no-cache-dir --target /private/tmp/hippo_bref_monai160 monai==1.6.0
rtk proxy env HIPPO_PROFILE_MONAI_PATH=/private/tmp/hippo_bref_monai160 MPLCONFIGDIR=/private/tmp/hippo_bref_mpl .venv/bin/python reports/compute_budget_5pct_fold0_seed0_20261003/profile_reference.py
rtk proxy env HIPPO_PROFILE_DEVICE=cpu HIPPO_PROFILE_MONAI_PATH=/private/tmp/hippo_bref_monai160 MPLCONFIGDIR=/private/tmp/hippo_bref_mpl .venv/bin/python reports/compute_budget_5pct_fold0_seed0_20261003/profile_reference.py
```

## Suggested paper wording

“We anchored the training-compute allowance to a preliminary unconstrained
SwinUNETR experiment on the 5% training subset (fold 0, seed 0). Its 75-epoch
allowance comprised 750 updates, with an estimated cost of 155 GFLOPs per
update, yielding a reference allowance of approximately 1.16 × 10^14 FLOPs.
Counts include forward, backward, supervised loss and optimizer arithmetic under
an explicitly documented counting convention; validation is reported separately.”
