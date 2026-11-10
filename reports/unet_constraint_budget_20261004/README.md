# U-Net constraints under the official reference budget

The user adopted `B_ref = 4247 * 9,580,978,301 = 40,690,414,844,347 FLOPs` on 2026-10-04. Its displayed value is approximately **40,690.4 GFLOPs**. Use the exact integer for scheduling; treating the rounded display as the literal cap loses one baseline update.

Same 10 training patients per fold, batch size 1, 64³ inputs, three output classes and MONAI U-Net architecture as the completed baseline study. One patient-pass epoch is 10 optimizer updates. Each run receives its own reference allowance.

| Objective | FLOPs/update | GFLOPs/update | Maximum updates | Patient-pass epochs | Complete epochs + final updates |
|---|---:|---:|---:|---:|---|
| Dice baseline | 9,580,978,301 | 9.580978301 | 4,247 | 424.7 | 424 + 7 |
| Dice + bands | 9,650,184,334 | 9.650184334 | 4,216 | 421.6 | 421 + 6 |
| Dice + original edge | 9,653,481,607 | 9.653481607 | 4,215 | 421.5 | 421 + 5 |
| Dice + bands + original edge | 9,722,687,640 | 9.722687640 | **4,185** | **418.5** | **418 + 5** |

For bands plus edge, `floor(40,690,414,844,347 / 9,722,687,640) = 4,185`. Consumption is 40,689.447773400 GFLOPs, leaving 0.967070947 GFLOPs. The next update would exceed the cap. Stop after five updates in epoch 419; do not train all 419 epochs.

## Profiled objective

Canonical two-step inner/outer balanced BCE bands (`focal_gamma=0`, `degree_alpha=0`) and the original pooled signed-face edge-consistency loss. Both use existing logits. Their weighted sum has one model forward, one combined backward and one AdamW update. Arithmetic counts do not depend on the numerical values of positive scalar coefficients. The profiler uses band weight 0.04, edge weight 0.1 and positive scale 0.2 only to exercise these operations; these are **not newly calibrated U-Net coefficients**.

Bands add approximately 0.722% update FLOPs; bands plus edge add 1.479%. Both constraints independently reconstruct morphology masks, matching existing code rather than assuming a cache. Dense convolution counts include zero kernel entries. No additional network forward is required. The arithmetic overhead is not a wall-clock estimate.

## Counting convention and verification

CPU profiling uses PyTorch 2.2.2 and MONAI 1.6.0, the original baseline arithmetic audit, dense MAC=2, the corrected convolution-backward formula, and scalar/normalization/optimizer estimates. Fused constraint logaddexp/log-sigmoid/sigmoid operations have explicit supplemental formulas in the profiling script. The baseline cost is reproduced exactly. Ellipsoid and box masks give identical counts for all four objectives; complete updates produce finite losses, gradients and parameters. `profiles.json` contains stage counts, operator audits and source hashes. These are declared arithmetic estimates, not exact GPU instructions or empirical convergence guarantees.

The training allowance includes network forward, losses (including GT morphology), backward and optimizer. It excludes validation, preprocessing/augmentation, checkpoint I/O, precision housekeeping, comparisons, integer/boolean operations and memory copies. Additional coefficient calibration or diagnostic gradient probes are **not included**. If performed as part of the constrained method, profile their cost separately and subtract it before scheduling training. Model-specific calibration must not reuse SwinUNETR coefficients without checking their suitability.

Scale the existing 5% learning-rate warmup and cosine schedule to each new maximum update count: 213, 211, 211 and 210 warmup updates, respectively. Freeze a constraint activation schedule before training; warmup does not remove the charged loss cost when the loss is still evaluated. Focal/degree bands, mask caching, additional probes or multiple backward passes require a new profile.

The anchor was selected after inspecting the nine pilot U-Net validation curves; it is pilot-derived. No previous result or frozen protocol was rewritten, and no new training or cluster job was submitted.

Reproduce:

```sh
rtk proxy env PYTHONPATH=/private/tmp/hippo_bref_monai160:/private/tmp/hippo_bref_networkdeps MPLCONFIGDIR=/private/tmp/hippo_bref_mpl .venv/bin/python scripts/profile_unet_constraint_budget.py --output reports/unet_constraint_budget_20261004
```

Temporary dependency paths refer to this machine's profiling environment. Another machine must provide the same MONAI/PyTorch versions and baseline profiler dependencies.

Original-edge checks: 12 tests passed across `thesis/new_constraints/test_edge_consistency.py`, `scripts/test_train_edge_consistency_experiment.py`, and `scripts/test_edge_lowdata_folds.py`. Generic PCGrad checks: five passed and the CUDA-only test was skipped on CPU. The separated-edge implementation and dependent active experiment scripts were retired separately; see `../edge_variant_retirement_20261004/README.md`.
