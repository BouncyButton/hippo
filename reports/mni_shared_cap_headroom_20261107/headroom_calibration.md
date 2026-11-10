# MNI training-headroom calibration

The user chose one shared FLOP cap and authorized model recipe adjustments. The user clarified that 85–90% training Dice is not a rigid requirement. Validation must exceed 50%; assess overfitting using train/validation gaps and curves. Validation now serves as a floor and diagnostic, not a score to maximize. Constraints remain stopped.

Current cap: 95,929,352,001,270 optimization FLOPs, 3× historical and 62.5% below the previous 8× cap. All models use cap-matched LR schedules, with no per-model early stopping.

| Pilot | Model | Cap / historical | Training Dice | Validation Dice |
|---|---|---:|---:|---:|
| mni_shared_cap1p5_train85_90_pilot_20261107 | basic_unet16 | 1.5× | 82.46% | 68.54% |
| mni_shared_cap1p5_train85_90_pilot_20261107 | nnunet | 1.5× | 41.15% | 39.03% |
| mni_shared_cap1p5_train85_90_pilot_20261107 | swinunetr | 1.5× | 91.33% | 74.92% |
| mni_shared_cap1p5_train85_90_pilot_20261107 | unetr_vit_b16 | 1.5× | 74.01% | 64.74% |
| mni_shared_cap2_train85_90_pilot_20261107 | basic_unet16 | 2× | 91.93% | 71.83% |
| mni_shared_cap2_train85_90_pilot_20261107 | nnunet | 2× | 81.74% | 72.72% |
| mni_shared_cap2_train85_90_pilot_20261107 | swinunetr | 2× | 94.87% | 74.25% |
| mni_shared_cap2_train85_90_pilot_20261107 | unetr_vit_b16 | 2× | 77.56% | 67.78% |
| mni_shared_cap3_headroom_lr_refine_20261107 | basic_unet16 | 3× | 87.49% | 67.64% |
| mni_shared_cap3_headroom_lr_refine_20261107 | nnunet | 3× | 88.61% | 74.11% |
| mni_shared_cap3_headroom_lr_refine_20261107 | unetr_vit_b16 | 3× | 90.26% | 71.90% |
| mni_shared_cap3_headroom_recipe_pilot_20261107 | basic_unet16 | 3× | 55.49% | 50.93% |
| mni_shared_cap3_headroom_recipe_pilot_20261107 | nnunet | 3× | 92.82% | 75.56% |
| mni_shared_cap3_headroom_recipe_pilot_20261107 | swinunetr | 3× | 86.28% | 75.56% |
| mni_shared_cap3_headroom_recipe_pilot_20261107 | unetr_vit_b16 | 3× | 79.57% | 70.17% |
| mni_shared_cap3_headroom_refine_20261107 | basic_unet16 | 3× | 90.03% | 69.19% |
| mni_shared_cap3_headroom_refine_20261107 | nnunet | 3× | 91.06% | 75.18% |
| mni_shared_cap3_headroom_refine_20261107 | unetr_vit_b16 | 3× | 90.49% | 70.63% |
| mni_shared_cap3_headroom_unetr_lr045_20261107 | unetr_vit_b16 | 3× | 91.26% | 71.31% |

Accepted calibration cells: nnUNet 88.61%, SwinUNETR 86.28%, UNet 87.49%. These are fold 0 / seed 0 results, not cross-fold confirmation.

UNETR with half-frequency D4 reached 90.49%, 90.26%, and 91.26% as peak LR changed from 0.0006 to 0.00055 to 0.00045. Lower LR did not reliably reduce fitting. The next test uses three-quarter D4 frequency at the original 0.0006 LR, between full D4 (79.57%) and half D4 (90.49%).

The small UNet full-D4 variant collapsed one subfield and was rejected; its accepted recipe uses no augmentation.

All completed cells are archived and verified before checkpoint deletion.

## Selected fixed recipes after clarification

| Model | Training Dice | Validation Dice | Gap |
|---|---:|---:|---:|
| nnunet | 88.61% | 74.11% | 14.50 pp |
| swinunetr | 86.28% | 75.56% | 10.72 pp |
| unetr_vit_b16 | 79.57% | 70.17% | 9.40 pp |
| basic_unet16 | 87.49% | 67.64% | 19.84 pp |

These are four fold-0/seed-0 pilot cells. The confirmation will add the remaining eight fold/seed cells per model. UNETR retains full paired D4 augmentation because weakening it merely to raise training Dice increased the gap. The unsubmitted three-quarter augmentation pilot was abandoned and paused, with no optimizer updates.

None of these pilot results proves absence of overfitting. At the final budget, validation was at its measured best for UNETR and UNet, within 0.01 pp of best for nnUNet, and about 1.01 pp below best for SwinUNETR. The full run will report this behavior across all folds and seeds.

## Confirmation submission

Job 744210 submitted the remaining 32 cells with two GPU workers at the same shared cap. Four selected pilot cells are reused, giving 36 total cells (four models × three folds × three seeds). This is a finite job with at most one queued continuation; completed cells are archived and cleaned individually. The constraint search remains paused.

Startup verified: nnUNet fold 1 / seed 0 completed its 269 updates with 89.40% training and 72.01% validation Dice. SwinUNETR fold 1 / seed 0 had completed 431 updates at the same check. Both workers are functioning; the remaining study is still running. Current free BeeGFS quota: approximately 46.7 GiB.
