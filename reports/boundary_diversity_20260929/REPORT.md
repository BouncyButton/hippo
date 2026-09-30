# Training diversity with matched updates and actual learning rates

The treatment changes unique case diversity and repeated exposure at fixed updates and actual learning rates. One seed and a reused validation fold; case intervals omit seed and selection uncertainty. Train totals must not be compared across different cohort sizes. Paired training scores use only the original common ten cases.

| Model | Selected/stop | Train cases | Train Dice% | Validation Dice% | Shell errors, 52 validation cases |
|---|---:|---:|---:|---:|---:|
| ten_repeated | 19/60 | 10 | 97.263 | 81.031 | 54144 |
| fifty_unique | 32/60 | 50 | 95.371 | 86.249 | 39077 |

All 60 passes match actual learning rates, warmup and updates; no AMP steps were skipped.

Fifty minus ten, validation effects:

- selected: Dice +5.217 percentage points; shell errors -15067; ASSD -0.1804 grid voxels.
- pass 15: Dice +3.384 percentage points; shell errors -9433; ASSD -0.1138 grid voxels.
- pass 30: Dice +5.423 percentage points; shell errors -15059; ASSD -0.1828 grid voxels.
- pass 45: Dice +5.778 percentage points; shell errors -15737; ASSD -0.1850 grid voxels.
- pass 60: Dice +5.759 percentage points; shell errors -15588; ASSD -0.1828 grid voxels.
