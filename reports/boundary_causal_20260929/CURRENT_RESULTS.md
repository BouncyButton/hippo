# Matched boundary pilot: current completed results

Case means for scores; validation error counts are totals across52 cases. All use identity-input audits. No incomplete arm is summarized.

| Model | Epoch best/stop | Train Dice% | Val Dice% | Val union Dice% | Shell FP+FN | Val cross correctness% | Val bands BCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed0/dice | 33/60 | 96.0920 | 86.2016 | 88.5550 | 38972 | 42.021 | 0.6286 |
| seed0/dice_aug | 38/60 | 93.0287 | 87.3730 | 89.9662 | 34629 | 46.262 | 0.4904 |
| seed0/sum | 32/60 | 95.3715 | 86.2486 | 88.6480 | 39077 | 42.261 | 0.6116 |
| seed0/sum_aug | 42/60 | 93.3097 | 87.4341 | 90.0089 | 34687 | 46.087 | 0.4735 |
| seed0/pcgrad_historical | 30/historical | 94.6439 | 85.3531 | 87.9573 | 40735 | 40.276 | 0.5450 |

Pairwise effects (candidate minus control):

## seed0/sum_aug_minus_sum

- macro_dice: 0.011854, descriptive paired-case95% interval [0.006789635424358526, 0.017135805951858596]
- boundary_union_errors: -84.423077, descriptive paired-case95% interval [-100.92307692307692, -68.03750000000001]
- assd_voxels: -0.055418, descriptive paired-case95% interval [-0.06739497670603327, -0.044041291114502217]
- bands_bce: -0.138129, descriptive paired-case95% interval [-0.15803645559801507, -0.11953268165771778]

## seed0/dice_aug_minus_dice

- macro_dice: 0.011714, descriptive paired-case95% interval [0.006506473689564709, 0.017262711973142063]
- boundary_union_errors: -83.519231, descriptive paired-case95% interval [-99.36538461538461, -67.90384615384616]
- assd_voxels: -0.056664, descriptive paired-case95% interval [-0.06877319886701701, -0.04523710071873797]
- bands_bce: -0.138192, descriptive paired-case95% interval [-0.15855173361129485, -0.11883195174428132]

## seed0/sum_minus_dice

- macro_dice: 0.000470, descriptive paired-case95% interval [-0.001081477864278604, 0.002028697472632411]
- boundary_union_errors: 2.019231, descriptive paired-case95% interval [-5.365384615384615, 9.346634615384609]
- assd_voxels: 0.001037, descriptive paired-case95% interval [-0.004238040809593995, 0.006363524115980577]
- bands_bce: -0.016959, descriptive paired-case95% interval [-0.027038753118652566, -0.007159830414904998]

## seed0/sum_aug_minus_dice_aug

- macro_dice: 0.000611, descriptive paired-case95% interval [-0.0017105177875596375, 0.002928502458446266]
- boundary_union_errors: 1.115385, descriptive paired-case95% interval [-5.769711538461538, 8.385096153846147]
- assd_voxels: 0.002283, descriptive paired-case95% interval [-0.00305793713086133, 0.00779826251248303]
- bands_bce: -0.016896, descriptive paired-case95% interval [-0.023486084018189173, -0.009925158092608823]

## seed0/sum_minus_pcgrad_historical

- macro_dice: 0.008956, descriptive paired-case95% interval [0.005604585437187743, 0.012455484536657152]
- boundary_union_errors: -31.884615, descriptive paired-case95% interval [-45.82740384615385, -17.980288461538468]
- assd_voxels: -0.019948, descriptive paired-case95% interval [-0.02923461561830862, -0.010766942283470568]
- bands_bce: 0.066567, descriptive paired-case95% interval [0.04766432256079637, 0.08659737383803495]

Intervals resample paired cases within this reused validation fold. They omit checkpoint selection, study selection, training-seed variability, and unknown subject dependence. Epochs are not independent replications.
