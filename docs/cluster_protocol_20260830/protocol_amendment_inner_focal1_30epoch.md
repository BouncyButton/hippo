# Protocol amendment: inner focal-1 / outer BCE, 30 epochs

Date: 2026-08-30

The original five-epoch gate rejected the inner focal-1 / outer BCE pilot solely
because its foreground false-negative count was 9,622, exceeding the gate's
9,503 maximum by 119 voxels. The pilot nevertheless improved hard Dice from
0.853874 to 0.856618, reduced foreground errors by 450, reduced total mislabeled
voxels by 821, and produced a paired Dice 95% bootstrap confidence interval of
[+0.000596, +0.005047].

After reviewing the gate and the five-epoch warm-up schedule, the user approved
a 30-epoch exploratory extension. This is a transparent post-pilot amendment,
not a claim that the original gate passed. The scientific purpose is to test
whether the observed benefit persists after sustained training at full
constraint strength.

The extension retains the calibrated pilot configuration:

- shared focal gamma: 0
- inner focal gamma: 1
- outer focal gamma: 0 (ordinary BCE)
- calibrated bands weight: 0.011383761103380578
- seed: 0
- constraint warm-up: 5 epochs
- validation/checkpoint cadence: epochs 5, 10, 15, 20, 25, and 30
- total epochs: 30

The run must be interpreted as exploratory because the amendment was made after
observing pilot results. Its trajectory should be compared against the matched
BCE trajectory at each recorded validation epoch, not only at epoch 30.
