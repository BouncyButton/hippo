# SwinUNETR unwrapped-input follow-up

This campaign tests the three authorized follow-ups to the completed 6 October
unwrapped-input study. It adds 24 training runs and four read-only checkpoint
audits. Completed runs, guide models, atlas geometry, and depth coefficients are
reused. The entire segmentation network and its auxiliary branch remain trainable.

## Registered experiments

| Question | 5% data | Full data | New runs |
| --- | --- | --- | ---: |
| Does the branch still help without the depth loss? | Plane + position; unfolded + position; Dice only; seed 0 | Plane + attention; unfolded + attention; Dice only; seed 0 | 4 |
| Do the selected gains repeat? | Native + depth; plane + position; unfolded + position; seeds 1 and 2 | Native + depth; plane + attention; unfolded + attention; seeds 1 and 2 | 12 |
| Does uncertainty help choose where to apply the correction? | Both position branches, spatial entropy gate and mean-entropy control; seed 0 | Both attention branches, spatial entropy gate and mean-entropy control; seed 0 | 8 |

The selected unfolded variants had the highest validation Dice among unfolded
models at each data fraction. Each keeps its matched plane control. This is a
follow-up on a validation-selected configuration, not independent confirmation
on an untouched test set. The plane branch uses the same MRI samples and branch
capacity; the spatial arrangement differs from the actual unwrapped charts.

## Fixed protocol

MSD hippocampus, fold 0, identical train/validation case IDs to the prior study,
5% and full training sets, 52 validation cases, 75-epoch maximum, minimum 25 epochs,
early-stopping patience 8 and minimum improvement 0.0005. All original model,
preprocessing, optimization, precision, and checkpoint-selection settings are
retained. AdamW uses learning rate 0.0001 and weight decay 0.00001; StepLR halves
the learning rate every 20 epochs. No new augmentation or extra loss is added.

The Dice-only ablation removes depth loss completely. All other new runs use
Dice plus the original depth-balanced cross-entropy: five Dice-only warm-up
epochs, then the original five-epoch ramp and frozen per-fraction coefficients.
There is no SDF, topology loss, extra segmentation head, or branch-only training.

The seed replication changes segmentor initialization and training-case order.
It deliberately shares the completed seed-0 out-of-fold guide predictions,
unwrapped atlases, and loss coefficients. Results therefore measure segmentor
variability conditional on fixed geometry, not variability of the entire guide
generation pipeline. Seed-0 results are reused rather than retrained.

## Uncertainty fusion

The native SwinUNETR decoder produces features `F`; the existing branch and
fusion block produce a correction `R`. Ordinary fusion predicts from `F + R`.
The new variant predicts from `F + g * R`, where `g` is the detached entropy of
the native path's three-class softmax, divided by `log(3)`. Thus `g` lies between
zero and one. It changes the size of the correction, with no added parameters
or loss terms. The native path uses the same classifier as the final prediction;
its entropy is an uncalibrated confidence proxy, not an independently supervised
uncertainty estimate.

The control broadcasts the volume's mean entropy to every voxel. It controls
for average gate strength at a given model state, allowing comparison with
spatial selection. Independently trained models can develop different mean
entropies, so the control does not guarantee identical residual norms throughout
training. Confidently wrong native predictions are a known limitation of the
spatial gate; the diagnostics and missing-tissue counts help detect this failure.

## Existing-checkpoint diagnostics and evaluation

Before new training at each fraction, evaluate the two selected existing branch
checkpoints with (1) normal fusion, (2) mapped auxiliary features zeroed while
support and fusion remain, and (3) the native path alone. These are ablations of
the same jointly trained checkpoint. The native-only ablation is distinct from
the separately trained native baseline.

Record missing tissue introduced or recovered by fusion relative to that native
path, stratified by target depth bands and distance to the target surface, plus
native entropy at newly missed tissue. Foreground recovery does not necessarily
mean a correct anterior/posterior label. Annotations are used only for the loss
and evaluation, never to construct the auxiliary MRI inputs.

Every run saves native-space per-case anterior/posterior Dice, foreground Dice,
missing and extra voxels, label swaps, mean surface distance, HD95, and depth-band
errors. The report includes paired case differences and descriptive bootstrap
intervals; replications also report seed variability. Comparisons retain the
matched data fraction, architecture, and training seed. The main question is
whether a gain survives the matched plane control and avoids increasing missed
tissue, not merely whether one validation Dice is the largest.

## Execution, recovery, and storage

One sequential Slurm GPU job runs the campaign, after any already-running user
job. Source, protocol, dataset, frozen atlases, and selected old results are
checked by SHA-256 before use. Local tests cover objective isolation, entropy
semantics, feature-fusion equivalence, full-network gradients, interruption
recovery, and completed-run reuse. Production CUDA checks on the largest cached
atlas at each fraction run before its scientific training. Failure stops the
campaign. Each run can resume model, optimizer, scheduler, and RNG state; verified
completed runs are skipped.

The initial BeeGFS check showed 56.41 GiB used of 93.13 GiB (36.72 GiB free).
Submission rechecks the live quota and requires at least 6 GiB free. Atlases and
dependencies are shared with the previous campaign rather than duplicated. No
prior outputs need deletion with the measured quota. The submission receipt
records the final quota and queue dependency.

Machine-readable details: `protocol.json`, `local_checks.json`, and, after
submission, `submission.json`. Remote output is the dedicated
`unwrapped_followup_fold0_20261007_01` directory under the user's BeeGFS home.
