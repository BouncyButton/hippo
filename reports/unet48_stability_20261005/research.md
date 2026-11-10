# Width-48 U-Net stability investigation

## Observed failure
The completed budget pilots switch from ~45.8% to ~82% macro validation Dice between 4,000 and 4,100 updates under distinct budget-scaled learning-rate trajectories. This does not establish a compute threshold: the earlier updates have different learning rates.

The failed candidate06 predicts 2,565,755 posterior false positives across 10 training and 52 validation cases. Of these, 757,294 occur in zero-intensity background; 99.833% of that subset lie on odd D planes. Successful candidate07 has 19,137 posterior false positives and only six on zero-intensity background. An entirely zero input produces 52,165 posterior predictions in candidate06 versus 2,086 in candidate07. These are checkpoint diagnostics, not an image-level causal intervention.

## Primary-source research
- Odena, Dumoulin and Olah (2016), [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/): stride-2 kernel-3 transposed convolution has uneven overlap. Even-overlap kernels and resize-convolution are proposed remedies; even-overlap kernels can still learn artifacts. Their experiments concern image generation, so transfer to this segmentation failure is a hypothesis.
- [MONAI 1.6 UNet source](https://github.com/Project-MONAI/MONAI/blob/1.6.0/monai/networks/nets/unet.py): this residual U-Net uses stride-2 transposed convolution with default kernel size 3; the top upsampling operation has normalization and PReLU followed by a residual unit with a linear last convolution.
- [MONAI 1.6 Dice source](https://github.com/Project-MONAI/MONAI/blob/1.6.0/monai/losses/dice.py): background is included by default; documentation suggests excluding it when small foregrounds are overwhelmed.
- [MONAI discussion #7681](https://github.com/Project-MONAI/MONAI/discussions/7681): similar late-class learning in a residual 3D U-Net, but 36-40 classes and different normalization/activation. Widening did not eliminate delayed learning. No established fix for our exact architecture/recipe.
- [MONAI issue #807](https://github.com/Project-MONAI/MONAI/issues/807): historical Dice instability with zero gradients under older MONAI and a 2D model. Our checkpoint has finite, nonzero gradients and matched BF16/FP32 inference, so this report is not evidence of the same failure.
- Cabezas and Diez (2024), [An Analysis of Loss Functions for Heavily Imbalanced Lesion Segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10976047/): describes high recall with excessive false positives under Dice, and examines cross-entropy alternatives. Compound Dice/CE is a fallback intervention if schedule and pure-Dice adjustments are insufficient.

## Controlled screen
Frozen before execution: same fold0 ten-patient training subset, 52 validation patients, 64 cubed inputs, batch1, augmentation and seed0. Each run is fresh and has 3,000 updates. All use 100 absolute-update warmup then constant AdamW LR 1e-4, weight decay 1e-5.
1. Dice including background; AdamW beta=(0.9,0.999).
2. Dice excluding background; identical optimizer.
3. Dice including background; AdamW beta=(0.9,0.99).
The adapter changes one recipe factor at a time relative to the fixed-schedule control. SGD momentum 0.99 to 0.95 is not directly transferable to AdamW: beta1 is already 0.9 and beta2 controls squared-gradient averaging.

Synthetic FLOP audit: PyTorch2.2.2/MONAI1.6.0, both ellipsoid and box agree: including BG 78,849,850,205; excluding BG 78,848,277,315 FLOPs/update. Changing beta values and LR leaves arithmetic shapes unchanged. These are diagnostic experiments; official B_ref remains unchanged until a stable recipe is validated and a budget is explicitly selected.

Next: inspect full curves and replicate the simplest successful intervention across seeds0,1,2. If the posterior artifact persists, test the decoder intervention motivated by the spatial diagnosis rather than assuming an optimizer fix.

## Additional evidence and decoder intervention
The fixed-schedule BG-including control still has posterior Dice 6.95% at 2,400 updates, then rises to 78.07% at 2,500. Its final macro Dice is 83.89%; the schedule alone therefore does not remove the plateau/cliff.

The decoder intervention changes only the last upsampling convolution, using nearest resize x2 then a stride-1 padded 3x3x3 convolution. The residual network, activation, normalization, all initial weight scalars, other layers, and post-initialization RNG state are retained; final weights transpose their input/output channel axes. Parameter count remains 43,234,202. Regression checks verify preserved initialization and RNG, output shape, and finite backward gradients. Exact cost becomes 89,551,616,861 FLOPs/update with Dice including BG, or 89,550,043,971 excluding BG; both synthetic geometries agree. Nearest interpolation performs indexing/copying, excluded by the declared arithmetic convention. Architecture and cost changes will be explicit if this intervention is adopted.

## First screen outcome
All three recipes fail the predeclared requirement for both foreground Dice scores to reach 70% by 2,000 updates. Default BG Dice reaches it at 2,500 updates, beta2=0.99 at 2,200; foreground-only Dice never does by 3,000. The largest single 100-update macro jumps remain 35.28 and 31.28 percentage points for the BG and beta2 controls. These interventions are rejected for trajectory stabilization despite their ~83.9% final macro scores.

The resize-only test also retains a weak-label plateau: at update1,500, anterior Dice is 4.56% and posterior82.42%. This favors testing the final class-channel normalization as an independent factor. The linear-head test removes only the final InstanceNorm/PReLU, retaining the transposed convolution and all hidden blocks, fixed schedule, Dice including BG, and default AdamW. Parameter count falls by exactly one (the PReLU slope) to43,234,201. Declared FLOPs/update are78,836,480,841; geometric checks agree. Neither mathematical reasoning nor this partial curve establishes the linear-head intervention's success before execution.

## Resize-only rejection
Completed resize-only seed0 at3,000 updates: macro45.64%, anterior7.56%, posterior83.72%. Replacing the final uneven-overlap upsampler alone therefore does not fix the weak-class plateau. The lattice is a visible symptom and a contributing susceptibility; these results do not establish that uneven overlap alone caused the optimization failure.

The final-head normalization test starts from identical retained parameter tensors and the identical post-initialization RNG state. It removes only final InstanceNorm and PReLU (one learned slope) on the3 class channels. The top output block becomes linear; all normalization/activation in hidden features remains. This is a deliberate architecture intervention, not a claim that MONAI is incorrectly implemented.

## Verified replication
All three completed linear-head seeds pass. Final validation macro Dice: 84.73%, 84.69%, 84.28%; mean 84.57% and sampleSD 0.25 percentage points. Both labels exceed70% by the300-update milestone in every seed; late regressions are less than1 point. The corrected model costs only0.017% less compute per update. The factory `baselines/residual_unet.py` reproduces the tested model graph, initial tensors, RNG and forward output exactly; the width48 training entry point supports an explicit linear-head protocol and keeps legacy protocols distinct.

Reference selection is now a fresh seed0 baseline-only bounded search under this frozen recipe. The first180-update run ends at77.38%, and90 updates at32.07%; neither qualifies for the70-75% target. The search records all development compute separately from the chosen per-run allowance. The reference is exploratory on fold0seed0 and does not establish a useful or sufficient update allowance for every comparator.

Additional primary-source architecture comparison: the [nnU-Net decoder implementation](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/unet_decoder.py) predicts class logits through a plain1x1 convolution after normalized hidden feature processing. Our correction differs in retaining the residual U-Net final transpose convolution and linear residual unit, while removing their intervening class-channel InstanceNorm/PReLU. This source supports investigating class-head design; it does not establish that every normalized class block is faulty.
