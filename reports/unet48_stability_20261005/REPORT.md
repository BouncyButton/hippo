# Verified width-48 U-Net training correction

Retirement update (2026-10-05): the15 subsequent fix and stable-budget pilot run directories were deleted at the user's request, freeing12,973,310,984 bytes. Their560 archived metadata/source files were verified locally before deletion. Evidence is preserved in `fix_retirement_metadata.tar.gz`, `fix_retirement_metadata/`, and `fix_retirement_local_verification.json`; completed deletion is recorded in `fix_retirement_deletion_receipt.json`. Checkpoint binaries referenced below in those retired directories are no longer available. The new94,603.78GFLOP baseline allocation684354 uses a separate frozen source snapshot and run root.

The prolonged weak-label plateau is removed on the fixed fold0 ten-patient subset for all three tested seeds. The correction omits InstanceNorm and PReLU only in the final three-class upsampling block; all hidden feature blocks and residual connections remain. AdamW stays at betas(0.9,0.999), LR1e-4 and weight decay1e-5; Dice still includes background. A100 MIG BF16 autocast uses FP32 model and optimizer state.

A 100-update absolute warmup followed by constant LR ensures that changing the stopping budget does not change the learning rates of earlier updates. No early stopping is used.

| Seed | First validation with both class Dice ≥70% | Final macro Dice at3000 updates | Anterior | Posterior |
|---|---:|---:|---:|---:|
| 0 | 200 | 84.73% | 85.42% | 84.04% |
| 1 | 300 | 84.69% | 85.45% | 83.94% |
| 2 | 200 | 84.28% | 85.18% | 83.37% |

Mean final macro Dice: 84.57% ± 0.25 percentage points (sample SD over three seeds). The largest late macro regression across the three curves is0.94 points.

The matched seed0 controls retained the plateau: original class head with fixed LR; foreground-only Dice; AdamW beta2=0.99; and final resize-convolution. These controls change one recipe factor relative to the fixed-schedule control. The successful head intervention removes two operations jointly; this study does not isolate InstanceNorm from PReLU individually or establish a general MONAI bug.

Production implementation: `baselines/residual_unet.py`; launched by `scripts/train_unet48_budget_pilot.py` with protocol fields `unet_output_head=linear`, `unet_lr_policy=absolute_warmup_constant`, `unet_warmup_updates=100`, and hash-bound factory/adapter source. Legacy normalized-head protocols remain explicitly supported. Checkpoint loading uses `build_unet48()` with strict state-dictionary loading.

Parameters:43,234,201 (one fewer than the original head). FLOPs/update:78,836,480,841 under the existing declared dense-arithmetic convention. Both synthetic geometries agree. This is0.017% below the original cost. Fourteen targeted regression checks passed, including exact production/experimental model equivalence and legacy budget-controller behavior.

Primary sources: [MONAI1.6 residual U-Net output block](https://github.com/Project-MONAI/MONAI/blob/1.6.0/monai/networks/nets/unet.py), [uneven-overlap transposed convolution research](https://distill.pub/2016/deconv-checkerboard/), [Dice confident false-positive analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10976047/). The sources motivate hypotheses; the completed controlled runs establish the empirical correction for this fold.

The ten obsolete budget-search pilots were removed after local archive verification, freeing8,648,898,905 bytes. Their metrics, immutable sources, protocols, runtime identity, dataset hashes and checkpoint hashes are preserved in `../unet48_budget_selection_20261005/retirement_metadata.tar.gz`. nnU-Net and unrelated experiments were not part of this deletion.

The strict 70–75% reference pilot is a separate development experiment. Its selected budget must use the corrected cost and fixed LR trajectory. The old selected reference remains historical evidence; the failed normalized-head search supplied no valid width48 reference.

Budget-selection interpretation: repairing the head accelerates U-Net learning substantially. A reference near200 corrected U-Net updates supplies only about100 SwinUNETR and44 nnU-Net updates with the profiled64-cubed batch-one configurations. The fixed-compute comparison remains valid, but this reference cannot itself guarantee that all comparator models reach useful accuracy. Use explicitly predeclared larger budget tiers when investigating that question; slowing or weakening the baseline to inflate its reference cost would not demonstrate constraint efficiency.

All model-development training costs are recorded in `development_compute_ledger.json`; they are separate from a frozen final run's budget. The stability conclusions are limited to this fixed fold and three seeds. Constraints were not used to select the stable recipe or budget.

Budget-search staging repair: job684135 completed four candidates (180,90,135,157 updates), then its source hash guard stopped before training candidate04. The source snapshot mistakenly contained historical generated candidate protocols; creating candidate04 changed one of those files. Original source and manifest are preserved. The continuation imports the four completed runs with their original plan, history and completion hashes; only the exact overwritten generated protocol is explicitly hash-bound as an exception. Every training-source hash remains checked. New source manifests reject generated candidate directories before training. The 70–75% selection criterion, model, optimizer, data and fixed learning-rate trajectory are unchanged.

## Selected stable reference

Job684168 completed successfully. The predeclared baseline-only search selected171 updates (17.1 patient-pass epochs), with final macro foreground validation Dice73.3728%, anterior78.1516%, posterior68.5941%, across52 validation patients. The target applies to macro Dice; it does not require each class to individually reach70% at the selected endpoint. This baseline is freshly trained from initialization with pure Dice and the qualified corrected-head recipe.

B_ref =171 ×78,836,480,841 =13,481,038,223,811 FLOPs =13,481.038223811 GFLOPs.

| Model | FLOPs/update (GFLOPs) | Maximum updates | Patient-pass epochs | Unused budget (GFLOPs) |
|---|---:|---:|---:|---:|
| Corrected residual U-Net width48 |78.836480841|171|17.1|0|
| SwinUNETR feature size24 |155.012136022|86|8.6|149.994525919|
| nnU-Net planned64-cubed PlainConvUNet |355.293896301|37|3.7|335.164060674|

One patient-pass epoch is10 batch-one optimizer updates. Fractional epochs mean stopping partway through a ten-patient pass. Other-model allowances are floors of the same training-compute cap; their performance has not been inferred from these counts. Constraint schedules require their own corrected-model loss/calibration FLOP profiles before launch.

All seven completed short baseline candidates charged84,749.216904075 GFLOPs;71,268.178680264 GFLOPs are additional search compute beyond the selected171-update run. Development training and aborted allocations are separately recorded in the ledger. Local verification checked492 archived metadata files, selected metrics/completion hashes and every compute allowance. Selected protocol and reference are saved in `../../experiments/unet48_stable_reference_20261005/`; previous global reference is retained as historical evidence.

## Follow-up audit of rapid learning

171 updates correspond to17.1 passes over10 training volumes. The actual native image dimensions across the62 selected training/validation cases range32–43 by42–59 by27–47 voxels. They are localized hippocampus images, all smaller than64 in every dimension, and the current pipeline pads them to64 cubed. It does not perform whole-brain localization, or select crops from the ground-truth labels. The residual model has43.23 million parameters. These facts make rapid learning plausible; they do not independently establish its causal mechanism.

The controlled seed0 comparison at200 updates is14.3648% macro Dice with the original normalized output block, versus75.4443% with the corrected linear block, under the same data/initialization/schedule. This directly supports an output-block optimization bottleneck. The selected171-update run starts at1.617% macro Dice at update6 and ends at73.3728%; final soft macro Dice is65.0112%. Hard argmax Dice rises steeply when the posterior class begins being predicted reliably; the soft score also improves. The84.3–84.7% three-seed results belong to3000-update runs (300 volume-pass epochs), not the171-update reference.

The read-only audit finds no case-ID overlap and no exact image duplicates between train and validation. However, the pickle stores file-derived subject_id, image_data, label_data and dims only. It contains no original MRI patient grouping. The repository MSD split utility treats each case as an independent group. The [original MSD dataset paper](https://arxiv.org/html/1902.09063) describes195 originating MRI atlases, whereas the challenge release lists394 hippocampus volumes. These facts mean volume IDs cannot alone certify patient-level independence; they do not prove a cross-split patient overlap. Prior references to10 training patients and52 validation patients were too strong. Use10 training volumes and52 validation volumes until original-patient correspondence is verified. Archived protocols retain their historical field names for provenance.

[PyTorch InstanceNorm documentation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.InstanceNorm3d.html) confirms per-channel spatial mean/variance normalization. Removing it preserves the scale/offset of class-channel maps before the remaining linear residual unit; this is a mechanistic hypothesis for the observed acceleration because the intervention also removes PReLU.
