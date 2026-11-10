# Training schedules for ten patients under the reference compute cap

All schedules use **64 × 64 × 64 inputs, batch size 1, three classes, no
activation checkpointing and no gradient accumulation**. Counts follow the
same reconstructed arithmetic convention as the reference report.

The schedules enforce the user's stated maximum of **116,259 GFLOPs** literally.
The unrounded original reference allowance is **116,259.1020165 GFLOPs**; the
small rounding difference removes one SwinUNETR update under the literal cap.
Both interpretations are recorded in `training_schedules.json` and
`training_schedules.csv`.

| Model and declared configuration | GFLOPs/update | Maximum updates | Complete ten-update epochs | Additional updates | Epoch/exposure equivalent |
| --- | ---: | ---: | ---: | ---: | ---: |
| MONAI U-Net, channels 16/32/64/128/256, two residual units | 9.58098 | **12,134** | **1,213** | **4** | 1,213.4 |
| Original SwinUNETR, feature size 24 | 155.01214 | **749** | **74** | **9** | 74.9 |
| nnU-Net default topology for isotropic 64³ inputs, features 32/64/128/256/320 | 355.29390 | **327** | **32** | **7** | 32.7 |

If using the unrounded original reference instead, U-Net and nnU-Net remain
unchanged; SwinUNETR gets **750 updates = 75 complete epochs**.

## What an epoch means

With ten patients, batch one, one tensor per patient visit, and a loader that
visits each patient exactly once, **one dataset-pass epoch = ten optimizer
updates**. Thus:

- U-Net: 1,213 complete passes, then four updates in pass 1,214.
- SwinUNETR: 74 complete passes, then nine updates in pass 75.
- nnU-Net: 32 complete passes, then seven updates in pass 33.

The repository's default nnU-Net trainer instead defines an epoch as **250
random patch iterations**. With that unchanged convention, the same 327-update
cap is **one trainer epoch + 77 updates in the second**, not 32.7 native epochs.
Random sampling does not guarantee each patient is visited once; in that case,
32.7 means only the number of single-patient samples divided by ten. For an
actual dataset-pass protocol, explicitly schedule one visit to every patient
per ten-step block. The U-Net example's four-crops-per-patient transform and
batch-two loader must likewise be replaced by the agreed single-input,
batch-one protocol.

Use `max_steps` as the authoritative stopping condition. `max_epochs` alone
would allow a full final epoch and exceed the cap for U-Net and nnU-Net.

```python
for epoch in range(loop_epochs):
    for batch in train_loader:
        if global_step >= max_steps:
            break
        train_step(batch)
        global_step += 1
    if global_step >= max_steps:
        break
```

Every update includes forward, loss, backward and optimizer arithmetic. These
are baseline schedules: auxiliary constraints require a new profiled step cost
and, where applicable, subtraction of setup/calibration costs.

## Why the update counts differ

The total allowance is the same. This specific U-Net update is approximately
**16.18 times cheaper than SwinUNETR** and **37.08 times cheaper than nnU-Net**.

| Model | Parameters | Dense forward arithmetic, GFLOPs |
| --- | ---: | ---: |
| MONAI U-Net | 4,807,482 | 3.16 |
| SwinUNETR | 15,703,029 | 50.86 |
| nnU-Net matched-input topology | 16,548,076 | 118.18 |

The small MONAI U-Net starts with a stride-two convolution, so its first
16-channel feature map is 32³. nnU-Net begins with 32 channels at full 64³
resolution and uses two convolutions per encoder/decoder stage. The nnU-Net
decoder also processes many channels at full resolution; MONAI U-Net's final
upsampling produces a narrow three-class output. SwinUNETR additionally has
window attention and a substantial convolutional decoder, including 24-channel
processing at full resolution. Most of its dense forward FLOPs here come from
convolutions; the difference is not explained by attention alone.

For a dense 3-D convolution, approximate cost is
`2 × batch × output_D × output_H × output_W × C_in × C_out × kernel_volume`.
Halving all three spatial axes reduces the corresponding volume by eight.
Doubling both channel widths increases convolutional cost by four. This is why
parameter count alone does not predict training cost.

These are configuration-specific results, not a general claim that every
U-Net is 16–37 times cheaper. A wider U-Net, a different nnU-Net preset,
additional crops, a different input resolution, or checkpoint recomputation
requires recalculation. The high U-Net update allowance also does not create
new independent training data or guarantee better validation performance.

## Model and recipe provenance

U-Net uses the architecture in `baselines/3dunet/3dunet.py`, adapting its binary
output to the reference's three classes. Its original illustrative dataset
placeholders are not executed. Softmax Dice and AdamW match the SwinUNETR
reference; the original file's sigmoid loss/Adam settings are not profiled.

nnU-Net's old dataset-specific plan was not retained in the cluster archive.
This is a new, fully specified comparison configuration generated by the
repository's default topology rules: isotropic spacing, fixed 64³ patch,
bottleneck minimum edge four, base features 32, maximum 320, two convolutions
per stage. The full configuration is saved in
`nnunet_matched64_architecture_native.json`. It is not a recovered historical
nnU-Net run or an unrestricted automatic patch/batch plan.

The main nnU-Net schedule retains deep supervision, foreground Dice+CE,
Nesterov SGD and gradient clipping. Recurring momentum-update arithmetic is
profiled, conservatively bounding its slightly cheaper first update. An
additional shared Dice/AdamW configuration was profiled for architecture
isolation: **355.33509 GFLOPs/update, also 327 updates**.

The main table therefore compares declared baseline recipes. To attribute a
performance difference solely to architecture, use shared loss/optimizer/data
handling, and describe the resulting nnU-Net-derived architecture explicitly.

## Verification and reproduction

Meta-tensor and real CPU full-step counts agree for U-Net and both nnU-Net
recipes. The matched SwinUNETR count exactly reproduces the already verified
reference. `build_schedules.py` asserts that every included update fits and one
additional update exceeds the allowance. Counts are estimates under the
reference convention, not exact GPU hardware instruction counts; validation,
preprocessing and CUDA GradScaler overhead are excluded consistently.

The profiling environment is local PyTorch 2.2.2, temporarily downloaded
MONAI 1.6.0 and dynamic-network-architectures 0.4.1. No training job was launched
and no historical run or repository environment was modified.

```bash
rtk proxy .venv/bin/python -m pip install --no-deps --no-cache-dir --target /private/tmp/hippo_bref_networkdeps dynamic-network-architectures==0.4.1 batchgenerators
rtk proxy env HIPPO_PROFILE_MONAI_PATH=/private/tmp/hippo_bref_monai160 MPLCONFIGDIR=/private/tmp/hippo_bref_mpl .venv/bin/python reports/compute_budget_5pct_fold0_seed0_20261003/profile_matched_models.py --model unet --device cpu
rtk proxy env HIPPO_PROFILE_MONAI_PATH=/private/tmp/hippo_bref_monai160 MPLCONFIGDIR=/private/tmp/hippo_bref_mpl .venv/bin/python reports/compute_budget_5pct_fold0_seed0_20261003/profile_matched_models.py --model nnunet --recipe native --device cpu
rtk proxy .venv/bin/python reports/compute_budget_5pct_fold0_seed0_20261003/build_schedules.py
```

The reference README includes the temporary MONAI installation command. Run
each profiler with `--device meta` to regenerate shape-only comparisons. The
schedule exporter expects both meta and CPU records, plus the verified
reference and matched SwinUNETR record.
