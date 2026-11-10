# Midrange baseline profiles

Synthetic Dice-only training-step profiles at 64³, batch size one, three output classes. Same declared arithmetic convention, AdamW, and official 12,207.124453303 GFLOP budget.

| Model | Parameters | GFLOPs/update | Updates | Patient-pass epochs |
|---|---:|---:|---:|---:|
| segresnet_filters12_deconv | 2,669,271 | 62.166547 | 196 | 19.6 |
| unet_width40 | 30,025,986 | 55.166375 | 221 | 22.1 |
| unet_width48 | 43,234,202 | 78.849850 | 154 | 15.4 |

SegResNet-12 uses init_filters=12, four GroupNorm groups, and learned deconvolution upsampling. This is an explicit scaled configuration of MONAI SegResNet, not its default configuration.

The wider residual U-Nets use channels (40,80,160,320,640) and (48,96,192,384,768), with num_res_units=2.

Costs include forward, Dice loss, backward, and optimizer; validation, augmentation and I/O remain excluded. GroupNorm is charged with declared normalization estimates analogous to the reference audit.

These are baseline allowances. Constraint training requires objective-specific training and calibration profiles. No patient-data performance is available for these candidates; no new training jobs were submitted during this profiling.
