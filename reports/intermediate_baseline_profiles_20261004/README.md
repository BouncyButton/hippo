# Intermediate baseline FLOP screening

All candidates profiled at 64³, batch size 1, three output classes, MONAI 1.6.0 / PyTorch 2.2.2, using the production Dice graph and AdamW. Each cost agrees across ellipsoid and box label geometries.

Budget: 12,207.124453303 GFLOPs. Update allowances below apply to Dice-only training; constraint and calibration costs require their own profiles.

| Model | Parameters | GFLOPs/update | Updates | Patient-pass epochs |
|---|---:|---:|---:|---:|
| unet_width24 | 10,812,626 | 20.531939 | 594 | 59.4 |
| segresnet_filters8_deconv | 1,187,003 | 27.825641 | 438 | 43.8 |
| segresnet_filters16_deconv | 4,744,051 | 110.130459 | 110 | 11.0 |
| swinunetr_feature12 | 4,078,077 | 43.988272 | 277 | 27.7 |

SegResNet configurations use learned deconvolution upsampling instead of default interpolation. GroupNorm is included using the same declared fused-normalization estimates as the existing reference audit. These are arithmetic estimates, not elapsed-time measurements.

The wider U-Net uses channels (24,48,96,192,384) and two residual units. The compact SwinUNETR uses feature_size=12, retaining default stage depths and attention heads.

This screening used synthetic labels only. No candidate was trained on patient data or selected using constraint performance. Existing cluster runs and the official reference budget are unchanged.
