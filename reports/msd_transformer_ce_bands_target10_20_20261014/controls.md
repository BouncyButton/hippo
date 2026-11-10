# Active four-run transformer bands campaign

Submitted as job 701092. This is the only authorized active campaign.

Order: SwinUNETR 10%, UNETR 10%, SwinUNETR 20%, UNETR 20%. Fold0 seed0, 10 training/52 validation crops, no augmentation, Dice+CE, cap31976450667090 optimization FLOPs. Calibration overhead is separately reported. Same75% p95 ceiling as60% reference. Same50-update prefix and50-update warmup.

Use the frozen module `scripts.cross_dataset_campaign.transformer_bands_low_job` with `status`, `pause`, or `resume` and `--stage /mnt/beegfsstudents/home/3160552/msd_transformer_ce_bands_target10_20_20261014`, via SSH with PYTHONPATH set to its source directories. Pause preserves optimizer/RNG/sampling and cancels this campaign's pending continuation. Resume requires user authorization if manually paused.

Start/restart the local collector with `scripts.cross_dataset_campaign.manage_msd_ce_constraints.collect(name='msd_transformer_ce_bands_target10_20_20261014')`; it selects Python3.12. Generate validated comparisons with `python3 -m scripts.cross_dataset_campaign.report_transformer_bands_low`.

The old56-run MSD sweep is abandoned and its checkpoints retired. ADNI/seismic/MedSAM3 and the full replicated sweep remain held. Never resume them from this campaign.
