# SAM3: four-volume constraint strength comparison

Approved scope: GeoFWI3D only, seven runs on two isolated GPU workers. One matched native-loss baseline; bands and bands+edge at sequential gradient targets 10%, 20%, and 40%. The corresponding 95th-percentile ceilings are 50%, 100%, and 200%, so clipping in calibration does not nullify the target increase.

SAM3 uses its original native objective, which includes Dice plus mask/detection/presence terms. These percentages describe LoRA gradient norms, not scalar Dice-loss multipliers. For bands+edge, bands is calibrated against native gradients, then edge against the current native+bands gradient. The combined auxiliary effect can exceed either individual target.

Training support: model_0546, model_0093, model_0417, model_0103, the first four volumes of the original frozen support. All conditions retain the same 96 validation volumes, frozen initialization, 200 optimizer updates, fold 0, seed 0, no augmentation, and sampling schedule. Calibration uses four anchors per training volume (16 pairs); training uses 400 slab pairs with accumulation 2. This is an update-matched SAM3 experiment, distinct from the conventional FLOP-capped baseline campaign.

Primary outcome: final-update mean validation Dice against the matched four-volume baseline. Diagnostics: training Dice, total validation foreground FN/FP, paired per-volume Dice changes, actual initial calibration median/p95 ratios, initial gradient cosine alignment, and training clipping frequency. Calibration diagnostics do not claim that those ratios persist throughout training. Single fold/seed results are exploratory.

Run `python3 -m scripts.cross_dataset_campaign.report_sam3_four_volume` to build comparison.json from verified durable archives. The script checks source binding, archive hashes, complete final-update evaluation, and matching cohorts. The analysis source is preserved here as report_sam3_four_volume.py.

The experiment uses the successful earlier scientific training source unchanged. Only the support size and constraint calibration target/ceiling change. Each completed run is evaluated, copied to this durable folder, hash verified, and then its generated checkpoint is removed from BeeGFS. Failed/interrupted recovery files remain protected. A local archive collector must remain online; if unavailable, workers stop after the configured wait rather than continue accumulating artifacts.

Execution: queued after the authorized 24-run ADNI/MSD seed-3 baseline campaign. Maximum account occupancy is one running allocation and one queued allocation. See submission.json and live.json for recorded scheduler state.

Submitted as Slurm job **702900**, pending dependency `afterany:702897`. This preserves one running job and one queued job. Submission quota: 25.00/93.13 GiB used, 68.13 GiB free. The cluster preflight passed before submission; both archive collectors are active.

## Completed results (16 October 2026)

All 7 final-update runs completed and were verified/cleaned. Mean validation Dice over the same 96 volumes: native baseline 85.21%; bands at 10/20/40%: 86.06/86.29/85.59%; bands+edge at 10/20/40%: 77.39/82.19/78.81%. Bands20 is +1.08 percentage points against the matched four-volume baseline; all edge variants are below it. See comparison.json for paired case changes, FN/FP, gradient ratios and clipping. The 20% bands run reduced foreground FN by 297,354 and increased FP by 194,525 versus baseline. All calibrated auxiliary p95 ceilings bound the effective weights, so the achieved median ratios were below target. This is fold0/seed0 exploratory evidence. Generated checkpoint cleanup released 1.45GiB cumulatively.
