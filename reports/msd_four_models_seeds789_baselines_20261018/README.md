# MSD seeds 7, 8 and 9: matched baseline comparison

36 runs: nnU-Net, SwinUNETR, UNETR and U-Net × folds 0/1/2 × seeds 7/8/9. One batch allocation with two isolated GPU workers. Only MSD is included.

Each run inherits the archived seed 3 protocol: 31,976,450,667,090 optimization FLOPs, frozen training/validation cohorts (10 training and 52 validation volumes), batch size 1, BF16, original losses/optimizers/schedules and no augmentation or constraints. Update counts: nnU-Net 90, SwinUNETR 206, UNETR 245, U-Net 563. Final-budget validation hard Dice is primary; best-validation Dice is secondary. All folds and seeds, including poor outcomes, remain in the comparison.

The frozen source differs scientifically only in the accepted CLI seed range. A dedicated dispatcher assigns alternating tasks to two worker lanes so both GPUs serve MSD concurrently. Model initialization is independently fingerprinted in the original cluster environment and verified again before submission.

Conservative storage allowance: 8 GiB logical / 16 GiB mirrored plus 4 GiB reserve, requiring 20 GiB free. Pre-submission observation: 25.29/93.13 GiB used, 67.84 GiB free; no verified archives pending cleanup found among 30 campaign stages. Every completed run must finish Dice, FN/FP and A/P-swap evaluation, then archive metadata/curves/configuration/logs locally with hash verification before deleting checkpoints. The collector must remain connected; a 300-second archive timeout preserves recovery and stops new launches.

The job requests two A100 MIG 4g.40gb devices, eight CPU cores, 80 GiB host memory and a 24-hour limit. Account policy permits at most one running and one queued job; the submission controller checks this and adds a dependency if another job is running.

After collecting results, generate the complete fold/model comparison with `python3 -m scripts.cross_dataset_campaign.report_seed_baselines --destination reports/msd_four_models_seeds789_baselines_20261018`. This reports final/best Dice, FN/FP, A/P swaps, means and fold standard deviations for every seed.

Submitted as Slurm job **706250** at 2026-10-18T11:31:22.051277+02:00, with no dependency. Preflight passed all 36 CLI commands, 12 initialization hashes, dataset identity and scheduler validation. Quota at submission: 25.46/93.13 GiB used, 67.67 GiB free. Six concurrency/archive safety tests passed. The Python 3.12 archive collector is running locally.
