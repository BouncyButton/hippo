# ADNI and MSD seed-3 baselines

24 supplementary runs: ADNI and MSD × nnU-Net, SwinUNETR, UNETR, 3D U-Net × folds 0/1/2. Seed 3 throughout; baselines only. The user confirmed two GPU workers in one Slurm allocation and replacement of pending job 702776, which was cancelled before starting.

Each dataset has one isolated A100 MIG 4g.40gb worker, 4 CPUs and 40 GiB host memory. The allocation requests 2 GPUs, 8 CPUs, 80 GiB and 24 hours. No automatic follow-up allocations; interrupted recovery remains available.

The original 31,976,450,667,090 optimization-FLOP cap is preserved per run. Baseline updates are 90 for nnU-Net, 206 for SwinUNETR, 245 for UNETR and 563 for 3D U-Net. Original learning-rate horizons, precision (BF16), batch size 1, preprocessing and no augmentation are unchanged. nnU-Net retains foreground Dice+CE, deep supervision and SGD; the other models retain Dice including background and AdamW. No auxiliary constraints are active.

Original frozen cohorts are retained for every fold: MSD has 10 training and 52 validation volumes; ADNI has 5 training subjects/10 crops and 27 validation subjects/54 crops with left/right hemispheres kept together. Scientific training code is unchanged; the only upstream source edit expands the CLI seed choices to accept seed 3. The verifier reverses that single-line edit and checks the original source hash. Seed-3 initialization hashes are verified in the original cluster environment and again before each model trains.

Save final/best training and validation Dice, integer FN/FP, A/P swaps for MSD, curves, configurations, fold, seed, cohorts, budget ledgers, selected/stopping updates, source and logs. Final-budget results are primary. Every completed run is archived to this local reports directory and hash/readability verified before its cluster checkpoints or bulky outputs are deleted. Interrupted checkpoints are preserved. The archive collector requires this Mac to stay awake and connected; a 300-second archive timeout releases the allocation if it cannot archive safely.

Estimated peak storage: 8 GiB logical / 16 GiB mirrored for two active workers, plus 4 GiB reserve. Require at least 20 GiB personal quota headroom before submission and each new run. No dataset, original annotation, source repository or durable archive is a cleanup target.

These results are supplementary seed sensitivity checks. Keep seeds 0/1/2 in the original study; report seed 3 separately and record the cluster GPU environment when comparing with Colab.

Submission and preflight receipts are saved alongside this file. The controller permits at most one running and one queued job on the account.

Initial job **702843** failed before any training updates because the inherited CLI only accepted seeds 0–2. Its evidence is preserved in `attempts/702843` and `frozen_before_seed3_cli_repair`. The repair only adds seed 3 to the CLI choices; all 24 actual trainer commands now pass the parser check, and all eight dataset/model initialization hashes and original loss/optimizer settings pass preflight.

Replacement job **702897** completed all24 runs successfully on **gnode02** at22:43:38 Europe/Rome on15 October2026 (51minutes1second; exit0). All24 archives passed hash verification and their generated checkpoints were cleaned, releasing15.34GiB of file data cumulatively. SAM3 job **702900** started at22:43:39. Final-budget results and fold summaries are in completed_results.json. The original replacement job 702776 was cancelled before starting. Submission quota was 24.75/93.13 GiB used (68.38 GiB free). See the timestamped launch verification for current training progress.
