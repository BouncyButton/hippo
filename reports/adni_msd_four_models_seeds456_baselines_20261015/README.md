# Seeds 4, 5 and 6: matched baseline sensitivity experiment

The user confirmed 72 runs in a single Slurm allocation: nnU-Net, SwinUNETR, UNETR and 3D U-Net × ADNI/MSD × folds 0/1/2 × seeds 4/5/6. This replaces the unsubmitted ADNI U-Net-only seed 4 plan. Two isolated GPU workers advance ADNI and MSD independently.

All conditions inherit the completed seed 3 experiment. The only changes are seed, initialization reference and run identity; the trainer CLI accepts the additional seed values. Training code, preprocessing, original folds, losses, optimizers, learning-rate schedules, batch size 1, BF16 and no augmentation remain unchanged. Each run has a 31,976,450,667,090 optimization-FLOP cap: 90 updates for nnU-Net, 206 for SwinUNETR, 245 for UNETR and 563 for U-Net. Final-budget validation hard Dice is primary; best validation Dice is secondary.

Every model within a dataset/seed starts from its verified common initialization across all three folds. ADNI uses 5 training subjects/10 crops and 27 validation subjects/54 crops per fold, with hemispheres kept together. MSD uses 10 training and 52 validation volumes per fold. No constraints are active.

Run all 72 fixed conditions. Compare each seed across all eight model/dataset combinations, retaining every fold and poor result. No predefined numerical threshold for “good on every baseline” has been supplied, so this batch does not claim a seed meets that criterion or search further seeds automatically. Earlier seed results remain part of the evidence.

One running and one queued job maximum. This batch queues behind the active SAM3 experiment. It requests two A100 MIG 4g.40gb devices, eight CPUs, 80 GiB host memory and up to 24 hours; the job releases its allocation on completion. The preceding 24-run campaign took 51 minutes, suggesting roughly three hours for this batch, with runtime depending on I/O and archive transfer.

Storage headroom is checked before submission and each new run. Conservative peak estimate: 8 GiB logical/16 GiB mirrored plus 4 GiB reserve, requiring 20 GiB free. Each run saves final/best Dice, FN/FP, A/P swaps for MSD, curves, configuration, fold/seed, cohorts, initialization, budget ledger and logs. After local hash verification and archive acknowledgment, generated checkpoints are immediately retired. Failed/interrupted recovery remains protected. The local archive collector must remain connected; an archive wait timeout preserves results and stops further launches.

Frozen source and protocols are in frozen/. Initialization reference, preflight, local tests and scheduler receipts are saved alongside this file.

Generate comparison.json with `python3 -m scripts.cross_dataset_campaign.report_seed_baselines --destination reports/adni_msd_four_models_seeds456_baselines_20261015`. It verifies all available archives, reports every fold and computes cross-model seed summaries only when the corresponding24 runs are complete. The reporter reproduced all eight seed3 model/dataset means exactly.

Submitted as **703110**, dependent on SAM3 job **702900**. Verified one running and one queued job after submission. Quota at submission:25.47/93.13GiB used,67.66GiB free. Preflight passed all72 actual CLI commands,24 model initialization checks, both dataset hashes, original loss/optimizer checks and scheduler validation.

## Completed results (16 October 2026)

Slurm job703110 completed all72 runs with exit0. Every run has a verified local archive and completed checkpoint cleanup, releasing 46.02GiB of generated files cumulatively. Final-budget mean validation Dice by seed (ADNI nnU-Net/SwinUNETR/UNETR/U-Net; MSD nnU-Net/SwinUNETR/UNETR/U-Net):

- Seed4: 76.86/67.55/54.32/68.73%; 41.64/68.27/23.86/71.66%.
- Seed5: 77.52/66.34/65.93/74.06%; 74.66/40.24/55.16/78.76%.
- Seed6: 75.22/69.64/55.03/75.03%; 44.02/65.10/50.07/69.14%.

Seed5 has the highest equal-weight mean across eight model/dataset groups (66.58%) but MSD SwinUNETR falls to40.24%. Seed6 has the highest minimum group mean (44.02%) but MSD nnU-Net falls to44.02%. No seed is uniformly strong. ADNI U-Net recovers from seed3’s8.81% mean to68.73%,74.06%,75.03% for seeds4,5,6. See comparison.json for every fold, FN/FP and A/P swaps.
