ADNI folds 0/1/2, seeds 0/1/4; reuse completed work

Files:
- ADNI_SEEDS_0_1_4_RESUME_NO_AUGMENTATION.ipynb
- adni_seeds014_resume_noaug_code.zip

1. Keep the existing adni_data_verified.zip, adni_replication_code.zip and all results/recovery files.
2. Upload only the new code ZIP to My Drive/hippo_adni_3fold_3seed/input/ on the account containing ADNI.
3. When ready to switch, stop the old notebook once and wait for recovery uploads. Do not run both controllers.
4. Open the new notebook in Colab with the original validated A100-SXM4-80GB environment and run its cells in order.
5. The original seeds0/1 preflight is reused only if its environment matches. The new seed4 preflight is mandatory.
6. The run cell verifies existing seeds0/1 archives, resumes interrupted work, fills missing seeds0/1, then runs missing seed4.
7. Run the combined comparison cell after a session finishes or stops cleanly. It launches no training.

The final target is 504 cells. If seeds0/1 are complete, 168 new seed4 runs are required. No seed2 task launches.
Results and recovery for seeds0/1 stay in results_noaug_v1. Seed4 results/recovery, retained-seed snapshots and the combined comparison go to results_seeds014_noaug_v1. Seed2 results and recovery are preserved. Original comparison.json is not overwritten by this notebook; the separate combined view excludes seed2.

No dataset re-upload, accounting reset, archive migration, automatic lock deletion or preflight bypass. Seed0/1 retain their original frozen protocols and checkpoints; seed4 differs only in seed/initialization references and the seed CLI allowance. No augmentation. All scientific losses, calibration, precision, batch size, optimizers, schedules and FLOP caps are retained.

Validation: 102 CPU/local tests passed. Original A100 acceleration evidence is retained with an explicit seed-only derivation check. The new seed4 GPU preflight has not been run here. No training or cluster jobs were launched; a CPU-only initialization export matched the four existing cluster seed4 hashes. Existing cluster seed4 baseline scores are provenance, not mixed into Colab comparisons.

The seed replacement was requested after observing performance. Keep the original seeds0/1/2 study and describe the seeds0/1/4 comparison as post-selection exploratory analysis.
