# Seed-0 PCGrad with 50 examples: completed and audited

Job **676272** completed successfully on 29 September 2026 at 17:50 Europe/Rome, after 16m54s. Early stopping triggered at epoch 60 (3,000 updates); the selected checkpoint is epoch 30 (1,500 updates). Selected hard macro Dice is 94.6433% train and 85.3531% validation; final-epoch validation Dice is 85.1720%. No optimizer updates were skipped. Local and cluster LR/resume tests passed (two tests); all six GPU projection/AMP checks passed.

Exactly one fresh model: fold 0, seed 0, PCGrad only. Training includes the original ten cases plus forty deterministic additions from the same fold's training pool. The original 52 validation cases are unchanged. No extra control, seed, or fold was submitted.

Learning rate starts at 1e-4 and uses validation-driven ReduceLROnPlateau (halve after four checks without improvement beyond 0.0005; floor 1e-6). Early stopping retains minimum epoch 60, patience 8, min_delta 0.0005 and maximum epoch 75. Loss coefficients and PCGrad are unchanged. Epoch-based warmup remains five epochs, now 250 updates. A larger cohort also means more updates; the adaptive LR is another intentional change, so this does not isolate the effect of sample count alone.

Full training and validation anatomy/coherence audits were saved at epochs 5, 15, 30, 45 and 60 and at the selected checkpoint. The same original two cases remain the FP32 gradient probes. Final optimizer, scheduler, AMP and RNG state are retained for a possible explicitly authorized continuation.

Inference-only audit job **676288** completed successfully in 2m04s. It verified checkpoint, source and data hashes and recomputed original bands metrics plus surface distances for this model and the earlier seed-0 10-case PCGrad/sum controls, using the original CUDA runtime. No further training was launched. BeeGFS quota was checked before the audit: 2.37 GiB free; no cleanup was needed.

Full report: `reports/pcgrad_50cases_seed0_20260929/AUDIT_REPORT.md`. Machine-readable results: `AUDIT_SUMMARY.json`, `AUDIT_CASES.json`, `AUDIT_TEMPORAL_CASES.json`, and `AUDIT_INPUT_HASHES.json` in that directory. Supplementary inference outputs and verification: `supplement/results/`.

Interpretation: substantially improved geometry, outer/crossing correctness and bands confidence, with reduced overprediction and fragmentation. Inner correctness decreases as false negatives increase; constraints are not uniformly improved. Crossing and bands retain substantial train–validation gaps, and later training improves training fit without validation recovery. This single-seed comparison confounds sample count, update budget, warmup in updates and LR schedule; it does not establish PCGrad superiority over a matched 50-case control.

BeeGFS free space before submission: 2.90 GiB, above the 1.6 GiB requirement. No previous results were removed or modified.

Remote root: /mnt/beegfsstudents/home/3160552/pcgrad_50cases_seed0_20260929_01.
Logs: logs/676272.out and logs/676272.err.
Training outputs: fold0/seed0/pcgrad/.
Completion records: completion.json and LAUNCHER_EXIT.json.

No scheduled monitor or automatic follow-up training experiment was created. All original experiment results and selected checkpoints were preserved.
