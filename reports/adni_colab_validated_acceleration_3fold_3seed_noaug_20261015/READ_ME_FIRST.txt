ADNI v5: validated A100 acceleration, 3 folds x 3 seeds, NO AUGMENTATION

FILES
ADNI_VALIDATED_ACCELERATION_v5_3fold_3seed_NO_AUGMENTATION.ipynb
adni_validated_acceleration_control_v5.zip
SHA256SUMS.txt contains both SHA-256 hashes. The notebook pins the matching ZIP hash.

SWITCHING
1. If the old run is active, interrupt its run cell ONCE and wait for all workers
   to stop and finish verified recovery uploads. Otherwise leave the runtime idle.
   Nothing in this delivery has interrupted or accessed your current Colab runtime.
2. Upload ONLY adni_validated_acceleration_control_v5.zip to:
   My Drive/hippo_adni_3fold_3seed/input/
   Keep it zipped. Retain adni_replication_code.zip and adni_data_verified.zip there.
   Do not re-upload the dataset or replace/delete the original scientific ZIP.
3. Open the new complete notebook in Colab. Use the same A100-SXM4-80GB and software
   environment that passed your returned validation. Reuse the connected runtime
   if possible; a fresh runtime must match the existing binding exactly.
4. Run the notebook cells in order. Keep this exact results directory:
   /content/drive/MyDrive/hippo_adni_3fold_3seed/results_noaug_v1
   Setup preserves local work. Existing valid preflight receipts are reused;
   the new preflight cell is a read-only check of those receipts plus your returned
   A100 evidence. It runs no new GPU trials and cannot rebind the environment.
5. Leave MAX_WORKERS=0 to target the CPU-supported maximum immediately.
   Set SESSION_HOURS BELOW the runtime's remaining allowance, including time
   already used before switching. Default is 8 further hours; use 22 only with
   sufficient fresh-session allowance. Colab can end earlier.
6. Verified completed runs are skipped. Interrupted work is prioritized and must
   pass the original checkpoint/ledger audit. Other eligible tasks fill spare slots.
   Keep manual STOP files unless you explicitly intend to resume. Never delete
   locks, receipts or recovery files to bypass a failed check.

RETAINED
The existing 504-task study (four models, fourteen conditions, folds/seeds 0/1/2),
subject splits, initialization, no augmentation, binary evaluation, loss recipes,
calibration, batch size, BF16, optimizer/LR schedules and per-run optimization cap
31,976,450,667,090 FLOPs are unchanged. Calibration remains separately accounted.
All latest-checkpoint writes, durable charge writes/readbacks and recovery cadence
are preserved. Existing results, recovery and charge ledgers are not reset.
Existing process/Drive ownership, matched baseline and archive/cleanup gates remain.

CHANGES
- Immediate slot filling and refill; no mandatory solo memory profile, ramp-up,
  or throughput-driven target reduction. Resource/dependency checks still gate
  actual occupancy and explain each unfilled slot.
- New work uses a genuinely separate mounted /mnt/local-scratch when available;
  old work stays put. Physical recovery staging keeps hardlinks on the same disk.
- Each archive file is read from Drive once into a fresh temporary local snapshot;
  the unchanged full verifier checks hashes, sizes, readability, identity, source,
  acknowledgment and cleanup. Reports/profiles use those same verified bytes.
- The exact validated candidate caches prepared data on GPU and immutable SDF/depth
  annotations, defers evaluation metric readback, and skips only redundant writes
  of an unchanged best checkpoint. Latest/recovery checkpoints are never skipped.
- Local status and 30-second heartbeats report phases, files/counts, run IDs,
  updates, workers, GPU activity/memory, RAM/CPU, and waiting versus actual progress.
  Timings separate data, evaluation, checkpoints and Drive waits.
- Old verified ADNI memory records remain hints with extra cache allowance and
  provisional floors until an accelerated scientific run supplies full-run peaks.
  Diagnostic memory peaks are not treated as full-run profiles.

VALIDATION
Your returned A100-SXM4-80GB evidence passed exact original/candidate equality for
all four models on fold 0/seed 0 with all four auxiliary terms: losses, raw/clipped
gradients, calibration, model/optimizer state, binary metrics, RNG and checkpoint
replay. All 64 input crops matched and cache/write-suppression paths were exercised.
No tolerances were relaxed. Evidence and exact candidate source are included and
bound to the original scientific bundle and exact GPU/software environment.
The diagnostic's training_approval=false is retained: it did not write/replace the
campaign preflight. A separate read-only gate validates that evidence AND requires
an unchanged passed original preflight before any accelerated scientific run.

85 local CPU tests passed, including scheduling/refill, resource limits, baseline
requirements, duplicates, archive corruption/preservation, resume/ledger mismatch,
clean stop, failure/drain, storage placement, hook order and environment rejection.
All 649 frozen scientific files and 466 protected MSD/reference files match their
prior hashes. Notebook command routing, ZIP contents/CRC and pinned hash passed.
No extra scientific runs, GPU benchmarks or cluster jobs were launched.
Actual concurrent Colab throughput, full-sweep speedup and peak worker occupancy
remain unmeasured. The A100 evidence is a bounded equivalence check, not a guarantee
of bitwise equality across all future folds/seeds or a performance benchmark.
