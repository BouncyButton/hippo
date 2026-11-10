# Band-focus follow-up submission

**Update, 11 October:** job 694088 failed during the initial frozen-matrix check, before GPU preflight or any scientific training. Six multiscale entries used Python tuples that serialized as JSON lists; their numerical settings were identical. Recovery preserves all 193 configurations and is running as **job 694515**, documented in `../band_focus_20261011_recovery/`. The text below records the original submission state.

Submitted **job 694088** on 11 October 2026. At verification it was **PENDING (Dependency)** behind the existing `geofwi3d-3fold-3seed` job **694083**, which was running with a four-hour time limit. No scientific run from this campaign has started yet.

The job screens **193 predeclared experiments** using ordinary SwinUNETR, **5% data, fold 0, seed 0**. The target is to approach the historical auxiliary-branch 81–82% anterior/posterior mean Dice through the training objective. This campaign uses no auxiliary branch. Its validation set has already been used for development; any selected improvement needs confirmation on other seeds/folds.

## Scope and comparisons

- Five exact controls: Dice only, Dice + original bands, the 78.13% independent 1/3 mm pair formulation, the 78.94% focused-band winner, and the 78.74% stronger inside-target pair formulation.
- Hard-example fractions, their contribution to the loss, band widths, asymmetric inside/outside support, and independently calibrated band/pair strength.
- Neighborhood difficulty at several scales, selection within spatial tiles, smooth difficulty weighting, and multiscale bands. Neighborhood scores average nonnegative point errors, avoiding cancellation between missing and extra tissue.
- Alternative error penalties, pair distance/side combinations, actual A/P-interface supervision, small signed-distance/Chamfer contributions, and gradual focus schedules.
- **190 arms** retain the 75-epoch protocol. **Three separately labelled 150-epoch arms** compare Dice, original bands, and the focused-band winner under a longer schedule.

The complete frozen matrix is in [README.md](README.md). The source is `scripts/band_focus/`. Each arm starts from the verified common five-epoch Dice prefix. Calibration uses only the ten training cases; additional terms share one calibrated gradient budget.

## Verification and resources

- All **11 local scientific tests passed**, including exact original-control losses/gradients, empty/full support, neighborhood gradient direction, A/P masking, and interrupted/resumed training equivalence.
- A real ordinary SwinUNETR CPU check covered all **193 objectives** and geometries from all **ten training cases**. A combined spatial/A/P objective produced finite model gradients and an actual optimizer update.
- All **191 payload manifest entries** were verified after remote staging. The shared warmup hashes were verified before submission.
- **GPU preflight and full training-case calibration are pending allocation.** The worker requires both to pass before running the experiments and checks four control calibrations against the preceding campaign.
- Allocation: one `4g.40gb` GPU slice, four CPUs, 24 GiB RAM, **23 h 50 min** time limit. Estimated compute time is roughly **16–20 hours after allocation**, subject to actual arm runtimes; queue time is additional. Progress and completed results are resumable.

## Storage

Recorded and verified the completed 64-arm predecessor's results, then retired only **56 non-retained `best.pt` files**, freeing **3.672 GiB of logical checkpoint data**. BeeGFS quota accounting includes replication, so this differs from the quota-space change. Eight selected control/best checkpoints, the common warmup, all predictions, metrics, histories, calibration records, configurations, and source were preserved. Cleanup receipts are in `cleanup/`.

The submission-time quota was **84.04 / 93.13 GiB**, leaving **9.09 GiB**. This job retains its five controls and current top five model checkpoints, plus the shared warmup and active resume state; it preserves all results and checks quota before each run.

## Durable identifiers

- Remote root: `/mnt/beegfsstudents/home/3160552/band_focus_fold0_seed0_5pct_20261011_01`
- Protocol identity: `6247cbe0802d64720653bde27b50b19f7f297627dfab70d3f576c254329b8c9c`
- Protocol file SHA-256: `ff9bdabfc476d46d4a4f042e7bfcb5f966abbc39d60cc3511f12c58039db55a2`
- Payload archive SHA-256: `8bc96688ee0ff0d62962dd9e565b34b2db66a9c98f9207a32a8725b09d27c08a`
- Confirmed scheduler receipt: [submission.json](submission.json)
- Runtime progress: remote `status.json`, `logs/`, `RESULTS.md`, and `results.json` once the worker starts.

No changes were made to job 694083. The new job's `afterany:694083` dependency allows it to become eligible when that allocation ends.
