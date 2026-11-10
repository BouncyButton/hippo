> Historical submission: job 694515 was cancelled on 2026-10-11 at the user's request. The active campaign excludes the three longer-schedule runs. See [STANDARD75_RESUME.md](STANDARD75_RESUME.md).

# Recovered focused-band follow-up

Replacement job **694515** was submitted on **11 October 2026** and started on `gnode04` without a dependency. It replaces failed startup attempt **694088**; only one campaign allocation is active.

The initial worker verification, **GPU preflight for all 193 objectives**, and **calibration for all 193 objectives on the ten training cases** passed. GPU verification included an actual model update. All 193 calibration artifact hashes and their warmup hash were independently verified. The job has launched the first Dice-only control; no experiment has completed yet. Aggregate evidence is saved in [startup_verification.json](startup_verification.json).

The campaign preserves the exact original **193-arm matrix**, ordinary SwinUNETR, 5% data, fold 0, seed 0. Its controls include Dice only, Dice + bands, the 78.13% pair reference, the 78.94% focused-band winner, and the 78.74% stronger-pair reference. There are 190 standard 75-epoch arms and three separately labelled 150-epoch arms with matched controls.

## Repair and checks

Six multiscale configurations used Python tuples that became lists on JSON serialization. The original worker's strict equality check failed despite identical settings. The repair uses JSON-native lists, adds a regression test, and checks matrix equality before submission. No scientific configuration changed. The failed attempt trained no experiments. See [RECOVERY.md](RECOVERY.md) and the preserved original failure audit for details.

All **12 scientific tests** passed. All **193 objectives** and **ten training-case geometries** passed the real-model CPU smoke check, including an optimizer update. All **191 remote payload hashes** matched, and both local and cluster checks confirmed equality with the original frozen matrix. The original shared five-epoch Dice warmup was verified and migrated only to the new campaign identity.

## Resources and storage

One `4g.40gb` GPU slice, four CPUs, 24 GiB RAM, 23 h 50 min limit. Original estimate: approximately 16–20 hours of compute, subject to actual arm runtimes. Completed work and the active run are resumable.

Quota at recovery submission: **84.30 / 93.13 GiB**, leaving **8.83 GiB**. The latest startup verification reported **70.04 / 93.13 GiB**, or **23.09 GiB free**. This recovery performed no additional deletions; the cause of the intervening quota decrease was not investigated. The five fixed controls and current top five checkpoints are retained, with all metrics, predictions, configurations, logs, and calibration records.

## Identifiers

- Root: `/mnt/beegfsstudents/home/3160552/band_focus_fold0_seed0_5pct_20261011_02`
- Job: `694515`
- Protocol identity: `13e1b44d2dd58874eb61682411fdb7ba4827fc63f8ba03c5ea1d6e7b0c004bfb`
- Protocol SHA-256: `03c8a009c9c5484a7dc76cdd31ea7b59a08f5f03129788b4289e08ca9d2a142e`
- Archive SHA-256: `36a43d6d86027123e35c246fd83c9ede3b7a4982dd1f06e93712a8f6e8530470`
- [Scheduler receipt](submission.json)
- [Experiment matrix](README.md)

The 81–82% target remains a research goal. This reused validation set is development data; any selected improvement needs confirmation on other seeds/folds.
