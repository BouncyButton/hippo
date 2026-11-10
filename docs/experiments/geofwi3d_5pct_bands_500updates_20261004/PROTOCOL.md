# GeoFWI3D 5%: paired bands pilot

The user requested training a bands arm alongside the completed 5% short-budget baseline. The control is job **683068**, whose selected validation mean foreground Dice was **0.7373735061** at update 500.

This arm trains from scratch using the identical 19 training volumes, the same 96 native 96³ validation volumes, model/split seed 0, data ordering, augmentation, SwinUNETR architecture, optimizer, numerical precision and validation opportunities. Its fixed training cap is 500 updates, with the same 100-update learning-rate warmup and original 5,000-update cosine horizon. The baseline's frozen runtime source is copied byte for byte. Frozen protocol and framework-version checks require that only `methods: ["baseline"]` becomes `methods: ["bands"]`.

The sole training change is the existing auxiliary bands loss. It adds side-balanced binary cross entropy on two-voxel, six-connected inner and outer ground-truth bands to the foreground Dice+CE objective. Both band sides contribute equally where defined. A two-voxel frame at every observation-cube face is excluded so crop boundaries are not treated as geological interfaces. The auxiliary coefficient increases linearly from 0.0002 at update 1 to 0.1 at update 500. Bands are generated only from the selected training labels. Inference uses the seismic image alone and predicts the full salt region.

These band settings were specified in the original pipeline and are frozen before observing this arm's validation performance. There is no coefficient search or repeated trial until an improvement appears. Actual highest validation Dice at one of updates 100, 200, 300, 400 and 500 selects the checkpoint, regardless of whether the bands arm helps. The baseline budget was chosen after examining an earlier validation learning curve, so the paired study remains exploratory. There is no test evaluation.

The comparison matches labelled data and optimizer updates. Bands incur additional computation, so it does not assume equal wall time; training and validation timing records must be reported with results. Both earlier baseline campaigns and their checkpoints remain unchanged.

Cluster campaign: `/mnt/beegfsstudents/home/3160552/geofwi3d_swinunetr_5pct_bands_500updates_seed0_20261004_01`. Submission uses the existing empty-user-queue guard. No recurring monitor is created. Existing verified data/model preflight and regression evidence are reused because the copied source, data and architecture are unchanged.

The launcher confirmed an empty user queue and submitted job **683092**. Its startup check found it running on `gnode02` with method `bands` active and no stderr output. Local verification confirmed byte-identical frozen runtime source and identical framework versions to baseline job 683068. Frozen source/protocol/submission records and the paired-protocol verification receipt are saved under `reports/geofwi3d_5pct_bands_500updates_20261004/cluster/`.

The bands arm completed at update 500 with selected validation Dice **0.7684393824 (76.84%)**, versus **0.7373735061 (73.74%)** for the paired baseline. Both best checkpoints were selected at update 500. The measured improvement is **3.11 percentage points**. Completion hashes and all 96 validation IDs were verified. See [RESULTS.md](RESULTS.md) for the paired Dice CSV and recorded timing.
