# GeoFWI3D archive 0: low-data 3D SwinUNETR

The user requested archive 0, retirement of all LEGO dataset code, and training pipelines for 5%, 20% and 100% data. Subsequent instructions removed the test set and made Dice the success metric. This prepares baseline runs; it does not submit training or create monitoring.

## Data and supervision

Source: https://zenodo.org/records/20148778 and https://github.com/aaspip/geofwi3d. Archive `models_batch_0.tar.gz` is 7,358,564,728 bytes, with published MD5 `57013080521bfa96069db03b98e2e553`. It contains 1,000 different simulated geological scenes on native 96×96×96 grids. Metadata identifies 479 salt-containing scenes (249 without faults, 230 with faults). The experiment uses that positive cohort to match the target-present hippocampus setting. Its results therefore concern salt-containing scenes, not salt-presence detection across all 1,000 scenes.

We stratify an 80/20 training/validation split by fault presence, with split seed 0. All 479 simulation seeds in this cohort are distinct and training/validation seed separation is verified.

| Setting | Labelled training volumes | Validation volumes |
|---|---:|---:|
| 5% | 19 | 96 |
| 20% | 77 | 96 |
| 100% | 383 | 96 |

The percentages refer to the 383-volume eligible training pool, rounded to the nearest volume. The subsets are nested and stratified by fault presence. A 5% run reads labels from only its 19 selected training volumes and from the common validation cohort; the other eligible training volumes provide no model supervision. Validation labels are additional evaluation supervision, so this is a low-training-data experiment, not a claim that the entire annotation budget is 19 volumes. No test split or test evaluation is configured.

The output is a full binary salt mask: all salt material inside the observation window is foreground, every other material is background. Small physically valid disconnected components are retained. The authors' implicit label rule is `rgt3d == 0`; inspection found background null-RGT artifacts. Preparation audits the physical velocity model and uses `(rgt3d == 0) & (vp3d == float32(metadata.salt_vel))`, rejecting physical salt outside the implicit mask. This correction is recorded per case and in the manifest, rather than silently applying connected-component cleanup. Neither RGT nor velocity is a model input.

Only `image3d.bin` is used as input. It is normalized using its own 99.5th absolute-amplitude percentile and clipped to [-1,1]. Shape, orientation and native voxels are preserved. NIfTI files use identity affine and unknown physical units because the source does not supply calibrated voxel spacing. Z remains depth. The imagesTr/labelsTr folders contain the 383 eligible training cases; imagesVal/labelsVal contain the 96 validation cases.

## Training and Dice

Default runs use model seed 0, 3D SwinUNETR with feature size 24, two output classes, training from scratch, activation checkpointing, batch size 1 and CUDA bfloat16 AMP on the existing 40 GB GPU profile. Bfloat16 needs no gradient scaling. All runs use foreground Dice plus cross-entropy loss, AdamW (learning rate 1e-4, weight decay 1e-5), 100-update linear warmup and cosine decay. Augmentation is restricted to flips and 90-degree rotations in the XY plane; geological depth is preserved.

Every fraction has the same maximum **5,000 optimizer updates**. A deterministic cycle visits every selected volume before reshuffling; the 100% run consequently uses every eligible training case, even though the budget is expressed in updates. Validate every 100 updates on all 96 complete volumes. Early stopping monitors mean foreground hard Dice, with minimum 2,000 updates, patience 10 validation checks and min-delta 0.0005. The best checkpoint is the exact maximum validation Dice even when an improvement is smaller than the stopping delta. These are defaults chosen for a limited-budget comparison, not an assertion that all models will converge by 5,000 updates. Equal updates do not guarantee equal GPU time; per-interval training/validation times and invocation wall time are recorded.

Dice is calculated for each complete native 96³ validation volume as `2*intersection/(predicted_salt+true_salt)`, then averaged over volumes. Background Dice and pooled-voxel Dice are not the success metric. Best/final validation Dice, selection and stopping updates, per-volume Dice, histories and checksummed best/latest checkpoints are retained. There are no surface-metric or test-result claims. Validation is used for selection and stopping, so results must be described as selected validation performance.

The trainer also accepts `--methods baseline bands` for a future paired comparison. The optional bands arm derives two-voxel six-connected inner/outer bands only from the selected training masks, excluding a two-voxel frame at every observation-cube face. It adds side-balanced BCE with a configurable protocol coefficient (default 0.1) and 500-update auxiliary warmup. The default prepared campaign contains **baseline only**. No additional annotation or salt-mask input is used at inference. A later constraint study should freeze or calibrate its coefficient using only the selected training cases and include that cost when making speed claims.

## Local preparation

Run commands from the repository root, with the repository Python environment:

```bash
rtk proxy python3 scripts/download_geofwi3d.py --root datasets/Dataset203_GeoFWI3D
rtk proxy .venv/bin/python scripts/prepare_geofwi3d.py --root datasets/Dataset203_GeoFWI3D --split-seed 0
rtk proxy .venv/bin/python scripts/check_geofwi3d_pipeline.py --root datasets/Dataset203_GeoFWI3D --output reports/geofwi3d_swinunetr_20261004/preflight.json --model-smoke
rtk proxy .venv/bin/python scripts/prepare_geofwi3d_campaign.py --root datasets/Dataset203_GeoFWI3D --campaign reports/geofwi3d_swinunetr_20261004/campaign --preflight reports/geofwi3d_swinunetr_20261004/preflight.json
```

The downloader resumes partial transfers and checks the published archive MD5. Preparation streams the verified archive and only writes relevant images and masks, without extracting all raw modalities. The requested downloaded archive is retained. Re-running preparation verifies existing files and refuses to change the split.

## Cluster preparation and explicit submission

Copy the prepared dataset (excluding downloads) and the source scripts to the cluster. Run the check and campaign-preparation commands there using the existing research Python environment. Freezing records that host's interpreter, dependency versions, source hashes, dataset manifest and CPU model smoke result. Sources and dataset must remain unchanged after freeze.

```bash
rtk proxy python source/submit_geofwi3d.py --campaign /absolute/campaign/path
rtk proxy python source/submit_geofwi3d.py --campaign /absolute/campaign/path --submit
```

The first command verifies only. The second is the explicit submission entry point and refuses to submit if the user's entire Slurm queue is nonempty. One job runs the three arms sequentially on one GPU. It does not install monitoring. CPU verification is distinct from CUDA execution, which is checked when the job starts.

Unit checks:

```bash
rtk proxy .venv/bin/python -m pytest -q scripts/test_geofwi3d_pipeline.py
```

The regression checks cover nested/isolated splits, null-RGT label artifacts, genuine disconnected salt, exclusion of artificial cube faces, complete-volume foreground Dice averaging, reproducible sampling, exact continuation after a checkpoint interruption and rejection of changed completed-run configurations. The separate native SwinUNETR smoke check verifies a 96³ forward/backward optimizer step on an actual selected training case.

## Preparation evidence

Archive 0 was downloaded and verified against the published MD5. Its recorded SHA-256 is `3a9ac1bfc711952c4545e1cfc28bec170d0374dea11dd0a94bb2b071a0609639`. The 958 prepared image/mask files occupy 1,581,201,776 bytes, excluding the retained 7.36 GB download. The dataset manifest SHA-256 is `feab4145eacfc0b48437cc0ef1733a311a360a07072e0c6e10118e3674c5acf6`.

All 479 case pairs passed local and cluster file-hash, native-array and split-isolation checks. A native 96³ SwinUNETR CPU optimizer step passed on selected training case `model_0546` in both environments. The original six regression checks passed locally (13.78 seconds) and with the cluster environment (165.54 seconds). A subsequent campaign-path regression check also passed locally (3.77 seconds) and on the cluster (26.72 seconds), verifying that a preflight record already inside the campaign can be frozen safely. These checks validate the pipeline; the smoke-check loss is not a trained segmentation result. CUDA training has not yet been run.

The local frozen campaign is `reports/geofwi3d_swinunetr_20261004/campaign/`. Cluster training data is staged under `/mnt/beegfsstudents/home/3160552/Dataset203_GeoFWI3D`, excluding the download archive. The cluster campaign directory is `/mnt/beegfsstudents/home/3160552/geofwi3d_swinunetr_20261004_01`. The retained existing Python environment supplies framework dependencies and contains no LEGO experiment source.

Both frozen campaigns passed verification without submission during preparation. The cluster preflight, protocol, source hashes and dependency records were retrieved into `reports/geofwi3d_swinunetr_20261004/cluster/`; all eight frozen runtime source files match the current repository source. The completion receipt for that preparation is `reports/geofwi3d_swinunetr_20261004/prepared.json`. At that stage no training job or monitoring was started.

On 2026-10-04, following the user's explicit submission request, the frozen launcher verified its inputs, confirmed the entire user queue was empty and submitted Slurm job **682953**. The startup check found it running on `gnode02`, with `5pct_baseline` active and no failure record or stderr output. The 20% and 100% arms follow sequentially in the same job. The original frozen protocol remains unchanged. Submission and queue-check receipts are saved under `reports/geofwi3d_swinunetr_20261004/cluster/`. No recurring monitoring was created.

The subsequent requested status check found all three arms complete. Slurm reported `COMPLETED`, exit code `0:0`, and 47 minutes 28 seconds elapsed. Best mean validation Dice was 0.8327 (5%), 0.9288 (20%) and 0.9384 (100%). See [RESULTS.md](RESULTS.md) for selection/stopping updates, verification evidence and links to each per-volume Dice CSV.

LEGO executable source was retired locally and from its paused cluster campaign. Exact deletion receipts are under `reports/lego_retirement_20261004/`; earlier metrics and checkpoints remain available for provenance.
