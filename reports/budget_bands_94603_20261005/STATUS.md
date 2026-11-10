# Bands comparisons at94,603.78 GFLOPs

Three fresh fold0 seed0 runs, sequentially in one allocation, paired with completed baseline job684354. Same10 training volumes,52 validation volumes,64³ padded inputs, batch size1, augmentation, architecture, and model-specific optimization recipes. Each run retains its baseline loss and adds canonical two-step balanced BCE bands (gamma0, degree_alpha0). No edge or signed-distance term is enabled.

The first50 updates use the original baseline objective. Calibration uses one no-augmentation eval forward per training volume and detached FP32 logit gradients, targeting a median bands/base RMS ratio0.1 with p95 cap0.5. Calibration is charged to the per-run cap. Bands then ramp over50 constrained updates. For nnU-Net, its native Dice+CE and deep supervision are retained, bands act only at full resolution, and calibration compares gradients in that same full-resolution logit space, including the native deep-supervision weight.

| Model | GFLOPs per bands update | Calibration GFLOPs | Baseline prefix | Bands updates | Total updates | Volume-pass epochs | Total GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
|unet|78.905686874|262.198142134|50|1145|1195|119.5|94551.033654914|
|swinunetr|155.081342055|521.568187624|50|556|606|60.6|94497.401171304|
|nnunet|355.363888771|1186.803780354|50|212|262|26.2|94288.643014856|

Every baseline update cost was reproduced exactly with the established PyTorch2.2.2/MONAI1.6 profiling convention. Update and calibration counts agree on both synthetic ellipsoid and box geometries. Calibration includes a1024 FLOP scalar-selection allowance. Dense MAC=2. Validation, data preparation/augmentation, comparisons, memory operations, and checkpoint I/O are excluded consistently with the paired baselines.

Validation uses the same predefined absolute compute targets as the baselines, with calibration and the prefix included when deriving update milestones; the endpoint is always evaluated. Report both best and final validation Dice. These are one-fold, one-seed development experiments. Original patient grouping is unavailable; volume separation is verified.

21 distinct targeted checks passed across the initial suite and the final bands rerun, including native-loss preservation, bands gradient changes, calibration state/RNG purity, retry/resume accounting, controller tamper detection, existing constraints, baseline accounting, and corrected U-Net factory checks. The launcher passed bash syntax checking.

Submitted allocation **684431**. Last verified scheduler status: **RUNNING** on gnode02; allocation elapsed2min09sec, the controller entered the U-Net bands run, and stderr was empty at that check. Actual cluster model/optimizer/loss initialization preflight passed (Torch2.12.1+cu130, MONAI1.6.0, dynamic-network-architectures0.4.4). Submission and verification receipts are saved in submission.json and preflight.json; the receipt-bound preflight hash was verified locally. Training results are not yet recorded. Launcher: ../../scripts/run_budget_bands.sbatch. Cluster run root: /mnt/beegfsstudents/home/3160552/budget_bands_94603_684431.

## Progress snapshot

Allocation684431 remained RUNNING at9min43sec elapsed. U-Net completed1195 updates: final macro hard Dice82.67744%, best83.94532%; paired baseline final84.13358% (difference−1.45614 percentage points). SwinUNETR completed606 updates: final72.52529%, best72.66754%; paired baseline final67.26844% (difference+5.25685 points). nnU-Net had no output directory at that snapshot and is next in the sequential controller. Earlier stderr was empty, and no failure receipt was found. These are fold0 seed0 development results. The scheduler/metric snapshot and comparison are preserved in progress_snapshot.json.

## Completed allocation

Job684431 completed with exit code0:0 in11min30sec. The controller verified all three completion receipts, bound artifacts, frozen source, data split, runtime recipe, calibration, and compute ledgers. No discarded training updates or repeated calibrations occurred. Final verified results are saved in summary_final.json.

| Model | Updates | Volume-pass epochs | Baseline final Dice | Bands final Dice | Difference in points | Bands best Dice |
|---|---:|---:|---:|---:|---:|---:|
|unet|1195|119.5|84.13358%|82.67744%|-1.45614|83.94532%|
|swinunetr|606|60.6|67.26844%|72.52529%|+5.25685|72.66754%|
|nnunet|262|26.2|84.94933%|85.01506%|+0.06573|85.03288%|

nnU-Net selected update227 as its best checkpoint. Final anterior Dice85.55765%, posterior84.47246%. It consumed94,288.643014856 GFLOPs including1,186.803780354 GFLOPs for one training-only calibration, and stayed below the94,603.78GFLOP cap. The final change of+0.06573 percentage points is small in this fold/seed; no population-level or statistical-significance claim is supported. Earlier progress snapshots above are historical.
