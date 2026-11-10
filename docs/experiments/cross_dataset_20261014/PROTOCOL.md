# Cross-dataset exploration and gated replication

User decisions on 14 October 2026: no augmentation; Dice-only for all four conventional models; retain nnUNet deep supervision and its optimizer. SAM3 and MedSAM3 retain the previous native fine-tuning objective. The long study must not be queued until exploration is reviewed.

## Exploration: 50 experiments

| Dataset / model | Conditions | Data | Budget |
|---|---|---|---|
| MSD nnUNet | One baseline; all 13 constrained objectives at gradient targets 10%, 40%, 60% | Fold 0, seed 0; saved 10-volume 5% subset and 52 validation volumes | 31,976.45066709 GFLOPs per run |
| ADNI MedSAM3 | Baseline, bands, bands + original edge | Five individual crops, one from each saved training subject; 54 validation crops | 200 updates; two adjacent-slice pairs per update |
| GeoFWI3D SAM3 | Baseline, bands, bands + original edge | First five volumes from the frozen 19-volume subset; 96 validation volumes | 200 updates; two adjacent-slice pairs per update |
| GeoFWI3D four conventional models | Four baselines | Fold 0, seed 0; original 19 training / 96 validation volumes; native 96³ grids | **182,178.12654316 GFLOPs per run** |

The seismic cap was explicitly increased to the cost of 152 Dice-only nnUNet updates, exactly eight complete passes over 19 training volumes. The identical cap gives SwinUNETR 362 updates, UNETR 414 updates, and BasicUNet16 952 updates. MSD and ADNI keep the earlier cap. Fine-tuning stays at 200 updates. Profile arithmetic uses the existing dense-arithmetic convention; optimization is capped, while calibration, validation, preprocessing and I/O are reported separately.

The 10/40/60% settings are sequential gradient-calibration targets, not direct multipliers on loss values. Their p95 ceilings are respectively 0.5/0.5/0.75. Coefficients are recalibrated against the new Dice-only nnUNet objective. Earlier Dice+CE nnUNet results are not reused as matched baselines.

MSD fine-tuning previously used five individual hippocampus volumes; distinct person identities are unavailable. ADNI therefore uses five individual crops, one per saved training person, with alternating L/R choice in sorted subject order. Conventional ADNI training still uses both crops from each of the five people, as in the existing 5% study.

GeoFWI3D labels are background=0 and physical salt=1. Its existing normalization, exact split and native grid are retained. SAM3 starts with the general pretrained base and fresh zero-output LoRA; ADNI starts with the medical pretrained LoRA. All matched arms reset to identical initial weights and sampling seeds. Inference uses fixed text prompts (hippocampus / salt body) and images; labels are used only for training supervision and scoring.

## Evaluation and retention

Final-budget / final-update performance is primary. Conventional runs also retain best-validation metrics as secondary results, at common predefined compute checkpoints. Final training Dice helps distinguish insufficient fitting from poor validation generalization. Per-volume integer confusion matrices preserve class Dice, foreground FN/FP and, for MSD, A→P/P→A swaps.

After each experiment, the worker checks completion, required evaluation and checkpoint hashes. The local archive service copies all scalar records, curves, diagnostics, configuration, splits, weights, source and logs into `reports/cross_dataset_exploration_20261014/`. It verifies readability and SHA256 equality, then sends a receipt. Only then does the worker delete that experiment's generated checkpoints and write a cleanup journal. It never deletes datasets, original labels, shared weights, source, archives or failed-run recovery files. Prediction volumes are not persisted when audited confusion counts can be computed directly in memory.

The worker requires 8 GiB free personal BeeGFS quota: a conservative 6 GiB mirrored active-output estimate plus 2 GiB reserve. It rechecks before experiments and submissions. The base SAM3 checkpoint is staged outside BeeGFS and copied to node scratch for an allocation. The local archive service must be running and the Mac reachable. If it is unavailable, the worker waits at the verified-archive boundary and does not accumulate additional experiments.

## Pause, resume and changes

Run these from the repository root:

```bash
rtk proxy .venv/bin/python -m scripts.cross_dataset_campaign.manage status
rtk proxy .venv/bin/python -m scripts.cross_dataset_campaign.manage pause
rtk proxy .venv/bin/python -m scripts.cross_dataset_campaign.manage resume
rtk proxy .venv/bin/python -m scripts.cross_dataset_campaign.manage collect
```

Pause writes a persistent control flag, saves the active optimizer/RNG/sample state at the next safe boundary, and cancels this campaign's pending continuation. Wait for paused status before resuming. Resume restarts the archive service and submits a continuation if no campaign allocation remains. Completed archived runs are skipped. A changed experimental recipe must use a new run identity; never edit frozen protocols or sources in place. Future unstarted cells can be disabled through the campaign control file.

Each allocation requests 24 hours. The worker stops safely around 23 h 40 min; Slurm also sends a warning signal 15 minutes before the limit. Exactly one `afterany` continuation is queued while the campaign runs, so a wall-time termination has a successor. Completion, explicit pause or a training failure cancels the pending successor. Full model/adapter, optimizer, RNG, sample cursor and accumulated metrics persist on BeeGFS for the active experiment. Tests compare interrupted and uninterrupted trajectories, including optimizer tensors. Unexpected hard failures that leave unmatched charged attempts fail verification and preserve recovery artifacts.

## Full study: prepared, not submitted

`experiments/cross_dataset_full_20261014/plan.json` contains all 1,863 requested cells, 81 protocol drafts (including the three nnUNet target options), all saved fold/support selections, and explicitly disables submission. Local CPU profiling fills the four remaining native-96³ constraint cost tables; `reports/cross_dataset_full_20261014/assembly_status.json` tracks draft assembly.

- Conventional models: 3 datasets × 4 models × 14 objectives × 3 folds × 3 seeds = **1,512 cells**.
- MSD MedSAM3: 11 missing objectives × folds 0/1/3 × seeds 17/83/191 = **99 new cells**. Existing baseline/bands/bands+edge results remain references.
- ADNI MedSAM3: 14 objectives × folds 0/1/2 × seeds 0/1/2 = **126 cells**.
- GeoFWI3D SAM3: the same 14 × 3 × 3 = **126 cells**.
- Total: **1,863 new cells before exact pilot reuse**. Conventional MSD/ADNI/GeoFWI3D folds and seeds are 0/1/2.

The 14 objectives are baseline; bands; SDF; bands+SDF; bands+edge; bands+edge+SDF; depth CE; bands+depth; edge+depth; SDF+depth; bands+edge+depth; bands+SDF+depth; edge+SDF+depth; bands+edge+SDF+depth. “Edge” always means the original signed-face objective.

Release gates: select the nnUNet calibration setting from the exploration, review foundation adaptation and constraint benefit, and inspect seismic fitting under the increased cap. Binary nnUNet Dice-only constraint profiles and all fold/seed drafts are prepared. Native 96³ seismic constraint profiles are being measured locally, after which draft assembly reruns automatically without submitting anything. MSD foundation SDF/depth geometry compatibility must also be checked before release. These are explicit preparation gates, not permission to launch the long study.

## Provisional runtime

Exploration: approximately **4–8 GPU hours**, plus queue waits and transfers. The full matrix is likely on the order of **220–400 GPU hours (9–17 days)**, with additional uncertainty after the increased seismic cap. Prior conventional trajectories commonly took around a minute; prior three-class MedSAM3 training took roughly 29 minutes per arm before calibration and full-volume inference. Foundation evaluation, especially 96 seismic validation volumes, dominates the large matrix. Replace these estimates with measured per-stage pilot timings before releasing the long study.

## Submission receipt

Exploration job **700232** was submitted on 14 October 2026 with dependency `afterany:700231` (the existing storage-audit job). Its frozen manifest SHA256 is `1ee59d4604d5b92bb3bee68e6cd14a6c34fb47077ed004aee0f7e175704b08e1`. Seven cluster preflight tests passed. At submission, personal quota was 79.67/93.13 GiB (13.46 GiB free). No exploration result was complete at this point. The local archive collector is running; keep the Mac reachable. The full study has not been submitted.

First runtime verification: job 700232 started on gnode04 after job 700231 completed. Continuation 700235 is queued. The MSD Dice-only nnUNet baseline completed 90 updates with validation Dice 11.2193% and training Dice 11.3571%. Its archive passed hash/readability checks and both checkpoints were deleted (525,343,058 logical bytes). The 10% bands condition is active. Latest verified quota: 80.90/93.13 GiB, 12.23 GiB free. These initial scores show poor fitting; no experimental settings were changed in response.
