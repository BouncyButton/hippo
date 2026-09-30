# MedSAM3 five-shot: supervised bands and edge losses

Job **673386** completed on the Bocconi cluster in **53m 58s**, exit code 0.

All three arms independently fine-tuned the same medical LoRA initialization on five labelled volumes (011, 056, 146, 319, 363), with the frozen SAM3 backbone, seed 42, 200 updates and 800 slice presentations each. Bands and edge contributed to backpropagation. Predictions used plain MedSAM3 inference, without post-inference refinement.

Metric: 3D whole-hippocampus union Dice. Five separate development-validation volumes:

| Case | Baseline | + bands loss | + bands + edge losses |
|---|---:|---:|---:|
| hippocampus_017 | 0.882673 | 0.892030 | 0.894930 |
| hippocampus_019 | 0.844195 | 0.857060 | 0.866724 |
| hippocampus_033 | 0.886264 | 0.889827 | 0.894042 |
| hippocampus_035 | 0.879560 | 0.876435 | 0.890062 |
| hippocampus_037 | 0.832705 | 0.837156 | 0.842238 |
| **Mean** | **0.865080** | **0.870502** | **0.877599** |

Bands improved mean Dice by **0.542 percentage points** (4/5 cases); bands+edge improved it by **1.252 points** (5/5 cases). Adding edge to bands gained a further **0.710 points**. These are descriptive results from one seed and five existing validation volumes, not evidence of statistical significance or an independent test cohort.

Training-only gradient calibration fixed bands weight at 10.20054594 and edge weight at 456.95353169. Both ramped over 20 updates. Bands used the same coefficient in both constrained arms. The sampling estimator preserves the original full-volume 3D bands and signed-face objectives; MedSAM3 instance outputs are projected to a differentiable semantic union. See [protocol](PROTOCOL.md) for the exact representation and calibration procedure.

Verification: 17 local checks passed, four loss checks also passed on the cluster, and GPU calibration verified finite nonzero LoRA gradients for both constraints on all 20 support slabs. The independent [audit](AUDIT.json) reproduced every Dice/confusion count, verified all 30 prediction hashes and grids, all 600 training objective records, identical initial adapters/sampling/LR schedules, exact first native loss agreement, unchanged frozen parameters and three distinct trained adapters. Training image/label hashes and validation label hashes matched the local dataset.

Quota: 4.34 GiB free before submission; 3.92 GiB after completion. The base checkpoint and temporary caches used node-local storage. Probabilities, masks, plots and records remain on the cluster at:

`/mnt/beegfsstudents/home/3160552/medsam3_supervised_5shot_20260928/results_673386/`

Storage update for the replication: the three `final_lora_weights.pt` files were copied to the matching local report directories, SHA256-verified, and removed from BeeGFS to free 222,949,599 bytes. The cluster retains each arm's training trace and summary. See `../medsam3_replication_20260928/pilot_adapter_archive.json` for exact archive paths and hashes. Local predictions and full records are in `reports/medsam3_supervised_5shot_20260928/results_673386/`. The retired pilots are summarized in [this compact record](../medsam3_discarded_pilots.md); their generated result folders and temporary bundles were deleted. Original datasets, downloaded weights and reusable source/runtime were preserved.
