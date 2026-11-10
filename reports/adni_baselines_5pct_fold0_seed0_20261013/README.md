# ADNI 5% baseline pilot: fold 0, seed 0

Prepared on 2026-10-13 at the user's request. The user selected the same
31,976.45066709 GFLOP optimization cap as the current MSD four-model comparison.
Slurm job **698735 completed successfully** in 11 minutes 31 seconds, exit code
`0:0`, after its original `afterany:698711` dependency. All four baselines completed
their budgets with zero discarded updates. The user explicitly approved the ADNI data transfer;
the dataset and all 228 frozen stage files passed checksum verification on the
cluster before submission. See `submission.json` and `transfer_receipt.json`.

## Frozen experiment

All four models use the same five subjects selected from the 108 fold-0 training
subjects, retaining both hippocampi: ten training crops and 54 validation crops
from 27 other subjects. Five subjects is floor(5% of 108), or 4.63% of the training
pool. Selection uses a seed-0 NumPy RandomState permutation of sorted true subject
IDs. Both sides remain together, and there is no subject overlap with validation.
The exact IDs and mapping are in the staged `split_provenance.json`.

Inputs are the prepared 32 × 48 × 40 MRI crops, normalized per crop over nonzero
voxels and symmetrically zero-padded to 64³. They are not resized. Outputs have
two classes: background and whole hippocampus. The binary label and source
foreground voxel counts were checked for every training/validation crop.

| Model | Binary parameters | GFLOPs/update | Maximum updates | Initial LR |
| --- | ---: | ---: | ---: | ---: |
| SwinUNETR, feature size 24 | 15,703,004 | 154.969930463 | 206 | 0.0001 |
| UNETR ViT-B/16 | 92,685,538 | 130.031650493 | 245 | 0.0001 |
| 3D U-Net, BasicUNet width 16 | 1,438,802 | 56.697462661 | 563 | 0.0003 |
| Matched-input nnU-Net-derived topology | 16,547,592 | 355.220395319 | 90 | 0.01 |

The three MONAI models use softmax Dice including background and AdamW. The
nnU-Net-derived baseline retains foreground Dice+CE, four deep-supervision
outputs, Nesterov SGD with momentum 0.95, and clipping at norm 12. Architectures,
optimizer settings, mild_v1 augmentation and LR trajectories match the frozen
MSD comparison, with output heads changed to two labels. LR horizons remain
610, 477, 1,095 and 174 updates respectively. No pretrained weights, added
constraints, or early stopping.

Validation occurs at 20 shared compute targets and the final update. Save
training loss, validation hard/soft whole-hippocampus Dice, per-crop scores,
best/latest checkpoints, and a compute ledger. The optimization budget excludes
validation, preprocessing, initialization, offline profiling and checkpoint I/O.
The existing loop reserves each attempt before executing it and charges attempts
lost to interruption. A discarded-attempt run is flagged by the campaign verifier.

ADNI binary Dice and MSD anterior/posterior macro Dice measure different targets;
their absolute values are not directly equivalent. This pilot examines learning
trajectories under the same budget. Crops are reference-localized; no physical
distance metrics are calculated from their identity affines.

## Execution

One sequential Slurm job: partition `stud`, one `gpu:4g.40gb`, four CPUs, 24 GiB
host RAM, three-hour wall-time limit. Submission preserves the existing
one-running/one-queued policy by depending on the currently running job, if any.
No active jobs are cancelled or changed. Best/latest checkpoints are retained.

Local frozen stage:
`experiments/adni_baselines_5pct_fold0_seed0_20261013/`

Verified remote stage:
`/mnt/beegfsstudents/home/3160552/adni_baselines_5pct_fold0_seed0_20261013/`

Result directory:
`/mnt/beegfsstudents/home/3160552/adni_baselines_5pct_fold0_seed0_20261013_results/`

Completed transfer, explicitly approved by the user:

- Source: `/Users/filippofocaccia/Desktop/hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl`
- Destination host: `bocconi-cluster` (`slogin.hpc.unibocconi.it`), account `3160552`
- Destination: `/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl`
- Payload: 60,242,117 bytes, including MRI crops, masks and pseudonymous subject metadata.

The dataset and staged job bundle were transferred and verified before invoking
the staged `scripts.adni_baselines.submit` module. It refuses to submit
without the exact dataset hash, immutable source/protocol manifest, account and
queue checks, and at least 5 GiB of remaining storage. Submission is idempotent
through a lock and receipt.

## Verification

- Real CPU forward/backward/optimizer checks for all four binary architectures.
- Binary FLOP profiles use PyTorch 2.2.2 / MONAI 1.6.0 and the existing arithmetic
  counter. Three-class control profiles reproduce all four MSD costs exactly.
- Three focused tests cover true subject grouping, complete paired hemispheres,
  preserved foreground during preprocessing, rejection of subject leakage, and
  binary Dice against analytic perfect/one-false-positive examples.
- Two-step real-data 3D U-Net integration and completed-run resume check passed,
  recorded in `integration_check.json`.
- Python parsing and Slurm shell syntax checked.

The cluster production runtime was verified as PyTorch 2.12.1+cu130, MONAI 1.6.0,
and dynamic-network-architectures 0.4.4. Production training uses the existing
hippocampus environment with CUDA BF16 and FP32 optimizer state.
CPU checks establish executable training/data paths, not segmentation quality.

## Completed comparison with MSD

The exact fold-0/seed-0 native baselines in
`reports/four_models_six_optimization_31976_20261006/runs_metadata/` are the MSD
references. Completion hashes verify the saved metrics. Runtime versions,
augmentation, initial learning rates, per-model update limits, LR horizons, and
learning rates at common steps match across datasets.

| Model | ADNI final Dice | MSD final Dice | ADNI gain over final 20% of compute | MSD gain over final 20% |
| --- | ---: | ---: | ---: | ---: |
| SwinUNETR | 62.66% | 42.60% | +5.95 pp | +5.24 pp |
| UNETR | 51.34% | 43.15% | +1.75 pp | +3.26 pp |
| 3D U-Net | 64.18% | 70.01% | +3.08 pp | +3.47 pp |
| nnU-Net | 69.09% | 67.07% | +6.49 pp | +35.61 pp |

The overall picture is continued learning under a restricted budget, with
different progress by model and dataset. U-Net's trajectory is the most similar.
ADNI nnU-Net flattens near the end, while MSD nnU-Net jumps from 42.17% at step 85
to 67.07% at step 90; similar final scores therefore do not imply similar
convergence. ADNI nnU-Net's best checkpoint is step 85, at 69.53%.

ADNI's binary whole-hippocampus score and MSD's anterior/posterior macro score
are different targets. Absolute cross-dataset Dice differences are descriptive,
not improvements on a common task. MSD metadata does not contain sufficient
intersection/count information to reconstruct whole-hippocampus Dice from its
per-class scores. Ten ADNI crops also correspond to only five subjects, whereas
MSD uses ten named cases. This one-fold, one-seed pilot does not establish matched
convergence or an equivalent optimization basin.

Artifacts:

- `adni_vs_msd_learning_curves.png` and `.pdf`: all recorded validation points.
- `adni_vs_msd_training_loss.png` and `.pdf`: raw objective trajectories; loss
  levels depend on class count/objective and averaging windows differ.
- `comparison_summary.csv` / `.json`: endpoints, peaks, late gains and checks.
- `matched_compute_checkpoints.csv`: 20 common budget targets, each rounded down
  to an affordable update in the corresponding binary/three-class model.
- `results_snapshot.json`: verified ADNI result metadata, retaining exact CSV bytes.
- `compare_learning_curves.py`: reproducible analysis and plotting script.
