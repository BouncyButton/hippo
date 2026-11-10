# Stopped nnU-Net warmup experiment: job 682232

Job 682232 was cancelled at the user's request on 2026-10-04 at 11:58:11 Europe/Rome (09:58:11 UTC). Four of nine planned runs completed; the fifth was initializing and had no recorded optimizer updates. After analysing all eight saved best/latest checkpoints and verifying the evidence archive, the rerun directory was deleted at 10:13:12 UTC. No replacement training job was submitted.

The experiment shows that a 17-update warmup did not resolve the anterior-class failure. It also introduced a severe failure in fold 0/seed 1. The observed collapse is real, but its underlying cause remains unconfirmed.

## Completed runs and failure points

All four runs used 327 optimizer updates, batch size one, ten training patients, 64³ inputs, and 52 validation patients. An epoch here means one pass over the ten training patients: 327 updates are 32 complete passes plus seven updates (32.7 pass equivalents). Dice below is validation hard Dice at the checkpoint selected by mean anterior/posterior Dice; it is not an independent test result.

| Fold | Seed | Selected update | Anterior Dice | Posterior Dice | Mean Dice |
|---|---|---:|---:|---:|---:|
| 0 | 0 | 327 | 0.845209 | 0.827350 | 0.836279 |
| 0 | 1 | 17 | 0.001078 | 0.007773 | 0.004426 |
| 0 | 2 | 311 | 0 | 0.716571 | 0.358285 |
| 1 | 0 | 295 | 0.836156 | 0.819333 | 0.827744 |

For fold 0/seed 1, both classes have zero hard Dice at every recorded validation checkpoint from update 33 through 327. Thus collapse was observed by 3.3 pass equivalents; the metrics do not locate the exact intervening update. The latest mean Dice is zero, despite mean training loss falling from 1.223716 at update 17 to -0.274606 at update 327.

For fold 0/seed 2, the anterior remains zero from update 33 onward. Both classes are zero at recorded checkpoints 33–213; the posterior starts recovering at update 229 and reaches 0.716571 at the selected update 311. Warmup delayed posterior recovery without rescuing the anterior.

The original fold 0/seed 1 mean Dice was 0.841761, compared with 0.004426 after warmup. Original fold 0/seed 2 selected mean Dice was 0.386171, compared with 0.358285 after warmup. These are historical recipe comparisons, not a basis for selecting the better recipe independently for each seed. Four completed runs cannot be presented as the planned nine-run aggregate.

![Original versus warmup validation curves](failure_curves.png)

## Checkpoint audit

The audit reconstructed the frozen experiment and evaluated the eight best/latest checkpoints on CPU in FP32, without optimizer steps or changes to saved weights. Each checkpoint was examined on two validation patients and one training patient. The full validation curves above come from the original 52-patient evaluation, not from this smaller diagnostic sample.

All saved model tensors were finite. All four latest checkpoints' saved momentum buffers were finite. Both foreground outputs received nonzero gradients. Every audited training entry preserved both foreground labels through preprocessing and all active deep-supervision targets. Both labels also remained present in the 90 audited first-pass augmented samples. These checks do not support absent labels, disconnected foreground outputs, or a simple evaluation class-mapping error as explanations.

Fold 0/seed 1 at update 327 predicted all 262,144 voxels as background on both sampled validation patients and the sampled training patient. On ground-truth anterior voxels in the training example, mean probabilities were background 0.96013, anterior 0.01681, posterior 0.02305. On ground-truth posterior voxels they were 0.94399, 0.02380, 0.03221. This is failure to learn the training example too, not merely poor generalization.

Fold 0/seed 2 at update 327 likewise predicted no anterior on these three examples, while predicting useful posterior regions. Seed 0 provided successful controls in both completed folds.

FP32 evaluation rules out an inference-only BF16 explanation on the sampled cases. It does **not** rule out effects of BF16 during the preceding training trajectory. Finite optimizer buffers also do not rule out unsuitable momentum dynamics.

## Why decreasing loss concealed the failure

The native Dice component is negative, so a negative Dice+CE loss is expected and is not itself an implementation error. Deep supervision weights are [4/7, 2/7, 1/7, 0]. Evaluation uses the full-resolution output.

For the failed seed 1 latest checkpoint on the sampled training patient:

| Quantity | Value |
|---|---:|
| Full-resolution mean CE | 0.060591 |
| Full-resolution negative Dice | -0.034662 |
| Full-resolution Dice+CE | +0.025929 |
| Total weighted deep-supervision loss | -0.284856 |
| Implied weighted mean of the two active auxiliary losses | -0.699235 |

The auxiliary mean is derived from `(total - (4/7)*primary)/(3/7)`; it is not a separately measured auxiliary Dice score. It establishes that the auxiliary terms make the aggregate loss substantially lower while the evaluated full-resolution output predicts only background.

Full-resolution CE averaged over true background/anterior/posterior voxels was 0.005291 / 4.133686 / 3.652921. Background dominates the voxel average, obscuring the large errors on the two foreground classes. Across the 30 training entries, median combined foreground fraction drops from 5.383% in the raw images to 1.256% after padding to 64³. Foreground is preserved, but its relative contribution is diluted. This is a plausible contributor, not an established cause; successful seeds and other models use the same inputs.

## Related evidence from official nnU-Net sources

Sources checked on 2026-10-04. These establish related failure modes and useful hypotheses, not a diagnosis of this experiment.

- The historical **nnU-Net v1** trainer contains an explicit fallback for zero estimated validation Dice at epoch 100: it lowers momentum from 0.99 to 0.95 and reinitializes network weights. Its comments identify excessive momentum as a possible cause. The class is called `nnUNetTrainerV2`, but it belongs to the v1 branch; this is not evidence that the current nnU-Net v2 trainer has the same fallback. [Historical trainer, lines 380–389](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/training/network_training/nnUNetTrainerV2.py#L380-L389).
- A 2021 issue reports rare-class Dice collapsing to zero in low-resolution training while full-resolution training succeeds. The accessible issue body does not establish a resolved cause. [Issue #812](https://github.com/MIC-DKFZ/nnUNet/issues/812).
- A July 2026 issue reports sparse multiclass MRI training reaching near-background predictions while Dice+CE loss keeps improving, including with foreground oversampling increased to 0.9. Its reported foreground fraction (~0.07%), hardware, training duration, and early pseudo-Dice measurements differ from ours. [Issue #3048](https://github.com/MIC-DKFZ/nnUNet/issues/3048).
- The current native trainer defaults to 1,000 epochs with 250 training iterations per epoch, initial LR 0.01, SGD momentum 0.99, and CUDA autocast with gradient scaling. Our 327-update schedule is an extreme compression of that horizon. Our fixed architecture, shared input preparation, augmentation, and BF16 training form an **nnU-Net-derived baseline**, rather than the complete automatically planned native pipeline. [Current native trainer](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py).

## Interpretation and experiments that would distinguish causes

The strongest documented hypothesis to test next is the interaction between high momentum, the short optimization horizon, and class imbalance. Warmup alone was insufficient. A momentum coefficient of 0.99 retains gradient history on an approximate 100-update time scale; 327 updates span only about 3.3 such scales. This calculation motivates a control but does not prove causality. Initialization, data order, and augmentation all depend on the same seed, so their effects have not been isolated.

Before another full campaign, controlled diagnostics would separate momentum (e.g. 0.99 versus 0.95 with other choices fixed), training precision, and full-resolution versus auxiliary loss/gradient behavior. Changing all of them together would obscure which intervention matters. Foreground-presence oversampling alone does not address the observed problem: every audited input already contains both classes. Tighter ROI preparation changes voxel proportions and would require a separately disclosed protocol and consistent comparison inputs.

Any later revised baseline should apply one declared recipe to all folds/seeds, keep the per-run FLOP cap, and disclose the additional tuning compute. A restart or reinitialization must be charged for all attempted updates; a native epoch-100 recovery rule cannot be treated as free compute or translated into update 100. No such experiments were launched here.

## Evidence preservation and deletion

The four runs each consumed the declared 116181.104090427 GFLOPs, below the 116259-GFLOP cap. Their combined research training expenditure was 464724.416361708 GFLOPs. CPU diagnostics are additional analysis compute outside that training ledger.

Only `/mnt/beegfsstudents/home/3160552/nnunet_warmup_682232/nnunet` was removed: 994,052,730 bytes. Original U-Net and SwinUNETR run directories were verified present. The stopped job's source snapshot, logs, and retirement marker remain on the cluster.

The local archive contains 41 records: configurations, validation results, curves, budget ledgers, completions, logs, frozen protocol/trainer, and diagnostic records. Checkpoint binaries were deleted after analysis and are not included in the archive. Archive SHA-256 and all individual record hashes were verified remotely and locally.

- [Stopped run records](stopped_failure_records.json)
- [CPU FP32 checkpoint audit](checkpoint_audit.json)
- [Verified deletion manifest](deletion_manifest.json)
- [Preserved evidence archive](stopped_run_records.tar.gz)
- [Exportable failure curves](failure_curves.pdf)

The deletion manifest records cancellation, exact target, file inventory, preserved hashes, and deletion time. The output root has `NNUNET_RETIRED.json` with `do_not_resume: true`.
