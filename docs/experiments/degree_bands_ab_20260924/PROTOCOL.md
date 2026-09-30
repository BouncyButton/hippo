# Degree-weighted boundary supervision: A versus B

Requested scope: add optional degree weighting to the current bands objective;
submit **one Slurm job training only A and B**, with early stopping. Existing
Dice and Dice-plus-bands controls are reused, never retrained. This is a
single-seed, full-data fold-0 exploratory comparison on a repeatedly used
development fold, not an independent test-set claim.

## Fixed objective

Ground-truth whole foreground is anterior union posterior. Degree counts only
the six face-sharing foreground neighbours after the existing spatial
augmentation. The two-step six-connected inner and outer bands are unchanged.
Raw inner weight is `r = 1 + alpha * (6 - degree) / 6`, with **alpha = 2**.

- **A / inner:** divide raw weights by their per-case mean over the entire
  inner band. This redistributes weight both within the surface and from the
  deeper band to the surface.
- **B / surface:** divide raw weights by their per-case mean over the immediate
  foreground surface (degree < 6); keep deeper inner weights exactly one.

Inner weighted BCE and outer unweighted BCE are averaged separately, then
combined equally, with equal averaging over valid cases. The baseline
multiclass Dice is unchanged. Foreground logits are grouped using
`logsumexp(z_A,z_P)-z_BG`. No GNN, learned head, or inference postprocessing.
`--bands-degree-alpha 0` preserves existing loss/gradient behavior exactly.
`--bands-degree-normalization inner|surface` records the selected formula.
The existing focal option can compose with degree weights; both experiment
arms have focal gamma zero. Degree weighting is rejected for class-Tversky.
Foreground touching the crop edge fails clearly rather than treating an
unknown cropped neighbour as anatomical background. The fixed data and
augmentation must retain a background halo.

## Matched controls and training

Existing controls on the cluster:

- Dice: `/mnt/beegfsstudents/home/3160552/regularization_round1_20260916_01/runs/augmentation_seed0`.
- Bands: `/mnt/beegfsstudents/home/3160552/bands_augmented_20260916_04/runs/bands_augmented_seed0`.
- Epoch-five calibration source: `/mnt/beegfsstudents/home/3160552/bands_augmented_20260916_04/calibration_source/checkpoint_latest.pt`.

Both new arms use the controls' full 208/52 split, seed 0, 64-cubed padded
images without resize, `mild_v1` augmentation, batch 1, AMP, plain tensors,
no activation checkpointing, AdamW at 1e-4 with weight decay 1e-5, StepLR
period 20 and factor 0.5, and a five-epoch linear constraint ramp (0.2, 0.4,
0.6, 0.8, 1.0; not five Dice-only epochs). Training starts
from the same seeded random initialization independently in each arm. The
existing calibration checkpoint is used only to calibrate, never initialize.

Early stopping matches the historical controls: validation foreground macro
hard Dice, patience **8**, minimum epochs **25**, significant improvement
**0.0005**, maximum **50 epochs**. The best-Dice checkpoint is selected;
the latest checkpoint remains resumable. Early stopping limits continued
training after a plateau but cannot guarantee absence of overfitting.

Each variant gets a fresh, training-only gradient calibration on the same
deterministic 32 training cases and same augmented views from the existing
epoch-five checkpoint. The existing target ratio 0.10 and p95 safety cap 0.50
are retained. Lambda may differ between A/B because their gradient scales
differ; this compares normalized objectives under the same calibration policy,
not a fixed-lambda ablation. Reports bind alpha, normalization, source, inputs,
runtime and device. Historical reports lacking degree fields mean alpha=0,
normalization=inner and cannot silently calibrate a positive-alpha run.

The frozen cluster payload uses the historical `swin_unetr.py` source
(SHA256 `a95420e28c84fe682d737eee59aeecd9e61e4eb3efdda6893ecf7a3936ddd7fc`)
to match calibration/model/data provenance. Its differences from the current
workspace file concern the separate baseline training CLI. The user's local
baseline changes are preserved. The modified constraint trainer and loss
are frozen with a full payload hash manifest.

## Storage and submission

One `stud` allocation, one A100 MIG `4g.40gb`, six CPUs, 32 GB RAM, four-hour
wall limit. Calibrate both arms, then train A followed by B. CPU unit checks
and a no-training cluster preflight precede the single submission.

Initial BeeGFS quota: 90.67 / 93.13 GiB. Files use Buddy Mirror storage.
Logical output budget includes four full checkpoints (~196 MB each), two
selected model exports (~70 MB each), one atomic checkpoint replacement,
temporary input/checkpoint snapshots, source and small reports. Reserving
**3.5 GiB actual quota headroom** accounts for mirroring and margin; require
2.25 GiB before B. Quota is checked before submission, on job entry and before
each training run. No image arrays are exported by this study's final audit.

The user authorized cleanup if needed. Only hash-verified redundant latest
checkpoints from the completed 2026-09-23 boundary-auxiliary study are targeted;
all 12 fits selected epoch zero under their registered criteria. Their selected
weights, common states, reports, metrics, source and cleanup manifest remain.
No active job, baseline, calibration checkpoint or dataset is deleted.

## Evaluation

Re-score the two historical selected checkpoints and new A/B selected
checkpoints on the same 52 validation images, using identical CUDA/AMP inference.
Check historical selected-export hashes and checkpoint/config consistency.
Report whole union and A/P Dice, FP/FN counts, A/P swaps, and pooled
bidirectional six-face surface-voxel ASSD and HD95 at 1-mm isotropic spacing.

Report reference whole-graph degree strata 0-1, 2-3, 4-5, 6, both pooled and
per case. Anterior/posterior strata subset that same whole-graph degree;
they are not internal A/P interface degrees. Compare A-B, A-bands, B-bands,
A-Dice and B-Dice with paired per-case Dice differences and descriptive
bootstrap intervals. The higher validation macro-Dice arm is identified,
alongside false positives and surface errors; improved exposed-voxel recall
alone is not sufficient evidence of anatomical improvement.

The objectives already differ during the initial ramp, so early training
trajectories need not match. Neither historical control is rerun. Preserve
per-case audit data on the cluster; only non-identifying aggregate results
need be retrieved.

## GPU request update

At the user's request, the replacement allocation uses A100 MIG `3g.40gb`
and four CPU cores (OMP/MKL threads=4), retaining 40-GB GPU memory and the
32-GB host-memory/four-hour limits. The objectives, data, split, seed, batch
size, augmentation, optimizer, calibration policy and early stopping remain
unchanged. The original frozen source is preserved; the new payload is in
`degree_bands_ab_20260924_02_3g` on the cluster.

The epoch-five calibration checkpoint retains its original 4g provenance.
An explicit option permits transfer only between A100 80GB PCIe 3g.40gb and
4g.40gb with matching compute capability and numerical backend settings.
All source/runtime/input/checkpoint hashes remain checked. Calibration is
recomputed on the target device, and the resulting reports must exactly
match the execution device used to train both arms. Actual hardware is
recorded; numerical identity with historical 4g controls is not assumed.

Validation: 98 focused tests passed, including migration opt-in, rejection
of other hardware/numerical-policy changes, report validation for both
profiles, degree losses and legacy calibration compatibility. Shell syntax
and whitespace checks passed.
