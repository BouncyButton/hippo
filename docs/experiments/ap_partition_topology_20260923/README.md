# A/P partition topology and ordering: feasibility decision

## Decision

**Do not add a new topology or unfolding loss to the current training pipeline.**
The proposed division of labor is valid: supervised Dice assigns the A/P labels
and their location, while a label-symmetric structural loss could penalize an
incoherent partition. However, the available models already satisfy the weak
ordering rule almost everywhere, and most A/P errors occur in cases that pass
the connectivity checks. A stronger coronal-plane constraint already exists
and has already been trained in this repository.

Implemented a reusable, tested feasibility audit instead of introducing another
unvalidated training objective. This decision concerns the current MSD A/P task
and the inspected caches; it is not a claim that topological losses cannot help
other models, early training stages, or hippocampal subfield segmentation.

## What was checked

- 208 native fold-0 training masks: test whether the proposed rules agree with
  the labels before recommending them.
- 52 existing development cases, paired across the early-stopped unaugmented
  and translation-augmented seed-0 models. These are cached **local CPU** outputs,
  not the separately reported official CUDA results.
- Label values, RAS orientation, exact case membership, and cached reference
  agreement with symmetrically padded native NIfTI labels are checked.
- Recomputed baseline Dice and A/P swap totals must reproduce each cache's
  existing summary. Input hashes, checkpoint paths from the cache metadata,
  and audit library versions are recorded in `manifest.json`.

No checkpoint was retrained, no new inference ran, and no final holdout was
opened. Fold 0 has been repeatedly inspected in this project; all findings are
exploratory development evidence, not independent confirmation.

## Exact structural definitions

**Connectivity:** count 3D components of A, P, their union, and a contact band.
The contact band comprises A voxels face-adjacent to P and P voxels face-adjacent
to A. Report both 6- and 26-neighbor component counts. The primary summary
requires one component in each of these four masks under 26-connectivity.
This is a digital connectivity test, not proof that the interface is a manifold
disk. No tunnel, cavity, or genus invariant is inferred from these counts.

**Ordering:** along stored axis 1, split every ray into contiguous foreground
runs. Within each run allow A*P* or P*A*, including pure A or pure P runs.
Penalize only more than one class transition. Background gaps break runs and
are never bridged. Either semantic orientation is allowed, independently per
run: this directly tests the weak label-symmetric idea from the discussion.
It does not impose global orientation or transverse boundary smoothness.

**Diagnostic repairs:** find the closest valid hard labeling in Hamming
distance, preserving predicted foreground exactly. The ray repair uses the
rule above. A stronger plane repair fits one coronal plane across the whole
volume, considering both orientations and retaining occupied slices on both
sides. These two fits see predictions only. Ties have a deterministic rule
and are recorded; hard projection results are not training-loss results.

**Oracle comparison:** fit the best plane to the reference labels and render
it on the unchanged predicted foreground. This uses the answer and cannot be
deployed. It measures the opportunity in correct cut placement under this
specific projection; it is not a strict upper bound on achievable Dice.

## Findings

The training labels support a soft version of the rules: 207/208 pass all four
26-connectivity checks; all 208 have one posterior component and one contact
band component. One training case has repeated coronal transitions, comprising
seven one-voxel repairs. Thus neither complete connectivity nor single crossing
is an exception-free hard truth. Using 6-connectivity creates additional valid
label exceptions: 12/208 posterior masks and 8/208 contact bands are disconnected.

| Development endpoint | Unaugmented | Augmented |
|---|---:|---:|
| Cases | 52 | 52 |
| A/P macro Dice | 0.871443 | 0.886557 |
| A/P swap voxels | 4,292 | 3,702 |
| Cases passing all four 26-connectivity checks | 45 | 47 |
| Swaps in those passing cases | 3,893 (90.7%) | 3,440 (92.9%) |
| Cases with repeated coronal transitions | 1 | 0 |
| Foreground runs with repeated transitions | 1 / 19,348 | 0 / 19,121 |

All 52 development references pass the 26-connectivity tests and have no
repeated coronal transitions. The single defective baseline ray needs one
voxel edit, which happens to be correct. The augmented model needs none.

| Diagnostic intervention | Unaugmented Dice change, percentage points | Augmented Dice change, percentage points |
|---|---:|---:|
| Label-symmetric single-switch ray repair | +0.00075 | 0.00000 |
| Prediction-fitted single coronal plane | +0.04414 | +0.06296 |
| Reference-fitted plane on predicted foreground (oracle) | +2.29419 | +1.95284 |

The prediction-fitted plane improves/worsens Dice in 28/18 unaugmented cases
and 23/16 augmented cases. It corrects 514 errors while introducing 436 in the
unaugmented model; corresponding augmented counts are 280 and 170. This is a
small net effect with patient-level harms, not evidence for deploying a hard
projection. The oracle reduces A/P swaps to 235 and 232 respectively. Its
remaining errors include departures of the released labels from a perfect
plane. Foreground false positives and negatives are untouched throughout.

## How this relates to existing work

[`ExistentialAPPlaneLoss`](../../../thesis/new_constraints/ap_plane/existential.py)
already implements the main division of labor: foreground support comes from
training labels, the best cut is selected from predictions, and the ordinary
supervised objective locates the correct boundary. It is stronger than the
independent-ray rule and uses the documented dataset orientation.

The [completed plane-coherence experiment](../../../experiments/ap_plane_20260914/validation_audit/AUDIT_REPORT.md)
reported +0.277 percentage points of CUDA Dice on one seed/fold, with a paired
interval spanning zero. It reduced A/P swaps but did not improve overall
planarity. A perfectly coherent cut could still be several slices displaced.
Those results are separate from the CPU cache measurements above.

True [HippUnfold](https://pmc.ncbi.nlm.nih.gov/articles/PMC9831605/)
uses the cortical ribbon and anatomical boundary information to construct
intrinsic coordinates. MSD provides whole-hippocampus A/P masks, not the tissue
and boundary labels needed by that construction. A PCA axis or binary-mask
centerline would be a geometric proxy, not equivalent to HippUnfold. Moreover,
this dataset's head-versus-body/tail partition follows a coronal annotation
rule; a constant threshold in an intrinsic coordinate need not reproduce it.
We therefore did not fabricate an unfolding or assume its validity.

The evidence points toward **patient-specific cut localization**, not a lack
of basic partition structure. The oracle gain motivates that problem but does
not solve it: the earlier [frozen cut-head experiments](../../../semantic_constraints/ap_cut_head/EXPERIMENT_20260922.md)
also failed to establish a reliable new image-derived localization signal.
No additional head is justified merely by the existence of oracle headroom.

Hard-mask diagnostics cannot exclude a useful optimization effect from a soft
regularizer during training. Establishing that would require a matched study,
with gradient checks, training-only weight calibration, and comparisons against
the existing plane loss and conditional A/P cross-entropy. The current evidence
does not warrant that extra experiment as the next priority.

## Reproduce and verify

From the repository root:

```bash
.venv/bin/python evaluation/audit_ap_partition_topology.py \
  --output-dir docs/experiments/ap_partition_topology_20260923

.venv/bin/python -m pytest \
  evaluation/test_audit_ap_partition_topology.py \
  thesis/new_constraints/ap_plane/test_existential.py \
  thesis/new_constraints/ap_cut/test_posterior.py -q
```

The new tests exhaustively compare one-switch fits against all valid sequences
up to length seven, independently enumerate plane fits, verify empty/gapped
foreground behavior, distinguish 6/26-connectivity, preserve foreground during
repairs, and demonstrate the clean-but-displaced-boundary blind spot.

Artifacts: [summary](summary.json), [per-case measurements](cases.json),
[input and runtime manifest](manifest.json).
