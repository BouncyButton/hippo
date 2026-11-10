# Filippo_dev review and commit inventory — 10 November 2026

Reviewed the pending research backlog from `91aea0b0` on `Filippo_dev` and
split it into source, dataset, experiment-retirement, evidence, and thesis commits.
The remote branch matched the starting commit; `origin/main` was already an ancestor.

## Scope

- Add local dataset builds and physically aligned, subsampled COBRA crops.
- Preserve shape refiners, calibrated spatial objectives, compute-budget controls,
  cross-dataset experiments, MNI recipe selection, and resumable Colab/cluster runners.
- Record existing experiment retirements with their provenance and cleanup receipts.
- Preserve curated metrics, protocols, split information, and thesis sources/assets.
- Keep raw volumes, checkpoints, duplicate packages/source snapshots, runtime state,
  and Overleaf exports local. No local experiment artifacts were deleted in this review.
- Leave `papers/hippocampus_spatial_2027` out at the user's request because another
  chat is actively writing the paper. Live collectors may continue updating local reports;
  committed evidence represents the snapshot captured during staging.

## Review fixes

Added two synthetic COBRA crop-geometry tests. Corrected two test harnesses:
pytest flags no longer become a replication worker path, and the foundation-fold
check generates its dispatcher using the same transformation as campaign preparation.
Final whitespace cleanup preserves the Python AST of every touched source file.

## Validation

- Parsed selected Python sources and checked shell/Slurm source syntax successfully.
- Selected JSON evidence parsed successfully; all 23 thesis figure references resolve.
- Bounded credential-pattern scans found no recognizable provider tokens or private keys.
- Colab/cluster/dataset review: 351 tests and 34 subtests passed, 2 skipped.
- Portable numerical subset: 47 passed; standalone scientific/recovery checks: 10 passed.
- Corrected harnesses: 3 passed. Three objective checks that failed in a combined
  run passed in isolation.

The initial broad run reported 403 passed, 1 skipped, and 24 failures (including
subtests) before the harness fixes. Other failures were traced to the installed
MONAI 1.4.0 versus the profiled 1.6.0 requirement, or shared global state between
historical test modules. No version guard was bypassed. Run these historical
modules in separate processes with `PYTHONPATH=.:scripts` and a writable
`MPLCONFIGDIR`. Optional unwrapped tests need `skimage` and `trimesh`.

The full ADNI runtime module exceeded a 120-second review limit; its focused
recovery/setup subset passed. Thirteen archive-dependent checks were deselected.
GPU training, cluster submission, and full data/checkpoint smoke pipelines were
not run. No TeX compiler was available, so thesis validation checked assets and
source structure rather than producing a new PDF.

Final verification on a clean export of the committed sources: **52 tests passed**
in 28.33 seconds, including the numerical subset, both repaired harnesses, and
COBRA crop geometry. The export included the tracked evaluation dependencies;
it did not use local report bundles or untracked data. Source whitespace checks
also passed. The push contains 18 focused commits; the active paper remains local.
