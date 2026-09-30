# CST slice-error types and correction-action gate — 2026-09-22

This follow-up tested whether the trainable CST slice-QC component could be
turned into a reliable correction direction. It used the **early-best** frozen
Swin predictions, six frozen CST feature banks, and patient-disjoint CV within
each 52-case MSD validation fold. Both folds have already informed research
choices; this is discovery evidence, not a prospective test.

## Runs and operational targets

| Job | Purpose | Output | Status |
|---|---|---|---|
| `665517` | Derive auditable error-type targets, both folds | `/home/3160552/hippopotamus_runs/cst_error_types_665517` | completed |
| `665527` | Three-seed patient-grouped missing-vs-extra probe | `/home/3160552/hippopotamus_runs/cst_error_direction_665527` | completed |
| `665533` | Ground-truth oracle foreground-bias action bound | `/home/3160552/hippopotamus_runs/cst_action_oracle_665533` | completed |

The target taxonomy counts wrong voxels on each of the 32 sampled coronal
slices: reference foreground predicted as background (`missing`), background
predicted as foreground (`extra`), and anterior/posterior class confusion
(`ap_swap`). Under five wrong voxels is `minimal`. A type is dominant only when
it accounts for at least 60% of wrong voxels; other slices are `mixed`. These
are operational mask-comparison labels, **not** clinical anatomy labels.

| Fold | Minimal | Missing | Extra | A/P swap | Mixed | Isolated disappearance | Terminal extension |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 679 | 313 | 433 | 22 | 217 | 0 | 12 |
| 1 | 678 | 311 | 397 | 31 | 247 | 0 | 8 |

There are enough missing/extra slices for a directional probe, but not enough
isolated disappearances to validate that phenomenon. Terminal extensions and
A/P swaps are too rare for a trustworthy stand-alone correction model here.

## Direction is not reliably learnable from the current CST features

The probe fitted L2 logistic heads within five patient-held-out folds for
each teacher seed. It trained only on training-patient slices unambiguously
classified as missing or extra. No held-out patient labels entered fitting or
feature standardization. The table evaluates those directional slices among
the highest-ranked 20% according to the existing *out-of-fold* QC head.

| Fold | Model | Direction accuracy, mean ± seed SD | Majority baseline | AUC | Fraction scored with ≥0.8 confidence |
|---:|---|---:|---:|---:|---:|
| 0 | uncertainty | `0.489 ± 0.007` | `0.523 ± 0.016` | `0.493 ± 0.019` | `0.015` |
| 0 | portable CST relationships | `0.512 ± 0.016` | `0.523 ± 0.016` | `0.512 ± 0.012` | `0.036` |
| 0 | combined CST embedding | `0.539 ± 0.033` | `0.523 ± 0.016` | `0.550 ± 0.028` | `0.082` |
| 1 | uncertainty | `0.635 ± 0.008` | `0.576 ± 0.019` | `0.677 ± 0.009` | `0.008` |
| 1 | portable CST relationships | `0.615 ± 0.004` | `0.576 ± 0.019` | `0.632 ± 0.007` | `0.056` |
| 1 | combined CST embedding | `0.590 ± 0.023` | `0.576 ± 0.019` | `0.613 ± 0.009` | `0.147` |

The combined head barely beats the majority baseline on fold 0 and loses to
uncertainty on fold 1. Its high-confidence predictions have little coverage
and only `0.558` and `0.762` accuracy on folds 0 and 1, respectively. This
fails the predefined gate for a directional correction. No final action head
was fitted or deployed.

The QC component *does* make a narrower review contribution: at the same 333
of 1664 slice review budget, matched combined heads identify 20–21 of 22
swap-dominant slices on fold 0, versus 16 for uncertainty alone, and 30 of 31
on fold 1, versus 24. These are small absolute counts and not proof of a
deployable swap detector. The QC score should flag them for review, not
automatically switch labels.

## Simple probability shifts have a tiny ceiling

The action study changed foreground-vs-background log odds on the QC-selected
slices, preserving anterior/posterior odds. Tested shifts were `±0.25`,
`±0.5`, and `±1.0`. Both probes below use ground-truth labels and are **oracle
diagnostics, not achieved model improvements**:

| Fold | Perfect error-type sign, fixed `0.5` shift: mean case Dice gain | Patients harmed | Best label-selected shift per slice: mean gain |
|---:|---:|---:|---:|
| 0 | `+0.00016` to `+0.00022` | `12–15 / 52` | `+0.00087` to `+0.00095` |
| 1 | `+0.00028` to `+0.00029` | `9–10 / 52` | `+0.00105` to `+0.00107` |

The per-slice best-action oracle chose among every tested sign and strength,
including no-op, using the reference mask. Its gain is therefore an optimistic
upper bound for this limited action family. Even perfect missing/extra sign
is not safe at fixed strength. Do **not** train or deploy CST-gated foreground
bias from these features.

## Decision

Keep the matched CST ridge ensemble as **slice-level selective QC**. Do not
turn its scalar error score into an anatomy-matching loss or a missing/extra
correction. A next correction attempt would need a genuinely local image/mask
proposal that can represent boundary and interface geometry, plus a separate
no-harm acceptance test. The present data do not establish that such a model
will work. Before any clinical or prospective claim, freeze the pipeline and
evaluate on an untouched compatible labeled cohort. The available ADNI dataset
uses a single binary hippocampus class, so it is not a drop-in test of the
MSD anterior/posterior error taxonomy.
