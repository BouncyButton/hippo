# Cumulative ray supervision: gradient test

**Completed, 5 October 2026. Do not advance this cumulative-L1 formulation to
network training.** It detects arrangements that thickness and first moments
miss, but does not improve correction efficiency over the existing ray-moment
control. Plain CE is stronger in this frozen-logit comparison. The older ray
branch's stop decision remains supported.

## What ran

- Fifteen mathematical and implementation checks passed locally and on the
  cluster, including analytic cancellation examples, directional finite
  differences, physical scaling, ray reversal, oblique-coordinate interpolation,
  semantic swaps, high-confidence gradients, and MONAI Dice value/gradient parity.
- Separately checked that the plain wrong-only foreground BCE value and gradient
  match the September 15 implementation at strength zero.
- Slurm job **684544** completed with exit 0:0 in **3m39s**, one GPU. The probe
  itself took 109.8 seconds; the remainder includes startup and unit checks.
- **32 fold-0 training-partition cases:** the retained checkpoint's ten fitting
  cases plus 22 other training cases selected before inference. No validation
  labels were used by this probe.
- On eight of these annotations, tested correct masks, translations, tip
  truncations, protrusions, holes, and A/P shifts, plus two saturated translations.
- **820 gradient evaluations and 5,658 normalized logit updates.** All recorded
  small-step losses were non-increasing within the stated numerical tolerance.
- Verified output hashes, cohort membership, identical baselines, all error and
  per-class edit accounting, update budgets, metric deltas, and every aggregate.

The full predeclared design is in [PROTOCOL.md](PROTOCOL.md). Raw reports are in
the ignored `results_684544/` directory and retained on the cluster at
`/home/3160552/tomographic_gradient_20261005_01/results/`. `SUMMARY.json` and
`VERIFICATION.json` retain compact results and checks alongside this report.

## Real prediction results

The source is the current ten-case SwinUNETR baseline, not the retired mature
checkpoint used in the historical ray experiments. Baseline mean A/P Dice is
**0.679668**, with **96,521 semantic errors**: 93,494 FP, 896 FN, and 2,131 A/P
swaps. This is a strongly FP-dominated error distribution.

All rows below use the same per-case whole-logit update RMS **0.10**, without
network-parameter updates. Fixed/new counts are semantic errors; a foreground
error that becomes an A/P swap is not counted as a semantic fix.

| Loss alone | Errors fixed | New errors | Net errors removed | Mean A/P Dice change |
|---|---:|---:|---:|---:|
| Dice | 12,072 | 0 | 12,072 | +0.033753 |
| CE | 26,673 | 0 | 26,673 | +0.064680 |
| Wrong-only foreground BCE | 53,384 | 0 | 53,384 | +0.136745 |
| Legacy M0 | 12,486 | 266 | 12,220 | +0.025787 |
| Legacy M0+M1 | 13,591 | 243 | 13,348 | +0.028462 |
| Three-axis thickness L1 | 12,486 | 264 | 12,222 | +0.025798 |
| Three-axis cumulative L1 | 12,525 | 252 | 12,273 | +0.026090 |
| Seven-view thickness L1 | 12,300 | 267 | 12,033 | +0.025318 |
| Seven-view cumulative L1 | 12,316 | 247 | 12,069 | +0.025514 |
| Seven-view thickness + cumulative L1 | 12,305 | 259 | 12,046 | +0.025390 |

The cumulative term adds only **36 net fixes** over the same-view thickness
control, and loses **1,279 net fixes** to legacy M0+M1. It loses to M0+M1 in
**all 32 cases**, including the checkpoint's ten fitting cases. All ray arms
improve total error count against their unedited source; that is insufficient
evidence of added value.

This ordering is not a single-step-size artifact: the seven-view cumulative
loss loses to M0+M1 and CE at all six prespecified max/RMS step settings. Max-step
normalization changes some rankings among the legacy and axis-only controls,
which is why both normalizations are retained in the numerical reports.

### Added to Dice

Each auxiliary is rescaled to 10% of that case's Dice-gradient RMS, then the
combined update has total RMS .10. This is a diagnostic, not a training weight.

| Update | Net semantic errors removed | Additional fixes over Dice |
|---|---:|---:|
| Dice alone | 12,072 | 0 |
| Dice + legacy M0+M1 | 12,498 | 426 |
| Dice + seven-view cumulative L1 | 12,382 | 310 |
| Dice + seven-view thickness + cumulative L1 | 12,380 | 308 |
| Dice + CE | 13,354 | 1,282 |
| Dice + wrong-only foreground BCE | 16,120 | 4,048 |

These combined steps introduce no new semantic errors. The depth formulation's
small improvement over Dice is weaker than the legacy and simple controls.

## Where the gradient goes

For seven-view cumulative L1, the mean per-case fraction of squared logit-gradient
energy on **already-correct background is 98.624%**. It is 1.280% on FP, 0.026%
on FN, 0.009% on swaps, and 0.061% on correct foreground (rounding applies).

- All **896 false negatives** receive a harmful true-class versus strongest
  competitor margin direction; none receives a helpful direction.
- **96.04% of already-correct foreground voxels** receive a margin-decreasing
  direction. Most do not cross an argmax boundary at these small steps, but some
  create the new errors reported above.
- Nearly every FP receives a helpful direction. Since FP dominate this cohort,
  total repairs can look favorable while the gradient still opposes recovery of
  missing tissue.

These measurements support a broad foreground-suppression interpretation for
this checkpoint and loss. They do not prove the same distribution will occur at
every training stage. Correct-background energy can sharpen correct confidence;
it is not automatically useless, but it consumes nearly all this matched budget.

## Controlled errors on real training shapes

At RMS .10, pooled across eight annotations; entries are fixed/new semantic
errors. These are synthetic logits, not independent model-performance results.

| Error | Legacy M0+M1 | Seven-view cumulative | CE |
|---|---:|---:|---:|
| Translation | 734 / 5 | 302 / 3 | 11,590 / 0 |
| Missing tip | 509 / 2 | 154 / 0 | 1,262 / 0 |
| Protrusion | 580 / 16 | 106 / 1 | 814 / 0 |
| Internal hole | 199 / 0 | 45 / 0 | 264 / 0 |
| A/P shift | 815 / 0 | 473 / 0 | 1,228 / 0 |

The cumulative representation passes its intended algebraic falsifier: it
detects rearrangements preserving both total mass and first depth moment. That
extra representational information does not give better corrections in this
test. The correct-mask controls introduce no errors at RMS .10. Saturated
translation controls do not change hard labels at this magnitude for any arm;
raw gradients and continuous metrics remain available in the reports.

The independent local CPU sweep of the same eight shapes reproduces the table
above; local results are under `reports/tomographic_gradient_20261005/`.

## Decision and limits

**No training pilot for this version.** Adding cumulative profiles or four
oblique directions has not resolved the earlier branch's feedback-allocation
problem. Keep this implementation as a negative control and diagnostic.

This is not a rejection of every tomographic objective. It tests all-ray L1 mass
and full cumulative profiles, equally weighted union/A/P groups, the specified
sampling geometry, and one retained checkpoint. It does not test a carefully
localized residual loss, a different penalty, or full-network optimization.
Ordinary CE's stronger local corrections do not establish a new training result.
Wrong-only BCE's zero gradients at correct logits likewise do not imply that
shared network parameters would preserve correct predictions during training.

The historical checkpoint was unavailable after retirement. The current model
was fitted on ten cases and previously selected using validation; only ten of
the 32 probed cases were its fitting examples. The remaining 22 are still in
the designated training partition. The same qualitative comparison holds in
both groups. This run neither uses nor claims an untouched test set.

## Reproduction

```bash
rtk proxy .venv/bin/python -m unittest \
  semantic_constraints.tomographic.test_losses \
  semantic_constraints.tomographic.test_probe -v
rtk proxy .venv/bin/python -m semantic_constraints.tomographic.verify \
  docs/experiments/tomographic_gradient_20261005/results_684544
```

Cluster launcher: `scripts/run_tomographic_gradient.sh`. Implementation:
`semantic_constraints/tomographic/losses.py` and `probe.py`. The frozen run's
source hashes are in `SOURCE_MANIFEST.json`; the later results verifier records
its own source hash in `VERIFICATION.json`. The existing training runner and
legacy ray losses were not changed.
