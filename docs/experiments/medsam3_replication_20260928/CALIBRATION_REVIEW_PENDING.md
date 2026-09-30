# Calibration consistency review pending

During a manual progress check of job 673784 at about 48 minutes elapsed, fold0_seed83 was evaluating the bands model after both baseline and bands completed 200 updates. There were no runtime errors. Only fold0_seed17 had a completed, audited three-arm report.

A prospective final-aggregation issue was identified: calibration coefficients differ across these two seeds even though calibration cases/anchors match and the calibration RNG seed is fixed. The first comparison ran on gnode01; the resumed comparison runs on gnode02. This observation alone does not establish the cause.

| Coefficient | Seed 17 | Seed 83 | Relative difference |
|---|---:|---:|---:|
| Bands | 13.529893228346154 | 13.529747799720313 | 1.0748689837090067e-5 |
| Edge | 298.153307381751 | 298.1818065346947 | 9.558556701576152e-5 |

The current final aggregation gate uses rtol=1e-5, atol=1e-8 and will reject these differences after all comparisons finish. Training, per-comparison audits, and retention of raw results are unaffected. No tolerance, coefficient, training setting, source manifest, or running process was changed during this status check.

Before accepting a final aggregate, investigate calibration reproducibility and model initialization across runs using support-only evidence. Do not silently relax the gate or describe the coefficients as exactly identical. Preserve this record and disclose any eventual audit correction. The user requested manual checks; no monitoring automation is active.

## Completion update
All comparisons finished. See RESULTS.md and aggregate_observed.json. The original gate remains failed; final maximum coefficient deviation is 0.1054%, rather than the <0.01% observed in the first two seeds. Initial parameter hashes and calibration cases/anchors match. The cause and effect are unresolved; no threshold was loosened.
