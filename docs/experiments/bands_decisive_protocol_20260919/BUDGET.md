# Proposed compute budget

Generated from archived timing logs plus explicit planning allowances. These are allocated MIG-slice hours, not full-GPU hours, elapsed calendar time, or measured future costs.

| Stage | Production runs | Estimated slice-hours |
|---|---:|---:|
| development_horizon | 24 | 6.34 |
| legacy_anchor | 6 | 0.74 |
| development_ce_sensitivity | 3 | 1.02 |
| development_fraction_check | 5 | 1.70 |
| replication_10 | 48 | 16.36 |
| fraction_52 | 24 | 8.18 |
| boundary_placement_control | 12 | 4.09 |
| Shared calibration sources | 28 | 0.93 |
| Total before reserve | 122 production + 28 source | 39.37 |
| Total with 25% reserve | same matrix | 49.21 |

Four-arm plan without the additional matched-random control: 44.10 slice-hours including reserve. The matched-random control isolates boundary placement from merely supervising fewer voxels; decide whether to include it before replication results are revealed.

Reduced development-only plan: 27 production runs (24 horizon runs and three unit-CE checks) plus three calibration sources, 9.32 slice-hours including reserve. This omits historical anchors, the 52-case check and all replication.

Formula per production run: updates × 0.06452 s + validation events × (1.894 s + 2 × 0.899 s) + 120 s startup/final-evaluation allowance.

Calibration: 120 seconds per unique split/subset/model-seed/fraction shared within the experiment, with no validation-based weight selection. This is an allowance, not a measured calibration duration.

A standalone-method time-to-target must charge its full calibration-source and gradient-calibration time, even if the study reuses a cached source among methods. Research cost and standalone method cost are different accounting views.

If validation contains N case files, scale the validation term provisionally by N/52; measure it before launch. The budget assumes an update-based loop with validation every 50 updates. The current epoch-based runner is not equivalent and would have a different budget.

Rebenchmark a small preflight before committing to the full matrix. Do not spend the reserve selectively on unfavorable seeds. A budget reduction should remove a complete future stage or balanced blocks, not stop arms based on their results.

No job has been submitted. RUN_MATRIX.csv is intentionally non-launchable until manifests and source hashes are resolved.
