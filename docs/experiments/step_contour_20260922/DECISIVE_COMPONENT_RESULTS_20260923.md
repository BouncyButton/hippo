# Decisive presence experiments

Internal development results; not external confirmation.

## Saved-model component interventions

Oracle rows use labels and are non-deployable. Ray overlap requires a correctly predicted foreground voxel on the true ray.

| Intervention | ASSD mm | Thin ray overlap recall | Thin voxel recall | FP rays | FP voxels |
|---|---:|---:|---:|---:|---:|
| A | 0.495716 | 63.12% | 59.47% | 40.05 | 422.50 |
| E | 0.483148 | 55.55% | 53.84% | 28.94 | 393.16 |
| E_raw | 0.523650 | 83.12% | 78.79% | 80.75 | 605.09 |
| distance_renderer | 0.483140 | 55.65% | 53.91% | 29.02 | 393.41 |
| no_quadratic_fit | 0.482107 | 55.52% | 53.77% | 28.88 | 386.71 |
| oracle_edges | 0.380552 | 55.20% | 53.89% | 28.94 | 331.98 |
| oracle_gt_replace_empty_fp | 0.424209 | 55.55% | 53.84% | 0.00 | 293.58 |
| oracle_gt_replace_missed_thicker | 0.418974 | 55.55% | 53.84% | 28.94 | 393.16 |
| oracle_gt_replace_missed_thin | 0.459535 | 98.42% | 95.24% | 28.94 | 393.16 |
| oracle_gt_replace_present_wrong | 0.132504 | 57.13% | 58.61% | 28.94 | 99.59 |
| oracle_presence_and_edges | 0.262035 | 86.92% | 84.42% | 4.17 | 250.35 |
| oracle_presence_fit_and_renderer | 0.398784 | 90.60% | 85.20% | 4.17 | 341.11 |
| oracle_presence_fit_only | 0.483561 | 55.56% | 53.92% | 28.88 | 390.47 |
| oracle_presence_renderer | 0.405041 | 92.05% | 86.47% | 4.17 | 350.00 |
| oracle_raw_restore_erased_thin | 0.476112 | 81.89% | 75.62% | 28.94 | 402.75 |
| oracle_thin_presence | 0.483316 | 92.05% | 86.47% | 28.94 | 423.31 |

## Exact surface attribution

Each contribution is already divided by its method's full surface denominator; these are contributions to ASSD, not subgroup conditional means.

| Reference ray stratum / original E presence | A | E raw | E final | E minus A |
|---|---:|---:|---:|---:|
| empty | 0.053518 | 0.099231 | 0.045711 | -0.007807 |
| length3_5_E_missed | 0.031133 | 0.021752 | 0.035615 | +0.004482 |
| length3_5_E_present | 0.126078 | 0.131126 | 0.120153 | -0.005925 |
| length6plus_E_missed | 0.005032 | 0.004334 | 0.005060 | +0.000028 |
| length6plus_E_present | 0.244591 | 0.232330 | 0.239432 | -0.005159 |
| multi_run_E_missed | 0.000231 | 0.000208 | 0.000251 | +0.000020 |
| multi_run_E_present | 0.002730 | 0.002642 | 0.002756 | +0.000026 |
| thin1_E_missed | 0.003394 | 0.003105 | 0.003387 | -0.000006 |
| thin1_E_present | 0.004131 | 0.004809 | 0.004398 | +0.000267 |
| thin2_E_missed | 0.010300 | 0.006738 | 0.011659 | +0.001359 |
| thin2_E_present | 0.014578 | 0.017375 | 0.014725 | +0.000147 |

## Conditional paired ASSD uncertainty

- A: +0.012568 mm; conditional case-bootstrap interval [0.008200553174778103, 0.017376266678206836].
- E: +0.000000 mm; conditional case-bootstrap interval [0.0, 0.0].
- E_raw: +0.040502 mm; conditional case-bootstrap interval [0.03416800474585828, 0.047252627044610175].
- distance_renderer: -0.000008 mm; conditional case-bootstrap interval [-7.533791337135117e-05, 5.808387221038268e-05].
- no_quadratic_fit: -0.001041 mm; conditional case-bootstrap interval [-0.0017025522782366505, -0.00033662808977606424].
- oracle_edges: -0.102595 mm; conditional case-bootstrap interval [-0.10539796150163804, -0.09978363869121908].
- oracle_gt_replace_empty_fp: -0.058939 mm; conditional case-bootstrap interval [-0.0656601174515664, -0.05314170759845926].
- oracle_gt_replace_missed_thicker: -0.064174 mm; conditional case-bootstrap interval [-0.0706674132320904, -0.05849079174430621].
- oracle_gt_replace_missed_thin: -0.023613 mm; conditional case-bootstrap interval [-0.025647892989865757, -0.021654297386529728].
- oracle_gt_replace_present_wrong: -0.350643 mm; conditional case-bootstrap interval [-0.36094187690160473, -0.33993000400343343].
- oracle_presence_and_edges: -0.221112 mm; conditional case-bootstrap interval [-0.23039013561743923, -0.21282861914375692].
- oracle_presence_fit_and_renderer: -0.084363 mm; conditional case-bootstrap interval [-0.0927652200742988, -0.0765546252979981].
- oracle_presence_fit_only: +0.000414 mm; conditional case-bootstrap interval [5.5546012091186864e-05, 0.0007954341857336249].
- oracle_presence_renderer: -0.078107 mm; conditional case-bootstrap interval [-0.0869819368948557, -0.06927795643901263].
- oracle_raw_restore_erased_thin: -0.007036 mm; conditional case-bootstrap interval [-0.007853400717265525, -0.006268036289276671].
- oracle_thin_presence: +0.000169 mm; conditional case-bootstrap interval [-0.002297732328830823, 0.0029404721089956853].
