# Minimum-presence feasibility results

All interventions use reference labels and are non-deployable. Raw voxel logits, fitted edges, and beta remain fixed.

| Variant | ASSD mm | Correct thin-ray overlap | Thin voxel recall | FP rays/case | FP voxels/case |
|---|---:|---:|---:|---:|---:|
| E | 0.483148 | 55.55% | 53.84% | 28.94 | 393.16 |
| oracle_minimum_any | 0.473279 | 92.05% | 74.96% | 28.94 | 398.02 |
| oracle_minimum_all | 0.471946 | 83.00% | 81.25% | 28.94 | 398.60 |

The any endpoint asks for at least one correct voxel; the all endpoint asks for every true foreground voxel. For an infeasible endpoint, the original ray is left unchanged. The all variant can therefore recover fewer rays while recovering more voxels.

| Endpoint | Initially failing rays | Feasible by increasing presence | Feasible with zero extra FP voxels | Infeasible |
|---|---:|---:|---:|---:|
| oracle_minimum_any | 4078 | 3355 | 2685 | 723 |
| oracle_minimum_all | 4649 | 2881 | 2120 | 1768 |

For the any endpoint, 82.27% of 4,078 failing rays are conditionally recoverable; 65.84% of all failing rays (80.03% of recoverable rays) can be recovered without adding an FP voxel on that ray. The remaining 723 failures cannot be repaired by increasing scalar presence under the current raw logits and geometry.

These results establish representational feasibility, not learnability. They also expose a distinction between predicting whether a ray exists and choosing how strongly to modify its voxels. Setting true-thin presence to one added 30.15 FP voxels/case in the preceding component audit; minimum sufficient increases add 4.86/case for the any endpoint.

## Conditional paired ASSD intervals

- oracle_minimum_any: -0.009869 mm, conditional case-bootstrap interval [-0.011049077301014764, -0.008695847481999879].
- oracle_minimum_all: -0.011202 mm, conditional case-bootstrap interval [-0.012521739575566328, -0.009863043092091314].
