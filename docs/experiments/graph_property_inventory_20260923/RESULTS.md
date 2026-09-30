# Additional measurements from the property checklist

Computed 1,092 region descriptor archives: 260 reference cases plus 52 cases from each of two existing prediction caches, each with whole/anterior/posterior graphs. Training and validation are fold 0 (208/52). No segmentation model was trained.

## Exact distances and the centerline claim

| Region | Median exact LCC diameter, train / val (hops) | Median boundary fraction of chosen diameter path, train / val | Branched coronal slice graphs, train / val |
|---|---:|---:|---:|
| Whole | 68.0 / 69.0 | 74.6% / 71.6% | 78/208 / 20/52 |
| Anterior | 40.0 / 40.0 | 81.3% / 79.7% | 43/208 / 11/52 |
| Posterior | 53.0 / 54.0 | 90.9% / 92.9% | 51/208 / 13/52 |

The diameter is an exact longest-shortest-path distance within the largest component. Other components remain recorded. Each hop is a 1-mm face step, so the metric contains grid anisotropy. A deterministic shortest path between the selected diameter endpoints frequently runs on the boundary: **diameter does not specify a medial centerline**. Alternative equally short paths can exist. The slice graph is a coronal component-adjacency proxy, not an anatomical skeleton or a continuous Reeb graph.

## New coordinates and cut-location probe

| Coordinate | Training-median threshold | Best case-specific threshold error, train / val | Fixed training threshold error, train / val |
|---|---:|---:|---:|
| Fiedler [0,1] | 0.7430 | 1.91% / 1.87% | 4.10% / 3.79% |
| Fiedler volume rank | 0.4778 | 1.91% / 1.87% | 4.55% / 4.16% |
| Harmonic Y-end field | 0.6997 | 1.93% / 1.86% | 4.39% / 5.23% |

Errors are mean per-case A/P misclassification fractions on the **reference largest component**, not Dice and not deployable prediction results. The case-specific fits use GT labels and are oracle diagnostics. The common threshold uses training labels only, but evaluation still uses reference foreground. Fiedler is oriented by RAS Y, not a verified anatomical landmark. The harmonic field uses low/high 10% Y-extent anchors. Rank normalization preserves scalar ordering, so it cannot improve the optimal representable partition.

## Which reference voxels are missed?

| Reference whole-graph degree | Reference voxels across 52 cases | Missed foreground, unaugmented | Missed foreground, augmented |
|---|---:|---:|---:|
| 0–1 | 286 | 75.87% | 80.07% |
| 2–3 | 21,061 | 40.34% | 40.81% |
| 4–5 | 50,186 | 13.20% | 11.77% |
| 6 | 102,817 | 0.91% | 0.73% |

These are pooled voxel rates, not independent observations or proof of a useful loss. The strata use reference degree solely for scoring. A feature computed only on predicted foreground has no node at a missed voxel, which is why a full-grid depth/boundary auxiliary target or an explicitly extended correction field differs from adding a feature inside the current mask.

## Surface and topology claims

| Cached model | Mean per-case relative whole-surface change | Underestimated cases | Disconnected square interfaces | Good-minus-bad Dice association |
|---|---:|---:|---:|---:|
| Unaugmented | -4.10% | 42/52 | 4/52 | +0.003725 |
| Augmented | -4.54% | 42/52 | 1/52 | +0.026238 |

The Dice column compares different cases; it is **not an intervention gain**. Surface is exact exposed-voxel-face area. Other estimators and checkpoints can differ.

The mean whole-mask PCA-axis/coronal-normal angle is **24.58° training / 24.20° validation**, with 5th–95th percentiles 13.33–37.01° / 13.46–34.73°. Thus approximately 24° is a cohort descriptor under this definition, not a fixed anatomical requirement.

## Spectral checks and limitations

Maximum eigenpair residual: 5.04e-10; maximum orthogonality error: 4.66e-15. Twelve combinatorial graph modes and four HKS times are retained per largest component, with per-case truncation bounds. These are volume-graph descriptors, not a registered anatomical frame or verified surface ShapeDNA. Full-graph λ₂=0 on disconnected cases; LCC λ₂ is separately labeled.

![New property checks](../../../experiments/graph_property_inventory_20260923/property_checks.png)

![Per-voxel descriptor example](../../../experiments/graph_property_inventory_20260923/descriptor_maps.png)

See [the complete 44-property assessment](README.md), [machine-readable summary](summary.json), and [coordinate targets](coordinate_cases.json).
