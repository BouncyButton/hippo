# Voxel graphs of the whole, anterior, and posterior hippocampus

23 September 2026. **Completed: 780 explicit graphs from all 260 MSD reference
masks.** The primary split is the project's fold 0: 208 training and 52 validation
cases. Membership and separate summaries for all five existing folds are also
retained; the same hippocampus graph is reused across folds. These are case
counts, not a verified count of independent participants.

The most useful findings are a very large robust graph core, predominantly
connected A/P interfaces, and consistently greater anterior compactness.
They support **soft, scale-aware structural constraints**, but none supplies a
reliable A/P boundary locator. Reference exceptions rule out hard single-component,
zero-hole, disk-interface, and no-bridge requirements.

## Graphs and definitions

Each labeled voxel is a node. Two nodes share exactly one undirected, unweighted
edge when their voxel cubes share a face: six-neighbor adjacency, with no diagonal
edges. A graph edge represents a potential cut through that shared face. The
whole graph contains both labels; anterior and posterior are its induced
subgraphs. The A/P cut is the subset of whole-graph edges whose endpoint labels
differ. There are no exterior/background nodes or invented connections across gaps.

Graphs retain the original native 1-mm RAS lattice, every voxel, all islands,
and the NIfTI affine. There is no resampling, largest-component filtering, or
label repair. All 780 archives were reopened and independently checked against
the original masks, including every node, every edge, face axis, label, and affine.

Three distinct objects are measured:

1. **Voxel graph:** connected components, island mass, degree histogram, bridges,
   articulation voxels, largest biconnected block, cycle rank, and a physical
   shortest-path diameter lower bound. The double sweep is not an exact diameter;
   its path/chord ratio also includes the grid's Manhattan-distance effect.
2. **Closed voxel union:** all cubes and their faces, edges, and vertices. Its
   Betti numbers count components (β₀), tunnels (β₁), and enclosed cavities (β₂).
   Corner contacts connect this complex, so it uses 26-connected foreground and
   6-connected background. It is deliberately separate from the six-face graph.
3. **A/P interface:** the square complex of actual shared A/P faces. Exact
   boundary-matrix ranks over GF(2) give its homology. Edge incidence and every
   vertex link test manifoldness; boundary loops are checked before calling it
   a disk. A connected contact-voxel band alone cannot establish this property.

For the graph, β₁ = E − V + C counts lattice cycles. Even a solid 2×2×2 voxel
block has five graph cycles and **zero anatomical tunnels**. For the closed
voxel union, χ = n₀ − n₁ + n₂ − n₃ and β₁ = β₀ + β₂ − χ. Homology (1,0,0)
does not by itself prove that an object is a manifold ball.
The connectivity distinction is also documented by
[scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.euler_number).

![Representative whole, anterior, and posterior graphs](../../../experiments/voxel_graph_anatomy_20260923/example_graphs.png)

Orange nodes are anterior; green nodes are posterior; yellow edges cross the
A/P cut; magenta nodes lie outside a graph's largest connected component.
Coordinates in this figure are millimeters from the stored voxel-index origin,
not absolute scanner coordinates. All edges are drawn with low opacity.

## Connectivity and robust core

| Region | Disconnected graphs, training | Disconnected graphs, validation | Total island voxels, training / validation | Median graph cycles, training / validation |
|---|---:|---:|---:|---:|
| Whole | 13/208 | 1/52 | 19 / 1 | 5,278 / 5,292 |
| Anterior | 6/208 | 0/52 | 17 / 0 | 2,755 / 2,782 |
| Posterior | 12/208 | 1/52 | 27 / 1 | 2,415 / 2,432 |

These disconnected pieces are tiny: at most four whole-hippocampus voxels, or
eight voxels in either separate region, outside the largest component.
`hippocampus_156` is the only reference with two whole-mask components even under
26-connectivity; the other whole-graph exceptions touch via corners or edges.
`hippocampus_292` illustrates how a connected whole graph can yield disconnected
anterior and posterior induced subgraphs.

Bridge edges occur in **186/208 training and 47/52 validation whole graphs**.
Their median counts are four and five. Thus a rule banning all bridges or
articulation voxels would penalize normal reference boundaries almost everywhere.
Nevertheless, the largest biconnected block contains a median **99.87% / 99.85%**
of whole-mask voxels, with a minimum of 99.22% across all cases. Even the smallest
regional block contains more than 98.67% of its region. The bulk has redundant
paths; the fragile connections mostly involve a small amount of peripheral mass.

**Exploitable distinction:** penalize substantial detached mass or a severed
large core, while tolerating tiny peripheral contacts. Counting every component
or every bridge equally ignores this distinction. These observations do not
justify deleting graph bridges as a post-processing operation.

## Anatomical tunnels and cavities

| Region | Homology (1,0,0), training / validation | Cases with tunnels, training / validation | Cases with cavities, training / validation |
|---|---:|---:|---:|
| Whole | 178/208 / 43/52 | 19 / 5 | 11 / 4 |
| Anterior | 201/208 / 50/52 | 4 / 0 | 2 / 2 |
| Posterior | 189/208 / 47/52 | 12 / 5 | 7 / 0 |

One training whole mask has both a tunnel and a cavity. One validation whole
and posterior mask has two tunnels; other tunnel-bearing masks have one.
Whole-mask training cavities contain one voxel in eight cases and two voxels in
three cases. Validation cavities contain 1, 1, 4, and 17 voxels. These are
properties of the released annotations; masks alone cannot establish whether
they reflect anatomy, annotation choices, or voxelization artifacts.

A one-face dilation changes whole-mask homology to (1,0,0) in 201/208 training
and 50/52 validation cases. This demonstrates sensitivity to small morphological
changes, **not a persistence computation or a proposed repair**. Dilation can
also close an opening and create a new cavity, so its endpoint counts must not
be read as tracked survival of individual holes.

Raw graph cycle rank is particularly uninformative as a tunnel constraint:
within each region its correlation with node count is 0.990–0.998 in training.
A tree or acyclic-graph constraint would be fundamentally inappropriate.

## The actual A/P interface

All 208 training interfaces have homology (1,0,0), but only **199/208** are
manifold disks. Nine have nonmanifold contacts. Validation has **50/52 disks**;
the other two have a hole:

- `hippocampus_164`: a manifold annulus, β=(1,1,0), with two boundary loops.
- `hippocampus_338`: β=(1,1,0) with an additional nonmanifold corner contact.

Every reference interface is connected in the closed square complex. No
reference has a closed interface component enclosing an A/P bubble (β₂=0).
This supports connected-interface and no-bubble soft priors more directly than
an exception-free disk requirement. Point-touching faces still count as connected
here; manifold checks expose why connectedness alone is insufficient.

![Exact square interfaces, including exceptions](../../../experiments/voxel_graph_anatomy_20260923/interface_exceptions.png)

The median A/P interface contains 78 shared faces in training and 76 in
validation. Training anterior volume fractions span **40.1–66.4%**, so an
equal-volume partition is not an annotation rule. Forty-seven training and
eleven validation masks also depart from their closest coronal plane.

To test a geometric bottleneck explanation, minimize
`Ncut(A,P) = |cut| × (1/vol_G(A) + 1/vol_G(P))`, where graph volume is the sum of
whole-graph degrees in each part. Search coronal cuts leaving at least 10% of
voxels on each side, with no fitted parameter. This diagnostic is restricted
to coronal planes; it is not an unrestricted global normalized-cut solver.

The minimum agrees with the reference best-fit plane in only **13/208 training
and 2/52 validation cases**. Median absolute displacement is **4 mm in both
sets**; mean displacement is 4.22 and 4.44 mm. Even on the exact reference
foreground, a low-cost graph separator does not identify the anatomical cut.

## A consistent anterior/posterior graph asymmetry

Define `q(R) = number of degree-six nodes in G_R / number of nodes in G_R`.
This is the fraction of region voxels having all six neighbors in the same
region. On this 1-mm lattice it equals the retained fraction after removing
the one-face boundary layer. It measures compactness/thickness, rather than
a topological invariant under arbitrary shape deformation.

| Reference set | Median q(anterior) | Median q(posterior) | Cases with q(anterior) > q(posterior) |
|---|---:|---:|---:|
| Training | 59.28% | 52.55% | **208/208** |
| Validation | 59.92% | 51.22% | **52/52** |

The analogous Euclidean 2-mm erosion retains median 28.35% versus 19.70%
in training, and 29.00% versus 18.08% in validation, with the same ordering in
every case. The eroded posterior core is also occasionally much more fragmented:
its largest component falls below 95% of remaining mass in 9 training and 4
validation cases, versus zero anterior cases. Thus applying the same strong
thickness/core-connectivity requirement to A and P would be inappropriate.

This asymmetry is a plausible **weak semantic shape prior**: anterior should
usually be more compact than posterior. It could discourage reversed labels
or grossly implausible region shapes. However, a falsification probe shows its
limited localization power. Replacing labels by a plane six millimeters posterior
to the reference best-fit plane still satisfies the inequality in **all 260
cases**. A six-millimeter anterior shift also passes in 48/208 training and
16/52 validation cases. These are synthetic partitions on reference support;
the zero-shift plane does not reproduce every mixed-slice annotation exactly.
Both existing model arms already satisfy the inequality in all 52 validation
cases. An inequality penalty alone therefore has no hard-mask violations to
correct in these outputs. A fixed positive margin was not calibrated or tested.

![Compactness and displaced-cut falsification](../../../experiments/voxel_graph_anatomy_20260923/compactness_probe.png)

## Are the current predictions missing these properties?

The two existing seed-0 CPU prediction caches were audited on the same 52
validation cases. Every cached reference was checked against the symmetrically
padded native label. No new inference or training was required. These arms are
paired observations of the same cases, not independent cohorts.

| Validation diagnostic | Unaugmented | Augmented |
|---|---:|---:|
| Total A/P swap voxels | 4,292 | 3,702 |
| Cases matching reference cubical Betti triples for all three regions | 34/52 | 36/52 |
| Cases matching reference graph component counts for all three regions | 36/52 | 43/52 |
| Cases matching reference interface Betti numbers | 45/52 | 47/52 |
| Cases satisfying all three matching conditions | 31/52 | 33/52 |
| A/P swaps within those fully matching cases | **1,970 (45.9%)** | **2,081 (56.2%)** |
| Disk interfaces | 46/52 | 48/52 |

These are matches to each case's reference, not just agreement with an assumed
universal topology. They do not imply matching manifoldness, spatial location,
or all graph properties. Nor do errors in mismatching cases establish how much
a topology loss could recover: a topological defect and a misplaced cut may
coexist without sharing a cause.

The new homology measurements identify defects missed by simple connectivity
checks, but substantial A/P error survives even when all measured topological
counts agree with the reference. Topology can constrain coherence; it does not
determine the correct boundary location.

![Topology summary and existing-output diagnostic](../../../experiments/voxel_graph_anatomy_20260923/topology_summary.png)

## What I would exploit

1. **Mass- and scale-aware connectivity for whole/A/P probabilities.** Favor a
   dominant connected core and penalize large detached regions, with tolerance
   for small peripheral voxelization effects. A differentiable implementation
   could use H₀ persistence or a soft connectivity surrogate; the present hard
   component measurements are diagnostics, not differentiable losses.
2. **Joint A/P partition structure.** Couple the whole union and both regions,
   and softly discourage disconnected interfaces and internal label bubbles.
   Checking only the whole foreground misses A/P defects. Match trusted
   reference topology or use a persistence tolerance rather than forcing every
   case to be a disk or to have Betti numbers (1,0,0).
3. **Anterior/posterior compactness ordering as an auxiliary semantic prior.**
   For example, a soft approximation to `max(0, q(P) - q(A))`, with a separately
   checked gradient and training-only calibration. This is an untested proposal.
   It is already satisfied by the inspected outputs, and the shift probe shows
   that it cannot serve as an A/P cut locator on its own.

Persistent-homology losses provide an established route to soft topology
supervision ([Clough et al.](https://arxiv.org/abs/1910.01877)); spatially matched
3D topological features offer another implementation direction
([Stucki et al.](https://arxiv.org/abs/2407.04683)). These papers establish methods,
not efficacy for these hippocampi. A voxel graph with only foreground nodes
cannot optimize the exterior boundary by itself: a training loss must operate
on the full prediction lattice or another explicitly defined support that
permits adding missing voxels.

The smallest useful next experiment would compare a single soft structural
term against the current supervised objective and existing plane constraint,
using identical seeds/splits and reporting topology defects, A/P cut error,
foreground Dice, and harm to initially correct predictions separately. Freeze
the term and its weight using training data. This repeatedly inspected
validation fold is exploratory evidence, so improvement would require another
untouched evaluation. No training improvement is claimed from this audit.

## Artifacts and reproduction

- [Per-case measurements](cases.json), [fold summaries](summary.json),
  [input/source manifest](manifest.json), [graph verification and hashes](verification.json).
- [Morphological sensitivity](sensitivity.json), [erosion summaries](erosion_summary.json),
  [compactness shift probe](compactness_probe.json), [prediction measurements](prediction_cases.json).
- Graphs: local `experiments/voxel_graph_anatomy_20260923/graphs/<case>/whole.npz`,
  `anterior.npz`, `posterior.npz` (780 files, approximately 16 MB). These generated
  data-derived files and the figures remain ignored by Git.
- [Graph/audit implementation](../../../evaluation/voxel_graph_anatomy.py),
  [independent verification](../../../evaluation/verify_voxel_graph_anatomy.py),
  [figures](../../../evaluation/plot_voxel_graph_anatomy.py),
  [tests](../../../evaluation/test_voxel_graph_anatomy.py).

Each NPZ stores `coordinates_ijk` (N×3), `edges` (M×2, zero-based local node IDs,
each undirected edge once), `edge_axis`, physical center-to-center lengths,
shared-face areas, `node_labels`, `component_ids`, native `affine`, image `shape`,
`spacing`, and JSON provenance with source hash and all-fold membership.
The graph topology is unweighted; lengths and areas are additional attributes.
World voxel centers are `nibabel.affines.apply_affine(affine, coordinates_ijk)`.

```python
import numpy as np
from scipy.sparse import coo_matrix

with np.load("experiments/voxel_graph_anatomy_20260923/graphs/hippocampus_001/whole.npz",
             allow_pickle=False) as graph:
    xyz = graph["coordinates_ijk"]
    edges = graph["edges"]
    labels = graph["node_labels"]
    ap_cut = edges[labels[edges[:, 0]] != labels[edges[:, 1]]]
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(xyz), len(xyz)))
    adjacency = (adjacency + adjacency.T).tocsr()
```

From the repository root:

```bash
rtk proxy .venv/bin/python evaluation/voxel_graph_anatomy.py \
  --predictions experiments/uncal_fold_early_stopping_20260921/voxel_audit
rtk proxy .venv/bin/python evaluation/verify_voxel_graph_anatomy.py
rtk proxy env MPLCONFIGDIR=/tmp/hippo-graph-anatomy-mpl \
  .venv/bin/python evaluation/plot_voxel_graph_anatomy.py
rtk proxy .venv/bin/python -m pytest \
  evaluation/test_voxel_graph_anatomy.py \
  evaluation/test_audit_ap_partition_topology.py \
  evaluation/test_graph_partition_probe.py -q
```

**Verification:** 34 targeted tests passed, including 9 new graph/topology tests.
The new tests cover lattice cycles versus solid topology, diagonal contacts,
bridges, anisotropic path lengths, induced-region serialization, empty masks,
displaced cuts with unchanged topology, and disk/annulus/sphere/pinch surfaces.
Thirty-five random small voxel complexes agree with an independent explicit
boundary-matrix homology computation. All 780 saved graphs pass exhaustive
edge/node round-trip verification against the native labels.
