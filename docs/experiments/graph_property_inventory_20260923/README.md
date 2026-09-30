# Audit of the proposed graph-property list

This checks every item in the supplied ten-category inventory against the
existing experiments, new measurements, and the mathematical definition of the
property. **A descriptor being computable or predictable is not evidence that
adding it improves segmentation.** Several entries are reasonable candidates,
but some numerical claims mix graph definitions/model caches, and some proposed
anatomical interpretations do not follow from the descriptor.

The reference cohort remains the same 260 native MSD masks (208 training, 52
validation in fold 0). The new measurement pass also uses both existing 52-case
prediction caches, producing **1,092 region descriptor archives** across whole,
anterior, and posterior masks. No new network inference or training is involved.
See [new numerical results](RESULTS.md), [summary](summary.json), and
[provenance](manifest.json).

## Corrections to the quoted findings

| Statement in the list | Checked interpretation |
|---|---|
| Surface area is 5.5% too low; 42/52 underpredicted | The count **42/52** holds for both inspected models. With exact exposed-voxel-face area, mean per-case relative deficits are **4.10% unaugmented / 4.54% augmented**. Ratios of cohort mean areas give 4.38% / 4.77%. The 5.5% value needs its original checkpoint and surface estimator; it is not reproduced here. Lower area alone does not localize missing digitations. |
| Genus zero is false in 24/260 labels | **24/260 whole closed-voxel unions have β₁>0.** This is a tunnel count, not a directly verified boundary genus. A genus formula requires manifold boundary components; cavities, components, and nonmanifold contacts must be accounted for. Do not call this an established genus measurement. |
| Isoperimetric ratio has 0/52 range violations | Confirmed for both models against the **208-case training range**, using `S³/(36πV²)` and voxel-face S. This is weaker than matching the correct individual shape. |
| A/P interface: 260/260 connected; two prediction failures and +0.083 Dice gap | All reference **closed square interfaces** are connected. Prediction square-interface failures are **4/52 unaugmented, 1/52 augmented**. A 26-connected contact-voxel band instead gives **3/52 and 1/52**. The quoted two-case/+0.083 combination is not reproduced. A good-case versus bad-case Dice gap is an association, not improvement obtained by repairing topology. |
| Posterior connected in 260/260, three prediction failures | True for **26-connectivity**; failures are **5/52 unaugmented, 3/52 augmented**. Under the requested face graph, only **247/260 reference posterior masks** are connected, and prediction failures are **13/52 and 6/52**. |
| Anterior and union connected in 259/260 | Correct separately under **26-connectivity**. Under six-face connectivity, anterior is connected in **254/260**, union in **246/260**. |
| No cavities in 245/260 versus earlier 259/260 | **245/260 whole masks have β₂=0.** The 259/260 number describes 26-connected foreground (β₀=1), not cavity absence. These are different invariants, not discrepant estimates of the same property. |
| Coronal and exact in 202/260 | Confirmed: 202 masks equal their best-fit coronal A/P plane on reference foreground. The other 58 have mixed-slice deviations. |
| Interface normal 24° from long axis | Not a universal angle. A PCA-axis versus RAS-coronal-normal angle is now measured case by case; it is a geometric proxy, not an anatomical long-axis annotation. A nonplanar interface has multiple local normals. |
| Anterior fraction has 0/52 violations | Confirmed for both models using the training range, **0.4009–0.6639**. It cannot identify the correct cut merely by staying within these bounds. |
| Elongation has 0/52 violations | The earlier [elongation audit](../../../evaluation/elongation_frozen_audit_20260906/README.md) confirms this for its specific hard masks and descriptor, but the corresponding soft-probability measurement violates the bound in 52/52. A different PCA ratio is separately measured here. Never transfer the hard-mask result to an unspecified differentiable implementation or another checkpoint. |

## 1. Local node properties

| Property | Coverage and assessment |
|---|---|
| Degree / leafiness | **Already measured; now exported per voxel.** Degree one denotes a leaf in this graph. A rounded anatomical tip can contain no degree-one voxel, while a voxelization spur can create one. It is a local boundary/thinness descriptor, not a digitation detector. Input or auxiliary target is feasible; do not impose a universal leaf count. |
| Graph depth | **New exact per-voxel measurement on all regions.** Shortest graph distance to any node of degree <6; surface nodes have depth zero. This differs from Euclidean distance to background centers, which is one at a one-voxel boundary on this lattice. Both are saved to prevent conflation. |
| Local thickness | **New digital covering-ball estimate on all regions.** The usual local-thickness concept is the diameter of the largest inscribed ball *containing* the point, not necessarily centered there. We evaluate balls centered at foreground voxel centers, with radii from EDT to background centers. This is a documented lattice approximation, not exact continuous cube-union thickness. Thickness alone does not establish that the structure is sheet-like. |
| Eccentricity | **New exact per-voxel measurement.** Maximum finite graph distance within the voxel's connected component. Whole disconnected-graph eccentricity would be infinite; component IDs and disconnected status are retained explicitly. Extremal values suggest graph-peripheral locations, not necessarily anatomical ends. |
| Betweenness | **New approximate node map on whole graphs**, using 32 seeded source samples for every reference and prediction. It is not exact centrality, and regional A/P betweenness was not added. High values indicate shortest-path traffic under this graph definition; there is no guarantee of a medial spine or tail specificity. |
| HKS | **New truncated combinatorial-volume-graph HKS on the largest component of every region.** Twelve modes, four documented times, and a spectral tail bound are saved. Isometry invariance from [the original HKS work](https://geometry.stanford.edu/paper.php?id=sog-hks-09) does not mean arbitrary bending of a voxelized solid leaves this discrete graph invariant. Its geometry and adjacency can change. |
| First k eigenvectors | **New: up to 12 modes on each largest component**, with residual/orthogonality checks. The constant mode adds no positional information. Sign, repeated-eigenvalue basis rotations, and near-degenerate mode exchanges remain correspondence problems. Sign fixing for storage does not create an anatomical reference frame. |

HKS and positional fields on excluded small components are deliberately not
invented. Archives identify the largest-component nodes; an input pipeline
would need explicit validity channels/fallbacks for the other voxels.

## 2. Intrinsic coordinates

| Property | Coverage and assessment |
|---|---|
| Fiedler / harmonic long-axis coordinate | **New per-voxel reference and prediction fields**, plus earlier [harmonic intervention](../harmonic_ap_20260923/README.md). Fiedler sign is oriented toward increasing RAS Y. The new harmonic field uses low/high 10% Y-extent anchors, not GT A/P labels or an uncal landmark. Neither field inherently knows head/body/tail. The previous harmonic repair experiment already failed to improve these predictions. |
| Fraction of volume behind a voxel | **New Fiedler-rank volume coordinate.** It is an empirical rank transform of the chosen scalar field, with midranks for ties. It is not a unique anatomical definition of “behind.” Thresholding this coordinate imposes a volume fraction, so it does not provide independent evidence beyond ordering plus volume. |
| Distance to each end | **New exact graph-distance maps to a diameter pair**, oriented by Y. These are geometrically selected endpoints. Anatomically annotated head/tail-tip distances remain unavailable; diameter endpoints must not be relabeled as verified landmarks. |
| Angle around axis | **Not computed as an anatomical quantity.** Requires a centerline/frame, a medial/lateral reference, hemisphere handling, and rules at singularities or sign changes. A bare voxel graph is invariant to relabeling and cannot distinguish medial from lateral. RAS/world coordinates can help only when their anatomical registration is established. |
| Centerline distance and arc length | **A shortest-diameter-path proxy is newly saved**, including distance and nearest-path arc index. It is not a medial centerline. A valid diameter path may hug the boundary; nearest-path assignment can jump. A true centerline extraction and stability check are still needed before defining this auxiliary target. |

The new reference-field probe fits the best A/P threshold **for scoring only**,
then separately tests a threshold fixed to the training median. The former is
an oracle fit; the latter uses reference foreground and is still a geometry
diagnostic, not deployable validation of an image-to-segmentation system.

## 3. Cross-section profiles

| Property | Coverage and assessment |
|---|---|
| Area per slab | **Previously partly covered by coronal candidate geometry; now explicitly exported for every graph.** Current slabs are RAS-coronal, not perpendicular to an inferred curved axis. Intrinsic slabs need a binning and physical-area definition. |
| Medial width | **Not established.** Total left/right and superior/inferior extents per slab are now saved as geometric proxies. Total LR width is not medial protrusion; a hemispheric/anatomical anchor is needed. The claim that the medial-width change identifies the uncal landmark needs an actual localization test. |
| Components per section | **New complete 4-connected coronal profiles.** More than one component can arise from a curved unbranched tube, a cavity, segmentation noise, or a tangential slice. It does not by itself demonstrate an anatomical branch or digitation. |
| Perimeter / area | **Existing related descriptors; now exact exposed-pixel-edge perimeter/area profiles for all cases.** The ratio is scale-dependent and sensitive to sampling. Matching a target profile is defensible as a hypothesis; universally minimizing it promotes smoothing and may erase the details of interest. |
| Centerline curvature / torsion | **Not computed as anatomical descriptors.** The raw shortest path is a staircase and lacks a unique smooth Frenet frame. Smoothing scale, endpoints, resampling, and a real centerline must be defined first. Torsion is particularly unstable where curvature approaches zero. |

Earlier [surface/MRI descriptor tests](../graph_partition_20260923/README.md)
already evaluated related geometry as cut-localization input and failed their
incremental-information/preservation gates. New descriptors still need a
matched incremental test; their names alone do not establish new signal.

## 4. Skeleton and Reeb structure

| Property | Coverage and assessment |
|---|---|
| Skeleton leaves | **Not measured as anatomical landmarks.** Requires a specific 3D skeleton algorithm and physical pruning rule. Small surface digitations need not produce stable skeleton branches. [clDice](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation_CVPR_2021_paper.pdf) measures skeleton overlap/connectivity; it is not a leaf-count loss or a guarantee of preserving hippocampal digitations. |
| Branch count/length | **Not computed on a validated anatomical skeleton.** Raw counts can be dominated by discretization and pruning. No universal target count is supported by the current masks. |
| Diameter | **Upgraded from a double-sweep lower bound to exact diameter within each component.** It is the longest *shortest* path distance, not the longest simple path. Its path is not necessarily medial and its grid length is not anatomical centerline length. A numerical regression target is possible only with these distinctions intact. |
| Reeb graph | **New coronal slice-component adjacency graph**, clearly named a discrete proxy. Nodes are 4-connected slice components; edges record face overlap across consecutive slices. We report branches/cycles for whole/A/P separately. This is not the continuous Reeb graph of a validated intrinsic field. A “posterior chain, anterior branches” rule is not assumed. |

## 5. Surface properties

| Property | Coverage and assessment |
|---|---|
| Surface area | **New exact voxel-face recomputation and paired prediction comparison.** A scalar area loss is highly non-identifying: roughness, extra islands, or an incorrect boundary can restore area without recovering missing anatomy. Area should be an endpoint alongside spatial boundary errors, not a sufficient training target. |
| Curvature / shape index / convex-hull depth | **Partial existing coverage:** the graph-interface probe used smoothed signed-distance normals/curvature at two scales. It did not establish a validated surface graph, principal-curvature shape index, or a new convex-hull-depth field in this audit. Those require a surface estimator and smoothing convention. A concavity cue may be useful but is not automatically an uncal-apex annotation. |
| Genus | **Not directly established as a manifold-surface invariant.** Existing voxel-union Betti numbers must not be renamed genus without checking the boundary's manifoldness and components. |
| Isoperimetric ratio | **Recomputed.** It is insensitive to location and a loose population bound accepts both inspected models. Equal ratio does not imply equal shape. |

## 6. Topological invariants

These were already comprehensively measured in the
[voxel graph audit](../voxel_graph_anatomy_20260923/README.md). The main correction
is to specify **six-face graph**, **26-connected closed voxel union**, or
**actual square interface**, rather than mixing their component counts.

Connected interfaces and absence of internal label bubbles are plausible soft
priors. “All classes must always be one component with no cavities” contradicts
some reference labels. Bad topology and poor Dice can coexist without the former
causing the latter; a measured between-case Dice gap is not a repair effect.

## 7. A/P partition properties

| Property | Coverage and assessment |
|---|---|
| Single crossing on any tail-to-head path | **False as stated.** In a solid prism with a perfect planar P/A split, a simple path can cross P→A→P→A without revisiting any node. An explicit test demonstrates this. Single crossing along prescribed rays, or along paths monotone in a chosen scalar field, is a different property. The earlier ray audit covers the first; a threshold field enforces the second by construction. |
| Interface orientation | **Previously measured via face-axis counts and plane disagreement; PCA angle newly quantified.** Coronal orientation matches the annotation rule more closely than orthogonality to a generic intrinsic long axis. Use soft tolerance for the 58 nonexact cases. |
| Interface area / section area | **Partial existing coverage:** actual cut area and candidate coronal cross-face counts already exist. For an exact planar cut at that section these areas are identical by construction, so the ratio supplies no new localization information. A different intrinsic cross-section definition would require another measurement. |
| Cut fraction along long axis | **New per-case oracle Fiedler/harmonic thresholds and a training-only median-threshold probe.** Learnable as a target/prior after fixing coordinate orientation and normalization. Patient-specific anatomical localization remains unresolved. |
| Anterior volume fraction | **Already measured and rechecked.** A population range is too loose to discriminate current errors. Learning a patient-specific fraction from MRI is a different, unvalidated task. |

## 8. Spectral descriptors

| Property | Coverage and assessment |
|---|---|
| Spectrum / ShapeDNA | **New combinatorial graph spectra**, raw and multiplied by `N_lcc^(2/3)` as a volume-scale proxy. They are not identical to a discretized surface Laplace–Beltrami ShapeDNA. The [original method](https://graphics.stanford.edu/courses/cs468-08-fall/pdf/reuter.pdf) concerns a specified manifold operator. Spectra are not unique shape identifiers: isospectral shapes and symmetric ambiguities exist. No spectral training loss was tested. |
| Algebraic connectivity λ₂ | **New exact low eigenvalue solve with numerical checks.** Full-graph λ₂ is zero whenever the graph is disconnected; λ₂ of its largest component is separately saved. It depends on size, normalization, degree, and bottlenecks, so it is not a pure anatomical thinness parameter. |
| Elongation | **Previously studied; a new explicit PCA standard-deviation ratio is also measured.** The existing frozen-logit study found that bad changes could satisfy the original scalar rule. An apparently reasonable population range is insufficient evidence for a useful loss. |

## 9. Edge/face properties

| Property | Coverage and assessment |
|---|---|
| No cut / outer cut / A/P cut | **Labels are derivable; an edge head has not been trained.** The original foreground-only graph contains within-foreground and A/P edges but **no foreground/background edges**. Outer-cut targets need a graph over the full image lattice or an explicit exterior representation. Otherwise missing foreground cannot be added by graph inference. |
| Face orientation | **Already measured.** Face normals take the three lattice-axis directions; A/P normal sign additionally depends on endpoint label ordering. Orientation alone cannot locate the cut. |
| MALIS / edge betweenness | **Different concepts, not interchangeable.** Node betweenness is now sampled; edge betweenness is not computed here. [MALIS](https://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation) assigns learning responsibility through maximin paths in a predicted affinity graph. Uniform binary adjacency supplies many ties and no learned confidence ordering. MALIS weights cannot be meaningfully claimed without specifying that model/affinity field. |
| Alignment with long-axis field | **Computable from the newly saved whole-graph scalar fields and original edges.** `u_j-u_i` is a field difference; divide by edge length for a directional derivative. A normalized alignment needs a gradient/frame convention. It remains a shape-derived cue, not evidence of the correct anatomical separator. |
| MRI contrast | **Existing A/P tests performed and negative for the tested natural-error intervention.** That result does not establish failure or success for exterior-boundary correction. A separate outer-edge task, foreground/background edges, and held-out image evidence are needed. |

## 10. Population and atlas level

| Property | Coverage and assessment |
|---|---|
| Template correspondence | **Not computed: no registered template/correspondence protocol was supplied.** A graph or its eigenvectors does not automatically provide cross-subject anatomical correspondence. Atlas registration, hemisphere handling, topology exceptions, and inference-time inputs must be defined first. |
| Typical profile / cut-position distribution | **Profiles and coordinate-target distributions are now available.** Fit priors on the 208 training cases only; keep the 52 validation cases for exploratory checking. A prior computed across all 260 and then scored on those same 52 leaks validation information. The repeatedly inspected fold is not independent confirmation. |

## Practical rules, corrected

- `[input]` features must be available from MRI, a genuine first-pass prediction,
  or an independently available atlas at inference. During training, detached
  **out-of-fold** predictions are preferable for a separately trained correction
  stage. Detached in-sample outputs avoid backpropagation but do not remove
  overfitting; arbitrary GT noise need not resemble model errors. Missing-voxel
  locations require a defined extension outside predicted foreground.
- `[target]` avoids needing a graph solve at inference if the auxiliary head is
  discarded. But it does not necessarily mean one channel: eigenvectors,
  endpoint distances, orientations, and multi-scale HKS are multichannel, with
  ambiguity and normalization issues. Auxiliary prediction can encourage useful
  features; it does not *force* correct anatomy or prove a segmentation benefit.
- `[loss]` needs an explicit differentiable surrogate or a justified estimator.
  Hard argmax graphs, eigensolvers, skeletonization, and discrete component counts
  do not become end-to-end losses merely by comparing their scalar outputs.
- For missing surface detail, test spatial boundary/depth/thickness supervision
  before assuming that skeleton leaves or global area identify the errors.
  For A/P swaps, compare oriented-coordinate/profile cues against the existing
  image/logit/plane readout, preserving already correct cuts. Interface
  connectivity addresses incoherence, not a coherent but misplaced boundary.

The evidence supports treating the list as a **candidate descriptor inventory**,
with depth/thickness targets and oriented-coordinate/profile inputs worth
controlled tests. It does not support the proposed strongest-candidate ranking
as an established result. No new loss, edge head, skeleton rule, or atlas was
validated by this measurement audit.

The clearest current opportunity is **degree-aware boundary supervision**:
among reference foreground voxels, the two models miss roughly 76–80% of
degree-0–1 voxels and 40–41% of degree-2–3 voxels, versus less than 1% of
degree-six voxels. Degree 0–1 is a small group (286 voxels across 52 cases);
degree 2–3 contains 21,061 voxels. Degree also distinguishes exposure *within*
the boundary layer, whereas graph depth alone is zero throughout that layer.
Compare any such weighting/auxiliary target against ordinary boundary weighting
and focal supervision with matched gradient scale, and monitor added false
positives and preservation of correct boundaries. These rates establish an
error concentration, not a loss benefit or anatomical validity of every tip.
The descriptors re-express information already present in the masks; any benefit
would come from the form of supervision, not a newly observed anatomical landmark.

## Reproduction and archive schema

```bash
rtk proxy .venv/bin/python evaluation/audit_graph_property_inventory.py --workers 2
rtk proxy .venv/bin/python evaluation/summarize_graph_property_inventory.py
rtk proxy .venv/bin/python -m pytest evaluation/test_graph_property_inventory.py -q
```

Implementation: [measurement runner](../../../evaluation/audit_graph_property_inventory.py),
[paired summary](../../../evaluation/summarize_graph_property_inventory.py),
[tests](../../../evaluation/test_graph_property_inventory.py).

Per-case scalar measurements and complete coronal profiles are in
`case_metrics/<source>/<case>.json`. Per-voxel maps are in the ignored local
`experiments/graph_property_inventory_20260923/maps/<source>/<case>/<region>.npz`.
Sources are `reference`, `baseline_seed0`, and `augmentation_seed0`.
`coordinates_ijk` index the native reference array or the padded prediction
array respectively; cross-source spatial matching must account for that padding.
`lcc_node_ids` identifies the subset supporting spectral/intrinsic maps.
Per-component eccentricity is finite; distances to the selected diameter ends
are infinite on other components. All distances in this pass use the verified
1-mm isotropic lattice. Graph-hop paths and Euclidean distances are labeled
separately. No array here is a verified medial/lateral or uncal-apex coordinate.

Verification completed: all **1,092 map archives** were reopened and checked for
node counts, finite local fields, depth/degree consistency, covering-thickness
bounds, spectral subset dimensions, and diameter-path length. All 364 source
hashes agree with the previous topology audit. **17 targeted tests passed**
(8 new property tests plus 9 prior graph tests), including exact heat-kernel
comparison, truncated-HKS error bounds, disconnected-distance semantics,
covering-ball thickness, threshold ties, and a counterexample to the proposed
single-crossing-on-every-path rule.
