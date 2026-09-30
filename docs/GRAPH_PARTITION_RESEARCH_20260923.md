# Hippocampal A/P separation as graph partitioning

Follow-up: the [completed feasibility experiments](experiments/graph_partition_20260923/README.md)
tested all 208 training cases, 624 synthetic variants, and 832 uniform graph cuts.
New interface descriptors recover deliberately shifted cuts but fail to add natural
localization accuracy and disturb correct cuts. Both prespecified advancement gates
failed; full-network training was not launched. The assessment below records the
research formulation that preceded those experiments.

Research assessment, 23 September 2026. Scope: mathematical formulation,
correction of existing predictions, and training constraints. This note reviews
existing evidence and proposes experiments; no new training or segmentation
experiment was run for this assessment. Literature search is targeted, not
systematic. The approaches below are research candidates, not approved results.

## Assessment

The defensible hypothesis is that **surface geometry and image context can help
identify the semantic A/P interface, while a volume graph assembles that evidence
into a coherent partition**. A simplicial representation makes the interface
explicit, but does not supply its anatomical location.

The immediate priority is an information test: do boundary/edge features distinguish
the correct partition from a coherent but displaced partition better than existing
probabilities, geometry, and coronal-cut controls? Only a positive result justifies
a learned graph correction model. A separate training-only structured objective
can test whether comparing whole partitions improves learning even without adding
new anatomical information.

## What the target actually represents

The MSD data descriptor specifies 1-mm T1 MRI and defines the last head slice using
the uncal apex; posterior combines body and tail. Therefore the target is a
landmark-defined annotation, not necessarily a tissue boundary visible as a local
intensity discontinuity. [Simpson et al., 2019](https://arxiv.org/html/1902.09063v1#S2.SS1.SSS4)

In the local 260-label audit, 202 cases admit an exact coronal plane and 242 have
at most 1% wrong-side class voxels at their best plane. An intrinsic curved cut may
be anatomically attractive but still fit this annotation convention less well.
Retain all released labels when scoring; do not replace mixed slices with a plane.
See [existing geometry results](../semantic_constraints/cst_teacher/multiview/EXPERIMENT_20260922.md).

## Existing evidence that changes the recommendation

These are prior repository results, not new measurements from this assessment.
The two model arms below share 52 development cases and are not independent cohorts.

| Finding | Implication |
|---|---|
| 90.7% / 92.9% of A/P swaps occur in cases passing the four 26-connectivity checks | Basic connectivity is unlikely to identify most current errors. |
| Only one / zero foreground rays violate the weak one-transition rule | A one-switch rule has almost no hard-label repair opportunity here. |
| Prediction-fitted planes improve A/P Dice by only 0.044 / 0.063 percentage points | Coherence alone has limited demonstrated benefit. |
| Reference-fitted planes improve Dice by 2.294 / 1.953 points on unchanged predicted foreground | Correct localization matters, but this oracle is unavailable at inference and is not a strict Dice upper bound. |
| Raw-MRI harmonic fields reduce Dice by 0.500 / 1.037 points | Do not repeat intensity-driven diffusion as the default next experiment. |
| The tested MRI fields also lose to uniform graph weights | The tested affinities do not encode the needed semantic boundary. |

Sources: [partition topology audit](experiments/ap_partition_topology_20260923/README.md)
and [harmonic pilot](experiments/harmonic_ap_20260923/README.md).

The inspected incremental-context summary contains an unaugmented-model result:
adding MRI context to a logits/geometry cut ranker changes cut MAE by -0.0192 slices
and Dice by +0.0117 percentage points, with both paired case-bootstrap intervals
crossing zero. This does not establish useful incremental context. It also does
not exhaust spatial surface representations or nonlinear context models.
See [summary](experiments/context_increment_20260923/development_summary.json)
and [method](experiments/context_increment_20260923/METHODS.md).

The same development fold has been repeatedly studied. Participant linkage between
hippocampal crops is unverified; case IDs are not proven independent participants.
Confirmatory claims need a genuinely untouched, appropriately grouped evaluation.

## 1. Mathematical representation

Let K be a tetrahedral complex filling the hippocampal foreground. Its dual graph
G=(V,E) has one node per tetrahedron and one edge per shared triangular face.
Keep the complex as well as the graph if surface topology will be measured: the
adjacency graph alone does not encode all face/edge incidence information.

An initial voxel implementation instead uses a cubical complex and face adjacency.
This is a different discretization of the same partitioning problem. It avoids
meshing and interpolation as confounders. Six- and 26-neighbor connectivity must
not be silently interchanged: local labels already exhibit differences under them.

For a binary labeling z, the interface is the union of shared faces whose incident
cells have different labels:

    Sigma(z) = union {F_ij : (i,j) in E and z_i != z_j}.

In a suitable manifold domain, the intended interface is a properly embedded
surface whose boundary lies on the external hippocampal boundary. A connected
disk-like transverse interface is a candidate prior, not something guaranteed
by an ordinary cut. Internal bubbles, disconnected pieces, and nonmanifold
intersections remain possible in unconstrained discrete partitions.

### What “outer contour as leaves” means

For each external triangle, add a boundary node connected only to its owning
tetrahedron. This does produce leaves. But a free leaf contributes no evidence.
For a nonnegative Potts edge, minimizing over its unknown label gives

    min_b w * 1[b != z_i] = 0.

For harmonic energy, the free leaf simply takes its neighbour's value. Thus
adding unlabeled leaves changes neither interior optimum in these formulations.

A leaf becomes useful when it carries measured or predicted evidence D_b:

    induced unary on cell i = min_b {D_b(b) + w * 1[b != z_i]}.

Independent leaves therefore reduce to extra unary costs; they are not themselves
a new graph mechanism. If neighbouring surface nodes also exchange information,
they form a surface graph and cease to be leaves. A useful richer design has a
surface graph for fold/context features, a volume graph for partitioning, and
explicit surface-to-volume correspondences.

Do not pin the entire exterior to one A/P class. The exterior contains both
regions. Anterior/posterior tips can orient a problem but do not locate the apex.

### What simplicial topology can and cannot supply

With mod-2 chains, the boundary of the selected anterior volume contains both
external faces and the internal interface. Because boundary-of-boundary is zero,
the interface has no chain boundary in the interior; its boundary is on the
external surface. This is a relative-cycle consistency statement.

It does not guarantee a single disk or the correct cut location. If the interface
is already generated from a volume labeling, merely penalizing the identity
boundary-of-boundary=0 is vacuous. Such consistency is useful when independently
predicting faces, where incidence violations can actually occur.

A simple identifiability counterexample is a uniform tube. Every transverse cut
away from the ends separates the same anchors into two connected regions with a
disk interface of equal area. Topology and minimum area cannot distinguish their
locations. Nonuniform shape can break that tie, but its narrowest cross-section
need not coincide with the annotation's landmark.

## 2. Correction through a graph cut

Within fixed predicted foreground, define q_i=P(A | foreground) from the network's
A/P logits. For a cell of physical volume m_i, use stable conditional log costs:

    D_i(A) = -m_i log q_i
    D_i(P) = -m_i log(1-q_i).

For shared-face area a_ij and nonnegative affinity w_ij, minimize

    E(z) = sum_i D_i(z_i)
           + lambda sum_(i,j) a_ij w_ij 1[z_i != z_j].

An affinity is the cost of separating two cells, not the probability that they
should be separated. Strong candidate boundary evidence should lower this cost.
Areas, volumes, and lambda must use consistent units and scaling across cases.

For fixed binary unaries and nonnegative Potts terms, an s-t minimum cut finds
the global energy minimum. This guarantee does not extend automatically to added
volume-balance terms, arbitrary connectivity constraints, or topology penalties.
It is an optimization guarantee, not an anatomical one.
[Boykov and Jolly, 2001](https://www.eng.utah.edu/~cs7640/readings/boykov_iccv01.pdf)

The minimal baseline uses uniform affinities and original logits. Raw-intensity
affinities are a historical negative control here. The stronger hypothesis learns
bounded positive affinities from context near both endpoints and, potentially,
surface-derived features. Retain unary evidence rather than replacing the model
by interpolation between distant endpoints.

This initial correction preserves foreground exactly, so it cannot fix foreground
false positives or negatives. Evaluate those separately. On disconnected components
with inadequate evidence, preserve the original labeling and report coverage.
Automatic seeds, confidence rules, and any abstention threshold must be fixed using
training data. A minimum-energy gap is not a calibrated confidence by itself.

## 3. Three research approaches

| Approach | Question it answers | Effort / risk | Reuse and limitations |
|---|---|---|---|
| A. Unary-preserving voxel graph cut | Does neighbourhood regularization improve the current A/P partition? | Small / moderate scientific risk | Reuse fixed-support metrics and caches. Few new components; little methodological novelty; can shrink regions or preserve displaced cuts. |
| B. Contextual surface and volume graph | Does surface folding/image context add boundary-location evidence, and does graph inference use it better? | Large / high scientific risk | Reuse cut/edge targets and geometry checks. Most ambitious architecture; requires matched nongraph controls, robust surface correspondence, and independent validation. |
| C. Graph-based structured training loss | Does penalizing globally competing partitions improve the existing network without a new inference head? | Medium / moderate-to-high scientific risk | Reuse A/P logits, existing cut posterior, and conditional CE. Training-time discrete inference has usable subgradients; benefit may be ordinary extra supervision. |

Recommendation: establish an edge/candidate information test for B and compare
against A as a minimal inference baseline. Keep C as a separate training hypothesis;
success of an inference corrector does not establish training-only improvement.
No approach has yet demonstrated the required new gain.

## 4. Two routes into training

### Differentiable harmonic refinement

The existing offline solver can motivate, but is not itself, an autograd module.
With soft unary anchoring rather than endpoint-only interpolation, consider

    E(u) = 1/2 sum_edges w_ij (u_i-u_j)^2
           + lambda/2 sum_i c_i (u_i-q_i)^2
    (L_w + lambda C)u = lambda Cq.

Here C is diagonal and c_i are fixed/detached positive reliability weights. With
sufficient anchoring in every component the system is nonsingular; implicit
differentiation gives gradients through the solve. Compare supervised u with the
label, and bound/normalize conductances so edge scale cannot evade the objective.
Raw confidence is not automatically reliability, especially for confidently wrong
predictions. A loss only asking q to agree with a field fitted to q adds no
independent target information.

Learned random-walker inference already exists: [Cerrone et al., CVPR 2019](https://arxiv.org/abs/1905.09045).
The potential contribution here is task-specific evidence and evaluation, not the
ability to differentiate a Laplacian system. That paper studies seeded segmentation;
it does not establish automatic hippocampal A/P localization.

If refinement runs only during training, retain the ordinary segmentation loss on
raw predictions and explicitly evaluate raw inference. Improvements in the refined
field do not necessarily transfer to the unrefined network.

### Structured training without an inference head

Let E_theta(z) use the existing network's unary logits, initially with fixed graph
weights. Let y be the annotated partition on a fixed training support. Define

    L_struct = max_z [Delta(z,y) + E_theta(y) - E_theta(z)].

The maximizing z is the most competitive wrong partition after adding a task
penalty. With weighted Hamming Delta, finding it amounts to minimizing E-Delta:
only unaries change, so nonnegative Potts pairwise terms remain graph-representable.
After solving for z_hat, backpropagate through E_theta(y)-E_theta(z_hat), treating
the selected discrete labeling as fixed. This gives a subgradient where appropriate;
it is not smooth differentiation through the hard argmin.

This structured-learning idea is established, including graph-cut loss-augmented
inference. [Szummer, Kohli and Hoiem, ECCV 2008](https://alumni.media.mit.edu/~szummer/papers/SzummerKohliHoiem-learning-crf-cuts-eccv08.pdf)

Use a local physical interface band for loss weighting only if specified before
testing. Hamming loss permits the graph-cut construction; arbitrary Dice or topology
loss augmentation must not be claimed exact without a separate derivation.
For a finite candidate set, a smooth log-sum-exp over competing energies is another
option, but it searches only that set, not all graph cuts.

The repository already implements a supervised posterior over coronal cuts in
[ap_cut/posterior.py](../thesis/new_constraints/ap_cut/posterior.py). This is a required
baseline. It currently rejects/skips mixed-label slices under its configured
policy; comparisons must account for valid-case coverage rather than quietly
training/evaluating different cohorts. Extending from planes to graph partitions
needs demonstrated value beyond that existing structured supervision.

## 5. Closest literature and implications for novelty

| Primary source | Relevant established contribution | What remains unproven here |
|---|---|---|
| [Boykov and Jolly, 2001](https://www.eng.utah.edu/~cs7640/readings/boykov_iccv01.pdf) | Seeded binary cuts combining region and boundary costs | Costs that locate the MSD A/P landmark |
| [Grady, 2006](https://pubmed.ncbi.nlm.nih.gov/17063682/) | Harmonic hitting probabilities for seeded segmentation | Whether diffusion carries the needed semantic evidence |
| [Szummer et al., 2008](https://alumni.media.mit.edu/~szummer/papers/SzummerKohliHoiem-learning-crf-cuts-eccv08.pdf) | Structured parameter learning with graph-cut inference | Additional value for this task over its plane posterior |
| [Shi et al., 2009](https://pubmed.ncbi.nlm.nih.gov/19694286/) | Reeb/eigenfunction features for hippocampal surface correspondence | Whether corresponding morphology identifies the uncal-apex slice |
| [Cerrone et al., 2019](https://arxiv.org/abs/1905.09045) | Learned contextual conductances through random-walker inference | Useful automatic anchors/affinities on MSD |
| [HippUnfold algorithms](https://hippunfold.khanlab.ca/en/latest/pipeline/algorithms.html) | Anatomically anchored intrinsic coordinates | Availability of the necessary tissue/boundary labels in whole-mask MSD data |
| [DeKraker et al., July 2026 preprint](https://pubmed.ncbi.nlm.nih.gov/42619716/) | Surface-intrinsic Laplace coordinates and hippocampal correspondence | Evidence for head/body-tail cut prediction on MSD |

The July 2026 study is a preprint, and its alignment/morphometry results should not
be presented as validation of this proposed A/P segmentation method. Likewise,
HippUnfold is not reproduced by taking a whole-mask PCA axis or a centerline.

A Reeb graph is a possible compact descriptor: it tracks connected components of
level sets of a scalar field. Its branches/endpoints depend on that field and on
noise; they are not automatically anatomical regions. Existing hippocampal Reeb
work means the representation itself is not a novelty claim. A surface-fold
transition descriptor could be screened, but is currently speculative.

Candidate thesis claim, conditional on positive experiments:

> Surface/context-derived semantic affinities improve the localization of an
> annotation-defined internal hippocampal interface, and structured graph inference
> or training adds benefit beyond matched voxel and plane alternatives.

An equally valid negative-result contribution would distinguish representation,
topological validity, and semantic localization, and show exactly which additional
information is missing. This search does not establish publication novelty.

## 6. Decisive experiment sequence

### Stage 0: representation and failure-mode audit

Reuse existing topology and plane audits. Measure additional interface properties
only if claiming them: number of interface components, boundary loops,
nonmanifold edges, and Euler characteristic on a verified mesh. Connectivity of a
voxel contact band is not a proof that the interface is a manifold disk.

On training cases, construct controlled wrong partitions: displaced coherent cuts
(both directions), islands, fragmented interfaces, and correct partitions. Test
whether a proposed score distinguishes location errors from structural defects.
Oracle fitting is a representation diagnostic, not a deployable accuracy estimate.

### Stage 1: semantic-edge and cut-ranking information test

Define targets on foreground face-neighbour pairs: different A/P labels versus
same label. Compare coordinates/geometry/logits with added local MRI context and
then surface context. The relevant challenge is near plausible competing cuts;
random negatives across the organ make the classification too easy.

Hold out whole cases, and participants if verified linkage becomes available.
Use out-of-fold backbone predictions/features for downstream training when feasible.
Never split edges from one case between train and validation. Fix feature extraction,
candidate generation, selection, and acceptance criteria before viewing outcomes.

Report class-balanced edge precision/recall, but make candidate cut localization
the decision endpoint: edge accuracy can be high while cut location remains wrong.
Compare the exact same base features and capacity with/without context. Include
the existing incremental-context ranker rather than repeat it under a graph name.
Matched spatially shuffled features are a negative control, not automatically a
permutation significance test.

Advance only if added evidence improves cut ranking beyond logits/geometry on
held-out training cases, survives correct-cut and bidirectional displacement
checks, and retains value on predicted rather than only reference foreground.
Freeze a minimum practically useful effect before running this screen; the current
evidence does not justify inventing an expected effect size.

### Stage 2: correction study

Compare raw model, model-fitted plane, a cut-likelihood/plane ranker, uniform
graph+unaries, contextual graph+identical unaries, and a matched-capacity nongraph
corrector. Include a reference-informed diagnostic separately and prominently mark
it as oracle. Keep foreground identical for every correction arm.

Primary endpoint: case-balanced raw A/P Dice. Report cut error in mm, A/P swap
counts, boundary-band error, helped/harmed cases, severe failures, solver runtime,
coverage, and foreground preservation. Score curved partitions directly; fitting
a plane afterward is a predeclared secondary analysis, not endpoint rescue.
Any abstention rule is selected on training data and evaluated at its actual coverage.

Advance only on a frozen method beating both the original model and its appropriate
simple/graph controls, with acceptable case-level harm. Confirm on fresh grouped
data before interpreting reused development-fold intervals as validation.

### Stage 3: training study

Compare the same backbone and training budget under ordinary supervision,
conditional A/P CE, existing plane/cut supervision, supervised local edge loss,
and the chosen structured graph objective. For a learned affinity branch, add a
matched-capacity auxiliary nongraph branch. These controls distinguish graph
structure from extra parameters, extra targets, and ordinary boundary emphasis.

Evaluate raw segmentation without a graph at inference as the primary result for
the training-only claim. Report refined inference separately if offered. Use
matched seeds and stopping rules; select weights using training-only calibration.
Monitor union Dice despite conditional A/P logits: shared network parameters can
still change the foreground. Audit support/affinity collapse and gradient scale.

## 7. Immediate next action

Prepare a fixed, training-only comparison of correct and displaced partitions,
scored by existing logits/plane models versus one explicitly specified contextual
edge representation. Reuse the current caches and protocols, and first establish
which training features are genuinely out of fold. If the edge representation
adds no cut-location signal, stop that affinity branch before meshing the organ,
training a GNN, or adding a solver to the main training loop.

Open decisions for implementation: source of reliable surface evidence; grouping
metadata; availability of out-of-fold features; and the smallest effect worth
advancing. No external data annotation, compute spending, or training run is
implied by this research assessment.
