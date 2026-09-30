# Harmonic fields for A/P separation: research assessment

## Outcome

**Implemented and tested an offline harmonic solver and ran a controlled
feasibility pilot. Do not add the tested raw-MRI random walker to training or
inference.** It worsened A/P Dice on both inspected models and failed to beat
the same graph with uniform connections. A learned, context-dependent affinity
model remains a different hypothesis; this pilot does not validate it.

The useful research question is whether image/context-derived connections can
locate the annotated boundary better than the network already does. A smooth
field and reliable endpoints are not sufficient.

## What the literature supports

[Grady (2006)](https://pubmed.ncbi.nlm.nih.gov/17063682/) formulates seeded
segmentation using random-walk hitting probabilities, computed through a
Dirichlet problem on an image graph. This provides the inference mechanism;
the seeds and connections supply its information.

[Cerrone, Zeilmann and Hamprecht (CVPR 2019)](https://arxiv.org/abs/1905.09045)
learn CNN-predicted graph weights through the random-walker solve. Their
reported task is seeded CREMI neuron segmentation, not automatically seeded
hippocampal head/body separation. Their [author implementation](https://github.com/sciai-lab/pytorch-LearnedRandomWalker)
is useful prior art, not evidence that its seed protocol or affinities transfer
to our task. No external package or code from it was installed or vendored.

[Vernaza and Chandraker (CVPR 2017, revised manuscript 2018)](https://arxiv.org/abs/1802.00470)
also learn random-walk propagation jointly with a segmentation predictor using
sparse supervision. This motivates a semantic affinity model rather than
assuming every local intensity edge is an anatomical class boundary.

Thus differentiating through harmonic inference is established methodology.
A research contribution here would need to establish useful A/P-specific
affinities, automatic anchors, and gains over matched non-graph controls.

## Implemented experiment

The [protocol](PROTOCOL.md) was written before running the pilot. The solver
uses the six-face voxel graph inside fixed foreground, with A cores at 0 and
P cores at 1. It solves the sparse linear system in float64. Background and
insufficiently seeded components retain the original prediction.

Connections use normalized MRI intensity differences with a positive weight
floor. Beta=0 is the uniform-conductance control; beta=1 and 10 were screened.
Two automatic core recipes were considered: the outer 20% of the coronal
extent, or interiors at least 4 mm either side of the model's own cut. Both
avoid the outermost layer of foreground through one-face erosion.

Selection used 32 deterministically sampled fold-0 training cases, with
synthetic -2/0/+2-slice cut displacements on clean reference foreground. This
is a mechanism screen, not realistic network-error validation. The chosen
image candidate was **cut-centered cores, beta=10**, selected by cut MAE with
Dice as a tie-breaker. Selection was frozen before opening the development
prediction caches. Choosing by cut MAE does not guarantee better raw-field
Dice; both are reported below.

The development check used 52 cases with two early-stopped seed-0 models,
unaugmented and augmented, from existing **local CPU** caches. No model was
trained or rerun. Development labels never define cores, conductances, or
thresholds; they only score results and verify alignment. Foreground is
preserved exactly, including predicted islands and voxels outside the native
crop. The native MRI is padded consistently with the cached volume.

## Training screen

End-based cores placed cuts 4.72–4.88 slices from the target, irrespective of
the synthetic starting cut. They do not recover the annotation's anatomical
landmark from geometry/intensity alone in this screen.

| Cut-centered cores | Cut MAE, slices | Raw A/P Dice |
|---|---:|---:|
| Original synthetic prediction | 1.333 | 0.961647 |
| Uniform connections | 1.333 | 0.960311 |
| MRI beta=1 | 1.292 | 0.960871 |
| MRI beta=10, selected | 1.250 | 0.959008 |

The selected field has a directional effect. For initial displacements
-2/0/+2, cut MAE becomes **2.375 / 0.5625 / 0.8125**. It helps one displacement
direction, harms the other, and perturbs correct cuts. The selected setting's
slight average cut-MAE gain already trades against worse raw Dice.

## Development results

Scores average the two foreground-class Dice values within each case and
then average cases. All methods use the identical original foreground.

| Method | Unaugmented Dice | Augmented Dice | Unaugmented cut MAE | Augmented cut MAE |
|---|---:|---:|---:|---:|
| Original model | 0.871443 | 0.886557 | 1.019 | 0.904 |
| Original model, fitted plane | 0.871884 | 0.887186 | 1.019 | 0.904 |
| Uniform field, raw | 0.869360 | 0.880636 | 1.077 | 1.115 |
| MRI field, raw — primary | 0.866446 | 0.876186 | 1.058 | 0.981 |
| Shuffled-MRI field, raw | 0.863192 | 0.872660 | 1.096 | 1.192 |
| MRI field, fitted plane — secondary | 0.871854 | 0.884964 | 1.058 | 0.981 |

The **primary MRI-field Dice change is -0.500 and -1.037 percentage points**.
Its paired case-bootstrap 95% intervals are [-0.864, -0.130] and
[-1.386, -0.681] percentage points. It helps/harms **16/36** and **12/40** cases.
Raw A/P swaps rise from **4,292 to 5,200** and **3,702 to 5,470**.

Against uniform connections, MRI weights lose **0.291 and 0.445 percentage
points**, with intervals [-0.461, -0.121] and [-0.635, -0.251]. Against shuffled
MRI they gain 0.325 and 0.353 points. The image therefore affects the field
usefully relative to this destructive control, but that does not make the
field better than the original prediction or the geometry-only control.
The shuffle is one fixed realization, not a permutation significance test.

Fitting a coronal plane to the field removes much of its raw-label damage,
but still does not beat the original model's plane projection in either arm.
This supports the interpretation that some image-induced spatial variation is
inappropriate for these mostly planar labels. It is a mechanism interpretation,
not proof that all MRI features lack boundary information.

## What failed, and what did not

- **The numerical solve worked.** Maximum linear residuals were below
  4e-15; all solved values respected the [0,1] maximum principle and the cores.
  More than 99.98% of predicted foreground was in solved components.
- **A/P core purity was high.** Among cores inside reference foreground, only
  41 had the wrong A/P class for the unaugmented model (about 0.05%); the
  augmented model had **zero**. There were also 1,732/1,157 active cores on
  false-positive foreground; fixing foreground prevents this pilot from
  correcting those outer errors. Correct A/P cores are not enough to make
  the interpolation correct.
- **The geometric field already had a bias.** Uniform weights moved the fitted
  cut toward lower Y in 31/52 and 21/52 cases, and toward higher Y in none.
  MRI weighting moved it toward lower Y in 33/52 and 27/52 cases, with only one
  higher-Y move in each arm. Interpolation between symmetric core bands is
  affected by the support's geometry; it is not an unbiased cut estimator.
- **Raw intensity differences were not the needed semantic signal.** The
  observed failure is consistent with diffusion following local texture and
  shape rather than the annotation's coronal landmark. It does not distinguish
  all possible feature representations or all choices of beta/core width.

## Stronger research formulation, if revisited

The more defensible future model would keep the existing network's conditional
posterior probability `q_i` as a soft anchor and learn bounded positive
conductances `w_ij` from MRI plus local 3D context. For example:

`E(u) = 1/2 sum_edges w_ij (u_i-u_j)^2 + lambda/2 sum_i c_i (u_i-q_i)^2`

This yields `(L_w + lambda C) u = lambda C q`. It can be differentiated by
implicit differentiation. Unlike imposing a new midpoint between distant
cores, the unary term preserves evidence from the original classifier. This
is a proposed model, **not implemented or validated in this pilot**.

Several controls would be essential before crediting the graph:

1. Build automatic anchors from out-of-fold probabilities or jointly trained
   outputs, with confidence rules calibrated on training data. Label-derived
   cores at evaluation would change the task into an oracle-assisted one.
2. Fix or detach confidence weights initially. Bound conductances and control
   their scale relative to the unary term; otherwise learned weights can
   weaken connections simply to avoid consistency penalties.
3. Supervise the field's actual A/P predictions. Pure self-consistency between
   a prediction and a field fitted to that same prediction supplies no new
   anatomical evidence.
4. Compare uniform graph + unaries, learned graph + identical unaries, and a
   matched-capacity convolutional correction head. Include conditional A/P
   cross-entropy as a simple supervision control. A benefit from extra capacity
   or extra supervision is not automatically a graph benefit.
5. Require simultaneous improvement in raw A/P Dice, cut error, and patient
   harm rate, with foreground errors reported separately. Do not switch to
   plane-projected scoring only after raw field predictions disappoint.

The **next justified prerequisite is a training-only semantic-edge probe**:
can contextual features distinguish true cross-interface neighbor pairs from
same-class pairs, especially on already well-formed but misplaced cuts?
Compare MRI/context against coordinates and existing logits, using held-out
training cases and out-of-fold backbone features. If no additional signal is
demonstrated, there is little reason to insert a sparse solve into training.
The present raw-intensity pilot fails the advancement gate, so no learned
training run was launched.

## Scope and reproducibility

The same development fold has been repeatedly examined in this project, and
the two model arms are not independent cohorts. Bootstrap units are crops;
participant grouping is unavailable. These are exploratory local-cache
results, not independent confirmatory estimates or official CUDA scores.
The screen examined only a small fixed parameter grid. It rejects the tested
candidate as a current addition; it does not prove a universal impossibility.

Implementation:

- [Solver and core recipes](../../../thesis/new_constraints/harmonic_partition.py)
- [Two-stage pilot](../../../evaluation/pilot_harmonic_ap.py)
- [Analytical and integration tests](../../../thesis/new_constraints/test_harmonic_partition.py)

Run from the repository root:

```bash
.venv/bin/python evaluation/pilot_harmonic_ap.py --stage screen
.venv/bin/python evaluation/pilot_harmonic_ap.py --stage evaluate
.venv/bin/python -m pytest \
  thesis/new_constraints/test_harmonic_partition.py \
  evaluation/test_audit_ap_partition_topology.py \
  thesis/new_constraints/ap_plane/test_existential.py \
  thesis/new_constraints/ap_cut/test_posterior.py -q
```

**63 targeted tests pass.** They include exact 1D resistance solutions,
3D linear fields, seed-complement symmetry, affine-intensity invariance,
insufficient-seed fallback, spacing-aware cores, foreground preservation,
and a check that reference scoring cannot mutate the inferred field.

The evaluation refuses changed solver/pilot/metric/protocol hashes or changed
training inputs since selection. Cached baseline Dice and swap counts must
reproduce the original summaries. Generated evidence: [selection](selection.json),
[training cases](training_cases.json), [development summary](development_summary.json),
and [development cases](development_cases.json). JSON evidence contains metrics
and file hashes, not raw MRI, masks, or checkpoints.
