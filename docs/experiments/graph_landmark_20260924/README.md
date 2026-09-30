# Whole-graph local landmark search for the A/P boundary

24 September 2026. **There is a modest localization signal in a combination of
whole-graph profiles. There is no demonstrated single landmark that identifies
the cut in every volume, and the tested readout does not improve the existing
network cuts when applied to predicted foreground.**

This tests the previously missing question: can local changes along the whole
hippocampus graph locate the transition, without first giving the graph its A/P
partition? All extraction uses whole-graph arrays only. A/P labels supply targets
and scoring, never descriptor construction. MRI, logits, original network cut
locations, and anterior/posterior graph archives are excluded from the features.

## What ran

- 208 fold-0 training cases: fourfold case-level cross-validation, with all
  candidate rows from a case kept together.
- A frozen check on 52 reference validation masks, followed by transfer to the
  same 52 cases' existing unaugmented and augmented prediction masks.
- Eight fixed landmark rules, two population position/volume priors, four
  learned feature combinations, and a shifted-graph negative control.
- A subsequent **training-only** descriptor ablation to distinguish local
  structure from intrinsic position. It did not replace the primary selected
  model or rerun model selection on validation.

The [primary protocol](PROTOCOL.md) was written before fitting. Selection and
training results were saved before validation feature extraction. All 208
training and all 52 cases in each validation source had their reference target
inside the label-independent candidate set. No network training or new inference
was run. This is the same repeatedly inspected validation fold; these results
are exploratory rather than independent confirmation, and participant linkage
has not been established.

![Experiment results](../../../experiments/graph_landmark_20260924/results.png)

## Does a single graph landmark work?

No tested rule reliably identifies the reference transition. All rules search
the full candidate interval, without a window centered on the reference cut.

| Fixed landmark rule | Training CV MAE, mm | Reference validation MAE, mm |
|---|---:|---:|
| Largest coronal area rise | 1.81 | 1.73 |
| Largest thickness rise | 4.60 | 4.67 |
| Largest degree-six fraction rise | 6.15 | 6.02 |
| Betweenness peak | 4.09 | 3.83 |
| Slice-component branching peak | 14.79 | 15.13 |
| Minimum geodesic normalized cut | 6.12 | 6.38 |
| Largest geodesic-bin thickness rise | 4.40 | 3.38 |
| Largest standardized multifeature change | 9.42 | 9.81 |

Area rise was selected as the best fixed rule using training only. The branching
profile is constant across candidates in 132/208 training and 32/52 validation
cases. Its large error partly reflects the declared lower-candidate tie break
when there is no distinct peak at all; it should not be interpreted as a
consistently identifiable anatomical branch far from the target.

The branching statistic is a component-quotient graph over slices or geodesic
bins, not a verified medial skeleton. Diameter endpoints and their geodesic
coordinates are also geometric proxies, not annotated anatomical landmarks.

## Does combining local information help?

Yes, relative to the tested ordinary-shape readout and the spatially shifted
graph control. The learned model scores each possible coronal cut using fixed
local values, derivatives, and +/-2-bin changes. It selects the highest score.
This is a regularized logistic readout, not a GNN.

| Method | Training CV MAE, mm | Reference validation MAE, mm | Validation exact | Validation within 1 mm |
|---|---:|---:|---:|---:|
| Training-median normalized position | 1.17 | 1.13 | 14/52 | 37/52 |
| Training-median posterior volume fraction | 1.51 | 1.31 | 11/52 | 31/52 |
| Learned position only | 1.75 | 1.73 | 4/52 | 22/52 |
| Position + ordinary shape | 1.29 | 1.10 | 19/52 | 39/52 |
| Position + graph profiles | 1.09 | 1.00 | 19/52 | 40/52 |
| **Position + shape + graph** | **1.00** | **0.85** | **27/52** | **42/52** |
| Shape + spatially shifted graph control | 1.29 | 1.12 | 19/52 | 37/52 |

The combined model was selected by training CV. Adding graph features reduces
MAE against the shape readout by **0.288 mm** in training CV, with paired
case-bootstrap 95% interval **[-0.433, -0.149] mm**, improving three of four folds.
This passes the declared exploratory graph-increment gate.

On reference validation, the reduction is **0.250 mm**, interval
**[-0.500, -0.019] mm**: 19 cases improve, eight worsen, and 25 tie. Versus
the shifted-graph control, the reduction is 0.269 mm, interval
[-0.519, -0.019] mm. The control keeps the same number and distribution of
graph features while disturbing their alignment to the candidate cuts.

The stronger simple-position comparison is less conclusive: versus the
training-median normalized-position prior, combined MAE changes by -0.168 mm
in training CV (interval [-0.337, +0.005]) and -0.288 mm in validation
([-0.673, +0.096]). The learned position-only readout is weaker than this
simple prior; beating that learned baseline alone would overstate the result.
The fitted median cut lies at 57.75% of the occupied RAS Y extent from posterior
to anterior, illustrating how informative a basic population prior already is.

The combined model still misses the exact cut in **25/52** reference validation
cases. Ten errors exceed 1 mm, six exceed 2 mm, and the maximum is 4 mm.
Its 95th-percentile error is 3.45 mm, versus 3.0 mm for the median-position prior.
Better average localization does not mean a universally consistent landmark.
These bootstrap intervals are conditional on fitted models; they omit model
refitting and participant-level uncertainty.

## Is the gain just an intrinsic-coordinate prior?

The [post-primary training-only ablation](ABLATION_PROTOCOL.md) provides some
separation. Ordinary position and shape are retained in every arm below.

| Graph information retained | Training CV MAE, mm |
|---|---:|
| Full graph block | 1.005 |
| Only harmonic/Fiedler/geodesic position summaries and validity | 1.231 |
| Full block without intrinsic position summaries | 1.010 |
| Full block without degree/depth/thickness | 1.072 |
| Full block without geodesic section profiles | 1.058 |
| Full block without branching proxies | 1.000 |
| Full block without betweenness/eccentricity | 0.995 |

The intrinsic-only block is worse than the full block by 0.226 mm, interval
[+0.096, +0.356]; removing the intrinsic-position summaries scarcely changes
the full model. Thus the primary increment is not explained solely by the
explicit intrinsic-coordinate channels. Geodesic section profiles still use
an intrinsic coordinate to define their sampling, so this is not a claim that
all coordinate dependence has been removed.

Degree/depth/thickness and geodesic section profiles are plausible contributors,
but individual group-removal intervals include zero. Features are correlated
and can compensate. Neither the ablations nor large fitted coefficients identify
one unique causal landmark. There is no evidence here that the branch or
betweenness peak is the missing universal anatomical event.

![Successful, near, and failed reference examples](../../../experiments/graph_landmark_20260924/landmark_profiles.png)

These examples were selected after evaluation to illustrate zero, one, and
maximum cut error; they are not an additional test set. The horizontal reference
alignment is for plotting only. Descriptors and candidate search did not see it.

## Does it help correct current predictions?

The frozen combined readout was trained on reference geometry and then applied
to whole predicted foreground. This introduces foreground domain shift, but
directly tests whether the discovered signal can currently replace a model cut.

| Validation measurement | Unaugmented model | Augmented model |
|---|---:|---:|
| Original network cut MAE | 1.019 mm | 0.904 mm |
| Frozen combined graph readout MAE | 1.038 mm | 1.000 mm |
| Median-position prior on predicted support MAE | 1.019 mm | 0.962 mm |
| Originally correct cuts moved | **2/16** | **7/19** |
| Original A/P Dice | 0.871443 | 0.886557 |
| Graph-readout A/P Dice | 0.871950 | 0.883890 |
| Original A/P swaps | 4,292 | 3,702 |
| Graph-readout A/P swaps | 4,056 | 4,073 |

The unaugmented Dice change is +0.051 percentage points, interval
[-0.545, +0.601]. Against merely rendering the network's own fitted plane,
the increment is just +0.0065 points. Augmented Dice changes by -0.267 points,
interval [-0.735, +0.184], or -0.330 points versus its fitted-plane rendering.
Neither cut-MAE comparison establishes improvement over the existing network.

All correction renderings retain the original predicted foreground. Therefore
foreground additions/deletions are not credited to the graph. The graph-only
feature arm happens to tie the augmented network's mean cut error, but it was
not the selected model; switching to it after seeing validation would be model
selection on the evaluation set.

## Consequence for anatomical constraints

The whole graph does contain boundary-related information in a **combination
of local structural profiles**. The evidence supports a probabilistic shape
cue, rather than a hard rule such as “cut at the branch,” “cut at the narrowest
neck,” or “cut at the largest thickness change.”

That cue has not demonstrated reliable incremental correction of existing model
outputs. It should not yet be used as a hard projection or a training penalty
that forces agreement with this readout: a teacher with these errors could
move correct boundaries. Testing an auxiliary profile target or combining
these cues with image-derived evidence would be a distinct experiment. It would
need training-only calibration on realistic predicted foreground, appropriate
out-of-fold backbone outputs for fitting a correction stage, and a fresh
evaluation of correction versus preservation. No such efficacy is claimed here.

This result limits the tested descriptors, candidate-plane formulation, and
readout. It does not establish that no graph representation or anatomical
landmark could provide stronger localization.

## Definition and verification details

The input is the six-face graph of every whole-foreground voxel. Graph maps
come from the [previous property inventory](../graph_property_inventory_20260923/README.md).
Both labels are merged before that whole-graph construction. Coordinate
orientation uses RAS Y; no claim of a purely unembedded, rotation-invariant
graph landmark is made.

Each candidate has six position columns, 15 ordinary-shape columns, and 69
graph columns. Ordinary shape uses area, perimeter/area, LR/SI width, and
normalized cut. Graph features use degree frequencies, graph depth, digital
local thickness, 32-source approximate betweenness, eccentricity, component
quotients, harmonic/Fiedler positions, and 32-bin geodesic profiles. Geodesic
bin occupancy/crossing counts are graph descriptors, not physical orthogonal
cross-sectional area measurements. Intrinsic fields are valid only on their
supported component; finite-node fractions are explicit.

Extraction is invariant to image padding. Profiles use fixed Gaussian smoothing
and support-relative margins. The cut target is the best coronal plane for the
reference labels; it need not reproduce every mixed-slice annotation. Prediction
cache references were checked against padded native labels. Source/map hashes,
runtime versions, protocol hashes, and frozen selection are retained.

**23 tests passed**, including six new tests of translation/label independence,
disconnected intrinsic maps, quotient branching, exhaustive target fitting,
shifted-control integrity, and prediction independence from test targets.
The training-only full ablation reproduces all primary combined OOF cuts exactly.

## Files and reproduction

- [Primary protocol](PROTOCOL.md), [frozen selection](selection.json).
- [Training summary](training_summary.json), [training OOF cases](training_oof.json),
  [training manifest](training_manifest.json).
- [Validation summary](validation_summary.json), [validation cases](validation_cases.json),
  [validation manifest](validation_manifest.json), [additional paired comparisons](paired_details.json).
- [Verification of 364 feature archives, memberships, and frozen provenance](verification.json).
- [Ablation summary](ablation_summary.json), [ablation cases](ablation_cases.json),
  [ablation manifest](ablation_manifest.json).
- [Runner](../../../evaluation/probe_graph_landmark.py),
  [tests](../../../evaluation/test_probe_graph_landmark.py),
  [training-only ablation](../../../evaluation/explain_graph_landmark.py),
  [figures](../../../evaluation/plot_graph_landmark.py).

Generated feature arrays and figures remain ignored under
`experiments/graph_landmark_20260924/`. Run in this order from the repository root:

```bash
rtk proxy .venv/bin/python evaluation/probe_graph_landmark.py --stage train
rtk proxy .venv/bin/python evaluation/probe_graph_landmark.py --stage evaluate
rtk proxy .venv/bin/python evaluation/explain_graph_landmark.py
rtk proxy env MPLCONFIGDIR=/tmp/hippo-graph-landmark-mpl \
  XDG_CACHE_HOME=/tmp/hippo-graph-landmark-cache \
  .venv/bin/python evaluation/plot_graph_landmark.py
rtk proxy .venv/bin/python -m pytest \
  evaluation/test_probe_graph_landmark.py \
  evaluation/test_voxel_graph_anatomy.py \
  evaluation/test_graph_property_inventory.py -q
```

Reproduction requires the native MSD labels, existing whole-graph property maps,
and existing prediction caches. The evaluation stage refuses changed source,
protocol, or selection provenance from the preceding training stage.
