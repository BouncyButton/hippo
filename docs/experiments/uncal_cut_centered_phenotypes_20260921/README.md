# Cut-centred phenotype discovery

Date: 2026-09-21. This experiment implements unsupervised phenotype discovery
around a supervised A/P landmark. It is an exploratory analysis and does not
modify the segmentation model or reference labels.

## Question

Do local geometry and MRI appearance at the annotated MSD cut form multiple
reproducible phenotypes, and can those phenotypes improve cut localisation on
unseen subjects?

## Leakage control

- All 260 Task04 training volumes were evaluated through the five predefined
  subject-held-out folds.
- In each outer training fold, the annotated A/P cut selected one local feature
  vector per subject. The vector contained whole-foreground-union geometry,
  adjacent-slice transitions and local T1 appearance. It did not contain the
  anterior/posterior class identity.
- Cluster count was selected from 2-5 using training-fold silhouette only.
- The positional weight for prototype localisation was selected using
  three-fold cross-validation inside the outer training fold.
- Deployable validation methods never saw the validation cut. An oracle gating
  result that uses it is named explicitly and is diagnostic only.

This is unsupervised clustering conditional on a supervised landmark. It is not
a fully unsupervised discovery of both the landmark and phenotype.

## Phenotypes found

The selected number of clusters was 2, 4, 2, 2 and 3 across the five folds.
Selected-cluster silhouettes ranged from 0.154 to 0.177. Median bootstrap
adjusted Rand stability ranged from 0.698 to 0.880. There is recurring structure,
but the changing cluster count indicates a continuum or unstable subdivisions
rather than a fixed phenotype taxonomy.

In fold 0, the two broad groups were:

- phenotype 0, 78/208 training cases: darker intensity immediately superior to
  the foreground and a stronger new superior-side change between slices;
- phenotype 1, 130/208 cases: brighter superior-band intensity, smoother
  adjacent-slice overlap and weaker side-specific emergence.

The medoids do not support naming these groups “separate profile” and
“connected lip.” Both contain varied shapes, and intensity context contributes
strongly to their separation.

![Fold 0 phenotype 0 medoids](fold0_cluster_0_medoids.png)

![Fold 0 phenotype 1 medoids](fold0_cluster_1_medoids.png)

The figures show raw MRI with a white whole-hippocampus outline. They do not
show A/P class colours. The pink sagittal line marks the training annotation
used to centre the window.

## Held-out localisation

| Method | MAE slices | Exact | Within 1 | Within 2 | p90 |
|---|---:|---:|---:|---:|---:|
| Global relative-position median | 1.162 | 25.8% | 70.0% | **90.4%** | **2.0** |
| Global compact geometry + MRI model | **0.969** | 40.0% | **79.2%** | 89.2% | 3.0 |
| Cut prototypes, no position | 3.008 | 14.6% | 38.1% | 63.8% | 6.1 |
| Cut prototypes + inner-tuned position | 1.219 | 26.2% | 65.8% | 89.2% | 3.0 |
| Predicted-gate cluster experts | 1.981 | 35.4% | 74.6% | 85.4% | 3.0 |
| Oracle-gate cluster experts, diagnostic only | 2.100 | **48.5%** | 76.2% | 86.9% | 3.0 |

The tuned prototype method was 0.250 slices worse than the global compact model
(paired bootstrap 95% interval 0.077 to 0.419). It improved 70 cases, tied 62
and worsened 128.

Four folds selected a positional weight of 8 or 16; the fifth selected 4. The
prototype evidence alone therefore did not uniquely identify the cut. Inner
tuning improved it mainly by restoring the population position prior.

Hard phenotype-specific experts were unsafe. The oracle gate increased exact
predictions but produced endpoint failures up to 26 slices, yielding worse mean
error even when phenotype assignment had access to the true validation cut.
Thus gating error is not the primary explanation: fitting separate experts to
smaller phenotype subsets is itself unstable.

## Decision

The experiment supports the presence of recurring local appearance patterns,
but it does **not** support injecting hard unsupervised phenotype classes or
separate cluster-specific constraints into the segmentation model.

If this direction continues, use a continuous, soft phenotype representation
with strong shrinkage to a shared global cut model. Prototype evidence should
be a bounded residual or confidence modifier, never a standalone locator. A
reviewer should first inspect cluster medoids and label whether they correspond
to a separate medial profile, connected returning lip, other plausible anatomy,
or image/segmentation artefact. Those small semantic labels would convert the
problem into weakly supervised phenotype learning and prevent intensity-driven
clusters from being mistaken for anatomy.

## Reproduction

Audit implementation:
[audit_cut_centered_phenotypes.py](../../../thesis/new_constraints/uncal_fold/audit_cut_centered_phenotypes.py).

Medoid renderer:
[visualize_cut_centered_phenotypes.py](../../../thesis/new_constraints/uncal_fold/visualize_cut_centered_phenotypes.py).

```bash
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_cut_centered_phenotypes
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.visualize_cut_centered_phenotypes
```

Complete fold assignments, predictions, tuning curves and cluster descriptors:
[summary.json](summary.json).
