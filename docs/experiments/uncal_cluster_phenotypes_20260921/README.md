# Unsupervised hippocampal phenotype pilot

This audit tests whether whole-hippocampus morphology naturally separates MSD
Task04 cases into groups that support different anterior/posterior cut rules.
It is an exploratory analysis, not a trained segmentation result.

## Design

- All 260 labeled Task04 training volumes were evaluated once through the five
  predefined subject-held-out folds.
- Clustering used the union of anterior and posterior foreground and local T1
  appearance. The A/P class labels and annotated cut were hidden from the
  clustering algorithm.
- Per-slice geometry and image transition features were normalized within each
  case, sampled at 16 relative long-axis positions, standardized on each outer
  training fold and reduced to 12 principal components.
- The number of clusters was selected independently in each training fold from
  2-5 using silhouette score. Validation subjects were assigned using the
  training transform and centroids.
- A/P labels were revealed only after clustering to describe the groups and to
  score global versus cluster-conditional cut predictors.
- This proof of concept uses reference foreground. A deployable version would
  need predicted foreground or a raw-image representation.

## Result

Every training fold selected two clusters. The silhouette scores were low
(0.115-0.125), although subsampling stability was moderate: median adjusted
Rand index ranged from 0.799 to 0.887. This means the algorithm repeatedly
found a broad division, but the division was weak rather than a clean set of
anatomical phenotypes.

The clusters did not exhibit useful cut-position differences. Across folds,
cluster median cut positions occupied essentially the same relative range
(approximately 0.588-0.600 of the foreground long-axis extent), with strongly
overlapping interquartile ranges.

| Predictor | MAE (slices) | Exact | Within 1 | Within 2 |
|---|---:|---:|---:|---:|
| Global relative-position median | 1.162 | 25.8% | 70.0% | 90.4% |
| Cluster-specific position median | 1.181 | 25.0% | 68.8% | 90.4% |
| Global compact geometry + MRI model | **0.969** | **40.0%** | **79.2%** | 89.2% |
| Cluster-specific compact models | 1.154 | 38.1% | 78.8% | **89.6%** |

The cluster-specific compact model increased MAE by 0.185 slices. A paired
case bootstrap gave a 95% interval of -0.015 to 0.442 slices, so the audit does
not establish significant harm, but it provides no evidence of improvement.
It was better for 20 cases, unchanged for 218 and worse for 22.

The visually discussed fold-0 examples also did not define the clusters:
185 and 205 were assigned together, while the other separate-profile example
327 was grouped with ambiguous case 164. The discovered division therefore
does not correspond directly to “separate profile”, “connected lip” and
“ambiguous” cut appearances.

## Interpretation

The hypothesis that the dataset contains multiple boundary phenotypes remains
plausible. This pilot says that clustering patients by their entire normalized
hippocampal sequence is too broad. It can group curvature, intensity,
segmentation style and other variation that is unrelated to the local A/P
transition. Splitting only 208 training cases per outer fold also reduces the
sample available to each conditional predictor.

Unsupervised clustering has recovered meaningful hippocampal organization from
much richer histological morphology, so the general scientific idea is sound;
the 1-mm Task04 T1 crops provide a harder and less specific signal. See
[DeKraker et al., 2020](https://pubmed.ncbi.nlm.nih.gov/31682982/).

## Recommended next test

Cluster **candidate transition windows**, not whole patients:

1. Generate every plausible coronal candidate, or use the current global model
   to produce a small candidate set.
2. Encode a short window around each candidate using coronal MRI, sagittal
   continuity and predicted whole-hippocampus shape. Do not include A/P colours.
3. Learn soft prototypes for appearances such as a separate medial profile, a
   connected returning lip and a weak/ambiguous transition. These names are
   hypotheses for later visual interpretation, not preset cluster labels.
4. Let prototype membership gate several fuzzy predicates and retain an
   uncertainty/fallback predicate. Do not force each patient into one permanent
   hard cluster.
5. Compare the soft mixture with the same global model in nested subject-wise
   cross-validation. Require stable prototypes, visually coherent medoids and
   improved held-out boundary error before adding the mechanism to segmentation
   training.

If training windows are centred using the annotated cut, this is unsupervised
**phenotype discovery conditional on a supervised landmark**, rather than a
fully unsupervised method. A strictly unsupervised version must cluster windows
from all candidate positions and then learn which prototype-position pair
matches the cut, which is considerably less identifiable.

For an LTN-style mixture, a useful form is

`cut_truth(c) = sum_k gate_k(c) * predicate_k(c)`

where the gates form a probability distribution and include an uncertain
fallback. A spatial constraint can then encourage anterior probability above
the candidate plane and posterior probability below it, with a soft band around
the plane. Gate entropy, minimum prototype usage and an unconstrained fallback
must be monitored so the model cannot satisfy the objective by collapsing all
cases into one easy group or marking every case uncertain.

## Reproduce

Implementation: [audit_cluster_phenotypes.py](../../../thesis/new_constraints/uncal_fold/audit_cluster_phenotypes.py)

```bash
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_cluster_phenotypes
```

Complete fold assignments, predictions and diagnostics:
[summary.json](summary.json).
