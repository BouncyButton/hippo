# Fold 0 validation: native-slice profile audit

Date: 2026-09-21. This is an exploratory, label-guided inspection, not an independently annotated anatomical benchmark.

## Conclusion

Native slices reveal useful local geometry that an opaque 3D surface can hide. Cases 185 and 327 have a separate small profile at the last anterior slice; case 205 has a plausible returning lip that remains connected. Case 164 does not show an equally convincing configuration at its annotated boundary. These observations support investigating the landmark using multiple planes, but do not establish a universally observable double-profile rule in this dataset.

The literal condition “two disconnected hippocampal mask components at the last anterior slice” holds in only 2/52 validation volumes. That condition is narrower than the anatomical description: two levels of a folded structure can remain connected. This count is **not** the prevalence of an identifiable uncal apex, and an ambiguous image does not establish anatomical absence, age-related disappearance, or an annotation error.

![Selected examples](../../../experiments/uncal_validation_profiles_20260921/comparison.png)

## Scope and method

- All 52 validation volumes from entry 0 of `datasets/Dataset101_MSD/splits_final.json`; no overlap with its 208 training volumes. Case identifiers denote dataset volumes, not independently verified subject identities.
- Split-file SHA256: `1d6a3fe993359ad85e9c28a29901162251c3359449b710c09cb9e92d66dda472`.
- Native image and label arrays agree in shape and affine. Their affine linear parts are identity and voxel spacing is 1 mm. This establishes native array alignment, not independent AC–PC alignment or original hemisphere provenance.
- All 52 cases were visually screened in seven coronal planes, from three slices anterior to three slices posterior to the best-fit reference-label interface. Raw MRI and thin label contours were displayed separately, with nearest-neighbour rendering and a fixed 1st–99th percentile intensity window per volume. There was no interpolation intended to create anatomical detail.
- Extended coronal, sagittal and axial sheets were generated for 017, 019, 033, 035, 037, 049, 052, 123, 157, 164, 185, 205, 261 and 327. Selected detailed sheets were inspected to investigate positive and ambiguous appearances; generation of a sheet does not imply exhaustive review of every plane.
- Blue = anterior label 1; orange = posterior label 2. Slice indices are **zero-based native indices**, before any viewer padding. Increasing native y is anterior. The last anterior slice is the minimum y containing label 1. On sagittal panels, the pink line is the fitted annotation interface, not an independently identified apex.

## Findings that can be revisited directly

| Volume | Last anterior y | Observation | Interpretation |
|---|---:|---|---|
| 185 | 27 | At y=27, one 78-voxel profile and a separate 6-voxel profile; the small profile disappears at y=26. Sagittal x=18 shows a small lip near the interface. | Good candidate for the multiple-profile cue; anatomical identity remains unverified. |
| 327 | 27 | At y=27, one 83-voxel profile and a separate 5-voxel profile; the small profile disappears at y=26. | A second geometric positive; two mask islands alone do not identify the uncus. |
| 205 | 27 | Indentation and upper protrusion in adjacent coronal planes; sagittal x=17 shows a returning upper lip. The coronal mask remains connected at y=27. | Plausible folded morphology missed by a disconnected-component test. |
| 164 | 31 | Broad continuous profile around y=31; the reviewed sagittal/axial views do not establish a convincing returning lip at the interface. Small holes/indentations at y=28–30 generate separated column runs posterior to the cut. | Ambiguous apex, and a counterexample to using generic holes or separated runs as a specific apex detector. |
| 017 | 26 | A superior shoulder reduces across the interface; the sagittal appearance is less decisive than in 205. | Weak candidate, not a confirmed landmark. |

The current viewer mesh code (`utils/fold_3d_viewer.js`) emits exposed native voxel faces without smoothing or decimation. It uses opaque surfaces and depth testing. Thus presentation and viewing angle can conceal a recess, but the inspected code does not remove a fold through downsampling. This code review does not prove the exact historical screenshot used an identical build.

## Quantitative checks and their limitations

The mask used for geometry is the union of both hippocampal classes. Eight-neighbour connectivity is used within each coronal plane. A separate-profile flag requires at least two components of at least two voxels each.

| Geometric diagnostic | Volumes |
|---|---:|
| Separate profiles at the actual last anterior slice | 2/52 |
| Separate profiles within ±3 slices of the fitted interface | 2/52 |
| Separate profiles anywhere on an anterior-containing plane | 7/52 |
| Separate profiles anywhere on a posterior-containing plane | 7/52 |
| At least two columns containing two tissue runs of length ≥2, at the fitted cut | 0/52 |
| That column-run condition within ±3 slices | 7/52 |
| Reference classes not exactly separable by one coronal plane | 11/52 |

The separate-profile count at the actual last anterior slice stays 2/52 for minimum component sizes 1, 2, 3 and 5. The two cases are always 185 and 327. The threshold check does not validate anatomical specificity.

The 11 nonplanar cases are 035, 049, 229, 244, 280, 296, 305, 317, 349, 358 and 394. Their minimum disagreement with a single coronal separating plane ranges from 3 to 51 voxels. Local label irregularity must be represented explicitly rather than silently assuming an exact global cut. No relabelling is justified by this check alone.

Synthetic checks confirmed that the diagnostic distinguishes separated profiles from a connected bridge, rejects a singleton under the two-voxel threshold, and preserves component counts under horizontal reflection. This checks the implementation, not its anatomical validity.

## Consequences for the LTN experiment

Do not enforce “every last-anterior slice has two disconnected components.” It would disagree with most reference masks, including a plausible connected-fold example. Do not replace it with “there is a hole”: case 164 illustrates posterior holes that are not specific to the annotated transition.

A defensible next experiment is a confidence-weighted landmark predicate derived from the MRI across neighbouring coronal and sagittal planes, with an explicit uncertain state. Independently review a small training subset for a returning lip/medial profile and its disappearance, without displaying the anterior/posterior colour boundary during initial landmark selection. Then measure distance to the existing class interface. This separates anatomical supervision from recovering the annotation one started with.

The present figures locate the annotation first and inspect nearby anatomy. They are useful for discovery, but cannot demonstrate independent apex detection or improved generalisation. Fold 0 validation has now informed method development; keep another split untouched for the eventual performance claim. No model, loss, or reference labels were changed during this audit.

## Reproduce and inspect

Implementation: [audit_validation_profiles.py](../../../thesis/new_constraints/uncal_fold/audit_validation_profiles.py).

```bash
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_validation_profiles \
  --details 017 019 033 035 037 049 052 123 157 164 185 205 261 327
```

All measurements: [audit.json](../../../experiments/uncal_validation_profiles_20260921/audit.json).

All 52 screening notes and linked sheets: [screening.md](screening.md).

Protocol context: [Woolard and Heckers, 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3289761/) describes the multiple-profile/medial-protrusion cue in a multi-planar tracing procedure. [MSD dataset description](https://arxiv.org/html/1902.09063v1) defines the last hippocampal head slice using the uncal apex. The geometry diagnostics here operationalise only simplified image properties, not that complete anatomical procedure.
