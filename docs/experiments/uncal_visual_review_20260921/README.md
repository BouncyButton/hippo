# Raw MRI uncal-fold annotation pilot — 2026-09-21

**Outcome: the provisional AI visual annotations failed the MSD label-agreement check and should not supervise training.** This is a useful negative result: converting an ambiguous visual impression into an LTN predicate does not make that predicate anatomically valid.

## What “independently annotated” means

The desired extra supervision is an observation of fold visibility/disappearance made from the MRI without using the existing anterior/posterior mask to select the answer. Copying the MSD division into a landmark target adds no independent anatomical evidence. An AI review without the masks is still not an independent qualified human rater. This reviewer already knew aggregate results and general positional context from preceding analyses.

The intended anatomical cue follows the [HarP user manual](https://www.hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/02_Appendix-II_HarP-UserManual.pdf): folded appearance in sagittal context and double-level appearance in coronal views. Its operational interpretation on these small T1 crops requires expert adjudication. The [MSD descriptor](https://arxiv.org/html/1902.09063v1) describes the head division using the uncal apex; agreement with the supplied segmentation is nevertheless a dataset compatibility check, not a separate anatomical reference.

## Procedure

- Predetermined random sample of 12 of the 208 fold-0 training crops; seed 20260922. No validation crops were used.
- Recorded case IDs, source image hashes, and split hash before review.
- Reviewed full raw coronal sequences and sagittal context; inspected image-selected detail windows. No pilot masks, model predictions, or detector scores were consulted before recording the annotations.
- Recorded 11 candidate last-visible slices with uncertainty intervals, selected slice-level observations, and one abstention. The latter has a search window only.
- Saved `annotations_frozen.json` and its SHA256 at 13:18:28 UTC, before opening the pilot labels for comparison at 13:19:19 UTC. This is an integrity record, not external certification of blinding.
- All annotations remain `eligible_for_training=false` and `expert_validated=false`. No original annotation was corrected after the label comparison.

Coordinates are native 1-mm RAS array indices. Moving anterior to posterior decreases y. A candidate last-visible slice c belongs to the anterior partition; its partition plane is c−0.5. Subjective uncertainty intervals are not calibrated confidence intervals. Unlisted slices are not annotated, and “fold not resolved” does not prove anatomical absence.

## Label-agreement results

Reference: best-fitting coronal cut assigning class 1 to y ≥ c and class 2 to y < c, minimizing disagreements within the supplied foreground. Four of the twelve masks do not admit a perfectly planar cut, so this reference is an approximation for those cases.

| Measure | Result |
|---|---:|
| Reviewed / candidate / abstained | 12 / 11 / 1 |
| Mean absolute difference | 5.27 mm |
| Median absolute difference | 6 mm |
| Mean signed difference | +5.27 mm, anterior |
| Within 1 mm / within 2 mm | 0/11 / 1/11 |
| Candidate intervals containing reference | 1/11 |
| Mean interval width | 4.09 mm |

| Case | AI candidate y | AI range | MSD reference y | Signed difference, mm |
|---|---:|---|---:|---:|
| 274 | 27 | 25–28 | 20 | +7 |
| 249 | 34 | 32–36 | 29 | +5 |
| 345 | 30 | 28–33 | 27 | +3 |
| 341 | 34 | 32–35 | 26 | +8 |
| 145 | 33 | 31–35 | 29 | +4 |
| 375 | 34 | 31–36 | 32 | +2 |
| 287 | 36 | 34–38 | 29 | +7 |
| 231 | 33 | 31–35 | 27 | +6 |
| 343 | 31 | 29–33 | 25 | +6 |
| 257 | 34 | 32–36 | 31 | +3 |
| 321 | 30 | 28–33 | 23 | +7 |
| 334 | Abstain | 28–33 search only | 23 | — |

All eleven candidates were anterior to the reference. One possible explanation is that the reviewer followed a superior cap disappearing earlier than the actual uncal endpoint. This is a hypothesis, not an established anatomical identification. The observations do not justify assuming that MSD is wrong, applying a fixed offset, or fitting a new detector to these candidates. Because the reviewer has now seen the reference cuts, subsequent revisions of these cases would be unblinded.

## How valid annotations would enter the constraint

After qualified review, use visibility observations to supervise an image-based fold predicate and an uncertain disappearance location. For a validated candidate interval [l,h], the corresponding partition-plane interval is [l−0.5,h−0.5]. Within a local hippocampal foreground region, the soft rules are:

- tissue anterior to the entire interval ⇒ anterior class;
- tissue posterior to the entire interval ⇒ posterior class;
- tissue inside the interval ⇒ no landmark-derived class constraint.

The existing `UncalIntervalLogLTNLoss` implements this interval-based class constraint. Its mathematical validity does not validate its landmark input. A learned fold detector needs external supervision or a separately validated anchor: an unconstrained fold predicate can collapse to avoid a logical penalty. Do not turn lack of fold visibility into a universal posterior label, because the cue need not be resolved on every anterior slice.

The next necessary evidence is an independent qualified review of raw images, preferably with a second rater on a subset, followed by detector evaluation on untouched cases. Generalization must then be measured with controlled segmentation experiments and local boundary metrics. This training-only pilot establishes neither anatomical accuracy nor generalization improvement. No segmenter was retrained or training loss modified in this pilot.

## Artifacts and reproduction

Generated artifacts are under `experiments/uncal_visual_review_20260921/` (ignored by git): raw coronal/sagittal plates, detail plates, `manifest.json`, `annotations_frozen.json`, `annotations.csv`, `freeze.json`, and `label_comparison.json`.

`review_raw.html` presents full raw images without AI candidates or MSD comparisons. `expert_annotation_template.json` is a blank template for a new reviewer; save completed reviews separately. A reviewer should avoid the comparison report and frozen AI annotations until submitting their own judgment. For final annotation use the original NIfTI data in an MRI viewer, because montage resolution and viewing controls are limited.

```bash
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.prepare_visual_review
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.build_visual_review_packet
# Only after annotations are recorded and frozen:
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_visual_annotations
```

The audit completed successfully, checking annotation/source hashes, training membership, matching image/label affines and shapes, 1-mm RAS orientation, and interval bounds before scoring.
