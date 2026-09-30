# Case 164: visible fold versus annotated division

This is label-guided exploratory inspection, not an independent uncal-apex annotation.
Blue denotes anterior and orange posterior, matching the user's current screenshots.

## Native data

The aligned image and label arrays are 41 x 48 x 47 with 1-mm axis-aligned
affines. The dataset maps anterior to label 1 and posterior to label 2.
The last labelled anterior slice moving posteriorly is native y=31 (190 anterior
voxels, no posterior voxels); y=30 has 165 posterior voxels and no anterior voxels.
The class interface therefore lies between these slices, at y=30.5 in native
voxel coordinates. This identifies the annotation, not its anatomical justification.
The crop affine does not establish AC-PC alignment or recover the original
whole-brain location/hemisphere by itself.

Generated and visually inspected:

- `experiments/uncal_interface_inspection_20260921/case164/coronal.png`:
  raw MRI and labels, native y=38 through 26, anterior to posterior.
- `experiments/uncal_interface_inspection_20260921/case164/sagittal.png`:
  raw MRI and labels across the full labelled x extent, x=6 through 28.

An obvious schematic-like returning hook is not apparent in the inspected mask
sections. The raw MRI contains additional structure above the labelled contour;
its anatomical identity is not established. No such structure has been designated
as uncus, amygdala, vessel, or an annotation omission. These images do not justify
concluding that the uncus is anatomically absent, that age caused the appearance,
or that the reference boundary is wrong. No confident independent apex location
was obtained in this inspection.

## What the sources establish

Poppenk (2020), DOI 10.1002/hipo.23196, Figure 1 illustrates how displacement of
the uncus endpoint could change anterior/posterior volume assignment. It is a
schematic, not a required silhouette of every mask. Section 2.2 describes raters
identifying the last posterior slice showing the uncus in AC-PC space, supported
by sagittal folded or coronal double-level appearance. Training addressed
nearby distractors such as blood vessels. Initial reliability exercises used
10 HCP participants (20 hippocampi); reported final exact-slice agreement was
0.80-0.85, with ICC 0.97-0.99. These are rater-agreement statistics, not proof
of histological accuracy or a validation of an automated shape rule.

The Discussion explicitly states that the analysis is cross-sectional and
requires longitudinal confirmation of within-person change. It does not establish
complete disappearance of the uncus with normal aging or an explanation for 164.
There is no requirement that the landmark lie at the hippocampal midpoint.

The MSD data description cites Pruessner (2000) and Woolard & Heckers (2012):
https://arxiv.org/html/1902.09063v1 (Task04_Hippocampus).
It explicitly defines the final head slice by the uncal apex.

Woolard & Heckers' methods describe sagittal tracing followed by coronal
refinement, identification of multiple hippocampal sections plus a medial
protrusion toward the crus cerebri, and sagittal verification of the division:
https://pmc.ncbi.nlm.nih.gov/articles/PMC3289761/ .
This establishes the cited workflow, not the individual rater's decision for 164.

## Consequence for the constraint

A binary-mask hook or two-component requirement is not yet justified as a
universal constraint. Landmark recognition must be grounded in raw MRI and
spatial context, with visibility/uncertainty distinct from anatomical absence.
The annotated cut remains available for supervised boundary learning, but does
not independently label the uncal tissue itself. No loss, labels, checkpoint,
or training configuration was changed during this inspection.
