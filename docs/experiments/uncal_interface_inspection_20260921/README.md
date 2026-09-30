# Inspecting anatomy at the annotated A/P interface

This is the user's proposed **label-guided anatomical discovery**: start with the
last anterior voxels, inspect the raw MRI at the same coordinates in three planes,
and identify candidate image features. It is not a blinded annotation exercise,
training experiment, or independent landmark validation. Using labels to guide this
discovery is intentional. Any resulting detector must later be tested without those
labels supplying its location.

## Case 017: correspondence with the supplied screenshots

The user identified the screenshots as fold-0 validation case `hippocampus_017`.
The native volume is 35×48×32; the viewer uses a 64³ padded volume. The padding
offset is `(14,8,16)`. Cropping the saved fold-0 ground truth back to native space
was verified to reproduce the NIfTI labels exactly.

| Location | Native indices | Padded viewer indices |
|---|---|---|
| Screenshot sagittal plane | x=14 | x=28 |
| Screenshot axial plane | z=13 | z=29 |
| Last anterior coronal plane | y=26 | y=34 |
| Screenshot-plane intersection on the interface | (14,26,13) | (28,34,29) |
| Selected upper-side projection candidate | (18,26,15) | (32,34,31) |

Coronal y=26 contains 88 anterior and zero posterior voxels. The next posterior
slice y=25 contains 74 posterior and zero anterior voxels. Thus this case has a
clean annotated coronal division. The whole coronal partition is much larger than
the anatomical feature that might have justified choosing it.

## What the linked views suggest

At the central interface voxel `(14,26,13)`, the MRI shows continuing tissue rather
than an obvious sharp intensity boundary matching the red/blue division. The
surrounding contour is more informative than the voxel's intensity alone.

A specific candidate appears on the high-x side of the displayed coronal section:
the superior extent of the labelled tissue recedes as we cross the annotation
posteriorly. At x=18 the uppermost foreground z coordinate is:

| Coronal y, anterior → posterior | 28 | 27 | 26 | 25 | 24 |
|---|---:|---:|---:|---:|---:|
| Uppermost foreground z at x=18 | 16 | 16 | 15 | 12 | 12 |

The voxel `(18,26,15)` is anterior; `(18,25,15)` is background, while lower tissue
at x=18 continues posteriorly. This is an example of why examining only voxels
where A directly touches P can miss a terminal protrusion: its next posterior
neighbour can be background instead.

In the raw images, the candidate sits beside a dark cleft/space along the superior
side contour. Sagittal x=18 supplies continuity and the local contour change;
axial z=15 supplies the side contour; coronal y=26 supplies its cross-section.
The raw neighbouring sections are included to inspect whether that appearance
corresponds to a terminating folded component. **The mask's endpoint is established;
its identification as the uncal apex is not.** Partial volume, surrounding tissue,
and the chosen segmentation contour remain alternative explanations.

This gives a concrete hypothesis to investigate: a small superior-side component
terminates posteriorly while the main hippocampal body continues. Its 3-D continuity
and surrounding cleft may be more informative than mean intensity or an aggregate
coronal shape score. It is not yet a dataset-wide rule. In particular, a last-visible
landmark can be a small remnant, rather than a fully developed double-level profile.

## Other inspected cases

Linked and raw-neighbour plates were also inspected for 001, 274, 345 and 321.
Their central interfaces similarly do not establish a universal raw-intensity edge.
The label interfaces occupy native y=31, y=20, y=26–27 and y=22–24 respectively.
The latter two illustrate why the full interface and neighbouring sections matter.
No cross-case landmark annotations or consistency claims were generated from these
four examples. Case 017 is now explicitly a development/discovery example despite
its membership in the saved validation fold.

## Artifacts

Generated files are under `experiments/uncal_interface_inspection_20260921/`:

- Top-level: four comparison cases, each with linked raw/labelled views, raw
  neighbouring sections, and manifest coordinates/counts.
- `case017_screenshot/`: exact sagittal/axial screenshot planes, intersecting at
  the last annotated anterior slice.
- `case017_projection/`: linked views centred on `(18,26,15)`, plus adjacent raw
  sagittal, coronal and axial sections.
- `case017/`: initial interface-centroid inspection.

The cyan crosshairs always indicate the same native voxel. Yellow outlines indicate
anterior voxels directly touching posterior along y; they are not apex annotations.
The projection candidate can lie outside that yellow outline.

```bash
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.inspect_annotated_interface
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.inspect_annotated_interface \
  --cases 017 --anchor 14 26 13 \
  --output-dir experiments/uncal_interface_inspection_20260921/case017_screenshot
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.inspect_annotated_interface \
  --cases 017 --anchor 18 26 15 \
  --output-dir experiments/uncal_interface_inspection_20260921/case017_projection
```

All plates were visually inspected. Image/label affine agreement, native 1-mm RAS
axes and selected voxel labels were checked during generation. No detector, LTN
loss, model checkpoint or training configuration was changed in this inspection.
