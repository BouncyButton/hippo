# Prediction-guided MRI straightening prototype

This geometry prototype samples the actual MSD MRI into a second 3D view and
checks its correspondence with the original volume. It does not train a model.
The three inspected cases are hippocampus_164, hippocampus_017, and hippocampus_330.

## Construction

The frozen full-data depth-only SwinUNETR prediction defines a longitudinal plane
and a smooth transverse centreline. With longitudinal position `s`, transverse
position `t`, and signed distance from the plane `d`, the inverse transform is
`local = (s, t + c(s), d)`. The forward transform subtracts `c(s)` from `t`.
This transverse shear has an analytic inverse and determinant one. It preserves
signed depth and physical volume, but changes angles and distances. It is not an
arc-length, geodesic, or laminar unfolding of the hippocampus.

The complete MRI intensity field is sampled inside the prediction-derived grid;
it is not masked to predicted foreground. The grid covers the predicted region
plus 8 mm of context and a 2 mm sampling halo, with 64 samples per axis and spacing
of at least 1 mm. Annotation is loaded only after geometry and MRI sampling are
constructed, for overlays and coverage checks. Background outside the MRI field
of view is masked explicitly.

## Validation

The tests cover coordinate inversion, preservation of signed depth, numerical
Jacobian, ROI coverage, interpolation, and independence from predicted A/P labels.
The build measures MRI reconstruction on native voxel centres, with a matched
rotation-only control, and exports forward/inverse maps and validity masks.
Intensity error is RMSE divided by the shared MRI P1–P99 intensity range.

All annotated foreground, predicted foreground, and false-negative tissue voxels
have valid reconstruction support in these three cases. Case 164 has 4.59% MRI
reconstruction error versus 4.64% for the rotation-only control. These numbers
describe interpolation, not segmentation improvement. Geometry source and file
hashes are recorded in the generated audit.

## Reproduce locally

Run from the repository root, using the existing environment:

```sh
rtk proxy env MPLCONFIGDIR=/private/tmp/hippo-input-prototype-mpl .venv/bin/python -m scripts.unwrapped_input_prototype.test_geometry
rtk proxy env MPLCONFIGDIR=/private/tmp/hippo-input-prototype-mpl .venv/bin/python -m scripts.unwrapped_input_prototype.build
rtk proxy .venv/bin/python -m http.server 8767 --bind 127.0.0.1 --directory evaluation_output/unwrapped_input_prototype
```

The build requires Dataset101_MSD, the downloaded full-data depth prediction
archive under `evaluation_output/unwrapped_input_prototype/source/`, the existing
full-data completed snapshot, and the existing visualization dependencies in
`/private/tmp/hippo-unwrap-deps`. Open `http://127.0.0.1:8767/index.html`.
Generated medical volumes and visualization assets remain under ignored
`evaluation_output/`.

## Before a learning experiment

A 5% experiment needs a guide trained only on that subset. Training-case guides
should be cross-fitted, with validation annotations excluded from input geometry.
The proposed comparison holds the original Dice + depth loss fixed and compares
the native model, a second branch with a rotation-only MRI view, and a second
branch with this straightened MRI view. Mapped-back features would enter the
native decoder. No training jobs are submitted by this prototype.
