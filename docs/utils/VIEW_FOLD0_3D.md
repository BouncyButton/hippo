# Offline 3D validation viewer

Run from the repository root, in the project's Python environment:

```bash
python utils/view_fold0_3d.py
```

This loads all **52 fold 0 validation cases** from `Dataset101_MSD/splits_final.json`,
uses the saved `baseline_seed0/error_maps` predictions, and opens the viewer in your
default browser. The initial patient is the one with the most A/P swaps (164 in
this baseline). It generates `evaluation_output/fold0_3d/index.html` and a sibling
`.provenance.json` containing input paths, hashes, alignment details, and metrics.

The generated HTML works offline. You can copy it to another PC and open it
directly, without Python or a running server. To regenerate it on another PC,
copy this repository's script and sibling HTML/JS templates, together with the
dataset and prediction files. Dependencies: `numpy`, `nibabel`, and `scipy`.
The browser must support WebGL for 3D; the MRI slice panel also works without it.

```bash
python -m pip install numpy nibabel scipy
python utils/view_fold0_3d.py --initial-case 164
```

## Other saved predictions

```bash
python utils/view_fold0_3d.py \
  --dataset-dir /path/to/Dataset101_MSD \
  --prediction-dir /path/to/error_maps \
  --prediction-name "My model" \
  --fold 0
```

On Windows, use the same command on one line and quote paths containing spaces.
Paths explicitly supplied on the command line are relative to the current working
directory; defaults are relative to the repository, not to a particular computer.

Each validation case needs exactly one of:

- `hippocampus_164.npz`, containing `ground_truth` and `prediction` arrays. These
  are the audited predictions produced by `evaluation/fold0_voxel_errors.py`.
  Native or center-padded grids are accepted only when the entire reference mask
  exactly matches the padded native annotation. Errors outside the MRI crop are
  preserved and the unavailable MRI region is shown as a checkerboard.
- `hippocampus_164.nii.gz` or `.nii`: predicted labels on exactly the same shape
  and affine as the corresponding native MRI. This supports exported nnU-Net
  masks and other predictions already mapped back to the native grid.

Class IDs must be `0=background`, `1=anterior`, `2=posterior`. Inputs must use an
axis-aligned RAS grid, as the current MSD files do. Bare `case_0018_pred.npy` files
are not accepted because they lack a verified spatial reference; use the audited
NPZs or a correctly restored native-space NIfTI. No inference is run.

## Controls

- **Patient / sort:** select any validation patient, ranked by A/P swaps, total
  errors, or patient ID.
- **Coronal slice:** move in the anterior-to-posterior direction, retaining
  original zero-based NIfTI `y` indices. Buttons move one slice at a time.
- **Display:** annotation A/P labels, binary foreground, predicted A/P labels,
  predicted mask plus errors, or errors on neutral foreground.
- **Predicted mask + errors:** correct predictions keep their A/P colors. Extra
  foreground, missed foreground, and A/P swaps receive distinct colors. The
  comparison surface includes both masks so missed voxels remain visible.
- **View:** rotate by dragging, or select sagittal, coronal, or axial orientation.
- **MRI plane in 3D:** place the actual MRI slice inside the volume.
- **Annotation contour:** overlay the reference foreground outline on the MRI.

An opaque 3D surface can hide interior errors; the synchronized MRI slice shows
all errors on that plane. The dashed reference cut is shown only if the annotation
has an exactly planar, ordered A/P partition. The surface uses exact exposed
voxel faces with physical spacing, without smoothing or mask resampling. MRI
display uses a per-case 1st–99th percentile window and 8-bit display intensities.

## Additional options

```bash
python utils/view_fold0_3d.py --ground-truth-only
python utils/view_fold0_3d.py --fold 0 --no-open --output evaluation_output/viewer.html
python utils/view_fold0_3d.py --help
python -m pytest utils/test_view_fold0_3d.py -q
```

Generated pages contain MRI and mask data. Keep them out of Git, as with the
underlying datasets. The default `evaluation_output/` location is already ignored.
