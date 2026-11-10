# GeoFWI3D ground-truth preview

These are actual dataset samples `model_0001` and `model_0002`, retrieved from a bounded prefix of the public batch-0 archive. They are generated geological scenes, not field acquisitions or model predictions. No training job was submitted.

- Source: https://zenodo.org/records/20148778
- Metadata and authors' mask rule: https://github.com/aaspip/geofwi3d
- Paper: https://arxiv.org/html/2610.01033v1
- Native array convention: little-endian float32, C-order `(X,Y,Z)`, shape `(96,96,96)`.
- Target displayed faithfully follows the authors' rule `rgt3d == 0`, with all salt material sharing label 1.
- The inline preview samples every second voxel, yielding `(48,48,48)`. Its 3D faces represent the sampled voxel mask. Native arrays are preserved locally.
- Display amplitudes are symmetrically clipped at the per-volume 99.5th percentile of absolute amplitude and quantized to 8 bits; these are display transformations only.
- Optional inner/outer bands use 6-connected morphology, two native voxels, excluding the outer two-voxel observation frame. They illustrate training emphasis, not a separate segmentation output.

## Label audit to retain before any training preparation

Both examples contain small disconnected foreground components. Most agree with the supplied salt velocity and should not automatically be deleted as annotation errors. Each example also has exactly one RGT-zero voxel whose velocity is background rather than salt. `model_0001/label_audit.json` and `model_0002/label_audit.json` record the comparison. The displayed labels have not been changed. A full preparation should audit the implicit RGT rule, especially for nominally salt-free cases, rather than silently assuming every zero means salt.

Salt intersects observation-cube faces in these samples. These faces must not be treated as geological boundaries by a boundary loss; the current hippocampus loss cannot simply be reused without checking this condition.

The viewer was inspected in Chrome at 736px and 320px widths. Both cases, slice changes, rotation and boundary overlay rendered without JavaScript errors or horizontal overflow. These checks verify the preview, not model performance or full dataset quality.
