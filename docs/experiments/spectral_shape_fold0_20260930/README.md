# Spectral hippocampal shape and error analysis

**Completed: anatomical PCA directions can express useful residual corrections,
but projecting a prediction onto the anatomical shape model makes it worse.**
The positive results below fit coefficients using validation annotations and
are capacity tests, not learned predictions. The family of modes itself was
learned exclusively from the 208 training cases.

This is a representation diagnostic, not a new neural-network training run.
The aim is to distinguish compact anatomical variation from compact correctable
segmentation errors. All numerical jobs run through Slurm `sbatch`.

## Frozen protocol

- Same canonical Dice-only, early-stopped, unaugmented SwinUNETR and original
  fold-0 split: 208 training and 52 validation cases. Hash checks reuse the
  spline pilot's canonical checkpoint, historical dataset, and split checks.
- Generate frozen predictions in CUDA FP32. Whole-hippocampus Dice excludes
  background and merges anterior/posterior labels.
- Translate each mask pair by the integer displacement placing the prediction
  centroid near the centre of an 80-cubed, 1 mm grid. Apply the exact same shift
  to its annotation. No subvoxel resampling, scale normalization, or
  annotation-derived alignment. Keep common dataset axes; residual pose and
  left/right variation are limitations rather than silently removed anatomy.
  Assert that no foreground is cropped and that shifted edit bands agree.
- Compute signed distance, clipped at +/-8 mm to reduce influence of distant
  background. The boundary is its zero level set.
- Fit two separate PCAs using only training cases: annotated signed-distance
  fields and annotation-minus-prediction distance residuals. Fit before band
  masking, to avoid treating a hard mask edge as intrinsic frequency content.
- Compare ranks 8, 32, 128. Fixed cosine basis uses DCT-II frequencies ordered
  by squared frequency norm and then lexicographic tie-breaking; also test 512.
- Report train and validation annotation reconstruction Dice and residual
  projection errors, training PCA variance spectra, and DCT error energy.
- Oracle correction coefficients minimize squared residual in the existing
  prediction-derived two-step, six-connected bands. Column RMS normalization
  precedes fixed ridge 0.001. Every family gets the same fixed training mean
  error offset. Compare residual PCA, anatomical PCA directions, and DCT.
- Validation annotations are used to fit oracle coefficients. These results
  measure representation capacity on held-out shapes, NOT predictive accuracy.
- Two annotation-free controls: add the mean training error; project the
  predicted distance field onto the training annotation PCA space. Neither
  control sees the validation annotation.
- Identity and perfect band-edit oracle define reference endpoints. Every
  corrected mask retains the original foreground outside the prediction bands.
  These are implicit boundary corrections, not invertible coordinate maps.
- Report all ranks and cases, Dice, boundary-voxel ASSD/HD95, errors fixed and
  introduced, and connected-component changes. No validation-selected winner.
  Connected-component stability alone is not a topology guarantee.

## Interpretation limits

The segmenter's training predictions are in-sample, and fold 0 has been used
repeatedly for development. PCA variance explained is not segmentation quality.
A distance-field PCA reconstruction need not itself be a true signed-distance
field; we use its zero level only. Clipping and pose conventions influence the
observed spectra. Large oracle gains establish capacity, not coefficient
predictability. Annotation-free controls are distinguished in every report.

Source: `semantic_constraints/spectral_shape/`.
Launcher: `scripts/run_spectral_shape.sh`.
Raw data, checkpoints, cached volumes, and figures remain untracked.

## Execution

Cluster directory: `/home/3160552/spectral_shape_fold0_20260930_01`.
Slurm analysis job: 677487. Independent saved-mask verification job: 677490,
submitted with an `afterok` dependency. No numerical analysis runs on the
login node or locally.

Unit checks cover the cosine basis against SciPy's inverse DCT, low-rank PCA
reconstruction and an unseen orthogonal component, coefficient least-squares
recovery, and exact integer translation of masks and bands.

The report also examines training-only correlations between the first eight
shape/error scores and annotated volume, axis extents, and prediction volume
bias. Correlations are descriptive; mode signs are arbitrary. The first three
anatomical and residual modes are visualized on a common slice.

## Results

Analysis job 677487 completed with exit 0:0 in 5m23s. Verification job 677490
completed with exit 0:0 in 5s. All four numerical unit checks passed. The
baseline Dice exactly matches the preceding CUDA FP32 spline experiment.

### Anatomical variation is more compact than error variation

| Modes | Training anatomical variance explained | Training error variance explained |
| --- | ---: | ---: |
| 8 | 77.66% | 21.32% |
| 32 | 92.23% | 46.89% |
| 128 | 98.75% | 86.14% |

These are pooled training covariance fractions. They do not imply that the
reconstructed boundaries are accurate. Whole annotated shapes reconstructed
from 128 anatomical modes have mean Dice 0.99401 on training cases but only
0.91304 on validation cases. At 32 modes, those values are 0.92483 and 0.89595.
Fine subject-specific structure is not fully captured by this linear model.

The second anatomical mode correlates with annotated volume (r=0.713).
The first correlates with extent along axes 1 and 2 (r=0.792 and -0.858).
Because only translation was standardized, this can mix pose and anatomical
shape variation; it must not be called a pure biological mode. Eigenvector
signs are arbitrary. Error modes have more distributed structure; no broad
claim that high-frequency content is noise is supported by these data.

On validation residuals, the first 128 DCT frequencies contain on average
18.95% of full-field residual energy; the first 512 contain 46.86%. Clipping,
the 80-cubed canvas, and spatial support influence these fractions. Band-fitted
coefficients below are distinct from simply truncating a full-field DCT.

### Oracle correction capacity on the 52 validation cases

All entries in this table fit each case's coefficients using its annotation.
Whole-hippocampus Dice excludes background. Original Dice is **0.900978**.

| Coefficients | Anatomical PCA directions | Error PCA directions | Cosine modes |
| --- | ---: | ---: | ---: |
| 8 | 0.904683 | 0.902411 | 0.905493 |
| 32 | 0.914494 | 0.905971 | 0.912323 |
| 128 | 0.923685 | 0.912678 | 0.918080 |
| 512 | Not tested | Not tested | 0.924074 |

Anatomical PCA at 32 and 128 modes improves all 52 cases. At 128 modes it fixes
10,937 errors and introduces 2,749: a net reduction of 8,188. Mean boundary-voxel
ASSD decreases from 0.421466 to 0.316143 mm, and HD95 from 1.141528 to 1.0 mm.
Cosine modes at 512 reach similar Dice using four times as many fitted
coefficients. This is a descriptive comparison under the chosen alignment,
SDF clipping, band objective, and ridge penalty, not universal superiority.

The perfect band-edit ceiling is 0.998265 Dice. Therefore even these oracle
spectral fits leave considerable residual error; the study does not establish
that a very small basis captures almost every mistake.

### Controls without validation annotation access

| Method | Mean validation Dice | Improved / worsened |
| --- | ---: | ---: |
| Original prediction | 0.900978 | — |
| Add mean training error | 0.900978 | 0 / 0 |
| Project prediction into 8-mode anatomical model | 0.849835 | 0 / 52 |
| Project prediction into 32-mode anatomical model | 0.875390 | 0 / 52 |
| Project prediction into 128-mode anatomical model | 0.882857 | 1 / 51 |

The mean error field changes no hard labels. Projection onto a plausible
shape family does not reliably recover the case-specific true boundary. The
positive oracle result uses the anatomical modes as correction directions
around the existing segmentation; it does not replace that segmentation with
a reconstruction of an average shape.

### Case 164 and geometry checks

For case 164, Dice rises from 0.907598 to 0.932098 with 128 anatomical modes;
ASSD falls from 0.436413 to 0.308258 mm. It fixes 296 errors and introduces 77.
With 512 cosine modes, Dice is 0.932560. These are annotation-guided fits.

No method changes foreground labels outside the two prediction-derived bands.
Connected-component counts change in 12/52 anatomical-PCA-128 outputs, 20/52
error-PCA-128 outputs, and 6/52 cosine-512 outputs. A change is not necessarily
an error (the perfect band oracle changes component counts in 3/52 cases), but
these are not topology-preserving deformations. No claim about holes or mesh
injectivity is made. The result JSON's `topology_changed_cases` field counts
only connected-component changes.

The independent verifier checked all 832 metric rows and summary arithmetic,
recomputed bands and band ceilings for all 52 cases, and rechecked 66 saved
output masks (error-PCA-32 for every case plus the other saved case-164 methods).
It confirmed split disjointness, saved-mask preservation, and baseline parity.
Both fitted PCA bases are orthonormal to maximum error below 5e-7.

## Interpretation and next experiment

This supports using a training-derived anatomical basis to parameterize
**residual corrections**, with image information choosing the coefficients.
It does not support replacing predictions with their nearest low-rank shape
or claim that a coefficient regressor will achieve the oracle gains.

A reasonable next diagnostic is a regressor for 32 anatomical correction
coefficients, first checking deliberate fitting on a small training cohort.
Only then would a full-cohort training run test transfer. This additional
training has not been launched.

Artifacts are retained locally in `results_677487/`: full protocol, spectra,
per-case metrics, reconstruction/energy measurements, mode-property scores,
the first eight mode arrays, verification, saved masks, and `overview.png`
and `modes.png`. Figures were inspected; plotting-only batch job 677504
shortens mode titles to avoid overlap, without changing numerical results.
