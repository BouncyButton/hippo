# HFH hippocampus segmentation

Downloaded from the [official NITRC release](https://www.nitrc.org/projects/hippseg_2011/)
on 11 November 2026. The original `downloads/HFH.rar` and extracted `raw/`
are retained locally and excluded from Git.

The release contains 50 T1 MRI subjects: 25 labelled training subjects and
25 test subjects without public labels. The supplied ReadMe describes
40 subjects with temporal lobe epilepsy and 10 nonepileptic subjects.
The labels describe the hippocampus, not anatomical subfields.

| Header voxel spacing, array axis order (mm) | Training | Test | Total |
| --- | ---: | ---: | ---: |
| 0.781 × 2.0 × 0.781 | 14 | 15 | 29 |
| 0.859 × 2.0 × 0.859 | 1 | 0 | 1 |
| 0.390 × 2.0 × 0.390 | 10 | 10 | 20 |

These are stored voxel spacings, not a claim about effective acquisition
resolution. Matrices are 256 × N × 256 or 512 × N × 512, with variable N.
The source ReadMe identifies the 20 finer-grid scans as coming from a 3T system.

## Prepared data

- `imagesTr/`: 25 full-brain native-grid NIfTI images.
- `labelsTr/`: 25 binary hippocampus masks, merging original values 1 and 2.
- `imagesTs/`: 25 unlabelled official test images.
- `splits_final.json`: five subject-disjoint folds using only the labelled pool;
  each has 20 training and five validation subjects, including three 256-matrix
  and two 512-matrix validation subjects.
- `split_provenance.json`: seed 42 and the official train/test membership.

Conversion removes a singleton fourth dimension where present. It does not
resample, crop, normalize, or register images. The original two label values
remain in `raw/Train/Labels/`. Every saved image is reloaded and checked against
the original voxel values and affine; each binary mask is checked against the
union of the original labels. The inventory records source/output SHA-256 hashes.

Analyze headers do not independently establish scanner-world orientation or
anatomical laterality. The conversion preserves NiBabel's derived affine and
does not imply verified MNI registration or anatomical left/right. Preview plane
names follow these header conventions. Confirm orientation before applying
direction-dependent constraints.

To reproduce after extracting the archive into `raw/`:

```bash
.venv/bin/python datasets/Dataset104_HFH/prepare_hfh_dataset.py
.venv/bin/python reports/hfh_dataset_20261111/render_preview.py
```

See `reports/hfh_dataset_20261111/inventory.json` for per-subject shapes,
spacing, hashes, label counts, and verification results. Experiment-specific
cropping, compute budgets and low-data support selection are not defined here.
Local evaluation is possible on CV validation subjects; official test-set Dice
cannot be calculated without the withheld labels or evaluation by the provider.

Citation requested by the dataset authors: K. Jafari-Khouzani, K. Elisevich,
S. Patel, and H. Soltanian-Zadeh, “Database of magnetic resonance images of
nonepileptic subjects and temporal lobe epilepsy patients for validation of
hippocampal segmentation techniques,” Neuroinformatics, 2011.
