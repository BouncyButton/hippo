# Two-step outer-boundary bands

This constraint adds balanced foreground/background supervision on a fixed
ground-truth band. It merges anterior and posterior into foreground, constructs
exactly two 6-connected morphological steps inside and outside the final
transformed label, and applies equal-weight means to the two sides. The default
`--bands-focal-gamma 0` is the original BCE exactly; positive gamma multiplies
each voxel BCE by `(1 - p_t) ** gamma` before the existing reductions.

For the fold-0 evidence, full mathematical derivation, gradient behavior, weight
calibration, and evaluation contract, see
[`SCIENTIFIC_RATIONALE_AND_FORMULATION.md`](SCIENTIFIC_RATIONALE_AND_FORMULATION.md).

## Fixed experiment contract

- band steps: exactly `2`;
- calibration cases: at most `32` deterministic training cases;
- target gradient ratio: `0.10`;
- safety gradient ratio: `0.50`;
- calibration source: epoch-5 checkpoint from a matching `none` run;
- focal gamma: explicit and bound into both calibration and run provenance;
- invalid patients: excluded and reported;
- all-invalid/degenerate calibration: nonzero exit plus a saved failure report.

The source checkpoint must be the full `checkpoint_latest.pt`, not a standalone
state dictionary. Calibration verifies its epoch, dataset, fold, spatial size,
resize mode, constraint weights, dataset pickle, split file, embedded metadata,
and neighbouring `config.json`.

## Calibration and training

First create the separate Dice-only calibration checkpoint:

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set none \
  --epochs 5 \
  --output-dir /path/to/msd_fold0_none_calibration
```

Then calibrate using training cases only:

```bash
python thesis/new_constraints/bands/calibrate_weight.py \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --fold 0 \
  --checkpoint /path/to/msd_fold0_none_calibration/checkpoint_latest.pt \
  --checkpoint-config /path/to/msd_fold0_none_calibration/config.json \
  --output /path/to/bands_calibration.json \
  --bands-focal-gamma 1 \
  --device cuda
```

Calibration defaults to CUDA AMP so the checkpoint, calibration, and bands run
share the same logit rounding policy. Use `--no-amp` only when both the source
`none` run and the final bands run were also launched without AMP.

Finally use `recommended_bands_weight` from the successful JSON report:

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set bands \
  --bands-weight WEIGHT_FROM_CALIBRATION_JSON \
  --bands-calibration-json /path/to/bands_calibration.json \
  --bands-focal-gamma 1
```

The training command rejects failed or internally inconsistent reports, changed
source checkpoint/config files, stale inputs, wrong-fold or wrong-dataset reports,
wrong-runtime/device reports, different class groupings, and a different focal
gamma. The numeric weight must match the report exactly. For positive focal
gamma, the frozen Dice-only checkpoint keeps its original source identity while
the calibration report separately binds the revised loss source; the model and
data-pipeline source must remain identical.

These checks protect scientific provenance and accidental mutation. They are not
a cryptographic signature against a malicious local operator who can replace a
report and all referenced artifacts; calibration inputs are trusted local
experiment files.

The training pipeline retains its existing five-epoch linear constraint ramp:
`0.2, 0.4, 0.6, 0.8, 1.0`. It does not contain five Dice-only epochs.

## Tests

```bash
PYTHONPATH=. python -m pytest thesis/new_constraints/bands/test_outer_boundary.py
```
