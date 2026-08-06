# Two-step outer-boundary bands

This constraint adds balanced foreground/background supervision on a fixed
ground-truth band. It merges anterior and posterior into foreground, constructs
exactly two 6-connected morphological steps inside and outside the final
transformed label, and applies equal-weight BCE means to the two sides.

## Fixed experiment contract

- band steps: exactly `2`;
- calibration cases: at most `32` deterministic training cases;
- target gradient ratio: `0.10`;
- safety gradient ratio: `0.50`;
- calibration source: epoch-5 checkpoint from a matching `none` run;
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
  --device cuda
```

Finally use `recommended_bands_weight` from the successful JSON report:

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set bands \
  --bands-weight WEIGHT_FROM_CALIBRATION_JSON
```

The training pipeline retains its existing five-epoch linear constraint ramp:
`0.2, 0.4, 0.6, 0.8, 1.0`. It does not contain five Dice-only epochs.

## Tests

```bash
PYTHONPATH=. python -m pytest thesis/new_constraints/bands/test_outer_boundary.py
```
