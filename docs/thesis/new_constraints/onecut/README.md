# Outer-surface one-cut LogLTN pilot

This is the trainable form selected by the 52-case pre-training audit in
`thesis/Surface-normal ordinal LogLTN`. It supervises only the outer union
`H = anterior or posterior`; the A/P interface is deliberately excluded from
the first pilot because its counterfactual swap improvement came with a small
outer-surface cost.

For every valid ground-truth surface face, a spacing-aware outward normal ray
is sampled from -3 to +3 mm every 0.5 mm. Each candidate cut within ±1 mm of
the annotated surface defines one exact assignment: all samples before the cut
are hippocampus and all samples after it are background. The objective is the
negative log probability mass of the allowed assignments, divided by the
number of samples, averaged over rays and then patients.

## Locked pilot sequence

1. Train a fresh five-epoch `none` arm from scratch with the exact current
   source, inputs, AMP policy, seed, and GPU type.
2. On at most 32 deterministic fold-0 training cases, measure Dice and one-cut
   logit-gradient RMS from that frozen checkpoint.
3. Set the one-cut weight to the smaller of:
   - the weight matching median one-cut RMS to 10% of median Dice RMS;
   - the weight limiting the 95th-percentile one-cut RMS to 50% of median Dice
     RMS.
4. Run a matched five-epoch one-cut arm from scratch with the same warmup,
   optimizer, data split, seed, AMP policy, and model constructor.

The five-epoch pilot is a screening experiment, not evidence for a final
performance claim. It advances only if it preserves ordinary validation Dice
while improving independent decoded boundary metrics. A/P one-cut remains a
separate later ablation.

## Calibration command

```bash
python thesis/new_constraints/onecut/calibrate_weight.py \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --dataset MSD \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --fold 0 \
  --checkpoint /path/to/fresh_none_run/checkpoint_latest.pt \
  --output /path/to/onecut_calibration.json \
  --max-cases 32 \
  --seed 0 \
  --spatial-size 64 64 64 \
  --device cuda \
  --amp
```

The training launcher then requires both the report and its exact recommended
weight:

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set onecut \
  --epochs 5 \
  --onecut-weight CALIBRATED_WEIGHT \
  --onecut-calibration-json /path/to/onecut_calibration.json \
  --output-dir /path/to/onecut_pilot_seed0
```

## Tests

```bash
PYTHONPATH=. python -m pytest -q \
  thesis/new_constraints/onecut/test_outer_onecut.py
```

The suite verifies exact one-cut semantics, shifted/missing/multiple-crossing
penalties, gradient direction, A/P-swap invariance, deterministic cached ray
geometry, invalid-patient handling, patient balancing, and unified-objective
integration.
