# Momentum 0.95 experiment validation

The default local environment was rejected by the pinned MONAI version check;
no production training was performed there. Repeating with the profiled MONAI
1.6.0 and dynamic-network-architectures 0.4.1 dependencies passed all ten tests
in 47.791 seconds:

```bash
rtk proxy env PYTHONPATH=/tmp/hippo_bref_monai160:/tmp/hippo_bref_networkdeps \
  MPLCONFIGDIR=/tmp/hippo_bref_mpl .venv/bin/python \
  -m unittest scripts.test_compute_matched_baselines -v
```

Checks include unchanged FLOP caps and original schedule, momentum 0.95 used
by SGD and retained in saved optimizer/configuration, actual model training
smoke checks on synthetic data, foreground loss/validation class semantics,
checkpoint selection, resume accounting, and artifact tamper detection.
`bash -n scripts/run_nnunet_momentum095.sbatch` also passed.

Production cluster submission additionally checks the uploaded archive against
all staged files, dataset SHA-256, and three focused optimizer/loss checks.
The submission receipt records their outcome. The helper holds a submission
lock, checks queue occupancy, and records an attempt before sbatch to avoid
unintended duplicate jobs if its response is lost.
