# Translation equivariance

The implementation previously stored at
`thesis/new_constraints/translation_equivariance.py` now lives in this package.
The old module remains as a compatibility import so existing analysis scripts
and checkpoints continue to work.

Use the canonical training flag:

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set equivariance
```

`--constraint-set translation` remains accepted as a deprecated alias.

Historical reports and plots are stored in `equivariance/results/`.
