# Translation equivariance

The implementation previously stored at
`thesis/new_constraints/translation_equivariance.py` now lives in this package.
That compatibility shim has been removed; no caller in the repository imported
it. Import from `thesis.new_constraints.equivariance` instead.

Use the canonical training flag:

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set equivariance
```

`--constraint-set translation` remains accepted as a deprecated alias.

Historical reports and plots are stored in `equivariance/results/`.
