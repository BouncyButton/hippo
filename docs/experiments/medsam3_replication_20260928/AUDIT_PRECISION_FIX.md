# Audit precision correction after job 673549

Job 673549 stopped after 1:23:46 at the independent audit for fold 0, seed 17. Training and evaluation of all three arms had completed. The original audit required exact equality of the first native-loss values.

Observed initial values:

- Baseline: 144.15442657470703.
- Bands and bands+edge: 144.15442276000977.
- Absolute difference: 0.000003814697265625 (relative difference about 2.65e-8).

All three arms have exactly matching SHA256 digests for initial adapter parameters, frozen parameters, and sampled training schedules. Learning-rate sequences match exactly. Initial bands and edge losses also match exactly. The native-loss difference is consistent with floating-point reduction rounding; the audit should not require bitwise equality of floating-point reductions.

The correction allows relative and absolute tolerance of four float32 epsilons (4.76837158203125e-7), while retaining exact parameter-hash, data, schedule, and learning-rate checks. It additionally checks the initial bands and edge components, records every difference and the tolerance in the audit output, and rejects nonfinite values. Regression tests cover the observed rounding case, a meaningful mismatch in every component, and nonfinite values.

No training code, data selection, seeds, loss coefficients, predictions, raw result files, or frozen study protocol were changed. No outcomes were used to tune the experiment. The original audit and source manifest are preserved on the cluster in `revisions/audit_precision_fix/`. The original failed job and its logs remain intact. Resume is allowed only after the entire completed comparison passes the corrected independent audit; it will skip those three completed training runs.

The corrected independent audit passed: all 276 saved NIfTI files, all 600 training-update records, matching initialization/schedules, frozen parameters and checkpoint/prediction integrity. Mean Dice over the 46 fold-0 validation volumes: baseline 0.8354525228, bands 0.8491065304, bands+edge 0.8654511104. These are one fold/seed only, not the final replication result.
