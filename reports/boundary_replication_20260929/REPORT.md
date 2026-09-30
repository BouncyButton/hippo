# Augmentation replication

Two initializations on the same 50 training and 52 validation cases. Descriptive case intervals condition on selected checkpoints and omit training and selection uncertainty. Do not treat 104 scores as 104 independent validation cases.

| Seed | Dice control% | Dice augmented% | Change pp | Control shell errors | Augmented shell errors | Change | Cases with fewer errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 86.2016 | 87.3730 | +1.1714 | 38972 | 34629 | -4343 | 48/52 |
| 1 | 86.0414 | 87.4333 | +1.3919 | 39058 | 34046 | -5012 | 50/52 |

Equal-seed mean gain: 1.2816 Dice points and 4677.5 fewer shell errors per 52-case evaluation.

Fixed-budget checks:

- Seed0, pass30: +0.5887 Dice points; -4525 shell errors.
- Seed0, pass60: +1.2607 Dice points; -5168 shell errors.
- Seed0: matching case order for 60 passes, zero AMP skips; actual LR first differs at pass 30 under the shared adaptive policy.
- Seed1, pass30: +1.2448 Dice points; -4646 shell errors.
- Seed1, pass60: +1.3517 Dice points; -5350 shell errors.
- Seed1: matching case order for 60 passes, zero AMP skips; actual LR first differs at pass 36 under the shared adaptive policy.
