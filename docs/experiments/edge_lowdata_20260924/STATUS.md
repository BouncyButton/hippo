# 5%-data edge experiment status

Job **667831 COMPLETED**, exit 0, 19m36s, all seeds and audits successful.
Bands + edge improved validation Dice over bands-only in every seed:
+0.670, +0.397, +1.015 percentage points; mean +0.694.
Mean validation Dice: bands 77.081%, edge 77.775%.
New selected / stopped epochs: seed 0 73/75, seed 1 63/71, seed 2 74/75.

Bands + edge first reached each historical bands best score at epochs
48/49/50 versus bands 68/74/74. Boundary FN+FP decreased in every seed,
principally through reduced FP, with increased FN. Full evidence and
limitations: REPORT.md, RESULTS.json. All pre-submission protocol details
remain in PROTOCOL.md; no new baselines were trained.
