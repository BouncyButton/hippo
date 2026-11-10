# Job 682319 final status

**Retired:** the three run directories and checkpoints were deleted at the user’s request after preserving their records. See [retirement record](RETIREMENT.md). Results below are historical.

Checked 2026-10-04T10:44:45.985345+00:00.
Slurm COMPLETED, exit 0:0, elapsed 00:07:46, ended 2026-10-04 12:42:58 Europe/Rome.
All three fold-0 seeds completed with momentum 0.95 and no warmup. Stderr was empty.

| Seed | Selected update | Anterior Dice | Posterior Dice | Mean Dice |
|---|---:|---:|---:|---:|
| 0 | 278 | 0.858346 | 0.845096 | 0.851721 |
| 1 | 295 | 0.859335 | 0.845231 | 0.852283 |
| 2 | 327 | 0.853002 | 0.846005 | 0.849503 |

Mean selected validation Dice across three seeds: 0.851169 ± 0.001470 (sample SD).
Each run completed all 327 updates (32.7 patient passes), 20 validation checkpoints,
and consumed 116181.104090427 GFLOPs, below the 116259-GFLOP cap; discarded
attempts were zero. The frozen-source summarizer verified all run artifacts and
budgets before writing the complete aggregate.

Seed 2 now learns the anterior, compared with zero anterior Dice in both previous
recipes. This supports reducing momentum as a useful intervention in this fold;
it does not establish robustness across the other folds. Scores were selected
on 52 validation patients and are not independent held-out test results.

See [final aggregate](summary.json) and [submission record](submission.json).
