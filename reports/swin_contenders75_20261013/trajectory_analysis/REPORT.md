# Validation trajectories: capped winner versus longer-run winner

SwinUNETR, 5% training data, fold 0, seed 0; retrospective validation analysis.

The old capped-table winner is Dice + original bands + edge + signed distance + depth CE. The focused recipe is Dice + focused bands + endpoints + depth CE. All table entries below are observed epoch-end validation scores, not running maxima.

| Epoch | Focused recipe | Original all four | Original bands + edge | Dice only |
|---:|---:|---:|---:|---:|
| 10 | 35.216% | 35.244% | 35.504% | 33.109% |
| 12 | 44.749% | 48.796% | 40.959% | 38.489% |
| 13 | 57.461% | 58.743% | 42.460% | 39.565% |
| 14 | 68.804% | 64.253% | 43.888% | 40.461% |
| 15 | 74.489% | 67.370% | 44.693% | 41.658% |
| 20 | 80.061% | 70.707% | 64.091% | 53.333% |
| 21 | 80.028% | 71.453% | 66.435% | 57.028% |
| 30 | 81.074% | 73.715% | 72.996% | 66.183% |
| 41 | 81.502% | 75.273% | 75.479% | 71.119% |
| 49 | 81.318% | 76.312% | 76.353% | 72.337% |
| 58 | — | 77.204% | 76.817% | 73.539% |
| 66 | — | 77.138% | 76.998% | 73.913% |
| 69 | — | — | 77.303% | 74.151% |
| 75 | — | — | 77.245% | 74.286% |

The old recipe leads at epochs 10–13; its largest lead is 4.047 percentage points at epoch 12. The focused recipe overtakes it at epoch 14 and stays ahead throughout the remaining shared epochs. At epoch 18, focused Dice already exceeds the eventual best checkpoint of every original contender.

The capped experiment used 206 updates (20.6 patient-pass epochs). The longer experiment has recorded validation at 200 and 210 updates, so epochs 20 and 21 bracket this point. They are update-matched approximations, not exact FLOPs-matched measurements.

| Recipe | Best validation Dice | Best epoch | Last epoch | Stopped by patience? |
|---|---:|---:|---:|---|
| Focused bands + endpoints + depth CE | 81.502% | 41 | 49 | Yes |
| Original bands + edge + SDF + depth CE | 77.204% | 58 | 66 | Yes |
| Original bands + edge | 77.303% | 69 | 75 | No: epoch limit |
| Dice only | 74.304% | 73 | 75 | No: epoch limit |

The focused recipe reaches its best and stops earlier. The hypothesis that it wins only because it trains longer than the old all-four recipe is not supported. Early stopping used minimum 25 epochs, patience 8, and minimum improvement 0.0005 (0.05 percentage points); exact best-checkpoint selection was separate.

Among the original contenders, there is a smaller speed/final-score tradeoff: at epoch 20 the all-four recipe leads bands + edge by 6.617 percentage points. They become close around epochs 30–40 and trade places. The all-four best is 77.204% at epoch 58; bands + edge ultimately reaches 77.303% at epoch 69, after the all-four run stopped at 66. That 0.099-point difference is descriptive evidence from one tuned fold/seed, not proof of a reproducible ranking.

## Actual capped results and protocol differences

Under the actual cap, fold 0 / seed 0 also favors the focused recipe: 70.576% versus 65.298% at the final budget checkpoint. Selecting each capped run's best recorded checkpoint gives 70.576% versus 67.175%.

Across all nine capped fold/seed runs, the focused mean is 63.905% versus 63.659%, but focused wins only 2/9 paired comparisons. The original all-four recipe therefore wins more individual pairs despite having the slightly lower mean.

Capped training used mild_v1 augmentation, a 31-update LR warmup and a fixed 610-update cosine horizon. The longer comparison used no augmentation, initial LR 1e-4, and halving every 20 epochs. The capped old recipe delayed SDF to update 116; the long run included it in the shared update-51-to-100 constraint ramp. Calibration coefficients were also recalculated. Thus these are not the same trajectory stopped at different times, and the available records cannot separate the causal effects of augmentation, schedule, calibration, and SDF timing.

To isolate training duration, a future experiment would hold all of those settings fixed, validate exactly at update 206, and continue the same trajectories to later checkpoints. A factorial follow-up would be needed to attribute effects to individual schedule/augmentation changes. No training was launched for this analysis.

## Reproduction and provenance

Run `.venv/bin/python reports/swin_contenders75_20261013/trajectory_analysis/analyze.py` from the repository root. `analysis.json` records source hashes and derived results; `all_matched_epoch_curves.csv` contains all 15 trajectories and their running bests. The plot is available as PNG and PDF. These exports contain aggregate scores, not patient images.

Additional limitations:

- Equal update count is not exactly equal FLOPs: objective arithmetic differs.
- Capped runs used mild_v1 augmentation and 31-update LR warmup then a 610-update cosine horizon; long runs used no augmentation and StepLR every 20 epochs.
- The old capped all-four recipe activated SDF at update 116; the long-run contender included SDF in the common update-51-to-100 constraint ramp. Recalibrated coefficients also differ.
- Capped headline comparisons use final-budget checkpoints; long-run headline comparisons select best validation checkpoints.
- This exploratory fold/seed was used to select the focused constraint. Results do not establish ranking on independent data.
- Early stopping does not establish what would happen if training were continued beyond stopping.
