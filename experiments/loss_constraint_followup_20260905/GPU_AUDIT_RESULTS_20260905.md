# Official baseline GPU audit and project direction

Slurm **650074**, completed successfully (exit 0), **4 minutes 31 seconds** job
wall time; GPU computation program 180 seconds. Device: NVIDIA A100 MIG 4g.40gb.
All model inference and logit-gradient audits ran on CUDA on the cluster. Model
weights stayed on the cluster and no training ran.

Source snapshot and outputs:
`/mnt/beegfsstudents/home/3160552/loss_audit_official_20260905_01/`.
The source archive SHA256 is
`f24b60cea15db6a0855f6e15c2fee8ee21fccd4561915b06fcfcd155bcb17a2a`.
The official model SHA256 matches its completed protocol manifest:
`8c66f93525145be4efce4eb3578ffab39d1ede8035f728684d0355a62298d7f9`.
Data and split hashes were checked before inference. The preprocessing source
hash also matches the original training record. See `gpu_official_summary.json`
for aggregate results and hashes of the four complete cluster reports.

## Matched measurement

Official control: `matched_control_20260902/msd_fold0_none_matched_50epoch_seed0`,
epoch 50, fold 0, seed 0, batch size 1, AMP. Recorded hard Dice:
**0.8747624709**. Re-evaluated through the current evaluator:
**0.8747659463**, difference **3.4754e-6**. This passes the prespecified 1e-5
reproduction tolerance, but is not bit-identical. The decoded FP/FN/swap totals
reproduce the handoff exactly: 16,517 / 18,937 / 3,788.

The tables below compare results within this one audit: AMP model inference,
float32 probability/diagnostic arithmetic, and per-case-averaged foreground
macro Dice on all 52 validation cases. Do not compare these absolute values or
deltas directly with the earlier local fp32/voxel-pooled TTA results.

| Prediction | Macro Dice | Delta | Better cases | FP change | FN change | Swap change |
|---|---:|---:|---:|---:|---:|---:|
| Official baseline, re-evaluated | 0.874766 | — | — | — | — | — |
| Full 12-shift teacher | 0.881712 | +0.006946 | 41/52 | -1,418 | -919 | +8 |
| Sampled two-shift teacher | 0.876187 | +0.001421 | 29/52 | -1,218 | +57 | +320 |
| Identity + 12 shifts (13-view TTA) | 0.881602 | +0.006837 | 43/52 | -1,385 | -892 | -16 |

The full teacher reduces total errors by **2,329 (5.93%)**, almost entirely at
the outer foreground boundary. This is useful remaining error, despite the
boundary's apparent difficulty. It is not a measured training-side constraint
gain. The two-view row is one deterministic sampled pair per case (seed
20260905), not an exhaustive estimate over all possible pairs. Uncovered voxels
retain the baseline in its decoded prediction; the KL audit uses valid support.

## What the gradients establish

Full-teacher KL alone improves frozen logits at every tested maximum absolute
logit step:

| Maximum logit change | Teacher KL alone: delta macro |
|---|---:|
| 0.25 | +0.000249 |
| 1 | +0.001431 |
| 4 | +0.004739 |

Thus the real saturated baseline is reachable by this smooth auxiliary loss.
No preliminary Dice+CE training was necessary to demonstrate that fact.

However, the incremental effect of adding the teacher to a supervised gradient
is much smaller. At maximum step 4, Dice alone gives +0.011508 and Dice plus the
diagnostically scaled teacher gives +0.011849: **+0.000341 incremental**.
Dice+CE alone gives +0.012638 and Dice+CE plus teacher gives +0.013078:
**+0.000440 incremental**. The coefficients were chosen on these validation
logits only for mechanism diagnostics; they are explicitly invalid as training
calibration. Each direction is separately normalized to the same maximum logit
change, so these are bounded counterfactual comparisons, not optimizer learning
rates or estimates of training effect size.

Median full-teacher/supervised gradient cosine is **0.073 with Dice** and
**0.212 with Dice+CE**. This shows only partial alignment; low cosine does not by
itself prove useful complementarity. The measured incremental repairs provide
the more relevant caution. These calculations differentiate logit leaves,
not the complete model parameters. Whether shared network weights can realize
the repairs remains an empirical training question.

The cheaper two-view teacher has smaller decoded benefit and its strongest
tested auxiliary step increases swaps. This does **not** prove stochastic
two-view KL training is ineffective. On fixed common support, the KL gradient
is linear in the detached target; uniform view sampling can estimate the
full-average gradient in expectation. Its hard argmax quality is a nonlinear,
different measurement. This implementation also has sampled, varying valid
support and denominators at crop edges, so exact whole-crop unbiasedness should
not be asserted without a separate check. Measure gradient variance and
coverage before deciding the view budget; do not assume two views inherit the
full teacher's +0.00695 prediction improvement.

## Saturation is real, but the proposed universal explanation was wrong

On this checkpoint, max-class confidence >=0.99 occurs in **99.87% of the whole
crop**, **94.64% of GT foreground**, and **90.72% of the multiclass GT boundary**.
Among incorrectly decoded voxels, **79.07%** are saturated, and average
confidence is **96.69%**. Foreground accuracy is only **86.97%** against mean
confidence **99.19%**. Conditional A/P confidence on GT foreground is saturated
in **94.51%**; among conditional A/P mistakes, **67.42%** are saturated. These
are defined per-voxel confidence diagnostics, not clinical uncertainty estimates.

Dice+CE is therefore a justified control to investigate confident errors.
It is still a hypothesis about a future training run, not a proven cure for
saturation, and the existing Dice-trained checkpoint already responds to KL.

## A/P result and the label-pickle question

Direct inspection of the actual training pickle confirms the earlier raw-data
audit: **47/208 training cases** and **11/52 validation cases** contain mixed
A/P slices. Minority voxels total **1,104** and **284** respectively. The
discrepancy is not caused by reading a different NIfTI target representation.

The strict A/P posterior audit accepts only 41 validation cases. Its isolated
gradient can repair that compatible subset (up to +0.007990 at maximum logit
step 4), but it adds **no improvement** to Dice+CE in the tested combinations:
the incremental deltas versus supervised alone are approximately -0.000019,
-0.000081 and -0.000086 at steps 0.25, 1 and 4. Its target and foreground support
use GT, making these oracle diagnostics. The compatible subset also starts
at a higher baseline, 0.882523; never compare its result with the all-case mean.

This rejects the implemented strict all-case formulation. It does not reject
every approximate planar or cut-localization objective. The exceptions are
small (about 0.16% of foreground), so a tolerance-aware formulation remains
conceivable, but its incremental utility must be demonstrated before GPU
training. It should be a secondary research question rather than the next
expensive experiment.

## My recommendation

Make the thesis question precise: **which loss-based constraints add reproducible
value beyond matched supervised learning and ordinary augmentation, and why?**
Keep improvement through the training loss as the objective, while separating
that claim from useful inference ensembles and oracle repairs.

1. **Stop expanding the boundary-loss variant search.** Bands/focal/Tversky have
   not earned more tuning. Preserve their negative replication and denominator
   analysis as results. Do not claim the outer boundary is irreducible: the full
   teacher demonstrably corrects it.
2. **The next training block should establish the missing controls and replicate
   the strongest existing constraint.** Test Dice+CE alone with seeds 0 and 1,
   and move the existing equivariance formulation into Family B with the same
   two seeds. Analyze these as separate comparisons against matched Dice
   controls; do not simultaneously change the supervised loss and the
   equivariance formula and call the combined difference a constraint effect.
   Verify the new-source default-Dice bridge first. No such jobs were submitted.
3. **Test augmentation as a competing explanation.** On a frozen supervised
   objective, compare no translation, supervised translation augmentation,
   and the constraint with a common shift distribution, seeds and exposure.
   Account for extra views/compute. This distinguishes consistency from the
   benefit of finally showing the model translated inputs.
4. **Promote teacher KL only through a staged test.** It passes reachability and
   has a better full teacher, but neither proves useful training improvement.
   Resolve view/gradient variance and crop support, calibrate on training cases
   under the selected objective, time the extra forwards, then use two seeds
   at a fixed converged endpoint. Do not claim non-saturation or reuse the
   validation-derived diagnostic lambda. KL self-distillation is consistency
   regularization; calling it new anatomical knowledge would be inaccurate.
5. **Lock one candidate and confirm outside the repeatedly explored fold 0.**
   Use fixed endpoints and paired effects, not peak-epoch maxima. Two seeds are
   a minimum replication check, not a universal significance threshold. If no
   incremental benefit survives, report that limit instead of adding another
   almost-identical loss.

A later low-data extension could be scientifically motivated: the
[reference paper](https://arxiv.org/html/2509.22399v1) explicitly evaluates
reduced training-set fractions and reports its largest gain at 5% of training
data. That is a separate, preregistered setting with matched controls and
exposure; it should not become a post-hoc escape from a full-data null result.
