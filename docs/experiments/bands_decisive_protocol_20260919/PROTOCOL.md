# Boundary bands: proposed discriminating experiment

Draft 1, 19 September 2026. Planning only: no training jobs are authorized or submitted by this document. Proposed working budget: 30–50 allocated A100 MIG 3g.40gb hours. Numerical runtime estimates are generated in BUDGET.md; the complete unsubmitted matrix is RUN_MATRIX.csv. Split manifests, subject provenance and source hashes must be resolved before execution. The study is staged, so preparing the full matrix does not commit to spending the full budget.

## Questions and claims

The experiments should distinguish three explanations that the previous fixed-epoch result could not separate:

1. Bands accelerates optimization under a specified data and compute budget.
2. Bands improves held-out performance under a sufficiently long, fixed development procedure.
3. The advantage depends on placing supervision near the anatomical boundary, rather than merely adding another classification loss or balancing foreground/background.

A fourth, separate claim is that the benefit is greater with fewer labels. That requires the fraction interaction stage; a positive result at ten cases alone does not establish that interaction.

“Persistent” here means surviving the long, predefined training and checkpoint-selection procedure on held-out evaluation. It does not mean an asymptotic optimum or superiority over every tuned baseline. No finite experiment can prove global convergence. A separate equally resourced hyperparameter search would be required for a best-tuned-method claim.

## Loss controls

Keep the archived SwinUNETR architecture, preprocessing, Dice implementation, no-augmentation setting, AdamW learning rate 1e-4, weight decay 1e-5, batch size 1, and AMP. Preserve all Dice defaults, including its background treatment; scoring excludes background. This tests the original finding's setting. Augmentation and architecture changes are later robustness questions.

| Arm | Objective | What the comparison with bands resolves |
|---|---|---|
| D | Original Dice | Does the observed benefit replicate? |
| B | Dice + original two-step, six-neighbour, side-balanced bands BCE, gamma=0 | Candidate method |
| G | Dice + full-volume, side-balanced grouped-foreground BCE | Is restricting this supervision to the boundary useful beyond foreground/background balancing? |
| C | Dice + ordinary full-volume three-class CE | Does a conventional auxiliary classification loss explain the practical advantage? |
| R | Dice + matched-random foreground/background BCE | Is boundary placement useful beyond supervising the same number of randomly selected, correctly labeled voxels? |

D/B/G/C are the core four arms. R is a recommended additional control costing approximately five slice-hours including reserve. Include or omit R before replication evaluation is revealed; do not add it conditionally on a favorable test result. The full recommended matrix includes R and fits the provisional 50-hour envelope, subject to preflight timing.

Let h(v)=p_anterior(v)+p_posterior(v), and H be the training ground-truth foreground. Use stable grouped logits for BCE. Define:

    L_G = -0.5 * [mean over H of log h + mean over complement(H) of log(1-h)]
    L_B = -0.5 * [mean over inner band of log h + mean over outer band of log(1-h)]
    L_C = mean over all voxels of -log p_ground_truth_class

G and B have the same foreground grouping and equal weighting of the positive and negative sides. Use the same epsilon and empty-side handling. Do not use an unbalanced global BCE as the only localization control; that would also change class weighting.

For R, sample exactly as many distinct foreground voxels as the inner band contains and exactly as many distinct background voxels as the outer band contains. Sample uniformly within the corresponding correct ground-truth class, without replacement. Freeze each case's masks using a documented case/subset-specific seed independent of the model seed. Do not permute labels or assign false targets. No augmentation is used. B versus R tests placement while matching voxel counts, class grouping, side weighting and supervision labels. Background padding remains part of the shared field of view and must be identical across arms.

## Calibrating auxiliary strength without weakening the controls

For B, G, R and the calibrated C candidate, use one Dice-only calibration source trained for 50 optimizer updates on the same training subset. Reinitialize every production model to the paired initial state; do not warm-start a production arm from the calibration model.

On every training case, compute full-logit-tensor RMS gradients of the unchanged Dice and each raw auxiliary objective. Use all training cases, including all 52 in the larger fraction; the current 32-case calibration cap must be generalized. The common rule is:

    lambda_aux = min(0.10 * median(g_Dice) / median(g_aux),
                     0.50 * median(g_Dice) / q95(g_aux))

Record target and cap, valid case counts and per-case gradient magnitudes. No validation or evaluation annotations enter this calibration. Matching initial logit-gradient magnitude is a controlled scaling rule, not proof of equal parameter gradients or equal objective strength throughout training.

The current bands-specific report cannot simply be passed to G or C. Their gradients and loss identities require their own calibrated weights and provenance. Use stable fp32 auxiliary math with the same AMP policy across arms. In new update-based runs, multiply each auxiliary term by min(update/50, 1), once. Keep lambda fixed thereafter.

To avoid declaring a weak, specially scaled CE to be the strongest ordinary baseline, also run C with coefficient 1 and the same auxiliary warmup during development. Choose calibrated CE or unit-weight CE by the mean best development score over the three development seeds at 8,000 updates with ten cases; exact ties favor unit weight because it needs no gradient calibration. Freeze that choice for every replication block. Retain both candidates' development results. This gives the conventional control an explicit stress check; it is not an equal exhaustive hyperparameter search across all methods.

Cache the shared calibration model within a research block to save actual cluster time. For standalone-method speed, charge each calibrated method the full cost of obtaining its source model and its own gradients, even when the experiment used a cache.

## Data separation and the meaning of a final test

Before GPU runs, produce a case-to-subject/hemisphere provenance table. The current MSD helper treats full case names as patient IDs, so its existing “patient-disjoint” check does not settle biological independence. Do not infer subject identity from filename patterns without a documented mapping.

Where verified identities are available, split subjects first and keep all related cases together. Sample ten or 52 labeled case files within the eligible training pool; report the number of unique subjects as well. Ten case files are not automatically ten subjects. Use a fixed case permutation per subset seed so the ten-case set is contained in its 52-case set. Different subset seeds are separately drawn permutations; they are not required to be disjoint, and their overlap must be reported.

Proposed replication layout: form five subject-grouped outer folds with a frozen split seed, use outer folds 0–2 as R0–R2, and reserve 20% of each outer-training pool for checkpoint selection. The remaining outer-training pool supplies the scarce-data subsets. R0–R2 are new protocol roles, not assumed aliases for the old case-based folds. The three outer evaluation folds are disjoint in subjects, although their training pools overlap. Verify enough eligible cases remain for every 52-case subset before freezing manifests.

Within every replication run, training labels construct losses and calibration; development labels select the checkpoint; outer evaluation labels are used only after the procedure is frozen. Save two checkpoints for scoring: highest development score (earliest update on an exact tie) and the final checkpoint. Use the development-selected checkpoint as the primary endpoint. Never pick a checkpoint by outer evaluation Dice.

There are two evidence levels:

- **Retrospective internal replication:** new grouped splits and unseen subset/initialization seeds within this already-explored MSD cohort. This improves separation inside each new run, but prior dataset-level research choices and historical development remain part of the provenance. Historical D0 choices may also have seen cases later assigned to R folds. Do not advertise these as pristine independent confirmation or claim that new splits erase earlier selection.
- **Independent confirmation:** a demonstrably untouched, subject-disjoint evaluation cohort with compatible anterior/posterior labels and preprocessing, or genuinely new subjects. Freeze choices before revealing its results. A binary hippocampus dataset cannot silently replace an anterior/posterior endpoint. Extra external-cohort evaluation time depends on cohort size and is not precisely budgeted here.

If subject mapping cannot be obtained, preparation and explicitly case-level exploratory work can continue, but patient-generalization claims remain unresolved. The matrix deliberately contains no invented case assignments or purported clean test set. The checkpoint metric should be foreground macro Dice averaged within subject and then across subjects when mapping is verified; also retain the original case-macro metric for continuity. With no mapping, label case-macro results as such and do not use a subject bootstrap.

## Training budgets and schedules

Count successful optimizer updates, not epochs. At batch size one, the number of presentations is clear; separately log attempted steps and AMP-skipped steps. Advance the learning-rate scheduler only on successful updates. Pair initial weights, data order, preprocessing and precision across loss arms within each block. Use an independent data-loader generator and log the order hash. Calibrating one arm must not consume another arm's training RNG state.

| Schedule | Total updates | LR decays after updates | Purpose |
|---|---:|---|---|
| Historical anchor | 750 | 200, 400, 600 | Revisit the original ten-case budget on the chosen hardware |
| Long development | 4,000 | 1,060, 2,120, 3,180 | Bridge to the previously explored 400-epoch regime |
| Longer development and replication | 8,000 | 2,120, 4,240, 6,360 | Give every seed the same generous budget |

The initial LR is 1e-4 and each decay multiplies it by 0.5. Disable accuracy-based early stopping for these comparisons. Finishing every run at its registered cap prevents selectively extending a difficult or unfavorable seed. Infrastructure failures resume from exact saved state; numerical failures are reported and investigated without quietly replacing seeds.

At ten cases, 8,000 updates means 800 dataset passes; at 52 cases, about 154. This keeps update count, LR chronology and auxiliary warmup fixed across fractions while allowing distinct numbers of repeated exposures. Neither this schedule nor the original equal-epoch schedule represents universal convergence.

Validate every 50 updates, with the same cadence in every main arm and fraction. The two long schedules are distinct runs, not prefixes of one another: stretching decay changes the earlier learning trajectory. In particular, update 750 of the long schedule does not reproduce the original 75-epoch experiment. The historical anchor validates every ten updates and uses the original epochwise auxiliary warmup min(ceil(update/10)/5, 1); its different selection cadence is disclosed, and it is not the primary long-run comparison.

Before replication, inspect development trajectories for all arms. Flag the long budget as inadequately explored if any prespecified development run increases its best development Dice by more than 0.2 percentage points in the last 20% of the 8,000 updates. Also report the score at 4,000 versus 8,000 and the last-window trend. This is a conservative warning rule, not a theorem of convergence. If flagged, revise the protocol version on development data or proceed with an explicitly finite-budget claim; do not call the result converged or extend individual replication seeds.

## Stages and run counts

**Stage 0: manifests and implementation readiness.** Establish provenance and evaluation roles; implement and verify the missing controls and update-based loop; benchmark a short preflight. No full matrix starts before this passes. Engineering preflight cost comes from the reserve and is measured, not silently excluded.

**Stage 1: development, 38 production runs.** On a designated development split D0, use the historical ten-case subset and seeds 0/1/2 where subject provenance permits. Four arms at 4,000 and 8,000 updates give 24 runs; D/B historical anchors give six; unit-weight CE at 8,000 gives three. A 52-case feasibility check on seed 0 runs D/B/G/calibrated-C/unit-C for five more. If verified grouping invalidates D0's separation, rebuild D0 before using it for selection and record the change. This stage selects the CE policy, checks long-schedule behavior and verifies runtime. It is not a final generalization test.

**Stage 2: low-data replication, 48 production runs.** Three grouped evaluation splits × two subset seeds (101, 202) × two model seeds (11, 22) × D/B/G/selected-C, all at 8,000 updates. This yields 12 paired blocks, with explicit subset variation absent from the original study. It is still a modest design, not 12 independent datasets or a powered guarantee of detecting a 0.3-point effect.

**Stage 2R: placement control, 12 additional runs.** Run R in each of the same 12 blocks, using the same long schedule. This stage is recommended in the full plan. Decide inclusion before revealing replication evaluation. CPU checks and a short reserve-funded preflight cover its mask and calibration implementation before these runs.

**Stage 3: fraction interaction, 24 production runs.** D/B at 52 cases in the same 12 blocks, using nested subsets, identical development/evaluation cases and the same update schedule. The primary scarcity interaction is (B−D at ten cases) minus (B−D at 52 cases). This tests the fraction interaction of bands versus Dice. It does not establish how boundary specificity versus G/R/C changes with sample size; that would require those controls at the larger fraction too.

The full plan has 122 production runs and 28 distinct calibration-source blocks, estimated at about 49 allocated slice-hours including a 25% reserve. The four-arm version without R is about 44 hours. These are timing-based planning estimates for a new loop, not measured future costs or approval to spend. A stage can be deferred for budget reasons without selectively dropping unfavorable completed outcomes. If only about ten hours are available, use the reduced 27-run development set (24 horizon runs plus three unit-CE checks; defer the anchors and 52-case check), approximately 9.3 slice-hours including calibration allowance and reserve. Make no new confirmatory or fraction-interaction claim from that reduced pilot.

## Endpoints and decision rules

Primary accuracy endpoint: held-out foreground macro Dice of the development-selected checkpoint under the common 8,000-update procedure. Report percentage-point paired differences B−G, B−D, B−C and, if included, B−R. Report all arms' scores, per-class Dice, final-checkpoint scores, and the complete block table. Never choose whether to report best or final based on which is favorable.

Primary speed endpoint: time and successful updates until the third consecutive development evaluation at or above 78% foreground Dice. This credits only a target known to be sustained, not the earlier first crossing. Resolution is 50 updates. Secondary targets are 76% and 80%. A never-reached target is censored at the cap; it is not dropped and is not assigned the cap as if it were reached. First report attainment counts; do not average speeds over only successful runs. These are optimization endpoints on development data, not a substitute for the held-out accuracy endpoint.

Record two clocks: synchronized training-kernel time and end-to-end method time including source training, calibration, validation, checkpoints, startup and final inference. Queue time is reported separately. For cached sources, preserve actual research cost and standalone method cost. Randomize/counterbalance arm execution order across blocks; use the same GPU slice class and software environment. Do not compare a 4g baseline with a 3g bands timing.

Boundary endpoints: foreground-union average symmetric surface distance and HD95 in verified physical spacing, plus foreground FP/FN and anterior/posterior confusion. Define empty-prediction handling in the metric implementation (no silent dropping; report empty counts and a prespecified finite image-diagonal distance penalty). Use native-space/affine-consistent masks; do not assume a padded or resampled voxel equals one millimeter. Surface metrics are secondary and do not establish topology or anatomical validity.

Proposed practical margin: 0.5 Dice percentage points, explicitly a scientific planning threshold rather than a clinically validated cutoff. A positive point estimate is not a demonstrated effect. An interval excluding zero supports a positive difference for its stated target; an interval entirely above +0.5 supports the proposed meaningful margin. An interval wholly inside −0.5 to +0.5 is evidence compatible with practical similarity under that target and procedure. A wide interval spanning meaningful benefit and harm is inconclusive, not “no effect.”

| Pattern | Interpretation permitted |
|---|---|
| B improves early, but the long held-out difference is small with adequate precision | Primarily a speed benefit under this procedure |
| B and G/R perform similarly and both beat D | Additional foreground supervision helps; boundary placement has not added a demonstrated advantage |
| B beats G and R on long held-out endpoints | Evidence that boundary placement contributes beyond these matched supervision controls |
| C matches or exceeds B | No demonstrated practical advantage over the selected ordinary-CE recipe |
| B−D is reliably larger at ten than at 52 cases | Evidence of a scarcity interaction under the matched-update policy |
| Estimates are small and intervals remain wide | Insufficient precision; do not force a winner or claim equivalence |

## Statistical analysis and the small-effect problem

Keep subject, outer split, subset and model seed as distinct axes. Pair methods before averaging. Average seeds within each subset for a subset-level summary, then report each split and all subset results. The twelve blocks share evaluation subjects within splits, may overlap training cases, and are not independent trials for a naive t-test. Correlated epochs, voxels and repeated subject–seed scores are not independent sample sizes.

For a genuinely independent test cohort, first compute each subject's paired difference averaged over the fixed, prespecified set of trained replicas (average scores, not an undeclared prediction ensemble). A subject-cluster bootstrap of these differences estimates test-subject uncertainty conditional on those fitted replicas. Report it with an explicit conditional label; it does not by itself quantify uncertainty from new training datasets. Show subset/seed variability separately. For four declared accuracy contrasts, use simultaneous coverage or Bonferroni-adjusted 98.75% individual intervals; do not present four unadjusted 95% intervals as familywise confirmation. The primary B−G contrast is also reported individually. All surface and speed endpoints are secondary/exploratory unless separately preregistered.

For the retrospective R0–R2 evaluation, report paired block and subject summaries with overlap-aware limitations. Do not imply that bootstrapping three folds gives a reliable population interval. A conditional subject bootstrap can illustrate evaluation uncertainty, but prior cohort reuse and shared training remain outside its scope. An estimator of total training-plus-test uncertainty needs a separately specified crossed/resampling model and enough independent sampling units; no such population confidence guarantee is asserted here.

The existing three-seed +0.3-point residual cannot justify a prospective power calculation by itself. Before declaring an independent confirmation adequately powered, obtain the independent cohort size and a development-only estimate of paired subject variability, then assess precision for the 0.5-point margin and adjust the design before opening test results. If the achievable interval is too wide, the honest endpoint is inconclusive. Add training subsets rather than treating more initialization seeds as new subjects.

## Implementation work needed before launch

Existing support: original Dice, bands, ordinary three-class Dice+CE, casewise confusion/evaluation artifacts, frozen-source provenance and resume machinery. Current local training files contain unrelated uncommitted work; select an isolated source snapshot rather than silently using the working tree.

Required additions in an isolated implementation:

1. Global grouped BCE and matched-random BCE; independent tiny-tensor loss/gradient checks, correct labels, equal side weights, mask sizes and finite extreme-logit behavior.
2. A generalized training-only calibrator for each loss, all selected cases, common source identity, correct cap arithmetic, and zero validation/evaluation-label access. Unit CE bypasses calibration.
3. Successful-update-based LR, warmup, evaluation cadence and stopping cap. The existing epoch loop and `--constraint-eval-every` do not provide this: that flag concerns constraint metrics, not primary Dice validation cadence.
4. Common auxiliary warmup for CE without silently scaling the Dice term; preserve the archived Dice definition. The existing `DiceCrossEntropyLoss` adds CE directly and needs an explicit adapter for this protocol.
5. Exact resume of optimizer, scaler, LR, successful-update count, model/data RNG and sample order, including a mid-epoch state or replayable sampler. Test an interrupted toy run against uninterrupted execution before GPU replication.
6. Subject-grouped manifest checks; zero train/development/evaluation overlap within run; nested subset checks; initial-weight and sample-order equality between arms; registered hardware and sources.
7. Timers, per-update metrics, two-checkpoint evaluation, physical-space surface metrics and a report generator that implements the declared pairing and censoring rules.

The local CPU checks can be completed without spending the training budget. A short GPU preflight must subsequently measure the new losses and evaluation cadence. If projected total cost exceeds the budget, update BUDGET.md before launching replication. No command in this plan should be pasted as a submission instruction; actual cohort manifests and implementation do not yet exist.

## Research basis and deliverables

The development/evaluation separation addresses model-selection bias described by [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html). Explicit subset, initialization and tuning variation follows the benchmarking concerns in [Bouthillier et al., 2021](https://arxiv.org/abs/2103.03098). These papers motivate the design principles; they do not validate the particular budgets, margins or sample size chosen here.

Planning deliverables: PROTOCOL.md; PLAN.json with assumptions and unresolved prerequisites; RUN_MATRIX.csv with non-launchable rows; BUDGET.md; build_plan.py to regenerate counts and estimates from archived timings. The execution study should add subject provenance, split/source hashes, per-run manifests, unfiltered outcomes, paired tables, learning/time curves and a claim-by-claim conclusion. No original result or training code was changed in preparing this plan.
