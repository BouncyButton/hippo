"""Generate the numerical audit report from independently rebuilt AUDIT.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = Path('/Users/filippofocaccia/.codex/worktrees/5063/hippo/experiments')
a = json.loads((HERE/'AUDIT.json').read_text())
g = a['aggregate']


def link(name, file):
    return f'[{file}]({EXP/name/file})'


aggregate_rows = []
for key,label in [('four_fold_75','Original: four folds, three seeds, maximum 75 epochs'),
                  ('fold0_75','Original fold 0, three seeds'),
                  ('fold0_uniform_200','Fold 0, all seeds on maximum-200 schedule'),
                  ('fold0_adaptive_200_200_400','Fold 0, seeds 0/1 maximum 200; seed 2 maximum 400')]:
    r=g[key]
    aggregate_rows.append(f"| {label} | {r['baseline']['best_mean_pct']:.3f} | {r['bands']['best_mean_pct']:.3f} | {r['best_gain_pp']:+.3f} | {r['final_gain_pp']:+.3f} |")
long_rows=[]
for r in a['pairs']:
    if r['schedule']<=75:
        continue
    b,n=r['arms']['baseline'],r['arms']['bands']
    long_rows.append(f"| {r['seed']} | {r['schedule']} / {r['step_size']} | {b['stop_epoch']} / {n['stop_epoch']} | {b['best_epoch']} / {n['best_epoch']} | {r['best_gain_pp']:+.3f} | {r['latest_gain_pp']:+.3f} | {b['final_train_loss']:.3f} |")
frac=[]
for cases in (10,26,52):
    r=next(r for r in a['pairs'] if r['cases']==cases and r['fold']==0 and r['seed']==0 and r['schedule']==75)
    b,n=r['arms']['baseline'],r['arms']['bands']
    frac.append(f"| {cases} | {b['stop_epoch']} / {n['stop_epoch']} | {b['updates']} | {b['lr_sum_normalized']:.2f} | {r['best_gain_pp']:+.3f} |")
speed=[]
for r in a['pairs']:
    if r['schedule']>75:
        b,n=r['arms']['baseline'],r['arms']['bands']
        speed.append(f"| {r['seed']} | {r['schedule']} | {b['first_reach_epoch']['78'] or 'not reached'} / {n['first_reach_epoch']['78'] or 'not reached'} | {b['first_reach_epoch']['79'] or 'not reached'} / {n['first_reach_epoch']['79'] or 'not reached'} |")
hashes=sum(r['verified_files'] for r in a['checks']['hashes'])
shrink=100*(1-g['fold0_adaptive_200_200_400']['best_gain_pp']/g['fold0_75']['best_gain_pp'])
t0=next(t for t in a['exploratory_timings'] if t['seed']==0)
tb=t0['arms']['baseline']['logged_epoch_seconds_to_target']['80']
tn=t0['arms']['bands']['logged_epoch_seconds_to_target']['80']
cal=t0['calibration_source_epoch_seconds']
text=f'''# Independent audit of boundary bands under scarce training data

19 September 2026. The experimental values below are generated from AUDIT.json, independently rebuilt from archived confusion matrices, configurations and training histories. This audit does not modify the source experiments or the previous handoff. Interpretation is the auditor's reasoning, not an additional experimental measurement.

## Judgment

The original comparison is a valid, controlled demonstration that adding bands improves validation Dice with ten labeled case files under the specified training policy. Equal epochs and equal settings within each pair are appropriate for that question. It is too broad to say that this comparison is simply unfair or shows no training benefit.

It is also too broad to infer that bands reduce the amount of labeled data needed after adequate optimization, prevent overfitting, or improve the eventual attainable performance by the original margin. Those conclusions require additional controls. The longer runs provide strong evidence that the original benefit depends substantially on the optimization schedule. They leave a small positive best-checkpoint residual on fold 0 and do not establish equivalence or absence of a generalization effect.

The handoff's phrase “almost entirely an optimization-speed effect” should be narrowed to: **“On fold 0, the observed advantage becomes much smaller under longer, stretched learning-rate schedules; faster improvement is directly visible in the learning curves.”** Extending this causal explanation to all four folds is not supported by the completed long runs.

## What was independently verified

The audit rehashed {hashes:,} archived file instances against their saved manifests across six collections. These include repeated source files and are not independent scientific observations. All matched. It checked {a['checks']['runs']} production histories, rebuilt {a['checks']['evaluations']} checkpoint evaluations from {a['checks']['case_scores']:,} case-level confusion matrices, and recalculated {a['checks']['calibrations']} weights from saved casewise gradient summaries. The maximum discrepancy between a stored case Dice and reconstructed Dice was {a['checks']['max_case_dice_error']:.3g} on the 0–1 scale.

For every pair, the saved run specifications differ only in constraint selection, the bands weight, and calibration references. Best checkpoint indices agree with maxima of the logged validation Dice. Learning-rate chronology, five-epoch auxiliary warmup, loss arithmetic, and the stopping decisions agree with the histories. The ten-case subset is identical across fold-0 seeds and long controls; fraction subsets are nested (10 within 26 within 52), with the same validation cases. Case IDs are disjoint within each train/validation split. The principal reported gains agree with independent reconstruction.

Scope of assurance: these checks validate the consistency and arithmetic of the collected evidence. They do not rerun GPU training or inference, remeasure the original gradients, independently verify subject identities, or certify the current remote filesystem. Saved remote checks of raw data and weights remain provenance evidence; local checkpoints/raw MRI were not rehashed by this audit. The earlier implementation audit includes numerical gradient and geometry tests; those tests were reviewed as existing evidence, not rerun here. Current uncommitted training-code edits in the desktop repository were not used as the source of these historic results.

Reproduce the audit with `python3 audit.py`, then `python3 build_report.py` from this folder. The scripts require the original archive at `{EXP}`.

## What the experiments show

Dice is the average of anterior and posterior hard Dice per case, then averaged over validation cases; background is excluded. All gains are bands minus baseline in percentage points. “Best” uses each arm's own validation-selected checkpoint. “Final” uses each arm's actual stopping checkpoint, which can occur at different epochs.

| Comparison | Baseline best (%) | Bands best (%) | Best gain (pp) | Final gain (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(aggregate_rows)}

The original four-fold result is positive in all twelve paired configurations. This is useful replication across the selected folds and initialization seeds, but those twelve configurations are not twelve independent datasets. Three seeds share one training subset per fold and reuse validation cases. The low-data subset itself was not resampled within each fold. Four of the five original folds were run.

The same maximum-200 schedule for all three fold-0 seeds yields the intermediate result above. This deserves equal visibility with the mixed 200/200/400 summary: the latter incorporates an additional experiment selected after seed 2 was seen to train poorly. Both arms of that seed received the extension, so the comparison within the pair remains matched. The combined mean is nevertheless an exploratory result under different seed-specific schedules, not the performance of one fixed training policy applied to all three seeds.

| Seed | Maximum epochs / StepLR step | Actual stop baseline / bands | Best epoch baseline / bands | Best gain (pp) | Final gain (pp) | Final baseline supervised loss |
|---|---|---|---|---:|---:|---:|
{chr(10).join(long_rows)}

The fold-0 best-checkpoint difference is {shrink:.1f}% smaller in the mixed long-run summary than in the original fold-0 summary. This is an arithmetic reduction in an observed difference; it is not an estimate that {shrink:.1f}% of the original causal effect is “optimization.” The follow-ups jointly change the training duration and decay timing, and the original-to-long comparison also changes GPU slice. Both arms improve with the revised policy, with the baseline improving more.

At 200 epochs, seed 2 still had high supervised loss and reached its baseline best near its stop. Its 400-epoch rerun is informative evidence that the previous plateau was premature. It is not a proof that any of the longer runs reached the best performance that a well-tuned baseline could attain. There is no universal convergence criterion at supervised loss 0.12–0.19; that range is an observation from these runs. Conversely, a validation plateau can occur before training loss is minimized because validation performance can peak and then deteriorate.

## Fairness depends on the question

**Performance under a fixed training policy.** For the same subset, batch size and epoch schedule, both methods see the same number of training presentations until any differing stopping decision. This supports a conditional benefit of bands. Earlier learning is a useful benefit; it does not become invalid because a longer baseline catches up. “Fixed policy” is more exact than “every arm trained 75 epochs”: some original bands runs stopped earlier.

**Performance at a fixed compute budget.** Equal epochs do not guarantee equal cost. Bands adds loss computation and gradient calibration, including a separate five-epoch source run. Runtime claims should include method overhead and use comparable hardware. A research implementation that trains a full baseline before its bands run is not necessarily the deployment recipe: the calibration needs its five-epoch source, not necessarily a completed baseline. State the recipe being timed.

**Benefit specifically caused by label scarcity.** The fraction pilots change both available examples and optimizer updates per epoch. This entanglement can make a slow-learning baseline look disproportionately weak at the smallest fraction. The observations support the fraction-specific results under that schedule, but do not isolate the interaction between labels and the auxiliary loss after adequate optimization.

| Training cases | Actual stop baseline / bands | Baseline updates | Sum of LR / initial LR over updates | Best gain (pp) |
|---|---|---:|---:|---:|
{chr(10).join(frac)}

These fraction pilots use only fold 0, seed 0. The calibration weight is also recalculated for each subset, and the 52-case run uses a capped 32-case calibration sample. Thus this is an evaluation of the calibrated method across fractions, not a fixed-scalar-loss experiment. Training a larger dataset for more updates is normal for an epoch-based protocol; the issue is the interpretation of the interaction, not that larger datasets must always receive exactly the same number of updates.

**Best attainable performance with scarce data.** Give each method an adequate, equally resourced tuning procedure, select settings/checkpoints on development data, and evaluate a frozen protocol on independent cases. Keeping every hyperparameter numerically identical is a useful controlled ablation; it does not by itself establish the best tuned comparison between different objectives.

## The matched-budget table is descriptive, not a compute control

The handoff's five matched-budget values reproduce from the CSV histories. Their labels need care:

- Its ten-case reference is a five-epoch average over epochs 71–75, centered on 730 updates, not a single checkpoint at the final 750 updates.
- Five-epoch windows cover different update ranges. The 52-case LR-sum point averages epochs 5–9, covering 260–468 updates; the ten-case reference covers 710–750 updates.
- Equal sums of learning rates do not make AdamW trajectories equivalent. Moment estimates, gradient directions/noise, sample order, repeated exposure, warmup and decoupled weight decay all matter. The scalar sum is a descriptive schedule proxy, not an established unit of training or compute.
- Matching update count at different fractions still exposes the models to different amounts of unique data and different epoch-indexed learning rates. It is useful exploratory evidence about early learning, but cannot by itself separate all causes of the fraction trend.

The matching analysis therefore strengthens the motivation to study optimization, but the claim that the larger-data advantage “disappears only as they keep training” is stronger than the design identifies. Prefer “the early advantage is larger and becomes small later in these trajectories.” Do not treat correlated epochs as independent replicates or compare a between-method mean to within-run epoch SD as a significance test.

## What remains of the speed claim

Within each long schedule, both arms use the same hardware slice and stopping policy. The histories directly show earlier attainment of several accuracy levels:

| Seed | Maximum schedule | First epoch at 78% baseline / bands | First epoch at 79% baseline / bands |
|---|---:|---|---|
{chr(10).join(speed)}

These are retrospective first crossings of noisy validation curves, not prespecified sustained targets. They show optimization progress in fewer epochs/updates. A confirmatory speed study should prespecify targets, require a sustained or appropriately smoothed crossing, and report unreached targets explicitly.

The archived timing logs also show that this distinction matters in practice. In the seed-0 200-epoch experiment, first reaching 80% takes {tb:.1f} logged epoch-seconds for baseline versus {tn:.1f} for bands. Bands' separate five-epoch source adds {cal:.1f} logged epoch-seconds, bringing that partial total to {tn+cal:.1f}, even before gradient calibration and startup. Thus a small epoch advantage at that target does not establish an end-to-end time advantage for the implemented calibration recipe. Other seeds and lower targets show larger early advantages. Full overhead-inclusive runtime has not been established by the handoff; AUDIT.json retains the extracted timing evidence and its exclusions.

## Relationship to the attached paper

Source: Bergamin, Dimitri and Aiolli, *Integrating Background Knowledge in Medical Semantic Segmentation with Logic Tensor Networks*, IJCNN 2025, DOI 10.1109/IJCNN64981.2025.11227885. The attached PDF was read and page 5 visually checked. Relevant locations are IV.C and Table I on page 5, the constraint definitions on page 4, and the limitations/conclusion on page 6. [Authors' code repository](https://github.com/BouncyButton/segmentation-with-ltn).

The paper specifies five-fold 80/20 cross-validation, fractions 100%, 25% and 5%, 100 epochs for each model, learning rate 1e-4, batch size 4, warmup cosine scheduling, and no augmentation. Table I reports 74.34% versus 77.21% at 5% (+2.87 pp), 83.31% versus 83.91% at 25% (+0.60 pp), and 85.47% versus 86.28% at full data (+0.81 pp). Those paper values are separately transcribed from Table I; they are not values produced by the local experiment audit.

Your main study follows the same broad fraction-and-fixed-epoch strategy and shares the model family, initial learning rate and no-augmentation choice. It is not an exact reproduction of the published settings: maximum 75 rather than 100 epochs, batch size 1 rather than 4, StepLR rather than warmup cosine, four evaluated folds rather than five, and a different auxiliary objective. The paper combines Dice with contact/distance, nesting and volume constraints; your experiment uses GT-derived inner and outer boundary supervision. The paper's method/results text does not fully specify checkpoint selection and does not clearly settle every implementation detail of foreground-only metric averaging. Numerical similarity of the results is not proof of protocol equivalence.

Following a published fixed-epoch design is a reasonable reason to run this experiment. It does not eliminate that design's inferential limitations. The paper interprets low-data improvements as regularization and better generalization, but its reported experiment does not isolate optimization speed from those explanations through extended or update-controlled training. That is a limitation of the causal interpretation, not a reason to discard its Table I or your corresponding conditional result. Your bands follow-ups also do not prove that the paper's different constraints would lose their benefit under longer training. The paper itself presents the work as preliminary and calls for more extensive evaluation.

The separate MNIST claim in the handoff was not independently audited here; it is unnecessary to settle the attached-paper comparison.

## Corrections and limits to carry into the thesis

1. Replace a blanket “unfair comparison” judgment with the exact claim being tested. There is direct evidence of a benefit under the original scarce-data training policy.
2. Replace “almost entirely optimization-speed effect” with strong evidence of schedule dependence on fold 0. Duration, decay timing, recalibration and hardware change prevent a clean causal decomposition. Seed-2 extensions share the smaller slice, which gives additional within-hardware evidence, but still change the schedule.
3. Label the final aggregate “exploratory mixed-schedule long-training results,” not “the converged result.” Include the all-seed maximum-200 aggregate alongside it.
4. The positive residual is neither an established population benefit nor evidence of no benefit. Three seeds, one subset and one fold do not establish a robust 0.3-point advantage or equivalence. The near-equal long-run seed SDs are descriptive and arise under different seed-specific budgets; they do not prove equal intrinsic stability.
5. Validation was reused for checkpoints, stopping and research decisions. An independently evaluated final checkpoint remains a validation-set result, not an independent test. Maximizing over a longer validation trajectory also changes selection opportunity. Saving the earlier best protects it from later weight drift, but does not make it immune to selection bias.
6. Case separation was verified; unique-subject separation was not. Ten case files must not automatically be called ten patients. Subject/hemisphere mapping remains a priority before claims about unseen subjects. No subject leakage was established by this audit.
7. Bands at gamma zero is numerically a side-balanced boundary BCE on grouped foreground probability. It reuses segmentation annotations with a boundary-focused inductive bias. It does not demonstrate an advantage of LTN notation over the identical conventional loss, or acquisition of independent anatomical information. Compare Dice+ordinary CE and, if isolating localization, a calibrated global grouped-foreground BCE control.
8. Smaller false-positive counts explain an observed change in error profile, but do not establish a complete causal mechanism. A widened train–validation gap does not support the simple claim that bands prevented overfitting; nor does that gap alone rule out every regularization effect. Dice/confusion counts do not establish surface accuracy or topology.
9. “Every number is read from results” is not literally true of the previous generator. It hard-codes established gains (3.882, 3.840, 3.935), source hashes and narrative quantities; its 400-epoch input generator also hard-codes rounded original seed gains. The important rounded results reproduce correctly, so this is a provenance-description issue, not a detected reversal or numerical fabrication. The new experiment tables are computed from the raw archived confusion counts.

## Defensible wording now

> With ten labeled training cases per fold and the specified 75-epoch maximum training policy, adding a calibrated boundary auxiliary loss improved foreground validation Dice across all twelve paired fold–seed configurations, averaging +{g['four_fold_75']['best_gain_pp']:.2f} percentage points. Exploratory longer-training controls on fold 0 substantially reduced the gain: +{g['fold0_uniform_200']['best_gain_pp']:.2f} points with a common maximum-200 schedule, and +{g['fold0_adaptive_200_200_400']['best_gain_pp']:.2f} points after an additional seed-2 schedule extension. The learning curves support faster early optimization. These experiments do not yet establish a comparable improvement after adequate method-specific tuning or on an independent test set.

## Most informative next experiment

Freeze one sufficiently long schedule or one common development/tuning policy before examining the next results, and apply it to both arms across folds and newly sampled low-data subsets. Report the original fixed-policy endpoint as well as validation-selected performance from that common procedure. Use a subject-disjoint test set once the protocol is fixed. A schedule-sensitivity grid is more informative than asserting a universal loss threshold as convergence.

At minimum retain Dice and Dice+bands; add Dice+CE to establish whether a conventional auxiliary loss supplies the same advantage. A grouped-foreground BCE arm with the same calibration framework would more directly test boundary localization. Give methods comparable tuning resources. Prespecify a practically meaningful gain and summarize uncertainty at the appropriate subject/subset/fold/seed levels, rather than bootstrapping cases as though all trained pairs were independent.

For a low-data interaction claim, repeat fractions with sufficient optimization and multiple subsets/seeds. For a speed claim, evaluate curves against optimizer steps and hardware-matched elapsed time, including calibration and auxiliary overhead. These answer complementary questions; neither equal epochs nor equal updates alone resolves all of them.

## Evidence index

- Original four-fold study: {link('final_four_fold_report_20260918','FINAL_REPORT.md')}.
- Previous handoff being assessed: {link('low_data_convergence_findings_20260919','FINDINGS.md')} and {link('low_data_convergence_findings_20260919','build_findings.py')}.
- Longer seed-2 design and adaptive rationale: {link('low_data_noaug_5pct_400ep_es_fold0_seed2_20260918','PROTOCOL.md')}.
- Reconstructed measurements and paths: [AUDIT.json]({HERE/'AUDIT.json'}).
- Independent recomputation: [audit.py]({HERE/'audit.py'}); report generator: [build_report.py]({HERE/'build_report.py'}).

No original result, training code, remote job, checkpoint, or storage-protection manifest was changed.
'''
(HERE/'AUDIT_REPORT.md').write_text(text)
print(f'Wrote {HERE / "AUDIT_REPORT.md"} ({len(text.split())} words)')
