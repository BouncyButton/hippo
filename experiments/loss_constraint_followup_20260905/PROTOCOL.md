# Loss-constraint follow-up, 2026-09-05

Status: the official-baseline GPU audit completed (650074). The user subsequently authorized the original Family-B equivariance replication: seeds 0/1 are submitted as **650078/650080** from exact frozen baseline source. The current launch order and implemented augmentation/common-support follow-up are in [NEXT_PROTOCOL.md](../equivariance_family_b_20260905/NEXT_PROTOCOL.md), which supersedes the proposed launch order below. See [GPU_AUDIT_RESULTS_20260905.md](GPU_AUDIT_RESULTS_20260905.md) for audit evidence. No teacher, A/P or Dice+CE training has been submitted.

## Decisions

1. Add an explicit Dice+CE control. Treat improved calibration or Dice as hypotheses, not consequences guaranteed by CE. Preserve the original Dice loss as the default.
2. Implement translation-teacher KL as the next trainable constraint candidate. It has a corrective logit gradient when teacher and student disagree, including when the student is confident. It can still converge to an equally wrong, translation-invariant fixed point.
3. Implement a supervised A/P cut-posterior candidate and test its ground-truth assumptions before training. **The strict all-case planar formulation fails the current label audit.** Keep it on hold; do not silently exclude incompatible cases or change labels to make the rule true.
4. Treat entropy-only cut sharpening as an insufficient objective: it can sharpen the wrong answer without changing its rank. The implemented A/P objective uses the training label to identify the target cut.

These changes are proposed mechanisms, not measured segmentation improvements.

## Loss and measurements

`--supervised-loss dice` constructs the exact original `DiceLoss(to_onehot_y=True, softmax=True)` with installed MONAI defaults, including background. `--supervised-loss dice_ce --ce-weight 1` adds ordinary mean multiclass cross-entropy over the entire crop. CE operates on float32 logits without probability clipping, class weighting, or label smoothing. Dice retains its original arithmetic. Record the explicit loss identity, CE coefficient, source/runtime/data/split digests, seed, batch size and precision in every new config/checkpoint.

CE adds a new term and changes relative gradient scales. Old constraint weights and Dice-only gradient calibrations are not portable automatically. Legacy bands/onecut validators now reject a Dice+CE source checkpoint; those presets currently require Dice-only calibration. New teacher/AP weights are explicit experimental inputs; the diagnostic audit's suggested coefficient is not an approved training calibration.

`--calibration-diagnostics` records unclipped NLL, multiclass Brier score (sum over classes), entropy, confidence, accuracy, fraction with max probability >= 0.99, and 15-bin top-label ECE. Strata are whole crop, GT foreground, union of GT/predicted foreground, GT multiclass boundary, correct voxels, and incorrect voxels. The boundary includes both sides of within-FOV six-neighbor class transitions. These diagnostics are float32 calculations on the logits produced by the selected inference mode: casting AMP logits does not turn them into a separate full-fp32 model inference.

Training CSV diagnostics are **voxel-pooled within each stratum**, with counts and null values for empty strata. The official hard-Dice endpoint remains the existing **per-case/class foreground mean** and its arithmetic is unchanged. Never substitute one pooling for another. Per-case calibration values are available through the diagnostic helper/audit; whole-volume ECE is not evidence about A/P uncertainty.

## Comparisons and endpoints

Use fold 0, batch size 1, AMP on, AdamW lr 1e-4/wd 1e-5, StepLR step 20/gamma 0.5, 64-cubed inputs, no resize, fixed split, and seeds 0 and 1. All other data transforms remain unchanged. Name the supervised-objective variants explicitly (`B-Dice` and `B-DiceCE`) rather than pooling them into one baseline.

Primary endpoint: paired constraint-versus-own-supervised-control difference in the mean epoch 41–50 hard Dice. Report epoch 50 separately for continuity with the earlier preregistered bands endpoint; report mean epoch 21–30 as secondary. Best epoch is exploratory. Record both per-seed deltas and their mean; two seeds support a replication check, not a precise universal variance estimate or significance threshold. Report the observed matched-control seed gap for the same window (the handoff reports 0.001207 mean absolute gap at epochs 41–50).

Do not gate on epoch-5 validation Dice. A short prefix can check numerical behavior, timing, source compatibility, and gradients; it cannot decide converged quality. Do not treat successful tiny-model tests or a matching five-epoch prefix as proof that all future source revisions preserve 50-epoch trajectories. New-code Dice controls need an explicit source bridge before old trajectories can be reused. Never compare a seed-1 candidate with only the seed-0 control.

For loss attribution, compare:

| Question | Treatment | Required control |
|---|---|---|
| Does CE help? | Dice+CE, no constraint | Dice, no constraint, same seed/config/exposure |
| Does legacy equivariance transfer? | Dice + existing equivariance, Family B | Family B Dice, no constraint |
| Does the new teacher help? | Chosen supervised loss + teacher | Same supervised loss, no constraint |
| Does teacher KL improve on the old rule? | Teacher KL | Existing equivariance under the same supervised loss and exposure |
| Is extra translated input the explanation? | Teacher KL | Matched translation-augmentation control before claiming a distinct consistency mechanism |

Do not change CE, the constraint formulation, shift set, schedule, and stopping rule in one comparison and attribute the total change to a single ingredient. Translation-only augmentation is now implemented; its exact exposure and limitations are in NEXT_PROTOCOL.md. Treat the repeatedly explored fold-0 validation cohort as development data. Freeze the candidate before confirmation on other folds; five-fold performance and comparison with the reference paper require matched protocols, not a comparison of isolated headline numbers.

## Proposed launch order (requires approval)

First approve a source/timing smoke check and the **two no-constraint Dice+CE controls**, seeds 0 and 1, in fresh run directories. Existing Dice controls can supply the comparison only after the source bridge passes. The previous 50-epoch control took about 18.5 hours; this is an estimate for the new loss, not a measured runtime. The pair is about 37 aggregate GPU hours before extra diagnostics. No automatic sweep or follow-up jobs are authorized by this document.

After that baseline is evaluated, select the supervised objective without claiming it must unsaturate the model. For a teacher run, first freeze a training-only calibration cohort: at most 32 deterministic fold-0 training cases, no validation cases, with source checkpoint/epoch/hash, preprocessing and shift/temperature settings recorded. Compute auxiliary and **selected supervised-loss** logit-gradient RMS; use the existing policy of a 10% median ratio with a 50% p95 cap. Measure these ratios again later as diagnostics rather than automatically changing lambda. A diagnostic report from arbitrary cached cases cannot substitute for this calibration.

The teacher defaults to two distinct sampled views from 12 nonzero shifts (±1 and ±2 along each axis). This differs from the old six-shift ±2 pairwise constraint and from the 13-view inference TTA that includes identity. Report the distinction. Teacher probabilities are inverse-mapped with per-voxel valid-view normalization; padded invalid voxels never become background targets. Student KL is averaged over valid voxels per patient, then patients. Teacher forwards use `no_grad`, temporary evaluation mode, and restored buffers/RNG/mode; student training state is preserved. The weight is external to the objective, with the existing five-epoch warmup.

Before a 50-epoch teacher submission, time a bounded step/epoch sample: two extra teacher forwards can exceed the 24h10m Slurm limit. Do not assume the control's 18.5h runtime applies. If a continuation protocol is needed, design and freeze it first. Validate two seeds at converged exposure before a quality claim.

Do not queue strict `ap_cut` on this dataset. The audit found mixed A/P slices in 47/208 training cases and 11/52 validation cases; all current validation GT exports match the NIfTI data exactly. A future approximate-plane formulation needs its own specification and falsifier. The current API's explicit `invalid_policy='skip'` is useful for diagnostic coverage counts; it is not permission to silently change the training cohort.

## Concrete control commands

Run these from a **new frozen source copy** on the cluster only after approval. Preserve completed protocol directories and active-run source manifests. Creating or modifying files under `thesis/new_constraints` changes the protected source digest.

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set none --supervised-loss dice_ce --ce-weight 1 \
  --calibration-diagnostics --batch-size 1 --epochs 50 --fold 0 --seed 0 \
  --output-dir /mnt/beegfsstudents/home/3160552/loss_followup_20260905/dice_ce_none_seed0

sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set none --supervised-loss dice_ce --ce-weight 1 \
  --calibration-diagnostics --batch-size 1 --epochs 50 --fold 0 --seed 1 \
  --output-dir /mnt/beegfsstudents/home/3160552/loss_followup_20260905/dice_ce_none_seed1
```

The wrapper enables AMP by default. Do not add `--no-amp`. Run output directories must be new; there is no submission script in this folder that queues these automatically.

## Frozen-logit falsifiers

`thesis/new_constraints/audit_followup.py` consumes cached logits and labels, and optional logits from translated views. Use `--purpose diagnostic`, a new JSON output path, and explicit loss/constraint/geometry options. It records source and input digests, same-pool Dice/error counts, gradient RMS/cosine and finite logit-step repairs. Its diagnostic coefficient is not a calibrated training launch parameter. It needs no optimization of model weights.

**Teacher:** reject the particular cached teacher/protocol if it adds padded-edge artifacts, is worse in the target error strata, has nonfinite/zero corrective gradients despite disagreement, or consistently points away from supervised repair. An identical wrong translation-invariant teacher must give zero gradient; this is an expected fixed point, not evidence of correctness. A direct logit repair gain is a mechanism check, not a prediction that network training will realize it.

**A/P:** validate every training label's selected tensor axis and orientation, mixed-slice counts, ambiguous gaps, and allowed cut set before inspecting improvement. The strict all-case formulation already fails. On compatible cases only, check corrective gradients on saturated wrong cuts, zero background-channel gradients and unchanged outside-support logits. Report support coverage and skipped reasons alongside any repair number. A GT-anchored frozen repair is an oracle diagnostic, not inference performance.

**CE mechanism:** compare gradients of Dice and Dice+CE on confidently wrong synthetic and real logits; record calibration by region over training. If CE does not improve calibration or the post-peak trajectory, reject that proposed mechanism without declaring all auxiliary losses impossible. Whether CE increases TTA gain must be measured separately, not assumed.

## Verification and remaining limits

New unit tests check exact legacy Dice values/gradients, stable CE/KL on confidently wrong logits, teacher alignment/validity/state restoration, A/P validity/orientation and gradients, and diagnostic arithmetic. Tiny-model integration checks cover CLI, scalar losses, CSV/JSON serialization, checkpoint identity and resume. These checks establish implementation behavior; they do not measure real-data training performance.

The pre-existing broad equivariance test collection references a missing `equivariance.evaluate_closure` module. The explicit core equivariance/bands/onecut/telemetry tests pass; the missing-module issue is outside this change.

The initial local-checkpoint copy was blocked by automatic approval review. The user subsequently requested execution on the cluster instead. **That GPU audit completed successfully**, with no checkpoint export or training launch. The full teacher improved same-pool inference Dice by +0.006946, while its incremental frozen-gradient repair over supervised alone was much smaller; the results document above gives the limits and revised next steps.
