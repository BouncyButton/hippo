# Separated edge rules: one fold, three seeds

Authorized pilot: fold 0, seeds 0/1/2, two paired arms per seed (six new models).
No fold-1/2 training or automatic expansion. Review results before another fold.

## Controlled comparison with the original 75-epoch runs

Use the historical frozen source from edge_lowdata_folds12_20260924_01/source.
Verify all original model, data, band-loss and training source hashes before fitting.
Same ten training and 52 validation cases, same split hash, no augmentation,
random initialization reset to the same seed for both arms. Use the original
AdamW learning rate 1e-4, weight decay 1e-5, StepLR every 20 epochs with gamma
0.5, batch size 1, AMP, original spatial padding/resize configuration and five-epoch
constraint warmup. Keep the original per-seed unary-band coefficient.
Preserve maximum 75 epochs, minimum 60, patience 8, minimum delta 0.0005;
select the greatest validation hard macro Dice exactly as before. The original
experiment was capped at 75 epochs, not necessarily trained/selected at epoch 75.

Pooled control: original BoundaryEdgeConsistencyLoss implementation AND exact
historical per-seed edge coefficient. It is rerun with observational telemetry.
Separated candidate: disjoint inner-inner, outer-outer and crossing face sets
inside the same two-step GT bands. Average each rule within each case, then
patients, then take one third of each rule. No change to edge definition,
foreground union, bands, model architecture, supervised loss or optimizer.
All three face sets must be nonempty for every calibration/training case;
otherwise stop. General loss implementation returns zero for empty rules.

## Same strength, training-only calibration

Use each seed's retained epoch-five Dice checkpoint and all ten training cases.
For each arm measure the RMS gradient of the raw edge loss with respect to logits,
divided by that of Dice plus the unchanged weighted band term. Preserve the pooled
coefficient, and set the separated coefficient so its median weighted ratio is
identical. Require both p95 ratios <= 0.5, matching the historical guardrail.
If exact matching violates that guardrail, stop for review; do not silently lower
or increase the control coefficient. Record all raw gradients, ratios and scalars.
This matches a calibration-time logit-gradient budget, not the entire training
trajectory or the actual AdamW parameter-step magnitude. Do not adapt weights
online in this first experiment.

## Monitoring and endpoints

At epochs 0,5,15,30,45,60,75 and final stop, probe the first two lexically sorted
training cases. Record loss values, logit and all-trainable-parameter gradient
norms and pairwise cosines for Dice, weighted unary bands, the three raw edge
rules and the active edge combination. Also record the scheduled base and weighted
edge gradients. Probes use FP32 eval mode to avoid AMP underflow in unscaled
per-rule derivatives, never step an optimizer, and restore Python,
NumPy, torch, CUDA and loader-generator RNG states. Equal formula weights do not
imply equal gradients. Negative cosine is descriptive, not proof of a harmful step.

Record each rule's training loss every epoch. Audit full validation geometry at
monitoring epochs and the selected model on all training and validation cases.
Retain a shared epoch-60 spatial endpoint and compare learning curves at common
epochs. Include original baseline, bands and bands+edge selected-checkpoint spatial
results from the completed coherence audit plus the original validation-Dice
histories. Compare pooled rerun versus original edge trajectories explicitly.

Primary comparison: separated minus pooled for inner same-label disagreement
and FN. Report FP, correct boundary transitions, macro/union Dice, individual
soft rule losses and fragmentation alongside them. Desirable outcome is lower
inner disagreement without increased FN, retaining FP and boundary improvements.
These are descriptive direction checks, not a statistical pass/fail guarantee.
Average three repeated seeds before paired case bootstrap (10,000 resamples).
This reused development fold is exploratory; no independent-test claim.

## Execution and storage

One sequential stud allocation, gpu:3g.40gb, 4 CPUs, 24 GiB RAM, maximum 3 hours.
Queue after the running MedSAM3 fold-3 job; do not interrupt it. Require 1.6 GiB
free at allocation start and 0.8 GiB before each active arm. Retain six selected
FP32 checkpoints (~0.4 GiB total) plus metrics, audits and calibration. Save full
optimizer/scheduler/scaler/RNG and stopping state for the active arm each epoch,
allowing an explicit rerun to resume it. Retire only that arm's latest intermediate
after its selected checkpoint and final audit are durable. Never delete prior
experiments or modify their weights. Hash the full source payload; stop on drift.
