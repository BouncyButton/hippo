# Adjacent voxel coherence audit

Inference-only audit requested on 28 September 2026. Compare Dice baseline,
Dice + bands, and Dice + bands + edge for folds 0–2 and seeds 0–2, ten training
cases per fold, no augmentation, maximum 75 epochs. Reuse each historical
validation-selected best checkpoint; do not retrain or reselect. Evaluate the
ten clean training and 52 validation cases per fold. The primary comparison
is bands + edge against baseline, with bands as the incremental control.

Use the frozen experiment model and data code, padded 64-cube inputs, eval mode
and the original AMP inference. Verify checkpoint hashes against the previous
audit, data/split hashes, model-source hash, fold, seed and training budget.
Record differences in macro Dice relative to historical inference.

Construct the same ground-truth two-step six-connected inner/outer bands. Count
each face once, requiring both endpoints in the union. Calculate per-case
means, then seed means within a fold and equal fold means. Report every fold
and seed. Do not pool the repeated seeds as independent validation subjects.

## Measurements fixed before inference

- Signed face-contrast mean squared error on probabilities, exactly the edge
  loss, split into inner–inner, outer–outer and true crossing faces.
- Hard foreground disagreement on inner–inner and outer–outer faces.
  Also measure both endpoints correct and both endpoints wrong. Agreement
  alone can reward a uniformly incorrect region.
- Correctly oriented hard transitions, reversed transitions, and signed soft
  jump at true crossing faces. These prevent indiscriminate smoothing from
  being mistaken for successful boundary coherence.
- Whole-volume six-connected foreground component count, voxels outside the
  largest component, singleton components and enclosed background voxels.
  These are descriptive fragmentation measures, not complete topology proofs.
- Foreground FN, FP, union Dice, and three-class foreground macro Dice.

Hard segmentation is the foreground union of the three-class argmax, matching
the original evaluation; it is not a 0.5 threshold on summed foreground scores.

## Descriptive volume control

For each case and seed, take the baseline foreground probability ordering and
select exactly as many foreground voxels as the corresponding bands + edge
hard prediction. This uses no GT to choose a volume or threshold. Compare
hard coherence against bands + edge to assess whether a change in foreground
extent alone can reproduce the result. This is a diagnostic decision-rule
change, not a trained model or a causal isolation of the edge term. Its soft
metrics equal the baseline's and its A/P macro Dice is undefined.

## Inference and interpretation

Bootstrap paired case effects after averaging the three seeds for each
case. Resample within folds, keeping folds equally weighted. Check case IDs
for overlap across validation folds; if overlap exists, share bootstrap
weights across occurrences rather than treating them as independent. These
intervals are conditional on the fitted, validation-selected models and do
not cover model selection, training-set, or new-fold uncertainty.

Higher Dice does not mathematically imply greater spatial coherence. A lower
same-side disagreement together with preserved or improved true transitions
supports local coherence; it does not establish global anatomical correctness.
The analysis is exploratory and these validation folds are development data.
