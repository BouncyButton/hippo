# Semantic Constraints: Architecture and Design Decisions

This document is the durable context for future Codex/GPT sessions working on
`semantic_constraints`. Read it before changing the pipeline. The
shorter `README.md` is the usage reference; this file records the reasoning,
contracts, experimental rules, and current limitations.

## Objective

The research goal is to let an LLM discover useful differentiable semantic
constraints for semantic segmentation without allowing it to generate
arbitrary PyTorch code.

The intended high-level procedure is:

1. Train a normal segmentation baseline.
2. Freeze it and probe predictions against ground truth on a discovery split.
3. Search over a small DSL of generic differentiable descriptors.
4. Evaluate and revise candidate constraints without retraining the model.
5. Select a compact, non-redundant set of constraints.
6. Retrain or fine-tune once with segmentation and constraint losses.
7. Compare against an equally fine-tuned unconstrained baseline on an untouched
   final holdout split.

The proposed contribution is therefore constraint-language search with
empirical feedback, not unrestricted loss-function generation.

## Core Design Decisions

### Constrained DSL, not generated code

The LLM should compose constraints from primitives implemented and tested once.
The currently executable grammar is deliberately narrow:

```text
primitive(class_i[, class_j]) <= alpha
primitive(class_i[, class_j]) >= alpha
```

Each rule has one semantic threshold, `alpha`. The JSON representation also
stores a normalization scale, a loss weight (`lambda`), and fuzzy-semantics
parameters. More expressive Boolean or arithmetic composition is an intended
extension, not currently implemented.

### Generic descriptors, not anatomy-specific rules

The primitive library describes geometry, topology, relations, and prediction
statistics. It contains no hard-coded hippocampus knowledge. Domain-specific
meaning is introduced by binding generic primitives to semantic classes and by
learning a direction and threshold from data.

### Discover before retraining

Candidate discovery operates on cached frozen-model predictions and ground
truth. Many rules and alpha values can therefore be screened cheaply. The
design avoids retraining once per proposal; retraining happens only after
selection.

### Rule induction uses both purity and coverage

A useful candidate should describe ground truth reliably and apply to enough
samples. Ground-truth purity alone can select rare or vacuous rules; coverage
alone does not establish correctness. Prediction violations then identify a
potential training signal.

### Fuzzy loss is the default

Selected constraints default to `soft_exp` fuzzy semantics. A hard one-sided
violation is still needed to determine whether a threshold rule is violated,
but the normalized violation is converted to a smooth truth value rather than
used only as an unbounded ReLU penalty.

### Final evaluation must remain untouched

The discovery fold may be used for probing, alpha selection, constraint
selection, and model selection. The final holdout fold may be used only after
all those choices are fixed. A constrained run must be compared with an
unconstrained run initialized from the same checkpoint and given the same
training data, optimizer settings, and number of epochs.

## Pipeline and Module Boundaries

```text
baseline checkpoint
        |
        v
probe_model.py
        |  probe_foldN_{val|train|both}.{json,csv}
        v
evaluate_candidates.py
        |  candidate_report.{json,md}
        v
counterfactual_repair.py
        |  repair_report.{json,md} and optional repaired-mask NPZ files
        v
select_constraints.py
        |  selected_constraints.{json,md}
        v
compiled_losses.py -----> train_with_constraints.py
                               | baseline and constrained checkpoints
                               v
                         final_evaluate.py
                               | final holdout report
```

| Module | Responsibility |
| --- | --- |
| `primitives.py` | Differentiable operations over batched 2D/3D soft masks. |
| `registry.py` | Primitive metadata for search/LLM-facing discovery. |
| `probe_model.py` | Frozen SwinUNETR inference and prediction-vs-GT descriptor collection. |
| `evaluate_candidates.py` | Observation extraction, alpha sweep, diagnostics, scoring, and redundancy estimation. |
| `counterfactual_repair.py` | Frozen prediction-logit repair for candidate screening and alpha/strength/fuzzy-setting diagnostics. |
| `select_constraints.py` | Rule-based filtering and executable JSON generation. |
| `compiled_losses.py` | Safe interpretation of selected JSON as differentiable fuzzy losses. |
| `train_with_constraints.py` | Matched constrained or unconstrained fine-tuning. |
| `final_evaluate.py` | One-time baseline-vs-constrained evaluation on the reserved holdout fold. |
| `on_grokking_behavior.py` | Independent exploratory modular-arithmetic experiment; not part of the segmentation pipeline. |

The current selector is deterministic and rule based. The intended LLM loop
would consume the compact candidate report, propose/revise DSL expressions or
selection settings, and receive the same diagnostics. That orchestration layer
does not exist yet.

Counterfactual repair is deliberately diagnostic: it tests whether enforcing a
candidate provides a useful local correction direction while preserving the
original prediction. It does not predict the exact Dice gain from fine-tuning,
because repair independently optimizes one logit tensor per sample whereas
training updates one shared parameter vector. Candidate and repair tuning must
remain on the discovery split; the final holdout is still untouched.

Repair ranking is feasibility-aware. A configuration must satisfy the configured
minimum repair-success rate, improve soft Dice, and avoid reducing mean hard
Dice before receiving a nonzero score. The score uses soft Dice as the sensitive
local signal and penalizes harmful repairs. Since finite softmax probabilities
cannot produce exact-zero overlap or containment, exact-zero `<=` candidates are
tested at explicit positive alpha floors. Aggregate trajectories and gradient
norms distinguish infeasible thresholds, saturated losses, slow optimization,
and genuinely weak correction directions.

## Primitive Contract

Primitives accept soft masks shaped `(B, *spatial)` or `(B, 1, *spatial)` and
normally return one scalar per sample. They support gradients with respect to
the masks.

Current primitives:

| Primitive | Meaning |
| --- | --- |
| `volume` | Sum of soft mask mass in physical units. |
| `centroid` | Soft center of mass in physical coordinates. |
| `distance` | Euclidean distance between two soft centroids. |
| `overlap` | Soft intersection, Dice, or IoU. |
| `contains` | Fraction of the second mask's mass inside the first. |
| `adjacent` | Soft boundary-contact score within a voxel radius. |
| `boundary_length` | Total-variation perimeter in 2D or surface measure in 3D. |
| `compactness` | Scale-normalized boundary-to-volume penalty; lower is more compact. |
| `entropy` | Binary entropy of the soft mask. |
| `connectedness` | Soft reachability from a learned seed, not an exact component count. |

Physical descriptors must receive spacing consistently. The current
segmentation defaults are `(1.5, 0.5, 1.5)`, with inputs resized to
`64 x 64 x 64`.

Primitive names describe measurements, not desired semantics. For example,
`contains(A, B)` returns a high value when `B` lies in `A`; whether a useful
rule is `<= alpha` or `>= alpha` is inferred from ground truth.

## Probe and Candidate Induction

`probe_model.py` runs the frozen model and records class and class-pair rows.
The JSON includes metadata and all rows; CSV is a flatter inspection format.
Candidate induction accepts either, although JSON is the normal pipeline
artifact.

Applicability is explicit:

- A class rule applies only when that class has ground-truth volume above
  `--min-gt-volume`.
- A pair rule applies only when both classes are present.
- `coverage = applicable samples / total samples`.
- `GT purity = fraction of applicable labels satisfying the rule`.

For each descriptor and each direction, candidate induction sweeps alpha over
ground-truth values and quantiles. It retains alpha values meeting minimum GT
purity, then favors candidates with:

- high coverage;
- high GT purity and few exceptions;
- frequent prediction violations;
- a large normalized prediction-vs-rule gap;
- positive Pearson alignment between violation and segmentation error;
- low correlation with higher-ranked candidates.

The current raw score is:

```text
coverage * purity * (
    0.45 * prediction_violation_rate
  + 0.35 * min(normalized_violation_gap, 1)
  + 0.20 * max(error_alignment, 0)
)
```

It is then reduced by the configured redundancy penalty. This score is a search
heuristic, not a statistical guarantee of improved generalization.

## Selected Constraint Schema and Loss

`selected_constraints.json` uses schema
`semantic_constraints.selected.v1`. Each constraint contains at least:

```json
{
  "name": "01_compactness_class_1_le",
  "primitive": "compactness",
  "args": ["class_1"],
  "direction": "<=",
  "alpha": 18.0,
  "scale": 4.7,
  "lambda": 0.1,
  "fuzzy": {"mode": "soft_exp", "margin": 1.0, "beta": 4.0}
}
```

For descriptor value `v`, threshold `alpha`, and positive scale `s`, the
compiler computes:

```text
gap = relu(v - alpha)       for v <= alpha
gap = relu(alpha - v)       for v >= alpha
z   = gap / s / margin
```

The default fuzzy truth and loss are:

```text
truth = exp(-beta * z^2)
loss  = 1 - truth
```

The compiler also supports `lukasiewicz`, `lukasiewicz_ste`, and `hinge`.
`lukasiewicz_ste` clamps truth to `[0, 1]` in the forward pass while preserving
the unclipped gradient. `soft_exp` remains the default because it is bounded
and smooth; its `beta` and normalization scale control saturation.

Training minimizes:

```text
L = L_seg + constraint_multiplier * sum_i(lambda_i * L_constraint_i)
```

`constraint_multiplier` is a global experiment-level control. Per-rule
`lambda` values live in the selected JSON.

## Data Splits and Experimental Validity

The scripts use a deterministic shuffled K-fold partition with NumPy seed 42.
With `--fold 1 --holdout-fold 2`:

- fold 1 is the discovery/validation fold;
- fold 2 is the final holdout fold;
- training uses the remaining folds;
- `--train-fraction` is applied to that remaining training pool.

The valid experimental sequence is:

1. Fit or choose the initial baseline without using the final holdout.
2. Probe only the discovery split.
3. Select constraints and all hyperparameters using discovery artifacts.
4. Fine-tune constrained and unconstrained models from identical initial
   weights with identical data exposure.
5. Select checkpoints without consulting the final holdout.
6. Run `final_evaluate.py` once on both fixed checkpoints.

Do not induce constraints from the final holdout. Re-probing a constrained
model is useful for diagnostics, but changing rules based on the holdout would
invalidate the generalization comparison.

## Artifacts and Logging

Default discovery artifacts are under `probe_outputs/`:

```text
probe_fold1_val.json/.csv
candidate_report.json/.md
selected_constraints.json/.md
final_holdout_evaluation.json/.md
```

Training runs are under `runs/<run-name>/` and contain:

```text
config.json
metrics.jsonl
metrics_history.json
latest_model_weights.pth
best_model_weights.pth
```

Training metrics include segmentation and constraint losses, per-class and
foreground Dice, per-constraint truth/violation/value, and
`train/weight_l2_squared`. The latter is `sum(p**2)` over trainable parameters.
AdamW applies decoupled shrinkage directly and does not add this scalar to the
reported training loss.

Use separate output directories for separate discovery rounds and training
runs. Defaults can overwrite earlier reports, and existing files may represent
historical experiments rather than the latest intended configuration.

## Standard Commands

Environment:

```bash
cd /Users/berga/PycharmProjects/remote_ideas
source baselines/unetr_plus_plus/py39/bin/activate
export MPLCONFIGDIR=/private/tmp/mpl
export XDG_CACHE_HOME=/private/tmp/xdg
export PYTHONPYCACHEPREFIX=/private/tmp/pycache
```

Discovery:

```bash
python semantic_constraints/probe_model.py \
  --device cpu \
  --load-weights <baseline-checkpoint>

python semantic_constraints/evaluate_candidates.py \
  --probe-report semantic_constraints/probe_outputs/probe_fold1_val.json \
  --top-k 30

python semantic_constraints/select_constraints.py \
  --candidate-report semantic_constraints/probe_outputs/candidate_report.json \
  --fuzzy-mode soft_exp \
  --max-constraints 5
```

Matched training and final comparison are documented in `README.md`. Preserve
the same initialization, folds, epochs, learning rate, weight decay, batch
size, and training fraction across constrained and unconstrained runs.

## Current Limitations and Technical Debt

- There is no LLM orchestration loop yet. Candidate generation and selection
  are currently fixed Python heuristics.
- The executable grammar supports one primitive and one threshold per rule; it
  does not yet support ratios, arithmetic composition, quantifiers, or fuzzy
  logical connectives.
- Lambda selection is fixed or score-scaled, not optimized independently.
- Split construction is duplicated across scripts instead of being represented
  by one persisted split manifest.
- Training checkpoints currently save model weights only. Optimizer state,
  scheduler state, epoch number, RNG state, and best-score state are not saved,
  so interrupted training can only be approximately continued from weights.
- SwinUNETR and the Decathlon data path are hard-coded in the executable
  scripts even though the primitive layer itself is model- and domain-agnostic.
- Soft topology and boundary primitives are surrogates. In particular,
  `connectedness` is expensive and is not an exact connected-component count.
- Constraint quality is estimated on one discovery fold and can overfit it.
  Repeated folds or nested validation are needed for stronger evidence.
- Current device choices are `auto`, `cpu`, and `cuda`; this environment has
  generally used CPU because MPS/3D-convolution compatibility is uncertain.

## Guidance for Future Sessions

Before modifying or running the pipeline:

1. Read this file and `README.md`.
2. Inspect the exact run's `config.json`; do not infer settings from its name.
3. Identify which checkpoint produced the probe report.
4. Confirm that the holdout fold was excluded from both training and discovery.
5. Inspect `candidate_report.md` before accepting automatic selections.
6. Treat generated reports and run directories as experimental artifacts, not
   source-of-truth configuration.
7. Keep the primitive library generic and keep generated/LLM output inside the
   constrained JSON schema.

The next architectural milestone should be a resumable checkpoint format and a
persisted data-split manifest. After those experimental-integrity issues, add
the LLM proposer/reviser around the existing compact candidate report rather
than bypassing the DSL compiler.
