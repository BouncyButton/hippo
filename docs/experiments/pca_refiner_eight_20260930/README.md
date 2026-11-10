# Eight-case anatomical PCA coefficient fitting diagnostic

The question is whether a CNN can reproduce known useful 32-mode corrections
on its own training cases before attempting generalization.

## Prespecified experiment

- Same frozen canonical Dice-only, no-augmentation, early-stopped fold-0
  segmenter and historical dataset, with checkpoint/data/split hashes checked.
- Rebuild the 32 anatomical PCA directions and mean-error offset from all 208
  training cases, using the spectral experiment's distance representation,
  prediction-centroid integer translation, 80-cubed canvas, and +/-8 mm clipping.
- Only eight cases train the regressor. Select two cases from each quartile of
  predicted foreground volume with seed 0; do not select by oracle gain or
  validation performance. No validation case is inferred or evaluated.
- Fit oracle coefficients using exactly the preceding band least-squares
  procedure, including RMS normalization and ridge 0.001. These use annotations.
- CNN inputs: MRI, baseline foreground probability, clipped source signed
  distance divided by 4, predicted inner band, predicted outer band. Translate
  all inputs identically; any clipped distant MRI context is zero-padded.
- CNN: four stride-2 convolutions, channels 16/32/64/64, GroupNorm/GELU;
  adaptive spatial pooling to 4x4x4; linear 4096->128->32 with GELU. Last layer
  initialized to zero. Modes are scaled by their RMS over the union of the eight
  predicted bands, independently of the oracle targets.
- Fixed segmenter, fixed basis, fixed oracle targets. Zero coefficients yield
  the fixed mean-error correction, not a trainable mean template.
- Primary loss: oracle-field mean squared error in each predicted band,
  normalized by that case's zero-coefficient oracle error (floor 0.0001), then
  averaged across cases. Add 0.05 whole-foreground soft Dice loss against the
  annotation; outside-band foreground is fixed even in the soft Dice term.
- Adam, lr 0.0003 cosine-decaying to 0.000003, 1,000 full-batch updates (batch 8),
  gradient norm clip 5, FP32, seed 0, no augmentation/dropout/weight decay.
- Evaluate every 25 updates. Select lowest training relative oracle-field MSE;
  report final step separately. No validation-based selection.
- Prespecified diagnostic success: recover >=80% of mean oracle Dice gain AND
  reduce normalized oracle-field squared error by >=95%. Report per-case gains,
  including oracle regressions or negligible gains if they occur.
- Preserve initial labels outside predicted two-step six-connected bands and
  AP labels on retained foreground. Assign new foreground the nearest original
  AP label. No invertibility or topology guarantee is claimed.

This is deliberate fitting, not a generalization or few-shot-learning claim.
The anatomical basis uses 208 training annotations, including the selected eight.
The original segmenter's predictions are in-sample. Positive oracle gain is
not assumed: a nonpositive mean gain makes gain recovery undefined and the
success criterion unmet.

## Execution and checks

All numerical computation runs via `sbatch`. Unit checks cover DCT/PCA and
oracle fitting, deterministic label-free selection, image translation, exact
oracle field targets, and gradients through the coefficient head. The complete
job independently verifies saved Dice/error counts, prediction bands, preserved
outside-band/AP labels, and exclusion of validation cases (including case 164).

Source: `semantic_constraints/pca_refiner/`.
Launcher: `scripts/run_pca_refiner_eight.sh`.

Cluster directory: `/home/3160552/pca_refiner_eight_20260930_01`.
Slurm job: 677517, completed successfully in 4 minutes 55 seconds. It reused the
canonical checkpoint in the spectral-analysis directory. Seven unit checks and
the independent saved-output verifier passed.

## Results

Selected training cases: 051, 057, 075, 106, 133, 215, 224, and 238.
The CNN completed 1,000 updates; step 625 had the lowest training field error.

| Mean foreground Dice (background excluded) | Value |
| --- | ---: |
| Frozen baseline | 0.94947132 |
| Annotation-guided 32-mode oracle | 0.94969372 |
| CNN-predicted 32-mode correction | 0.94969372 |

The CNN matched oracle Dice on all eight cases and reduced normalized oracle
field MSE to 4.35e-8. It therefore passed the prespecified fitting criterion.
However, the available oracle improvement was only 0.02224 percentage points:
13 errors were fixed and one introduced, two cases improved, and six masks
were unchanged. This demonstrates fitting of these targets, not useful
generalization or an ability to learn large corrections. The final step had
the same Dice as the selected step.

Independent verification confirmed exact preservation of labels outside the
bands and anterior/posterior labels on retained foreground. All eight baseline,
oracle, and learned masks had one connected component. Validation was not
evaluated; case 164 was excluded. The basis used all 208 training annotations,
so this is not a claim of learning anatomy from eight subjects.

Saved metrics, protocol, weights, basis, case outputs, plot, and verification:
`results_677517/`. The plot's Dice axis is magnified; the absolute gain is small.

## Follow-up teacher-strength audit

Slurm job 677522 applies the same fixed basis, mean-error offset, and
annotation-guided coefficient fitting to all 208 training predictions. This
post-hoc diagnostic checks whether the selected eight had unusually weak
oracle gains. It does not train another CNN or access validation cases.
Source: `semantic_constraints/pca_refiner/audit_teacher.py`;
launcher: `scripts/run_pca_teacher_audit.sh`.

The audit completed successfully in 2 minutes 6 seconds. Its selected-eight
scores reproduced the original run. Across all 208 training cases, baseline
mean Dice was 0.93833327 and oracle mean Dice was 0.93904025: a gain of
0.07070 percentage points. There were 98 improvements, six regressions, and
104 unchanged masks; the median gain was zero. Only 46 cases gained at least
0.1 percentage points and one gained at least one percentage point. The oracle
fixed 1,474 errors and introduced 410. Detailed rows and provenance are saved in
`results_677517/teacher_audit_208.json`.

The selected eight had weaker gains than the full training set, but weak
hard-mask correction targets are not confined to that sample. For context,
the preceding validation oracle analysis achieved 0.90097752 -> 0.91449356
with 32 anatomical modes. Those validation coefficients used annotations;
they are not learned predictions and are not new validation evaluation here.

## Interpretation and next experiment

The CNN can fit these continuous correction fields, but this diagnostic does
not establish that it can learn substantial boundary repairs. The original
segmenter saw all 208 training subjects, and its in-sample errors yield much
smaller oracle gains than the held-out validation errors. This is evidence of
a mismatch in correction targets, not proof of its cause or of a remedy.

Before full refiner training, generate out-of-fold baseline predictions within
the 208 training subjects: each prediction must come from a segmenter trained
without that subject. Keep the original 52 validation subjects outside this
process. Recheck oracle correction gains on those predictions, then repeat
the small fitting diagnostic if useful targets exist. This is a recommendation,
not a submitted run. The present signed-distance least-squares oracle is also
not a maximum-Dice optimizer, so its weak gain is not an absolute limit on what
32 coefficients could achieve under another fitting objective.

## Frozen-model validation evaluation

At the user's request, job 677530 evaluates the existing eight-case refiner on
all 52 fold-0 validation cases. The preceding fitting diagnostic and teacher
audit did not assess learned generalization; weak training oracle gains do not
substitute for this evaluation or establish that it will fail.

The saved step-625 checkpoint, 32-mode basis, mean-error offset, coefficient
normalization, zero threshold, and prediction-derived two-step bands are fixed.
The baseline and refiner run in evaluation/inference mode with no optimization.
The refiner function receives only MRI, baseline labels, and probabilities;
annotations enter only after the output is fixed, to compute metrics. No
validation oracle is fitted. Both label outputs are saved for independent
verification. Hash checks ensure checkpoint, basis, and training protocol are
unchanged after evaluation.

Primary comparison: paired whole-hippocampus foreground Dice, excluding
background. Also report ASSD, HD95, per-case gains, errors fixed/introduced,
and a subject-level paired bootstrap interval (10,000 resamples, seed 0).
This tests a regressor trained on eight cases with a basis learned from 208
training annotations; it is not a refiner trained on all 208 subjects.

Source: `semantic_constraints/pca_refiner/evaluate_validation.py` and
`verify_validation.py`; launcher: `scripts/run_pca_refiner_validation.sh`.

Job 677530 completed successfully in 1 minute 53 seconds. Saved outputs and
independent verification are in `validation_677530/`.

| Validation metric, 52 subjects | Baseline | Frozen eight-case refiner |
| --- | ---: | ---: |
| Mean foreground Dice | 0.9009775184 | 0.9009775184 |
| Mean ASSD (mm) | 0.4214657223 | 0.4214657223 |
| Mean HD95 (mm) | 1.1415282270 | 1.1415282270 |

All 52 binary masks were unchanged: zero improved, zero worsened, zero errors
fixed, and zero errors introduced. The paired mean Dice gain and bootstrap
interval were exactly zero for these saved masks. The baseline Dice matched
the previous canonical validation result. The independent verifier recomputed
Dice/error counts and confirmed preserved outside-band and retained AP labels.
Checkpoint, basis, and protocol hashes were unchanged by evaluation.

This directly establishes that the present eight-case refiner provides no
hard-mask improvement on fold-0 validation. Its predicted continuous distance
corrections do not cross any voxel's foreground/background threshold. It does
not establish that a refiner trained on all 208 subjects, a different objective,
or the broader anatomical correction approach cannot generalize. The earlier
validation oracle improvement used annotations to choose coefficients and
must not be reported as learned validation performance.
