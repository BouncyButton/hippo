# Does contextual information improve A/P boundary placement?

**Decision: the fixed screening criterion was not met. Do not add joint graph training on the strength of this result.**

We tested 208 training cases and 52 development cases using two frozen, early-stopped seed-0 checkpoints. The probe already sees model probabilities, its fitted A/P cut, coordinates, and predicted foreground geometry. Adding fixed MRI context gives a tiny improvement for the unaugmented checkpoint and a tiny deterioration for the augmented checkpoint. Decoder features are neutral for one checkpoint and weakly positive for the other. No contextual arm establishes an incremental improvement with a paired 95% interval excluding zero.

These are local CPU development results. They reproduce the earlier CPU hard-mask cache exactly; they are not the official CUDA evaluation.

## Development cut error

Mean absolute error in mm (equivalently, 1-mm native slices); lower is better.

| Checkpoint | Original cut | Base probe | + MRI context | + Decoder context | + Shuffled MRI |
|---|---:|---:|---:|---:|---:|
| unaugmented | 1.0192 | 0.9615 | 0.9423 | 0.9615 | 0.9615 |
| augmented | 0.9038 | 0.8846 | 0.9038 | 0.8462 | 0.9231 |

## Incremental effects over the base probe

Negative change in cut error is improvement. Positive Dice percentage-point change is improvement. Intervals use 10,000 paired case bootstrap resamples.

| Checkpoint | Added features | Change in cut MAE, 95% CI (mm) | Change in Dice, 95% CI (pp) | Cuts improved / worsened / equal |
|---|---|---:|---:|---:|
| unaugmented | MRI | -0.0192 [-0.0769, +0.0385] | +0.0117 [-0.1355, +0.1615] | 2 / 1 / 49 |
| unaugmented | Decoder | +0.0000 [-0.0962, +0.1154] | -0.0141 [-0.2541, +0.2104] | 4 / 4 / 44 |
| augmented | MRI | +0.0192 [+0.0000, +0.0577] | -0.0524 [-0.1573, +0.0000] | 0 / 1 / 51 |
| augmented | Decoder | -0.0385 [-0.1346, +0.0577] | +0.0715 [-0.1345, +0.2806] | 4 / 2 / 46 |

MRI context changes three unaugmented cuts (two improve, one worsens) and one augmented cut (it worsens). Against the raw shuffled-MRI control, MRI wins by one slice in one case for each checkpoint; both paired MAE intervals end at zero. This is insufficient evidence of a robust anatomical cue.

## A/P Dice at fixed predicted foreground

Values are macro A/P Dice × 100. Foreground false positives and false negatives cannot be repaired by this experiment.

| Checkpoint | Original segmentation | Original fitted plane | Base | + MRI | + Decoder |
|---|---:|---:|---:|---:|---:|
| unaugmented | 87.1443 | 87.1884 | 87.3811 | 87.3928 | 87.3670 |
| augmented | 88.6557 | 88.7186 | 88.7205 | 88.6681 | 88.7920 |

The base probe itself changes Dice by +0.237 pp (unaugmented) and +0.065 pp (augmented) versus the original segmentation; both paired intervals include zero. Improvements versus the original network must therefore not be attributed entirely to contextual features.

## Training-only selection

Four folds split the 208 training case IDs. Each cell gives selected C and training-CV cut MAE. Inner folds hold out the probe but not the pretrained backbone.

| Checkpoint | Original training MAE | Base: C / CV MAE | MRI: C / CV MAE | Decoder: C / CV MAE | Shuffled MRI: C / CV MAE |
|---|---:|---:|---:|---:|---:|
| unaugmented | 0.5000 | 1 / 0.4183 | 0.1 / 0.3990 | 0.01 / 0.3365 | 0.1 / 0.4135 |
| augmented | 0.4231 | 0.1 / 0.4038 | 1 / 0.4135 | 1 / 0.3942 | 0.01 / 0.4327 |

All selected arms beat the original training cut except shuffled MRI for the augmented checkpoint. The predefined training gate consequently leaves that arm at the original prediction. The tables above show the ungated arms to expose the feature comparison; the JSON also records every gated result. A training gate pass is not evidence of a development-set gain.

## Interpretation and next decision

This experiment provides no convincing evidence that these contextual descriptors add boundary information usable by the chosen small linear readout. It does not establish that contextual information is absent or that a nonlinear model could never use it. Together with the earlier negative harmonic pilot, it does not justify the additional complexity of learned graph affinities yet.

The next informative experiment would generate out-of-fold backbone predictions/features, train the same incremental probe on those realistic errors, and evaluate on a fresh participant-grouped fold or external set. Verify participant linkage first. The existing training predictions are in-sample, and the development set has already been used for model selection and earlier research. The two checkpoints share cases and are not independent replications. Bootstrap intervals are exploratory and are not adjusted for multiple comparisons.

See [METHOD_NOTE.md](METHOD_NOTE.md): case IDs are not verified participant IDs. The original protocol retains its wording and checksum; this terminology clarification changed no analysis choices. [METHODS.md](METHODS.md) describes feature construction and scoring.

## Implementation and verification

- Research-only entry point: `evaluation/probe_context_increment.py`; no default training or inference path changed.
- All 520 feature caches checked for finite values, complete candidate targets, feature dimensions, and decoder-cache hashes.
- Checkpoint hashes agree with the earlier frozen-decoder feature audit.
- All 104 development prediction/reference pairs matched the earlier CPU hard-mask caches during extraction.
- All 260 native image/label pairs have matching geometry and 1-mm RAS orientation.
- First-checkpoint choices and per-case results remained identical when the combined report was generated.
- 32 relevant tests passed, including nine new tests covering label-independent feature construction, shuffled controls, probability orientation, normalization, ranking, and paired bootstrap behavior.

```bash
rtk proxy env MPLCONFIGDIR=/tmp/hippo-context-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python evaluation/probe_context_increment.py --stage extract
rtk proxy env MPLCONFIGDIR=/tmp/hippo-context-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python evaluation/probe_context_increment.py --stage select
rtk proxy env MPLCONFIGDIR=/tmp/hippo-context-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python evaluation/probe_context_increment.py --stage evaluate
rtk proxy env MPLCONFIGDIR=/tmp/hippo-context-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest evaluation/test_probe_context_increment.py evaluation/test_audit_ap_partition_topology.py thesis/new_constraints/test_harmonic_partition.py -q
```

The extraction reuses the two existing checkpoints and frozen decoder caches. Derived feature arrays remain in ignored `experiments/context_increment_20260923/cache/`. The dated folder retains the fixed protocol, model selection grid, per-case results, summary intervals, and provenance hashes.
