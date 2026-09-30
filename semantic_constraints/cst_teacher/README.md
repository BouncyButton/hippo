# CST teacher for conditional hippocampus constraints

The first completed MSD fold-0 study and its decisions are recorded in
[`EXPERIMENT_20260921.md`](EXPERIMENT_20260921.md).

This directory is an isolated research MVP for using a Convolutional Set
Transformer (CST) as a frozen, case-conditioned teacher for the MSD
hippocampus SwinUNETR. It does **not** replace SwinUNETR and is not enabled in
the existing training runner.

The implementation follows the paper's central SetConv2D mechanism: a shared
2-D convolution processes every set element, pooled element features interact
through self-attention, and the contextualized vectors are added back to the
spatial feature maps as dynamic channel biases. The medical adaptation adds a
normalized coronal coordinate to every attention token. Coordinates travel
with their elements, so the network remains permutation-equivariant without
discarding anatomical position.

## What is implemented

- `model.py`: a small PyTorch SetConv2D encoder, an MRI-only descriptor/profile
  teacher, and a mask-conditioned anomaly teacher.
- `data.py`: fixed, label-free coronal slab sampling; exact coordinate
  metadata; variable-cardinality masks for combinatorial training; and
  synthetic mask corruptions.
- `descriptors.py`: differentiable case descriptors and longitudinal A/P area
  profiles shared by teacher training and future SwinUNETR constraints.
- `losses.py`: quantile/profile/anomaly training losses plus the three frozen
  teacher constraints intended for SwinUNETR integration.
- `train_teacher.py`: fold-exact training using the same MSD pickle,
  `splits_final.json`, nonzero intensity normalization, and 64-cube
  center-pad/crop path as the existing SwinUNETR baseline.
- `analyze_embeddings.py`: training-only BIC model selection, bootstrap
  clustering stability, soft validation memberships, conditional descriptor
  intervals, and observational morphology rules.

The MRI-only teacher predicts:

1. conditional 10th/50th/90th percentiles for anterior volume fraction, union
   volume fraction, signed A/P centroid gap, and union elongation;
2. anterior and posterior cross-sectional area profiles for the sampled
   coronal locations;
3. an embedding for frozen representation probes and soft clustering.

The second CST sees MRI slabs together with candidate A/P mask slices. It is
trained to identify contextual anomalies produced by class swaps,
translations, erasures, and isolated islands.

## Train a discovery-fold teacher

Fold 0 has already informed constraint development in this repository. Treat
this as discovery/probing only, not as a final generalization result.

```bash
.venv/bin/python -m semantic_constraints.cst_teacher.train_teacher \
  --fold 0 \
  --epochs 40 \
  --batch-size 4 \
  --output-dir semantic_constraints/cst_teacher/runs/fold0
```

The default 32×32 slab resolution is an intentionally lightweight CPU/GPU
teacher view. Descriptor and profile targets are still computed from the full
64³ masks. Preprocessed sets are cached in memory, and validation loss uses
early stopping with patience eight by default.

Small CPU integration run:

```bash
.venv/bin/python -m semantic_constraints.cst_teacher.train_teacher \
  --device cpu \
  --epochs 1 \
  --batch-size 2 \
  --channels 8 16 \
  --heads 2 \
  --set-size 6 \
  --minimum-set-size 3 \
  --max-train-cases 4 \
  --max-val-cases 2 \
  --output-dir /tmp/hippo-cst-smoke
```

The checkpoint records both teacher state dictionaries, the exact fold and
architecture configuration, descriptor names, selected epoch, and validation
metrics. Generated `runs/` contents remain ignored.

Analyze a trained teacher without refitting it on validation cases:

```bash
.venv/bin/python -m semantic_constraints.cst_teacher.analyze_embeddings \
  --checkpoint semantic_constraints/cst_teacher/runs/fold0/best_teacher.pt \
  --output-dir semantic_constraints/cst_teacher/runs/fold0/embedding_analysis
```

The Gaussian mixture, feature standardization, cluster-specific descriptor
intervals, and textual rule summaries are fitted from the training cases only.
Validation cases receive soft memberships from that fixed model. A high
silhouette or stable cluster is not sufficient evidence for a constraint; the
generated report marks all rules as observational candidates.
Mixtures containing fewer than ten training cases in any component are
rejected by default; change `--minimum-cluster-size` only as an explicit
sensitivity analysis (or set it to one for a tiny integration smoke test).

## Trainable slice-quality component

`slice_qc.py` defines two small heads that predict the per-coronal-slice mean
anterior/posterior Dice error of a frozen Swin prediction: a standardized ridge
regressor and a depthwise temporal-convolution model. The 13-feature
`portable` input combines Swin uncertainty with CST/Swin profile residuals;
the 77-feature `combined` input also includes the teacher's contextual slice
embedding. It is a *quality-control score*, not a segmentation correction or
an anatomical constraint. The ridge combined head is the current default:
the more complex temporal head did not consistently beat it.

For an existing feature bank containing labels and `slice_target`, run the
patient-grouped experiment and fit deployable heads with:

```bash
python -m semantic_constraints.cst_teacher.run_slice_qc_experiment \
  --fold0-features /path/to/fold0/risk_features_seed_0.npz \
  --fold1-features /path/to/fold1/risk_features_seed_0.npz \
  --output-dir /path/to/qc_run/seed_0
```

This writes patient-out-of-fold predictions, metrics, split records, and final
`models/fold{0,1}/ridge_{uncertainty,portable,combined}.npz` artifacts.
Hyperparameters are selected inside each patient-held-out split. Repeat for
each teacher seed and pair each head with the teacher that made its features.
The `combined` embeddings are **not** aligned between independently trained
teachers, so a head must not be applied to another seed's embedding. Use the
`portable` head for a cautious cross-teacher fallback.

For a new case, provide the *already preprocessed* MRI (`(1,X,Y,Z)` float
array after the same 64-cube center pad/crop and nonzero z-score used for
training), the matching Swin softmax probabilities (`(3,X,Y,Z)`), and the
matching CST checkpoint. No reference mask is read or needed:

```bash
python -m semantic_constraints.cst_teacher.extract_slice_qc_features \
  --teacher-checkpoint /path/to/best_teacher.pt \
  --preprocessed-image-npy /path/to/image.npy \
  --swin-probabilities-npy /path/to/probabilities.npy \
  --case-name hippocampus_NEW \
  --output /path/to/new_case_features.npz --device cuda
python -m semantic_constraints.cst_teacher.score_slice_qc \
  --model /path/to/qc_run/seed_0/models/fold0/ridge_combined.npz \
  --features /path/to/new_case_features.npz \
  --output /path/to/new_case_slice_risk.csv
```

The scorer rejects case names used to fit the head. For a three-teacher
ensemble, run extraction once per teacher and pass three matched `--models`
and `--features` paths to `score_slice_qc_ensemble`. It averages scores only
after confirming identical case/slice order. Higher score means greater
predicted segmentation error; the model does not identify the correction
direction. The numerical study and its limitations are in
[`EXPERIMENT_20260921.md`](EXPERIMENT_20260921.md).

The subsequent [error-type and action-gate study](EXPERIMENT_20260922_ERROR_TYPES.md)
found that the current features do **not** reliably distinguish missing from
extra tissue and that even a label-informed foreground-bias edit has a tiny
Dice ceiling with patient harms. It does not justify an automatic correction.

A further [local image-and-mask correction attempt](EXPERIMENT_20260922_LOCAL_CORRECTION.md)
trained a bounded spatial residual CNN with and without CST context under
patient-held-out evaluation. Both raw variants harmed most patients; the
inner-validation acceptance rule abstained on every split. No correction is
enabled in the Swin inference path.

## Required gates before SwinUNETR integration

1. **Representation gate.** Freeze the CST and compare equal-capacity probes
   against the existing frozen SwinUNETR feature probe. The CST must add
   held-out information for interface localization, shape descriptors,
   corruption detection, or segmentation-error prediction. UMAP is for
   visualization only; clustering and probes operate in the original embedding.
2. **Calibration gate.** Conditional 10--90% intervals must have useful
   held-out coverage. Recalibrate them on training-only calibration cases if
   necessary. Wide intervals that always cover are not useful constraints.
3. **Repair gate.** Optimize frozen SwinUNETR logits with each proposed CST
   constraint. Retain a constraint only if it reduces its violation, improves
   soft Dice, does not harm mean hard Dice, and has a positive bootstrap lower
   confidence bound under the existing counterfactual-repair protocol.
4. **Training gate.** Compare matched runs from the same checkpoint and data:
   baseline, existing semantic rules, CST rules, and both together. Keep a
   separate fold untouched until all choices are fixed.

Run the descriptor repair gate on an ensemble of independently trained
teachers with:

```bash
python -m semantic_constraints.cst_teacher.counterfactual_repair \
  --checkpoints runs/seed_0/best_teacher.pt runs/seed_1/best_teacher.pt runs/seed_2/best_teacher.pt \
  --inference-dir /path/to/inference_fold0_val \
  --output runs/counterfactual_repair.json \
  --device cuda
```

The optimizer preserves total foreground probability at every voxel and only
redistributes anterior versus posterior probability. Ground truth is read
after repair for measurement, never by the repair objective.

## Intended SwinUNETR use

For logits `logits` and frozen MRI-only teacher output `teacher_output`:

```python
probabilities = logits.softmax(dim=1)

descriptor_result = conditional_descriptor_constraint(
    probabilities,
    teacher_output.descriptor_quantiles,
)

profile_loss = contextual_profile_constraint(
    probabilities,
    positions,
    teacher_output.slice_profiles,
    valid_elements=valid_elements,
)
```

For the anomaly constraint, extract the SwinUNETR A/P probability slices at
the same positions, freeze the anomaly teacher with
`anomaly_teacher.requires_grad_(False)`, and call `anomaly_constraint`. Do not
wrap that call in `torch.no_grad()`: teacher parameters are frozen, but the
gradient must still reach the segmentation probabilities.

## Deliberate limitations

- This is 2.5-D, not a claim that the paper already supports medical 3-D
  volumes. A SetConv3D model should be considered only after this cheaper test
  establishes incremental representation value.
- The first set contains coronal slabs only. Tri-planar elements and view
  identifiers are a follow-up, not silent extra scope.
- Embeddings and attention weights are not themselves semantic rules. Every
  extracted cluster or relationship must be translated into a measurable mask
  predicate and pass cross-fitting plus counterfactual repair.
- The anomaly teacher can learn shortcuts. Synthetic corruption performance
  alone is insufficient evidence that its gradient improves real masks.
