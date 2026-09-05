# Surface-normal one-cut and ordinal LogLTN: pre-training audit

This folder answers one question before any new model is trained:

> Does surface-normal logical supervision observe and locally improve the boundary errors made by the current model?

The frozen-logit audit supported a short outer one-cut pilot. The strictly
matched trainable implementation now lives in `thesis/new_constraints/onecut`;
this folder remains the immutable exploratory audit and evidence trail.

## Tested hypothesis

For the outer hippocampal surface, the exact grouped semantic field is

\[
r_H=\operatorname{LSE}(z_A,z_P)-z_0.
\]

The original two-point baseline at a ground-truth surface point with outward physical normal \(n\) is

\[
t=\sigma\left(\frac{r_H(b-\delta n)-r_H(b+\delta n)-m}{T}\right),
\]

and the guarded universal LogLTN loss is the patient mean of \(-\log t\). The optional internal-interface audit uses \(r_{AP}=z_A-z_P\) and rays oriented from anterior to posterior.

The improved primary formulation samples an ordered ray \(x_0,\ldots,x_{S-1}\). For each admissible cut \(k\) near the annotation, it defines

\[
\Phi_k=\left(\bigwedge_{i\le k}H(x_i)\right)\land
\left(\bigwedge_{i>k}\neg H(x_i)\right),
\]

and existentially marginalizes the uncertain crossing position:

\[
\Phi_{\mathrm{onecut}}=\bigvee_{k\in C}\Phi_k.
\]

The implementation evaluates the disjoint assignments stably with `logsumexp` and normalizes the negative log truth by the number of ray literals. It therefore constrains transition existence, direction, multiplicity, and allowed location—not just the relative ordering of two samples.

The signed distance field is used only to obtain spacing-aware outer normals. It is not predicted by a new model head.

## What the audit tests

1. **Geometry validity** — physical spacing, face-centred surfaces, normal orientation, and in-bounds ray guards.
2. **Observability** — ordinal-pair coverage plus one-cut loss discrimination for shifted, missing, reversed, and multiple predicted crossings.
3. **Known blind spot** — steep but displaced boundaries may satisfy the ordinal rule. The audit measures this directly rather than assuming the loss can localize them.
4. **Gradient behaviour** — common-logit bias invariance, A/P-swap invariance of the outer rules, gradient direction, RMS magnitude, and cosine with Dice.
5. **Counterfactual utility** — equal-RMS frozen-logit updates for Dice, ordinal pairs, one-cut logic, and the existing two-step grouped band BCE.
6. **Decoded geometry** — union Dice, surface Dice at 1/2 mm, ASSD, HD95, FP, FN, swaps, volume, and connected components.

The final verdict is one of `GO_TO_SHORT_PILOT`, `NO_GO`, or `INCONCLUSIVE`. A GO authorizes only a short matched pilot, not a full experiment.

## Input bundles

Each case is a compressed `.npz` containing:

- `logits`: `[3,D,H,W]` float32;
- `labels`: `[D,H,W]` with values 0/1/2;
- `spacing`: three physical voxel sizes in array-axis order;
- `case_name` and a schema version.

Saved softmax probabilities can be reused. The exporter converts them to `log(probability)`, which is an equivalent logit representative for every objective in this audit because they are invariant to a common per-voxel logit offset.

## Run locally on the existing saved fold-0 probabilities

From the repository root:

```bash
source .venv/bin/activate

python "thesis/Surface-normal ordinal LogLTN/export_cases.py" \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --dataset MSD \
  --fold 0 \
  --split val \
  --inference-dir datasets/Dataset101_MSD/inference_fold0_val_600236 \
  --labels-dir datasets/Dataset101_MSD/labelsTr \
  --spatial-size 64 64 64 \
  --output-dir /tmp/ordinal_case_bundles

python "thesis/Surface-normal ordinal LogLTN/run_audit.py" \
  --input-dir /tmp/ordinal_case_bundles \
  --output-dir /tmp/ordinal_audit \
  --interfaces outer+ap \
  --device cpu
```

For a quick pipeline check, add `--max-cases 2` to the exporter. Do not interpret a two-case verdict scientifically.

## Export directly from a checkpoint

This performs inference but never updates model parameters:

```bash
python "thesis/Surface-normal ordinal LogLTN/export_cases.py" \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --dataset MSD \
  --fold 0 \
  --split val \
  --checkpoint /absolute/path/checkpoint_best.pt \
  --labels-dir datasets/Dataset101_MSD/labelsTr \
  --spatial-size 64 64 64 \
  --device cuda \
  --output-dir /absolute/path/ordinal_case_bundles
```

Use exactly the same `--resize`, spatial size, and AMP policy used to produce the checkpoint comparison. The MSD files currently report 1-mm isotropic spacing; spacing is nevertheless read per case and saved in every bundle. If labels are unavailable, pass an explicit transformed `--spacing D H W` instead.

## Run on the Bocconi cluster

```bash
ssh bocconi-cluster
cd /mnt/beegfsstudents/home/3160552/hippo

sbatch "thesis/Surface-normal ordinal LogLTN/run_pretraining_audit.sbatch" \
  --checkpoint /absolute/path/to/checkpoint_best.pt \
  --output-root /mnt/beegfsstudents/home/3160552/ordinal_audit_20260901 \
  --interfaces outer+ap
```

The cluster job refuses to reuse an existing output root, preventing stale bundles or reports from being mixed into a new audit.

## Outputs

- `REPORT.md`: concise decision and the failed/passed gates;
- `audit_summary.json`: complete configuration, provenance hashes, aggregates, and verdict;
- `ray_case_summary.csv`: per-case error coverage, precision, violation rate, and crossing categories;
- `counterfactual_metrics.csv`: every case and matched repair arm;
- `gradient_diagnostics.csv`: gradient RMS, calibrated comparator weights, and Dice cosine;
- `rays/*.npz`: sampled fields, pair margins, one-cut truths/losses, crossings, points, normals, and category codes for further inspection.

The primary outer-ray screening defaults are:

- one-cut loss ROC AUC at least 0.70 as a structural sanity check;
- one-cut high-loss error coverage at least 30% at a fixed 5% correct-ray false-positive rate;
- precision lift at least 1.25 over boundary-error prevalence;
- positive incremental 1-mm surface-Dice response in at least half the cases for matched `Dice + one-cut` versus `Dice only` repair.

These are practical go/no gates, not claims of statistical significance. Change them only before inspecting the corresponding results and record the change.

## Tests

```bash
source .venv/bin/activate
bash "thesis/Surface-normal ordinal LogLTN/run_tests.sh"
```

The tests cover anisotropic distance geometry, outer and A/P orientation, exact grouped probabilities, trilinear coordinate conventions, invariances, gradient signs, existential cut marginalization, shifted/missing/multiple transitions, crossing classification, the displaced pairwise blind spot, physical surface metrics, and all repair comparator arms.

## Research-validity warning

Using held-out validation labels to choose whether to pursue this method is exploratory model selection. Preserve an untouched test set—or confirm the final method on new folds/seeds—before making performance claims. For a stricter screening protocol, run the go/no audit on deterministic training cases and reserve fold-0 validation for a pre-specified short pilot.
