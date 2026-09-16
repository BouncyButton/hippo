# Hippo — Repository Guide

A first-read map of this repo: what each folder does, how the pieces connect,
and how to actually run things. Written for someone opening the project for the
first time.

---

## 1. What this project is

A **Python research codebase for hippocampus segmentation** in 3D medical MRI,
built around one idea: **inject medical background knowledge into a deep
segmentation model as extra, differentiable loss terms** ("neuro-symbolic"
segmentation). It has two halves:

1. **Baselines** — a collection of established 3D segmentation models (nnUNet,
   UNETR++, SwinUNETR, 3D U-Net, NCA variants) wired up so you can train and
   evaluate them on hippocampus datasets with one launch script each.
2. **Semantic constraints** (the novel research) — discovers *differentiable
   geometric/topological rules* that usually hold in the ground truth (e.g.
   "anterior and posterior parts stay connected", "one part doesn't contain the
   other", "the two parts have similar volume"), then adds them as extra training
   losses to see if they improve segmentation — especially when labeled data is
   scarce.

**Not in the repo** (intentionally git-ignored, must be created/downloaded on the
run machine): raw `.nii.gz` images, model checkpoints, W&B runs, `.venv`,
generated outputs.

---

## 1b. The published paper (the anchor for everything here)

The `semantic_constraints/` work descends directly from a published paper —
read it to understand *why* the code exists:

> **Integrating Background Knowledge in Medical Semantic Segmentation with Logic
> Tensor Networks** — Luca Bergamin (U. Padua), Giovanna Maria Dimitri (U.
> Siena), Fabio Aiolli (U. Padua). **IJCNN 2025**, IEEE
> (DOI 10.1109/IJCNN64981.2025.11227985). PDF is in the repo root. Paper code:
> `github.com/BouncyButton/segmentation-with-ltn` (the `berga`/Bergamin paths in
> `ARCHITECTURE.md` come from that author's machine).

**What the paper does.** It segments the hippocampus (3 classes: background,
anterior/head, posterior/body-tail) on the **Medical Segmentation Decathlon**
hippocampus set (394 scans, 3D MRI 64×64×64, 3T, 1×1×1 mm). The backbone is
**SwinUNETR** (via MONAI). On top of the normal Dice loss it adds **four
hand-written first-order-logic constraints**, encoded with a **Logic Tensor
Network (LTN)** — LTN's "Real Logic" is fully differentiable, so a logical
formula becomes a loss. The four constraints:

| Constraint (FOL) | LTN predicate | How it's measured |
| --- | --- | --- |
| Anterior is connected to posterior | `Connected(ŷ)` | Chamfer distance between the two masks → **connectedness** |
| Anterior can't contain posterior | `Nested(ŷ)` | point-in-interpolant sampling → **nesting** |
| Posterior can't contain anterior | `Nested(ŷ)` | same |
| Anterior/posterior volumes are similar | `SimVol(ŷ)` | nonzero-pixel-count tolerance → **volume similarity** |

Training minimizes `L(θ) = 1 − SatAgg(Dice, Connected, Nested, SimVol)`.

**What the paper found.** LTN helps most when data is scarce.

- Dice (full / 25% / 5% of training data): SwinUNet `0.8547 / 0.8331 / 0.7434`
  → SwinUNet+LTN `0.8628 / 0.8391 / 0.7721`. Biggest win is **+~3 points at 5%
  data** — the constraints act as a regularizer against overfitting.
- Constraint satisfaction (full data): the **nesting violation** drops
  `0.4322 → 0.3357` with LTN; connectedness barely moves; volume similarity is
  already saturated at `1.0`. LTN is **soft regularization** — it reduces
  violations, it does not hard-guarantee them.

**How this repo goes beyond the paper.** The paper hand-picks 4 constraints for
the hippocampus. This repo's `semantic_constraints/` pipeline is the **general,
automated successor**: instead of a human writing the rules, it *discovers* them
from data over a small library of generic differentiable primitives, screens
them, and only then retrains. The repo also adds **more datasets** (MNI, ADNI,
HFH, COBRA beyond MSD) and **more baselines** (nnUNet, UNETR++, 3D U-Net, NCA)
for comparison. Mapping from paper → repo primitives:

- `Connected` → `connectedness` primitive
- `Nested` → `contains` primitive
- `SimVol` → `volume` primitive (compared across classes)
- LTN Real-Logic connectives → the fuzzy loss modes in `compiled_losses.py`
  (`lukasiewicz`, plus `soft_exp` default and `hinge`); `LTNtorch` is still a
  dependency in `requirements.txt`.

So: **paper = 4 manual LTN rules on MSD; repo = automatic constraint discovery +
a DSL + a fuzzy-loss compiler, generalized across datasets and models.**

---

## 2. Folder map

| Folder | Role |
| --- | --- |
| `baselines/` | Model code + upload helpers for nnUNet, UNETR++, SwinUNETR, 3D U-Net, M3D_NCA, NCAdapt. Vendored upstream packages live here. |
| `datasets/` | Scripts + metadata to build dataset folders in **nnUNet-style layout**. Five datasets: MSD, MNI, ADNI, HFH/TLE, COBRA. Also split-creation and UNETR++ split-fix helpers. |
| `semantic_constraints/` | The constraint-discovery research pipeline (probe → rank → repair → select → train → evaluate). Has its own `README.md` + `ARCHITECTURE.md`. |
| `scripts/` | End-to-end launch scripts (`run_nnunet_pipeline.sh`, `run_unetrpp_pipeline.sh`, `setup_unetrpp_env.sh`) + W&B upload/download helpers. |
| `evaluation/` | Post-training evaluators (`evaluate.py`, `evaluate_unetrpp.py`) called automatically by the pipelines. |
| `.venv/` | Local Python env (ignored). UNETR++ needs a **separate** Python 3.8 conda env. |
| `.idea/` | PyCharm project files. Ignore. |

Component guides are collected under `docs/`, mirroring the source folders. Read the corresponding guide before running a component's code.

---

## 3. Dataset layout (the shared contract)

Everything expects the **nnUNet-style** dataset directory:

```text
datasets/DatasetXXX_NAME/
  dataset.json        # class labels + metadata
  imagesTr/           # case_0000.nii.gz  (raw images, NOT in repo)
  labelsTr/           # case.nii.gz       (ground-truth masks, NOT in repo)
  splits_final.json   # train/test + CV folds (committed for the 5 datasets)
```

The five datasets and their builders:

| Dataset | Builder script | Source |
| --- | --- | --- |
| `Dataset101_MSD` | `create_msd_dataset.py` | Public MSD hippocampus tar (or W&B) |
| `Dataset102_MNI` | `create_mni_dataset.py` | MNI HiSub25 download (or W&B) |
| `Dataset103_ADNI` | `create_adni_dataset.py` | ADNI hippocampus protocol release (or W&B) |
| `Dataset104_HFH` | `open_tle_dataset.py` (inspect only) | HFH/TLE — data not included |
| `Dataset105_COBRA` | `create_cobra_dataset.py` | COBRA resources + atlas repo (or W&B) |

Builders prefer a **W&B artifact** if you're logged in, otherwise rebuild from
the public source. `datasets/fix_unetrpp_*` scripts convert/repair split files
into the format UNETR++ wants.

---

## 4. First-time setup

From repo root:

```bash
# main environment (baselines + semantic constraints)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt      # monai, nibabel, scikit-learn, wandb, LTNtorch, ...

# separate env ONLY for UNETR++ (Python 3.8, pinned torch 1.11)
scripts/setup_unetrpp_env.sh                    # uses requirements_38.txt

# if using W&B artifacts for datasets/checkpoints
wandb login
```

Two requirements files exist on purpose:
- `requirements.txt` — modern, unpinned (main env).
- `requirements_38.txt` — Python 3.8, pinned (`torch==1.11.0`, `monai==0.7.0`), UNETR++ only.

nnUNet installs itself automatically inside `run_nnunet_pipeline.sh` if
`nnUNetv2_train` isn't on PATH.

---

## 5. How to run — baselines

Use the **launch scripts**, not the model code directly. They handle: find/build
dataset → deterministic train/test + CV splits → train-only dataset copy →
preprocess → train → upload W&B artifacts → evaluate on the test split.

```bash
# quick sanity check (nnUNet)
scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode sanity

# full run, all 5 folds
scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode full --folds "0 1 2 3 4"

# UNETR++ (needs the conda env from setup_unetrpp_env.sh first)
scripts/run_unetrpp_pipeline.sh --dataset Dataset102_MNI --mode sanity
```

Useful flags: `--folds "0 1 2 3 4"`, `--skip-train`, `--skip-eval`,
`--max-cases 2`, `--mode sanity|full`.

Build a dataset by hand first if needed:

```bash
python datasets/Dataset102_MNI/create_mni_dataset.py --target datasets/Dataset102_MNI
python datasets/Dataset102_MNI/create_mni_dataset.py --target datasets/Dataset102_MNI --rebuild  # ignore W&B
```

Evaluation output lands in the run folder, typically
`../hippopotamus_runs/.../evaluation_output`.

---

## 6. How to run — semantic constraints (the research pipeline)

This is the heart of the project — the automated, generalized version of the
paper's manual-LTN idea (see §1b). Flow (from `ARCHITECTURE.md`):

```text
baseline checkpoint
   → probe_model.py            measure prediction vs ground-truth descriptors
   → evaluate_candidates.py    rank candidate rules (alpha sweep, scoring)
   → counterfactual_repair.py  test if a rule can locally repair frozen preds
   → select_constraints.py     write final selected_constraints.json
   → compiled_losses.py        JSON rules → PyTorch fuzzy losses
   → train_with_constraints.py fine-tune WITH vs WITHOUT constraints
   → final_evaluate.py         one-time comparison on reserved holdout fold
```

Quick pipeline:

```bash
python semantic_constraints/probe_model.py
python semantic_constraints/evaluate_candidates.py --top-k 15
python semantic_constraints/counterfactual_repair.py --top-k 5
python semantic_constraints/select_constraints.py
python semantic_constraints/train_with_constraints.py
```

Or the strict one-shot cross-fitted version:

```bash
python semantic_constraints/automatic_constraint_discovery.py \
  --load-weights semantic_constraints/runs/baseline/best_model_weights.pth \
  --root-dir tmp --train-fraction 0.1
```

CPU smoke test (checks the code runs; **not** a real experiment):

```bash
python semantic_constraints/probe_model.py --max-batches 2 --device cpu
python semantic_constraints/evaluate_candidates.py --top-k 5
python semantic_constraints/counterfactual_repair.py \
  --candidate-ranks 1 --gammas 0.1 --steps 2 --max-batches 2 --device cpu
```

### Key files

| File | Purpose |
| --- | --- |
| `primitives.py` | Differentiable measurements: volume, centroid, distance, overlap, contains, adjacent, boundary_length, compactness, entropy, connectedness. Generic — **no** hard-coded anatomy. |
| `registry.py` | Names/metadata of measurements available to search. |
| `probe_model.py` | Runs frozen SwinUNETR, records class/class-pair rows. |
| `evaluate_candidates.py` | Alpha sweep + scoring + redundancy of candidate rules. |
| `counterfactual_repair.py` | Largest file (931 LOC); tests local repair feasibility. |
| `automatic_constraint_discovery.py` | Full cross-fitted discovery in one run. |
| `select_constraints.py` | Deterministic rule-based selection → JSON. |
| `compiled_losses.py` | Safe JSON→loss compiler (fuzzy `soft_exp` default; also lukasiewicz, hinge). |
| `train_with_constraints.py` | Matched constrained/unconstrained fine-tuning. |
| `final_evaluate.py` | One-time holdout comparison. |
| `on_grokking_behavior.py` | **Separate** toy modular-arithmetic experiment; NOT part of the segmentation pipeline. |

### Experimental-validity rule (important)

The **final holdout fold is sacred**. Splits are deterministic K-fold with
NumPy seed 42. With `--fold 1 --holdout-fold 2`: fold 1 = discovery, fold 2 =
final holdout, rest = training. Discover/select/tune only on discovery
artifacts; run `final_evaluate.py` **once** at the end. Always compare a
constrained run against an unconstrained run from the **same** init checkpoint,
data, optimizer, and epochs.

Outputs: discovery artifacts → `semantic_constraints/probe_outputs/`, training
runs → `semantic_constraints/runs/<name>/` (both git-ignored).

---

## 7. Testing

No central test suite. "Tests" = the documented smoke/sanity commands:
- baselines: any `--mode sanity` pipeline run.
- semantic constraints: the CPU smoke commands in §6.

---

## 8. Gotchas / notes

- **Two Python envs**: main `.venv` for everything, separate 3.8 conda env for
  UNETR++ only.
- SwinUNETR + the MONAI Decathlon data path are **hard-coded** in the executable
  semantic-constraint scripts, even though the primitive layer is model/domain
  agnostic.
- Checkpoints save **weights only** — no optimizer/scheduler/epoch/RNG state, so
  interrupted training resumes only approximately.
- `connectedness` and boundary primitives are soft **surrogates**, not exact
  topology; `connectedness` is expensive.
- Runs have generally used **CPU** (MPS / 3D-conv compatibility is uncertain).
- Don't commit raw data, `.nii.gz`, checkpoints, W&B output, envs, or run folders.
- `docs/semantic_constraints/ARCHITECTURE.md` has a stale absolute path
  (`/Users/berga/PycharmProjects/remote_ideas`) in its "Standard Commands"
  section — that's from another machine; substitute this repo root.

---

## 9. Where to read next

1. **The paper PDF** (`Integrating_Background_Knowledge_...pdf`) — the research
   motivation and results the whole `semantic_constraints/` folder generalizes.
2. `docs/semantic_constraints/ARCHITECTURE.md` — deepest design doc; read before
   touching that pipeline.
3. Each folder's `README.md` — inputs/outputs for its scripts.
4. `AGENTS.md` — contributor conventions (style, commits, PRs).
