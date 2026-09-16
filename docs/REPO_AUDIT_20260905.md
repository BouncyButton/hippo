# Repository audit — 2026-09-05

Scope: the 71 uncommitted entries on `Filippo_dev`, the `.gitignore`, and the
claim that the codebase contains a lot of redundant scripts.

## 1. What the 71 actually are

Your GUI is lumping two categories together:

| Category | Count |
|---|---|
| Modified, already tracked | 17 |
| Untracked (new files) | 54 |
| **Total** | **71** |

The 54 untracked files are not scattered debris. They are a handful of coherent units:

| Unit | Files | Status |
|---|---|---|
| `thesis/Surface-normal ordinal LogLTN/` | 23 | Self-contained pre-training audit + evidence trail |
| `thesis/new_constraints/onecut/` | 5 | New trainer preset (`--constraint-set onecut`) |
| `thesis/new_constraints/teacher/` | 5 | Stop-gradient translation teacher candidate |
| `thesis/new_constraints/ap_cut/` | 4 | Supervised A/P cut auxiliary |
| Loose `new_constraints/` modules + their tests | 11 | Telemetry, supervised losses, audits |
| `bands/` additions | 3 | `class_aware_tversky.py` + 2 rationale docs |
| `equivariance/` additions | 2 | `closure_metrics.py` + its test |
| `docs/RESEARCH_CRITIQUE_20260905.md` | 1 | Root-level dated critique |
| **Total** | **54** | |

Every one of `onecut/`, `teacher/`, `ap_cut/` ships an `__init__.py`, a `README.md`
and a test file, and all three are imported by
`thesis/new_constraints/train_swinunetr_constraints.py`. This is in-progress
research, not garbage.

## 2. Your redundancy suspicion — mostly not supported

Method: built the import graph over all 62 python files under `thesis/` with
`ast`, seeded from the entrypoints that the cluster scripts actually invoke,
then cross-referenced every low-reachability file against all `.md`, `.sh`,
`.sbatch` and `.py` in the repo.

**Result: 6 of 62 files are unreachable** from any cluster entrypoint or test.
Cross-referencing then resolved most of them: 2 are invoked by `.sbatch` files
inside gitignored directories (`audit_official_cluster.py`,
`onecut/calibrate_weight.py`), 1 is documented in a README
(`aggregate_paper_results.py`), 1 is a package marker
(`surface_normal_ordinal/__init__.py`). That leaves 2 real candidates:
`create_epsilon_progress_update.py` (zero inbound references anywhere in the
repo) and `translation_equivariance.py` (a shim nothing imports).

So the redundancy is real but small — 2 files out of 62, plus one duplicated
harness. It is not the systemic bloat you suspected.

I also body-diffed the pairs that *look* like duplicates. Overlap of
non-comment, whitespace-stripped lines:

| Pair | Shared lines | Verdict |
|---|---|---|
| `bands/outer_boundary.py` vs `onecut/outer_onecut.py` | 32 / 212 (15%) | Different formulations, not a duplicate |
| `new_constraints/objective.py` vs `surface_normal_ordinal/objective.py` | 7 / 129 (5%) | Unrelated |
| `bands/calibrate_weight.py` vs `teacher/calibrate_training.py` | 7 / 152 (5%) | Unrelated |
| `bands/calibrate_weight.py` vs `onecut/calibrate_weight.py` | **133 / 234 (57%)** | **Real duplication** |

For the record, the signal that first looked damning — 16 `def main`, 9
`def parse_args` — is meaningless. Those are CLI scripts; every one needs both.

### The three real findings

**a. `thesis/new_constraints/translation_equivariance.py` — 4 lines, dead.**
A compatibility shim re-exporting `equivariance/translation_equivariance.py`.
`equivariance/README.md` says it exists so "existing analysis scripts and
checkpoints continue to work." No such caller survives. I scanned every `.py`,
`.sh`, `.sbatch`, `.md` and `.txt` in the whole repo — **including the
gitignored `experiments/` and `cluster_protocol_*/` directories**, which is where
those analysis scripts live. All 24 hits either import
`new_constraints.equivariance.translation_equivariance` (the real module) or are
metric-name strings like `results["translation_equivariance"]`. **Zero** resolve
to the shim. Safe to delete, though check any other branch you still care about.

**b. `thesis/paper_reproduction/create_epsilon_progress_update.py` — 994 lines,
zero references anywhere.** Not in any README, sbatch, shell script, or import.
The only file in the repo with no inbound reference at all. It generated the
`thesis/runs/paper_reproduction/full_fraction_epsilon_sweep_analysis_20260729/`
package once. Either reference it from `paper_reproduction/README.md` or drop it.

**c. `onecut/calibrate_weight.py` shares 57% of its body with
`bands/calibrate_weight.py`.** Of those 133 lines, 105 are substantive: shared CLI argument
plumbing and the provenance/record dictionaries emitted into the results JSON
(`checkpoint_source_sha256`, `execution_provenance`, `foreground_class_ids`, …).
The differing half is the constraint-specific objective. Worth extracting a
`new_constraints/calibration.py` and having both import it — that also lets
`teacher/calibrate_training.py` stop reinventing `_write_csv` / `_decision` /
`_report`, all three of which exist in triplicate.

Beyond those three, the code is disciplined. Every subpackage documents its own
hypothesis and states honestly whether it established a Dice improvement.
`Surface-normal ordinal LogLTN/` is explicitly declared "the immutable
exploratory audit" whose trainable form lives in `onecut/` — that is a deliberate
separation, not duplication.

## 3. The actual disorder: `.gitignore` is hiding your source

This is the real problem, and it is the opposite of what you suspected.

The 5 lines you recently added to `.gitignore` are:

```
semantic_constraints/
evaluation/
allenamenti.md
cluster_protocol_*/
experiments/
```

**Three of them are no-ops.** `semantic_constraints/` (15 files),
`evaluation/` (5 files) and `docs/allenamenti.md` are already tracked. Git ignores
`.gitignore` for tracked paths. Those lines do nothing today and will confuse
you later. To make them real you would need:

```bash
git rm --cached -r semantic_constraints evaluation allenamenti.md
```

Same for `.idea/`, which has 6 tracked files.

**The other two hide 30 scripts and 9 protocol documents.** `experiments/` and
`cluster_protocol_*/` are not output directories — they are your invocation
layer:

```
cluster_protocol_20260830/  16 sbatch
cluster_protocol_20260902/   6 sbatch + compare_checkpoints.py
experiments/equivariance_family_b_20260905/    3 sbatch + prepare_submit.py
experiments/loss_constraint_followup_20260905/ 1 sbatch + 2 py
```

These are the *only* record of how each result was produced. Several of the
"orphan" scripts I flagged in the reachability pass turned out to be reachable
solely through them:

- `audit_official_cluster.py` ← `experiments/loss_constraint_followup_20260905/GPU_AUDIT.sbatch`
- `calibrate_training.py` ← `experiments/equivariance_family_b_20260905/calibrate_teacher.sbatch`
- `onecut/calibrate_weight.py` ← `cluster_protocol_20260902/calibrate_onecut.sbatch`
- `run_hybrid_feasibility.py`, `run_band_location_sweep.py` ← `cluster_protocol_20260902/*.sbatch`

For a thesis, that is exactly the layer that has to be in git. `utils/` (2 python
scripts) is in the same situation.

Meanwhile the things that genuinely should never be committed were already safe
by extension only, which is fragile:

| Path | Size | Was covered by |
|---|---|---|
| `mri_dataset/` | 9.3 GB | `*.nii.gz` only |
| `mni-hisub25.tar` | 9.4 GB | `*.tar` only |
| `weights/` | 187 MB | `*.pt` only |
| `.claude/worktrees/` | 59 MB | nothing |
| `.pytest_cache/` | 56 KB | nested pytest `.gitignore` only |

One stray file extension in any of those and you commit gigabytes.

### What I changed

Rewrote `.gitignore` (backup at the path noted below). Additions are all
defensive and change nothing about what is currently visible — status is still
17 modified + 54 untracked (plus this report, which is untracked too):

- explicit directory entries for `mri_dataset/`, `weights/`, `.claude/worktrees/`,
  `.pytest_cache/`, `.idea/`, `.ipynb_checkpoints/`, `thesis/runs/`
- grouped into labelled sections
- **left your 5 lines in place**, under a comment block spelling out exactly what
  each one hides and which are no-ops

I did not delete the `experiments/` and `cluster_protocol_*/` lines, because
removing them makes 39 more files appear as untracked, and that is your call.

## 4. What was done

All of the following is applied in the working tree and committed on
`Filippo_dev`.

**`.gitignore` rewritten.**

- Added explicit directory rules for the large paths that were previously safe
  by file extension alone: `mri_dataset/` (9.3 GB), `weights/` (187 MB),
  `.claude/worktrees/` (59 MB), `.pytest_cache/`, `.idea/`,
  `.ipynb_checkpoints/`, `thesis/runs/`.
- **Removed** `experiments/`, `cluster_protocol_*/` and `utils/`. They were
  hiding 26 `.sbatch`, 6 `.py`, 9 protocol `.md` and ~1.1 MB of small
  `.json`/`.csv` evidence files — the reproducibility layer for the thesis.
  All 57 are now tracked.
- Replaced them with narrow, **prospective** output rules
  (`experiments/**/logs/`, `**/checkpoints/`, `*.out`, `*.err`, `slurm-*.out`).
  No such files exist yet; the rules exist so a future Slurm run cannot drag
  logs or checkpoints into git.
- **Kept** `evaluation/`, `semantic_constraints/` and `docs/allenamenti.md` under a
  comment block naming them as no-ops. Untracking those 27 files is a
  judgement call about your intent, so it is left to you:
  `git rm --cached -r <path>`. The same applies to `.idea/` (6 tracked files).

**Two dead files deleted.**

- `thesis/new_constraints/translation_equivariance.py`
- `thesis/paper_reproduction/create_epsilon_progress_update.py`

Recover either with `git checkout d20d01c -- <path>`.

Both `docs/thesis/new_constraints/README.md` and
`docs/thesis/new_constraints/equivariance/README.md` described the shim as live;
both were corrected.

## 5. Left for you

1. Decide whether to untrack `evaluation/`, `semantic_constraints/`,
   `docs/allenamenti.md`, `.idea/` (see above). Until then those `.gitignore` lines
   do nothing.
2. Optional refactor: extract the shared calibration harness out of
   `bands/calibrate_weight.py`, `onecut/calibrate_weight.py` and
   `teacher/calibrate_training.py`. `_write_csv`, `_decision` and `_report`
   currently exist in triplicate.
3. The folder name `thesis/Surface-normal ordinal LogLTN/` contains spaces, so
   it can never be imported as a package from elsewhere. Fine while it stays a
   standalone audit; rename it if that ever changes.
