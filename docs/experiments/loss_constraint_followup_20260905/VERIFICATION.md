# Local verification, 2026-09-05

Subsequent user-requested cluster execution completed: see
[GPU_AUDIT_RESULTS_20260905.md](GPU_AUDIT_RESULTS_20260905.md).
The local-copy limitation recorded below is historical; the official model was
audited on CUDA without exporting it. The CUDA routing and refusal to fall back
to CPU are covered by the updated audit suite (9 tests passed).

**210 targeted tests passed** in one combined run (58.42 seconds). Interpreter:
`/Users/filippofocaccia/anaconda3/bin/python3`. No cluster training ran.

```bash
rtk proxy /Users/filippofocaccia/anaconda3/bin/python3 -m pytest \
  thesis/new_constraints/equivariance/test_equivariance.py \
  thesis/new_constraints/bands/test_outer_boundary.py \
  thesis/new_constraints/onecut/test_outer_onecut.py \
  thesis/new_constraints/test_training_telemetry.py \
  thesis/new_constraints/test_supervised.py \
  thesis/new_constraints/teacher/test_translation_teacher.py \
  thesis/new_constraints/ap_cut/test_posterior.py \
  thesis/new_constraints/test_followup_training.py \
  thesis/new_constraints/test_audit_followup.py -q
```

The 31 follow-up integration checks include actual trainer `_main` execution
with a tiny Conv3d model and synthetic loaders, two epochs/two batches, for
Dice+CE alone, Dice+CE plus teacher, and Dice+CE plus a compatible planar A/P
target. Production optimization, checkpoints, config serialization, completion
hashes, metrics and resume logic execute. Same-config resume succeeds without
duplicating epochs; changing the CE weight is rejected before artifact mutation.
These are integration tests, not SwinUNETR training quality measurements.

Legacy Dice value/gradient identity, logit-space numerical stability,
teacher-state/RNG restoration, translation overlap and A/P geometry/gradients
are covered. Legacy calibration creation/consumption rejects a Dice+CE source
checkpoint; old report fields that omit the new supervised-loss keys retain
their historical Dice-only interpretation.

`bash -n thesis/new_constraints/run_new_constraints_cluster.sh` passed.
`git diff --check -- thesis/new_constraints` passed. A full-worktree diff check
finds pre-existing trailing whitespace in `docs/allenamenti.md` lines 67–68, which was
left untouched. Broad equivariance test discovery also finds a pre-existing
missing `equivariance.evaluate_closure` import; the explicit core suite above
does not include that unrelated broken collection target.

## Real label audit

See `audit_gt_geometry.py` and `gt_geometry_audit.json` for the reproducible
260-case audit. Strict slice-wise A/P planarity fails on 47/208 training cases
and 11/52 validation cases. Mixed-slice minority voxels total 1,104 and 284,
respectively. The existing parser, independent direct-byte checks and both
52-case validation GT export sets agree. The strict A/P candidate fails its
prerequisite before any GPU time is spent.

## Synthetic frozen-logit audit

The three `synthetic_*.json` reports in this directory use two **artificial
8-cubed cases**, not medical images or baseline checkpoint outputs. Each has
96 foreground voxels, with 16 A/P swaps in the student. Logits are +8 for the
decoded class and -8 for the others. One teacher is correct; the other is the
identical wrong student field. The two cached views are exact ±1 shifts along
axis 0. All reported coefficients are diagnostic and explicitly marked as
invalid for training calibration.

| Quantity on the correct-teacher toy | Measured logit-gradient RMS |
|---|---:|
| Original Dice | 6.45035e-10 |
| Dice + CE, coefficient 1 | 2.81909e-4 |
| Translation-teacher KL | 2.81909e-4 |
| GT-anchored A/P cut NLL | 1.40840e-3 |

The equally wrong teacher gives exactly zero KL and zero auxiliary gradient.
This explicitly disproves the proposed non-saturability guarantee. The correct
teacher's gradient is nonzero despite saturated student probabilities. A large
12-unit normalized logit step repairs its 16 swaps; steps 0.25, 1 and 4 leave
argmax unchanged. A limited post-hoc step therefore cannot establish universal
argmax immobility. The A/P toy repairs both identical label/logit cases under the
large step because it has a GT correctness signal. These demonstrations say
nothing about real-world Dice gains or a useful training lambda.

To regenerate the cache in a **new** temporary directory:

```python
from pathlib import Path
import numpy as np
import torch
from thesis.new_constraints.equivariance import translate_3d

out = Path('/private/tmp/hippo-followup-synthetic-reproduction')
out.mkdir(exist_ok=False)
label = torch.zeros((8, 8, 8), dtype=torch.long)
label[2:6, 1:4, 2:6] = 2
label[2:6, 4:7, 2:6] = 1
prediction = label.clone()
prediction[2:6, 4, 2:6] = 2
def logits_for(array):
    return torch.nn.functional.one_hot(array, 3).permute(3, 0, 1, 2).float() * 16 - 8
base, ideal = logits_for(prediction), logits_for(label)
shifts = ((1, 0, 0), (-1, 0, 0))
for name, teacher in [('synthetic_correct_teacher', ideal),
                      ('synthetic_wrong_fixed_point', base)]:
    views = torch.stack([translate_3d(teacher, shift) for shift in shifts])
    np.savez_compressed(out / (name + '.npz'), logits=base.numpy(),
                        labels=label.numpy(), teacher_logits=views.numpy(),
                        shifts=np.array(shifts, dtype=np.int64))
```

Run `python -m thesis.new_constraints.audit_followup` with that cache as
`--input-dir`, a fresh `--output`, `--purpose diagnostic`, `--ce-weight 1`, and
`--step-sizes .25 1 4 12`. Use `--constraint teacher --supervised-loss dice_ce`;
repeat with `--supervised-loss dice --ce-weight 0`, or with
`--constraint ap_cut --supervised-loss dice_ce --ap-axis 1 --ap-anterior-side high`.

## Remaining external step

The official Family-B checkpoint was not available locally. A read-only cluster
queue check succeeded and returned no jobs, but automatic approval review
rejected copying the completed control's `MSD_fold0/model.pt` and `config.json`
to `/private/tmp/hippo-loss-followup-20260905/`. The stated reason was potential
sensitivity of the exported payload and lack of specific export authorization.
No payload was copied. A real baseline teacher audit requires an approved copy;
long training submissions separately require the user's approval under the
handoff. The older local probability cache is not substituted for the official
baseline or treated as original unclipped logits.
