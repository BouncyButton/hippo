# Bounded nnUNet and Swin bands refinement

Authorized window: 9 November 2026, 16:20:24–19:20:24 Europe/Amsterdam. Setup and queue time count against this window. One Slurm job requests two GPU workers; no automatic successor or full-matrix resumption is permitted.

The optimization budget remains 53,305,109,592,450 FLOPs per run, with calibration accounted for separately under the existing convention. Original bands definitions, data splits, objectives, optimizers and learning-rate recipes remain frozen. There are 36 archived baseline/original-bands controls and no new baseline runs.

| Model | Screen pairs (seed, fold) | Candidates | Fresh screen runs |
|---|---|---|---:|
| nnUNet | (0,2), (5,1), (19,2) | Delayed 60%, delayed 30%, original onset 30% | 9 |
| SwinUNETR | (0,2), (19,0) | 25% or 50% onset with 10% or 5% strength; original onset with 5% strength | 10 |

Strength is the calibrated constraint-to-supervised-gradient target, not a raw loss multiplier. nnUNet delayed onset is update 75 of 149, retaining its 20-update ramp. Swin delayed onsets are updates 86 and 172 of 343, retaining its 28-update ramp.

nnUNet candidates must improve macro Dice over the matched baseline in all three screen pairs, ranked by minimum gain then mean gain. Swin candidates must improve macro Dice over baseline in both screen pairs, improve mean macro Dice over the existing bands recipe, and avoid reducing mean CA4/DG Dice relative to that recipe. Swin ranking uses mean macro improvement over existing bands, then minimum baseline gain.

If time permits, eligible candidates are checked on remaining seed/fold pairs. Ranking is fixed from the screen. Any nonpositive baseline gain rejects a candidate; the next eligible candidate may be checked. Confirmation requires nine positive matched baseline gains; Swin also retains the aggregate macro/CA4 checks against original bands. This is development-validation selection, not independent evidence of generalization.

The controller admits no new runs after 19:10:24 and requests graceful stopping by 19:17:24. Slurm provides an additional time limit ending no later than the authorized deadline. Interrupted checkpoints and charged FLOPs are preserved. Completed runs are archived and verified before deletion; if the laptop is offline, artifacts remain on BeeGFS, with a storage floor preventing further launches if headroom becomes insufficient. The optional local collector never submits jobs and stops after completion or its bounded grace period.

See `submission.json` for the actual job and quota receipt, `live.json` for collected status, and `final_archive_cleanup.json` for eventual results and cleanup. The broader sweep remains paused with its existing recovery state unchanged.
