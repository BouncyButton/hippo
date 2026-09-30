# Atlas pilot: prospectively specified feasibility screen

Written 24 September 2026 before registration-label scoring or fusion results.
Motivation: van der Lijn et al., NeuroImage 43 (2008), 708–720,
https://doi.org/10.1016/j.neuroimage.2008.07.058. This is a modern small-crop
adaptation, not a reproduction of their whole-brain registration pipeline.

## Questions and scope

Does image-registered label transfer add information to the canonical frozen,
unaugmented fold-0 seed-0 SwinUNETR? Test whole-foreground repair and A/P
localization separately. Do not introduce a new training objective or retrain
the network on the strength of in-sample results.

The deterministic seed-20260924 permutation of sorted training IDs assigns 16
cases to the atlas bank, 24 to calibration, 24 to assessment, and leaves 144
unused. Exact membership is in cohort.json. Every target is excluded from its
atlas bank. Participant linkage is unavailable; case disjointness is not proof
of participant disjointness. All 48 targets were in the backbone training set.
They are held out of atlas construction, NOT out of backbone fitting. Assessment
labels must not select parameters. The familiar 52-case development fold is a
separate, conditional exploratory evaluation, not an untouched final test.

## Registration fixed before scoring

Native 1-mm RAS images, no label-derived crop, mask, center, or registration
metric. The source files are local crops with reset origins. Rebase image
centers in local physical coordinates. Clip intensities at image-only 1st and
99th percentiles and rescale to [0,1]. Match the target against all 16 atlases
under centered identity and left/right reflection, using normalized mutual
information over image overlap. Select the best reflection for each source,
then the three highest-scoring distinct sources; no target label is consulted.

Register these three using affine Mattes MI (32 bins), full deterministic
sampling, multiresolution shrink factors 2/1 and smoothing 1/0 mm, at most 60
iterations per level. Then a coarse B-spline residual (mesh 2×3×2), Mattes MI,
shrink factors 2/1, smoothing 1/0 mm, at most 25 L-BFGS-B iterations per level.
Use one registration thread to avoid excessive local resource use. Retain
centered, affine, and affine+B-spline priors for comparison.

Reject affine transforms with singular values outside [0.65,1.5], determinant
outside [0.4,2.5], or translation norm above 12 mm. Reject residual deformation
with any sampled nonpositive Jacobian, residual displacement above 8 mm, or
less than 50% image coverage. These are fixed engineering guards, not clinical
quality guarantees. A rejected stage falls back to the previous valid stage;
record every failure. Select no transforms by label overlap. Transfer one-hot
labels with linear interpolation, treat outside-source space as background,
average the three sources uniformly, and clip probabilities only for log costs.

Identity/known-transform phantoms and a training-image timing smoke test may
fix implementation defects before the main run. Record any parameter change as
an amendment before inspecting registration-label scores.

## Frozen network and interventions

Use baseline_seed0_checkpoint_best.pt from CANONICAL_BASELINES.json with CPU
inference and existing no-resize 64³ preprocessing. Verify hashes and reproduce
previous hard predictions exactly; crop output back to native geometry and
verify cached truth against the native labels. Cache logits for reproducibility.

Whole foreground uses binary graph cuts on ALL native image voxels, so it can
add or remove foreground. Network foreground log-margin is
max(logit_A,logit_P)-logit_background. This preserves the original multiclass
argmax at alpha=lambda=0. It is a profile over the best foreground class, not
the marginal probability p(A)+p(P). Add alpha*log(pi_H/(1-pi_H)) from the atlas.
Keep the network's conditional A/P choice at retained or added foreground
voxels, isolating outer-boundary repair. Six-face pairwise costs are
lambda*exp(-(Iu-Iv)^2/(2*0.1^2)) in the normalized native MRI; all voxels have
1-mm spacing. Integer max-flow scale 10,000; record the rounding bound.

Compare raw network; graph-only; centered-atlas unary fusion; affine-atlas unary
fusion; deformable-atlas unary fusion; deformable-atlas plus graph; atlas alone.
Alpha grid: 0, 0.1, 0.3, 1, 3. Lambda grid: 0, 0.1, 0.3. Choose within calibration
only, maximizing mean foreground Dice; ties choose smaller alpha then lambda.

A/P diagnostic preserves the raw network foreground. Pool negative log costs
over every nonempty coronal cut, with posterior below/anterior above. Compare
the model-only likelihood plane, atlas-only plane, centered-atlas plus model,
affine-atlas plus model, and deformable-atlas plus model. Atlas conditional
margin is log(pi_A/pi_P), clipped probabilities at 0.01. Alpha uses the same
grid, selected by mean cut MAE on calibration only (ties smaller alpha).
Compare against both raw labels and model-only plane; the latter controls for
the plane restriction itself. Reference targets are best-fit coronal planes;
all voxel metrics retain the original, sometimes nonplanar, labels.

## Endpoints and decision

Report foreground Dice, A/P macro Dice, average symmetric surface distance and
95th percentile symmetric surface distance (mm), relative foreground volume
error, A/P swap counts, cut MAE, and corrected/introduced voxels. Surface and
volume metrics apply to foreground; cut and swaps apply to A/P. Show paired
case-bootstrap 95% intervals (10,000 resamples), improved/worsened/tied cases,
and per-case outcomes. Intervals are exploratory conditional-on-fit summaries,
not refitting uncertainty or participant-level generalization.

Whole-foreground advancement: on the 24 assessment cases, atlas-containing
method gains >=0.1 Dice percentage point over raw network AND graph-only,
its Dice-delta interval versus raw excludes zero, mean ASSD does not worsen,
and cases improved are at least cases worsened. A/P advancement: >=0.1-mm MAE
gain versus model-only plane, interval excludes zero, mean A/P Dice does not
worsen, and no more than one reference-correct baseline cut is spoiled.

If either gate passes, evaluate its already selected settings once on all 52
development cases without retuning. Failure is still a completed pilot: report
the limitation, oracle opportunity, and error complementarity, without launching
a training sweep. Success warrants a future out-of-fold-backbone study, not a
claim of validated improvement. An augmented-model replication is a subsequent
step only if the unaugmented development improvement also holds.

## Opportunity diagnostics

On assessment, report per-case oracle selection between raw and each locked
candidate (not deployable), changes on initially correct cases, and disagreement
between atlas-only and network predictions. These diagnose whether a reliable
selection/gating method might have room to improve; do not fit a gate on the
assessment set. Record transform quality and compare registered priors with
the centered control to isolate the value of image registration.

All derived arrays/transforms go to ignored experiments/atlas_pilot_20260924/;
compact numerical evidence and protocol remain in docs/experiments/atlas_pilot_20260924/.
Preserve existing experiments and training defaults.
