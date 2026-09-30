# Signed distance and extrema follow-up

Requested after the local and broad shoulder audits. This is another development
analysis of the same fold, not an untouched confirmation. Fix the following feature
families before computing their localization results; use only training labels to
calibrate cut offsets and select family/position strength.

## Grounding

Reuse the earlier extrema representation: presence and U(x,y)=max occupied z,
L(x,y)=min occupied z. Empty columns have no edge measurement. Use the largest
26-connected union component and verified 1-mm RAS geometry, with 6-voxel background
padding for stable distance transforms at volume edges. Define phi = EDT(outside)
− EDT(inside): negative inside and positive outside. Smooth phi at sigma=1 mm.

Sampling phi itself at an edge is not a concavity descriptor. Sample signed
curvature at the upper half-voxel face U+0.5 instead:

1. **SDF AP curvature:** curvature of the y-z implicit level curve, derived from
   phi_y, phi_z and its y/z Hessian. Negative values mean inward curvature.
2. **SDF 3-D curvature:** divergence of the normalized 3-D SDF gradient. Again use
   its negative (concave) component at upper extrema.
3. **Extrema concavity depth:** on each contiguous sagittal upper-extrema profile,
   fit its least concave majorant (upper convex-hull chain). At each y, measure the
   perpendicular signed gap between its hull segment and the smoothed profile.
   This measures indentation depth while accounting for global slope. It is a
   distance-to-envelope feature, not the raw signed-distance-field value.
4. **SDF + extrema:** geometric mean of AP concavity curvature and depth scores.

At each y, curvature scores average positive inward curvature across occupied
columns, multiplied by the fraction with inward curvature. Depth averages valid
per-column gaps. Smooth resulting profiles at sigma=1 mm; candidate-plane scores
average the adjacent samples at c−1 and c. No A/P class enters feature extraction.
Lower extrema, thickness and occupancy are retained as diagnostics; the hypothesis
is specifically about the upper concavity, so no validation-based choice of side.

## Evaluation

Use the same 208 training / 52 validation crops and the three available 52-case
predicted-support sets. Report each family's raw maximum and training-offset
version, plus a family-and-position selected version. Four-fold training OOF
evaluation encloses a three-fold inner selection over the four families and the
same position strengths [0, .25, .5, 1, 2, 4, 8]. Final validation selection uses
four-fold CV only within all 208 training cases. Always compare with position alone
and the native model cut. Preserve original local/broad results.

Zero-score curves are explicitly flagged; their deterministic midpoint tie-break
is only a forced prediction, not a detected landmark. Report every case, per-candidate
scores, paired bootstrap intervals, and selected-family stability. Test sign on a
synthetic inward bend, reject a convex control, verify physical translations,
mirroring and target-label independence. No segmentation training in this audit.
