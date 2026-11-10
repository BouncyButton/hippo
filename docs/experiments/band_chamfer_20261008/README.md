# Band and Chamfer follow-up

64 ordinary SwinUNETR runs; 5% (10 training cases), fold 0, seed 0; 52 validation cases.
The same five-epoch Dice-only checkpoint initializes every arm, including two fresh controls.
75 epochs maximum, unchanged optimizer, schedule, early stopping, and A/P Dice checkpoint selection.

## Objective matrix

| ID | Family | Comparison | Hypothesis |
|---|---|---|---|
| bands | control | bands | Fresh exact Dice + canonical two-step bands. |
| pair13 | control | bands | Fresh previous 1 + 3 mm probability-pair winner. |
| chamfer_mean | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| ap_ce_full | ap_control | bands | Existing foreground-only original-label conditional A/P CE. |
| ap_chamfer_mean_ce | ap_chamfer | ap_ce_full | Actual A/P interface distance plus a semantic guard against label reversal; one shared budget. |
| ap_cut_w1_t01 | ap_distribution | ap_ce_full | Wasserstein-1 between candidate-cut probability and the actual interface Y distribution; retain mixed slices. |
| pair13_plus_ap_chamfer | ap_combination | pair13 | Split the extra budget equally between outer pairs and guarded internal-surface supervision. |
| pair13_plus_ap_cut_w1 | ap_combination | pair13 | Combine outer pairs with global A/P cut-distribution alignment. |
| band_euclid3 | band | bands | Physical distance to GT surface rather than six-neighbor graph distance. |
| band_shell135 | band | bands | Equal mass per nonempty 0-1, 1-3, 3-5 mm shell and side. |
| band_morph1_valid | band | bands | Change morphological support; exclude padding. Width 2 isolates the FOV policy. |
| band_morph2_valid | band | bands | Change morphological support; exclude padding. Width 2 isolates the FOV policy. |
| band_morph3_valid | band | bands | Change morphological support; exclude padding. Width 2 isolates the FOV policy. |
| band_morph5_valid | band | bands | Change morphological support; exclude padding. Width 2 isolates the FOV policy. |
| band_euclid1 | band | bands | Physical distance to GT surface rather than six-neighbor graph distance. |
| band_euclid2 | band | bands | Physical distance to GT surface rather than six-neighbor graph distance. |
| band_euclid5 | band | bands | Physical distance to GT surface rather than six-neighbor graph distance. |
| band_outer75 | band | band_euclid3 | More weight on extra tissue while retaining inner supervision. |
| band_inner1_outer3 | band | band_euclid3 | Separate inside versus outside reach, retaining equal side mass. |
| band_inner3_outer1 | band | band_euclid3 | Separate inside versus outside reach, retaining equal side mass. |
| band_soft05 | band | band_euclid3 | Soft sigmoid targets near the annotation; boundary remains at the same position. |
| pair_common_1 | pair | pair_common_13_equal | Same eligible oriented GT faces for distance and mixture comparisons. |
| pair_common_3 | pair | pair_common_13_equal | Same eligible oriented GT faces for distance and mixture comparisons. |
| pair_common_13_equal | pair | pair_common_13_equal | Same eligible oriented GT faces for distance and mixture comparisons. |
| pair_common_13_far75 | pair | pair_common_13_equal | Same eligible oriented GT faces for distance and mixture comparisons. |
| pair_independent13 | pair | pair13 | Remove the cross-product coupling; keep identical endpoint samples. |
| pair_shift_symmetric | pair | pair_shift_symmetric | Same six-voxel-valid faces; move the 3 mm endpoints only. |
| pair_shift_outward | pair | pair_shift_symmetric | Same six-voxel-valid faces; move the 3 mm endpoints only. |
| pair_shift_inward | pair | pair_shift_symmetric | Same six-voxel-valid faces; move the 3 mm endpoints only. |
| pair_gap08 | pair | pair13 | Stop pushing when the inside/outside probability gap is sufficient. |
| pair_gap09 | pair | pair13 | Stop pushing when the inside/outside probability gap is sufficient. |
| sdf3 | surface | bands | Outer-foreground signed-distance probability loss, normalized separately inside and outside. |
| sdf8 | surface | bands | Outer-foreground signed-distance probability loss, normalized separately inside and outside. |
| distance_brier1 | surface | bands | Distance-weighted squared probability errors; finite gradients without a predicted surface. |
| distance_brier2 | surface | bands | Distance-weighted squared probability errors; finite gradients without a predicted surface. |
| chamfer_squared | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| chamfer_huber1 | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| chamfer_tolerance05 | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| chamfer_pred75 | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| chamfer_gt75 | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| chamfer_pred_only | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| chamfer_gt_only | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| chamfer_tail10 | surface | chamfer_mean | Interpolated outer isosurface: alter distance penalty, directional balance, tolerance, or tail weight. |
| pair13_plus_mean | combination | pair13 | Pair and surface terms share ONE calibrated extra-gradient budget. |
| pair13_plus_squared | combination | pair13 | Pair and surface terms share ONE calibrated extra-gradient budget. |
| pair13_plus_tail10 | combination | pair13 | Pair and surface terms share ONE calibrated extra-gradient budget. |
| pair13_plus_sdf8 | combination | pair13 | Pair and surface terms share ONE calibrated extra-gradient budget. |
| shell135_plus_chamfer | combination | band_shell135 | Multiscale regional supervision plus actual predicted-surface displacement. |
| shell135_plus_pair13 | combination | band_shell135 | Multiscale bands plus local cross-boundary ordering. |
| inner1_outer3_plus_chamfer | combination | band_inner1_outer3 | Broader outside reach plus surface alignment. |
| ap_ce_band3 | ap_control | ap_ce_full | Same conditional CE within 3 mm of the actual annotated interface. |
| ap_pair13 | ap_control | ap_ce_full | A/P crossing pairs only; isolate internal ordering from outer-boundary terms. |
| ap_chamfer_shape_only | ap_chamfer | ap_chamfer_mean_ce | Diagnostic shape-only control: deliberately invariant to swapping A/P names. |
| ap_chamfer_squared_ce | ap_chamfer | ap_ce_full | Actual A/P interface distance plus a semantic guard against label reversal; one shared budget. |
| ap_chamfer_huber1_ce | ap_chamfer | ap_ce_full | Actual A/P interface distance plus a semantic guard against label reversal; one shared budget. |
| ap_chamfer_tolerance05_ce | ap_chamfer | ap_ce_full | Actual A/P interface distance plus a semantic guard against label reversal; one shared budget. |
| ap_chamfer_tail10_ce | ap_chamfer | ap_ce_full | Actual A/P interface distance plus a semantic guard against label reversal; one shared budget. |
| ap_chamfer_gt75_ce | ap_chamfer | ap_ce_full | Actual A/P interface distance plus a semantic guard against label reversal; one shared budget. |
| ap_chamfer_column_ce | ap_chamfer | ap_chamfer_mean_ce | Match transition points on the same X/Z column, preventing lateral nearest-point shortcuts. |
| ap_cut_w1_t1 | ap_distribution | ap_ce_full | Wasserstein-1 between candidate-cut probability and the actual interface Y distribution; retain mixed slices. |
| ap_cut_expected_l1 | ap_distribution | ap_cut_w1_t01 | Expected absolute cut displacement control; compare with distribution transport. |
| pair13_plus_ap_chamfer_outer75 | ap_combination | pair13_plus_ap_chamfer | Retain more outer-pair strength; test whether prior outer+AP degradation was a budget tradeoff. |
| outer_chamfer_plus_ap_chamfer | ap_combination | chamfer_mean | Geometric supervision of both outer surface and guarded internal transition. |
| ap_sdf_ce | ap_chamfer | ap_chamfer_mean_ce | Distance-weighted conditional A/P errors as a surface-free comparator. |

## Definitions and safeguards

GT geometry only in losses. Original 45 arms use outer foreground; 19 added arms target the A/P transition. Verified 1 mm native RAS voxels. New geometry excludes padding; canonical bands preserve the prior implementation exactly.

Exact nearest-point distances on all linear p_foreground=0.5 axis-edge intersections. Coordinates retain gradients; crossing and nearest-neighbor assignments are discrete. Piecewise differentiable, not a mesh-area integral. All edges inside native FOV. Slope denominator floor 1e-6. Empty predicted/GT surfaces zero only the Chamfer term with telemetry; Dice and bands remain active. Calibration rejects a zero-gradient term.

Reuse the verified epoch-5 Dice-only model/optimizer/scheduler/order/RNG. Each band definition gets median 0.1/p95 <=0.5 logit-gradient RMS relative to Dice. Pair/surface/AP/conditional-CE extras individually normalized, optional predeclared relative shares applied, then jointly rescaled to ONE median 0.1/p95 <=0.5 budget relative to Dice+bands. Fixed coefficients, ramp epochs 6-10. No validation tuning.

Physical bands use Euclidean distance to the GT axis-edge midpoint samples. This is a discrete surface approximation; shell widths are in mm, not erosion iterations.

Common3 pairs require offsets -1,0 inside and 1,2 outside at the same oriented crossing. Shift controls require -2,-1,0 inside and 1,2,3 outside. Missing long-distance support is reported, never invented.

Chamfer variants supervise the p_anterior+p_posterior=0.5 outer surface. Final masks/metrics use the unchanged three-class argmax policy. The surface loss does not enforce topology or A/P anatomy. Grid-edge samples are not uniform surface-area samples; this bias is shared by the Chamfer comparisons.

The empty-surface fallback is not assigned a geometric meaning: its contribution is zero and counted, while Dice/bands still train. Fixed-topology gradients cannot create a new surface by themselves.

## A/P evidence and design

Training-only inspection: 7/10 cases have exactly planar A/P labels; 3/10 have one mixed slice, with 39, 9, and 7 voxels disagreeing with the best plane. Affines are RAS, anterior high Y. Do not overwrite those mixed-slice labels with a fitted plane.

The repository already implements conditional A/P CE and a strict planar cut posterior (thesis/new_constraints/ap_plane/conditional_ce.py; thesis/new_constraints/ap_cut/posterior.py). The former is reused as a control. The latter supplies the score construction, generalized here to the actual interface-distribution target because strict planar validity would exclude 3/10 training cases.

Previous harmonic and graph-interface experiments did not pass their advancement criteria. This follow-up tests direct supervision and distance geometry, without assuming raw MRI gradients identify the anatomical landmark or adding a learned cut head.

Chamfer alone cannot identify which side is anterior: the shape-only arm exposes that limitation; the other A/P Chamfer arms include original-label conditional CE. All A/P terms are training-only on GT foreground. No GT ROI, plane or cut is supplied to the inference model.

Same-column matching prohibits lateral nearest-point shortcuts but cannot create a missing crossing; it logs missing columns. Cut-distribution objectives stay differentiable even without a hard A/P transition. Their targets describe the released annotation, not a verified uncal-apex landmark.

Final raw-prediction diagnostics include A/P point-interface Chamfer and HD95, best-plane cut error (descriptive when labels are mixed), near/far 3 mm swaps, and undefined-interface counts. These do not select checkpoints or modify inference.

Report A/P and foreground Dice, extra/missed/swapped voxels, mean surface distance, HD95, per-case differences, calibration, geometry counts, and runtime. All 64 arms remain visible.

Exploratory single fold/seed with 64 comparisons. Report every arm and paired deltas. Replication required before claiming improvement. No automatic best-only continuation.

Sources: [Chamfer definition](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html), [regional signed-distance boundary loss](https://proceedings.mlr.press/v102/kervadec19a.html).

## Storage

64 best checkpoints; one active resumable checkpoint; one retained common warmup; compressed validation masks. Reuse dataset. Preserve all earlier results.
Estimated additional peak: 5.0 GiB. Require 7.0 GiB free user quota before submission. No cleanup.
