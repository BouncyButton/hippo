# Independent targets and bands follow-up

189 ordinary SwinUNETR arms; 5%, fold 0, seed 0; no auxiliary branch.
Same epoch-5 prefix, 75-epoch cap, optimizer, early stopping and raw A/P Dice checkpoint selection as job 690825.

Annotation-derived native 1 mm outer boundaries. Original controls unchanged. Independent scales use oriented crossings; whole intervening side segments must remain in class and native FOV. Separate-support arms retain valid sides independently. Common-support arms match oriented faces. Endpoint steps are measured from immediate inside/outside voxel centers; actual separation is inside_steps + outside_steps + 1 mm.

Verified shared epoch-5 Dice-only prefix. Bands calibrated median logit-gradient RMS 0.1 and p95 <= 0.5 relative to Dice. Extras jointly calibrated at 0.1 relative to Dice+bands, except explicit 0.05/0.15 budget arms; fixed p95 cap 0.5. Training-only calibration on all 10 cases; coefficients fixed, original epoch6-10 ramp.

Retain three controls and current top five best checkpoints, one active resume checkpoint, and common warmup. Verify all completed artifacts before retiring other new-run checkpoints. Keep every prediction, metric, history, config, source and calibration. No deletion from previous campaigns.

Exploratory post-hoc follow-up: 189 predeclared arms, one fold/seed, 10 training and 52 validation cases. Hypotheses informed by prior validation errors; same validation set is development data. Report every result and paired differences; independent seed/fold confirmation required. No automatic outcome-based additions.

| Experiment | Family | Comparison | Hypothesis |
|---|---|---|---|
| bands | control | bands | Exact canonical Dice + bands. |
| pair13 | control | bands | Exact previous coupled comparison. |
| independent13 | control | independent13 | Exact 78.13% independent-target reference. |
| d1_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d3_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_25_75_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_equal_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| sides_i75_o50_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| d135_mid_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| support_separate_13_50 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| combo_sdf8_in50 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| band_brier2_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| d1_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d1_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d1_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d3_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d3_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d3_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d5_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d5_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d5_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d5_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d7_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d7_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d7_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d7_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_75_25_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_75_25_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_75_25_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_75_25_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_equal_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_equal_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_25_75_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_25_75_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d13_25_75_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d15_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d15_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d15_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d15_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d35_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d35_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d35_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d35_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_equal_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_equal_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_equal_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_equal_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_mid_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_mid_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_mid_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_near_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_near_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_near_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d135_near_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d1357_in40 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d1357_in50 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d1357_in60 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| d1357_in70 | distance_side | independent13 | Factorial distance mixture and inside/outside balance; per-scale eligible support. |
| sides_i25_o50_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i25_o50_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i25_o75_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i25_o75_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i25_o100_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i25_o100_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i50_o25_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i50_o25_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i50_o75_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i50_o75_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i50_o100_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i50_o100_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i75_o25_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i75_o25_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i75_o50_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i75_o100_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i75_o100_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i100_o25_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i100_o25_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i100_o50_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i100_o50_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i100_o75_in50 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| sides_i100_o75_in60 | side_scales | independent13 | Let missing tissue and extra tissue have different scale preferences. |
| support_common3_1_100 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_common3_3_100 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_common3_13_50 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_common3_13_25 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_common5_13_50 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_common5_135_20 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_common7_135_20 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_common7_1357_40 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_separate_135_20 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| support_separate_1357_40 | support | independent13 | Separate distance effects from eligibility; separate support retains a valid side when its opposite is too thin. |
| offset_i0_o1_in50 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i0_o1_in60 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i1_o0_in50 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i1_o0_in60 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i0_o2_in50 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i0_o2_in60 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i2_o0_in50 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i2_o0_in60 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i1_o2_in50 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i1_o2_in60 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i2_o1_in50 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| offset_i2_o1_in60 | offset | independent13 | Shift far-scale supervision toward one side; retain adjacent-voxel reference. |
| reduce_unique_voxel_in50 | reduction | independent13 | Control repeated endpoint counts, boundary-anchor counts, or equal A/P surface mass. |
| reduce_unique_voxel_in60 | reduction | independent13 | Control repeated endpoint counts, boundary-anchor counts, or equal A/P surface mass. |
| reduce_anchor_in50 | reduction | independent13 | Control repeated endpoint counts, boundary-anchor counts, or equal A/P surface mass. |
| reduce_anchor_in60 | reduction | independent13 | Control repeated endpoint counts, boundary-anchor counts, or equal A/P surface mass. |
| reduce_class_balanced_in50 | reduction | independent13 | Control repeated endpoint counts, boundary-anchor counts, or equal A/P surface mass. |
| reduce_class_balanced_in60 | reduction | independent13 | Control repeated endpoint counts, boundary-anchor counts, or equal A/P surface mass. |
| penalty_bce_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_bce_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_power15_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_power15_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_power3_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_power3_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_huber_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_huber_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_tol05_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_tol05_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_tol10_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_tol10_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_top25_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_top25_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_top10_in50 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| penalty_top10_in60 | penalty | independent13 | Alter confidence-error emphasis without changing sampled anatomy. |
| band_brier1_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier1_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier1_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier1_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier2_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier2_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier2_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier3_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier3_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier3_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_brier3_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce1_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce1_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce1_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce1_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce2_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce2_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce2_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce2_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce3_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce3_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce3_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_bce3_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell123_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell123_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell123_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell123_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell135_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell135_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell135_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_shell135_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner1outer2_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner1outer2_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner1outer2_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner1outer2_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner2outer1_alone_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner2outer1_alone_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner2outer1_pairs_in50 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_inner2outer1_pairs_in60 | regional_band | independent13 | Test regional Brier/BCE and distance-shell weighting with and without independent endpoints. |
| band_focus_top25_in50 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| band_focus_top25_in60 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| band_focus_class_in50 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| band_focus_class_in60 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| band_focus_decay1_in50 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| band_focus_decay1_in60 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| band_focus_decay2_in50 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| band_focus_decay2_in60 | regional_band | independent13 | Focus regional bands on harder errors, balanced class surfaces, or nearby voxels. |
| combo_sdf3_in50 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_sdf3_in60 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_sdf8_in60 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_brier_distance_in50 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_brier_distance_in60 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_chamfer_in50 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_chamfer_in60 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_mass3_in50 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_mass3_in60 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_mass5_in50 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| combo_mass5_in60 | combination | independent13 | Independent endpoints plus distance, surface or local foreground-volume feedback; one joint extra budget. |
| budget_equal_0p5 | budget | independent13 | Explicit strength ablation; target ratios 0.05 or 0.15, same absolute p95 cap 0.5. |
| budget_equal_1p5 | budget | independent13 | Explicit strength ablation; target ratios 0.05 or 0.15, same absolute p95 cap 0.5. |
| budget_far_0p5 | budget | independent13 | Explicit strength ablation; target ratios 0.05 or 0.15, same absolute p95 cap 0.5. |
| budget_far_1p5 | budget | independent13 | Explicit strength ablation; target ratios 0.05 or 0.15, same absolute p95 cap 0.5. |
| budget_protect_inside_0p5 | budget | independent13 | Explicit strength ablation; target ratios 0.05 or 0.15, same absolute p95 cap 0.5. |
| budget_protect_inside_1p5 | budget | independent13 | Explicit strength ablation; target ratios 0.05 or 0.15, same absolute p95 cap 0.5. |
