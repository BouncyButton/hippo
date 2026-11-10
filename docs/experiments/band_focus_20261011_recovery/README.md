# Focused-band and neighborhood follow-up

Active scope: **362 standard75 arms**: **172 new asymmetric-band combinations first**, followed by unfinished original experiments. See [ASYM32_EXTENSION.md](ASYM32_EXTENSION.md) for the complete additions and priority order.

Submitted jobs: **696517**, then dependent continuation **696518**. See [asym32_submission.json](asym32_submission.json).

The user requested cancellation of job 696104 to extend the study. All 57 completed results and the interrupted checkpoint are preserved. Every experiment remains 5%, fold 0, seed 0, ordinary SwinUNETR, max75 and StepLR20. The three longer-schedule runs remain deleted and excluded.

The immutable protocol.json/source matrix document the original submission. The active amendment is asym32_scope.json; use scripts.band_focus.asym32_resume and run_asym32.sh for this extended campaign. The original retained 190-arm list follows below; the 172 additions are listed in ASYM32_EXTENSION.md.

## Retained original families

| Family | Arms |
|---|---:|
| control | 5 |
| strength | 23 |
| neighborhood | 21 |
| spatial_coverage | 9 |
| smooth_focus | 11 |
| ap_focus | 12 |
| hardness | 17 |
| width | 24 |
| side_focus | 19 |
| pair_interaction | 12 |
| penalty | 12 |
| distance_focus | 9 |
| multiscale | 6 |
| surface_interaction | 4 |
| curriculum | 6 |

## Original order, after all additions

| Arm | Family | Idea |
|---|---|---|
| dice | control | Fresh Dice-only control from the common prefix. |
| bands | control | Exact Dice + original bands reference. |
| independent13 | control | Exact previous 78.13% reference. |
| focus25 | control | Exact winning 78.94% formulation. |
| strong_inside | control | Exact 78.74% stronger-pair reference. |
| budget_band1p0_pair1p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| band_only_strength1p0 | strength | Does hard-band supervision need the endpoint pairs? |
| neighbor_k3_q25_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| tile8_q25 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| smooth_power1p0_mix50 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| ap_bce_w3_share10 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| hard_q10_mix25 | hardness | Separate the selected fraction from its loss mass. |
| hard_q10_mix50 | hardness | Separate the selected fraction from its loss mass. |
| hard_q10_mix75 | hardness | Separate the selected fraction from its loss mass. |
| hard_q15_mix25 | hardness | Separate the selected fraction from its loss mass. |
| hard_q15_mix50 | hardness | Separate the selected fraction from its loss mass. |
| hard_q15_mix75 | hardness | Separate the selected fraction from its loss mass. |
| hard_q20_mix25 | hardness | Separate the selected fraction from its loss mass. |
| hard_q20_mix50 | hardness | Separate the selected fraction from its loss mass. |
| hard_q20_mix75 | hardness | Separate the selected fraction from its loss mass. |
| hard_q25_mix25 | hardness | Separate the selected fraction from its loss mass. |
| hard_q25_mix75 | hardness | Separate the selected fraction from its loss mass. |
| hard_q35_mix25 | hardness | Separate the selected fraction from its loss mass. |
| hard_q35_mix50 | hardness | Separate the selected fraction from its loss mass. |
| hard_q35_mix75 | hardness | Separate the selected fraction from its loss mass. |
| hard_q50_mix25 | hardness | Separate the selected fraction from its loss mass. |
| hard_q50_mix50 | hardness | Separate the selected fraction from its loss mass. |
| hard_q50_mix75 | hardness | Separate the selected fraction from its loss mass. |
| width_i1_o1_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i1_o1_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i1_o1_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i2_o2_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i2_o2_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i2_o2_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i4_o4_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i4_o4_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i4_o4_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i5_o5_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i5_o5_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i5_o5_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i1_o3_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i1_o3_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i1_o3_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i3_o1_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i3_o1_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i3_o1_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i2_o3_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i2_o3_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i2_o3_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i3_o2_q15 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i3_o2_q25 | width | Physical support and concentration interaction, including asymmetric widths. |
| width_i3_o2_q40 | width | Physical support and concentration interaction, including asymmetric widths. |
| side_qi15_qo15 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi15_qo25 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi15_qo40 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi15_qo60 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi25_qo15 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi25_qo40 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi25_qo60 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi40_qo15 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi40_qo25 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi40_qo40 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi40_qo60 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi60_qo15 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi60_qo25 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi60_qo40 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_qi60_qo60 | side_focus | Different error selection fractions for missing and extra tissue, equal side mass. |
| side_mass40 | side_focus | Side-mass sensitivity around the winning formulation. |
| side_mass45 | side_focus | Side-mass sensitivity around the winning formulation. |
| side_mass55 | side_focus | Side-mass sensitivity around the winning formulation. |
| side_mass60 | side_focus | Side-mass sensitivity around the winning formulation. |
| budget_band0p75_pair0p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band0p75_pair1p0 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band0p75_pair1p25 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band0p75_pair1p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band0p75_pair2p0 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p0_pair0p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p0_pair1p25 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p0_pair2p0 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p25_pair0p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p25_pair1p0 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p25_pair1p25 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p25_pair1p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p25_pair2p0 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p5_pair0p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p5_pair1p0 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p5_pair1p25 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p5_pair1p5 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| budget_band1p5_pair2p0 | strength | Independent calibrated band/pair strength grid; retain p95 safety cap. |
| band_only_strength0p5 | strength | Does hard-band supervision need the endpoint pairs? |
| band_only_strength1p5 | strength | Does hard-band supervision need the endpoint pairs? |
| band_only_strength2p0 | strength | Does hard-band supervision need the endpoint pairs? |
| pair_inside60_strength1p0 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_inside60_strength1p5 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_inside40_strength1p0 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_inside40_strength1p5 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_inside_far_strength1p0 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_inside_far_strength1p5 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_near60_30_10_strength1p0 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_near60_30_10_strength1p5 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_far75_strength1p0 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_far75_strength1p5 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_separate_strength1p0 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| pair_separate_strength1p5 | pair_interaction | Combine hard bands with supported distance/side/eligibility hypotheses. |
| penalty_bce_q15 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_bce_q25 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_hybrid25_q15 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_hybrid25_q25 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_hybrid50_q15 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_hybrid50_q25 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_power15_q15 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_power15_q25 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_power3_q15 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_power3_q25 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_huber_q15 | penalty | Change the confidence-error response within hard regional bands. |
| penalty_huber_q25 | penalty | Change the confidence-error response within hard regional bands. |
| decay0p75_q15 | distance_focus | Combine distance emphasis with hard-error selection. |
| decay0p75_q25 | distance_focus | Combine distance emphasis with hard-error selection. |
| decay1p0_q15 | distance_focus | Combine distance emphasis with hard-error selection. |
| decay1p0_q25 | distance_focus | Combine distance emphasis with hard-error selection. |
| decay2p0_q15 | distance_focus | Combine distance emphasis with hard-error selection. |
| decay2p0_q25 | distance_focus | Combine distance emphasis with hard-error selection. |
| shell123_50_35_15 | distance_focus | Select hard voxels independently within each physical shell. |
| shell123_33_33_33 | distance_focus | Select hard voxels independently within each physical shell. |
| shell123_20_50_30 | distance_focus | Select hard voxels independently within each physical shell. |
| neighbor_k3_q15_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k3_q15_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k3_q25_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k3_q50_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k3_q50_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k5_q15_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k5_q15_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k5_q25_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k5_q25_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k5_q50_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k5_q50_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k7_q15_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k7_q15_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k7_q25_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k7_q25_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k7_q50_point0 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_k7_q50_point50 | neighborhood | Rank side-normalized neighborhoods by mean point error, then supervise actual voxel errors. |
| neighbor_multi35 | neighborhood | Combine local error context at multiple scales. |
| neighbor_multi37 | neighborhood | Combine local error context at multiple scales. |
| neighbor_multi357 | neighborhood | Combine local error context at multiple scales. |
| tile4_q15 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| tile4_q25 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| tile4_q50 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| tile8_q15 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| tile8_q50 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| tile16_q15 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| tile16_q25 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| tile16_q50 | spatial_coverage | Keep a difficult subset in each spatial tile; weight tiles by their valid band mass. |
| smooth_power0p5_mix25 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_power0p5_mix50 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_power0p5_mix75 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_power1p0_mix25 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_power1p0_mix75 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_power2p0_mix25 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_power2p0_mix50 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_power2p0_mix75 | smooth_focus | Continuous detached difficulty weights, capped at 20, instead of a hard cutoff. |
| smooth_neighbor3 | smooth_focus | Smooth neighborhood weights preserve weak supervision everywhere. |
| smooth_neighbor5 | smooth_focus | Smooth neighborhood weights preserve weak supervision everywhere. |
| multi13_strength1p0 | multiscale | Average nested hard-band losses with one calibrated band budget. |
| multi13_strength1p5 | multiscale | Average nested hard-band losses with one calibrated band budget. |
| multi25_strength1p0 | multiscale | Average nested hard-band losses with one calibrated band budget. |
| multi25_strength1p5 | multiscale | Average nested hard-band losses with one calibrated band budget. |
| multi135_strength1p0 | multiscale | Average nested hard-band losses with one calibrated band budget. |
| multi135_strength1p5 | multiscale | Average nested hard-band losses with one calibrated band budget. |
| ap_bce_w1_share10 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_bce_w1_share25 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_brier_w1_share10 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_brier_w1_share25 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_bce_w3_share25 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_brier_w3_share10 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_brier_w3_share25 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_bce_w5_share10 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_bce_w5_share25 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_brier_w5_share10 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| ap_brier_w5_share25 | ap_focus | Actual A/P-label conditional errors near the interface; shares the existing extra budget. |
| surface_sdf8_share10 | surface_interaction | Low-share surface feedback with the new hard-band winner. |
| surface_sdf8_share25 | surface_interaction | Low-share surface feedback with the new hard-band winner. |
| surface_chamfer_share10 | surface_interaction | Low-share surface feedback with the new hard-band winner. |
| surface_chamfer_share25 | surface_interaction | Low-share surface feedback with the new hard-band winner. |
| curriculum_wide_e20 | curriculum | Gradually concentrate feedback; calibrate the final objective once at the common prefix. |
| curriculum_wide_e35 | curriculum | Gradually concentrate feedback; calibrate the final objective once at the common prefix. |
| curriculum_gentle_e20 | curriculum | Gradually concentrate feedback; calibrate the final objective once at the common prefix. |
| curriculum_gentle_e35 | curriculum | Gradually concentrate feedback; calibrate the final objective once at the common prefix. |
| curriculum_wide_gentle_e20 | curriculum | Gradually concentrate feedback; calibrate the final objective once at the common prefix. |
| curriculum_wide_gentle_e35 | curriculum | Gradually concentrate feedback; calibrate the final objective once at the common prefix. |
