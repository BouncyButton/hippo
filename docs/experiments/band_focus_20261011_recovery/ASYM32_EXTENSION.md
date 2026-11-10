# Asymmetric-band extension: additions run first

The user requested interruption of job 696104, addition of all relevant variants around the 3 mm inside / 2 mm outside result, and priority for every added variant. Slurm confirmed cancellation; 57 completed runs and the interrupted side_qi25_qo40 checkpoint at epoch 41 are preserved.

**172 distinct additions; 362 total arms. All 172 additions execute before the 133 unfinished original arms.** Existing completed runs are verified and skipped.

All additions use 3 mm inside / 2 mm outside bands. Fixed conditions: ordinary SwinUNETR, 5%, fold 0, seed 0, same ten training and 52 validation cases, max75, StepLR20 gamma0.5, early stopping min25/patience8/min_delta0.0005, no augmentation or pretraining. The three deleted extended-schedule runs remain excluded.

The additions combine hard-fraction/emphasis settings, separate inside/outside fractions and weights, single- and multiple-size neighborhood ranking, neighborhood × side interactions, and a local band/pair strength grid. Three explicitly labelled band-only ablations test whether the endpoint term remains necessary. All other additions retain the independent 1 mm + 3 mm endpoint term, with the two distances weighted 50/50.

Equivalent configurations are deduplicated against the original 190 arms and other additions, including equivalent explicit inside/outside fractions. The remaining original penalty, distance-weighting, A/P, surface, curriculum and other families stay in the study. The extension is focused on the combinations discussed; it does not add an unbounded Cartesian product of every existing loss family.

The original protocol and scientific source files are immutable. Per-run identities remain the original protocol identity plus each complete run configuration. A separate hashed asym32_scope.json binds the added entries, execution order, source files and preservation receipt. Original calibration remains unchanged. Each new arm is calibrated only on the same ten training cases from the same verified epoch-5 checkpoint, using the original gradient-budget procedure.

The cluster preservation check passed. Active scope SHA-256: `171b1c62b7a350ffc6edfd6c4b1defebcce012459c9c08188ed35b3df3faffda`. The epoch-41 resume checkpoint SHA-256 is `3eb11d4c50eb138b8a3c5b0f489a52cc13259ade986353df0feb722b6efa8c68`; its best checkpoint is also hash-bound in the scope receipt.

Validation results motivated this amendment. This is adaptive exploration on the same validation set, not independent confirmation. The selected methods require a new seed/fold evaluation before generalization claims.

## Added families

| Family | New arms |
|---|---:|
| a32_hardness | 18 |
| a32_neighborhood | 60 |
| a32_side_fraction | 26 |
| a32_side_mass | 12 |
| a32_neighborhood_side | 16 |
| a32_strength | 11 |
| a32_multineighborhood | 18 |
| a32_neighborhood_mass | 8 |
| a32_pair_ablation | 3 |

## Verification and scheduling

Submitted as job **696517**, followed by job **696518** with dependency `afterok:696517`. The submission receipt confirms all 172 additions run first, 362 total arms, 57 preserved completed results, 5%, fold 0, seed 0, max75 and StepLR20. See [asym32_submission.json](asym32_submission.json).

Startup was confirmed on gnode04; the continuation is pending on its dependency. The worker passed the original payload and protocol checks. The GPU gate then passed for all 172 additions on the ten training cases, including successful encoder/decoder optimizer updates for four representative combinations. See [asym32_calibration_preflight.json](asym32_calibration_preflight.json).

The first added arm, `a32_hard_q35_mix75`, restored the common epoch-5 checkpoint, loaded its frozen calibration, and trained through epoch 13 without an error in the startup snapshot. Its saved configuration confirms max75, StepLR20, the original early stopping, and no auxiliary branch. BeeGFS had 21.90 GiB free at submission. See [asym32_startup_verification.json](asym32_startup_verification.json).

Four local tests passed: immutable original configurations and source hashes, exact execution order with no equivalent duplicates, finite losses/gradients and calibrated gradient budgets for all 172 additions, asymmetric-support/finite-difference checks, and exact interrupted-versus-uninterrupted training for focus and mixed neighborhood objectives. Shell syntax passed. See [asym32_local_checks.json](asym32_local_checks.json).

Before scientific training, the allocated GPU checks and calibrates all 172 additions on the ten training cases, then performs real encoder/decoder optimizer updates for four representative combinations. This gate must pass before any added training run starts.

Two sequential allocations use the same one-MIG-GPU/4-CPU/24-GiB/23:50 resource limits. A worker stops launching new arms after 22.5 hours and exits successfully for its afterok continuation. The pending continuation is cancelled if the first allocation completes the sweep. Model training schedules are unchanged.

## Added execution order

| Arm | Idea |
|---|---|
| a32_hard_q35_mix75 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_neighbor_k3_q25_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_side_qi25_qo40_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_mass45_q25_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_hard_q40_mix75 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_neighbor_k3_q35_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q25_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_qi25_qo40_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_mass55_q25_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_budget_b1p0_p1p5 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_hard_q10_mix25 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q10_mix50 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q10_mix75 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q15_mix25 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q15_mix75 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q20_mix25 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q20_mix50 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q20_mix75 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q25_mix25 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q25_mix75 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q35_mix25 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q35_mix50 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q40_mix25 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q50_mix25 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q50_mix50 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_hard_q50_mix75 | Cross the asymmetric support with the full hard-fraction/emphasis grid; reuse existing equivalent arms. |
| a32_side_qi15_qo25_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi15_qo25_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi15_qo40_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi15_qo40_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi15_qo60_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi15_qo60_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi25_qo15_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi25_qo15_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi25_qo40_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi25_qo60_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi25_qo60_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi40_qo15_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi40_qo15_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi40_qo25_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi40_qo25_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi40_qo60_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi40_qo60_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo15_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo15_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo25_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo25_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo40_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo40_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo60_mix50 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_side_qi60_qo60_mix75 | Cross independent inside/outside hard fractions with moderate and stronger focus. |
| a32_mass40_q25_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass40_q35_mix75 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass40_q40_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass45_q35_mix75 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass45_q40_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass55_q35_mix75 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass55_q40_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass60_q25_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass60_q35_mix75 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_mass60_q40_mix50 | Test the missed/extra-tissue balance under three supported focus settings. |
| a32_neighbor_k3_q15_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q15_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q15_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q15_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q25_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q25_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q25_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q35_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q35_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q35_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q40_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q40_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q40_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q40_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q50_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q50_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q50_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k3_q50_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q15_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q15_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q15_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q15_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q25_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q25_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q25_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q35_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q35_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q35_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q35_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q40_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q40_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q40_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q40_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q50_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q50_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q50_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k5_q50_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q15_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q15_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q15_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q15_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q25_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q25_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q25_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q25_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q35_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q35_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q35_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q35_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q40_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q40_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q40_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q40_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q50_mix50_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q50_mix50_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q50_mix75_point0 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_k7_q50_mix75_point50 | Combine asymmetric bands with local error ranking, including pure and mixed point/neighborhood ranking. |
| a32_neighbor_multi35_q25_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi35_q25_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi35_q35_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi35_q35_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi35_q40_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi35_q40_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi37_q25_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi37_q25_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi37_q35_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi37_q35_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi37_q40_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi37_q40_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi357_q25_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi357_q25_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi357_q35_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi357_q35_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi357_q40_mix50 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_multi357_q40_mix75 | Combine local error context at multiple sizes on the asymmetric support. |
| a32_neighbor_k3_qi15_qo25_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k3_qi15_qo25_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k3_qi25_qo15_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k3_qi25_qo15_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k3_qi25_qo40_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k3_qi40_qo25_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k3_qi40_qo25_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi15_qo25_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi15_qo25_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi25_qo15_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi25_qo15_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi25_qo40_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi25_qo40_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi40_qo25_mix50 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k5_qi40_qo25_mix75 | Three-way interaction: asymmetric support, neighborhood context and unequal side-specific focus. |
| a32_neighbor_k3_mass45_q25_mix50 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_neighbor_k3_mass45_q35_mix75 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_neighbor_k3_mass55_q25_mix50 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_neighbor_k3_mass55_q35_mix75 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_neighbor_k5_mass45_q25_mix50 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_neighbor_k5_mass45_q35_mix75 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_neighbor_k5_mass55_q25_mix50 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_neighbor_k5_mass55_q35_mix75 | Test whether spatial selection and a modest side-mass adjustment complement each other. |
| a32_budget_b0p75_p0p75 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b0p75_p1p0 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b0p75_p1p25 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b0p75_p1p5 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b1p0_p0p75 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b1p0_p1p25 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b1p25_p0p75 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b1p25_p1p0 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b1p25_p1p25 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_budget_b1p25_p1p5 | Local calibrated band/pair strength grid around the asymmetric winner; distances stay 1+3 mm. |
| a32_band_only_strength0p5 | Matched ablation: test whether endpoint pairs remain necessary with asymmetric focused bands. |
| a32_band_only_strength1p0 | Matched ablation: test whether endpoint pairs remain necessary with asymmetric focused bands. |
| a32_band_only_strength1p5 | Matched ablation: test whether endpoint pairs remain necessary with asymmetric focused bands. |
