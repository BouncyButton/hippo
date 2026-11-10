# Targeted A/P follow-up after shared-tissue audit

All 30 screening runs are retained. The 36 selected outer-constraint runs keep their existing order. The 44 broad A/P refinements and optional 32 combinations are replaced with 20 fixed targeted runs.

## Selection

Compare only annotated hippocampal voxels predicted as hippocampus by both the candidate and the fixed outer winner. Record corrected and newly introduced A/P labels on this identical tissue, plus tissue gained/lost outside that overlap. Lower total swaps alone is insufficient.

Eligibility requires lower common-tissue swaps and mean case error rate, no decrease in foreground Dice, no increase in total outer FN+FP, improved actual-interface Chamfer, and no increase in interface HD95. These are development-set screening criteria, not statistical proof of generalization.

Selected parent: `ap82_s1_winner_b25_g100` — 25% A/P bands / 75% endpoints, 10% calibrated A/P gradient target, 3 mm bands, equal 1/3 mm endpoint weights.

On common tissue: 219 corrected A/P labels, 182 newly introduced mistakes, net 37 fewer swaps. Foreground Dice increases by 0.0387 percentage points; outer FN+FP decrease by 176. Both outer surface distance and outer HD95 also improve. Across cases, 24 improve, 24 worsen, and 4 tie, so this remains a small exploratory signal.

## Twenty targeted runs

Ten one-factor profiles are each tested with the frozen outer winner and with Dice alone. All other parameters retain the selected parent values.

| Factor | New values |
|---|---|
| A/P gradient target | 7.5%, 12.5% |
| A/P band share | 12.5%, 37.5% (endpoint share is the remainder) |
| A/P band width | 2 mm, 1 mm |
| Hard-voxel selection | Hardest 25% on each A/P side |
| Hard-voxel emphasis | 75% on each A/P side |
| Anterior share | 40%, 60% (posterior is the remainder) |

All runs remain ordinary image-only SwinUNETR, 5% data, fold 0, seed 0, with the same five-epoch Dice prefix, max 75 epochs, StepLR20, and existing early stopping. Outer coefficients stay fixed. No automatic stage-three expansion follows these runs. A shared-tissue comparison is produced again when the targeted stage finishes.

## Interpretation corrected by the audit

The 81.296% run does improve common-tissue A/P labels (122 fewer swaps), despite having 84 more total swaps. Added foreground contributes 225 swaps, while removed foreground loses 19 baseline swaps: -122 + 225 - 19 = +84. It is not selected because total outer errors and outer boundary distances worsen. The strong 50/50 run improves common-tissue swaps by 83, but foreground Dice and total outer errors worsen.

Local checks cover shared-tissue accounting (including disappearing swaps), selection rejection conditions, paired/deduplicated 20-run generation, unchanged outer coefficients, finite gradients and calibration, and exact training resume. The cluster applies a train-only GPU calibration gate before the targeted runs.

## Deployment

Replacement job 697750; sequential continuation 697751. The controller preserves the 308 completed results and resumes the active outer run from epoch 37. The frozen total is 362 experiments including preserved history: 34 remaining outer runs followed by 20 targeted A/P runs at handoff. No automatic stage-three combinations remain.
