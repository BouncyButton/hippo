# Atlas-guided refinement: completed pilot

24 September 2026.

This pilot tested image-registered label transfer from a 16-case atlas bank, with 24 separate cases for calibration and 24 for locked assessment. All are MSD fold-0 training cases; they were excluded from the atlas bank but were seen by the frozen backbone. This is an exploratory mechanism screen, not out-of-fold-backbone or independent-participant validation.

## Decision

Written advancement criteria, applied to every locked family: `{"centered_fusion": false, "affine_fusion": false, "deformable_fusion": false, "deformable_graph": false, "centered_ap": false, "affine_ap": false, "deformable_ap": false}`.

**The atlas pilot did not pass its advancement gate.** No development evaluation or model retraining was launched. The following results characterize this crop-registration and fusion implementation; they do not establish that all atlas methods are ineffective.

## Interpretation and next experiment

The calibration A/P signal did not transfer: affine fusion reduced calibration cut MAE from 0.4583 to 0.3333 mm, but increased assessment MAE from 0.5000 to 0.7917 mm. It improved one cut and worsened eight. Deformable fusion improved one cut (hippocampus_068) and worsened one previously correct cut (hippocampus_353); the other 22 were unchanged. Its A/P Dice gain versus the model-only plane was 0.0602 percentage points, with a paired 95% interval spanning -0.1578 to +0.3384. This does not support an A/P improvement claim.

Registration improved atlas-alone assessment foreground Dice from 70.927% to 79.348%, but the frozen network achieved 91.223%. The atlas-alone foreground corrected 5,282 network errors while introducing 22,725 new errors across the 24 cases. This shows some complementary voxel information, with a much larger cost when applied indiscriminately. These counts describe direct atlas replacement, not the calibrated fusion, which selected zero foreground atlas weight.

**Recommendation:** do not add this atlas prior as a training loss yet. If pursuing a second pilot, first improve registration on training-only cases with stronger deformation regularization and explicit local alignment checks; 65/144 nonlinear fits exceeded the displacement guard. Then test whether model uncertainty and agreement among independently registered atlases can identify useful boundary corrections. Keep individual atlas outputs to measure disagreement before averaging. Freeze any acceptance rule using out-of-fold backbone predictions, and evaluate on a fresh case/participant-separated cohort. These are proposed experiments, not improvements demonstrated here. The current locked A/P candidates offer only 0.0417 mm of gain even with perfect case selection, so a more elaborate case selector alone has little observed headroom.

![Assessment effect sizes](../../../experiments/atlas_pilot_20260924/assessment_results.png)

## Whole-hippocampus refinement

Settings were selected by foreground Dice on calibration cases only. Foreground can expand and contract across the entire native image grid. Conditional A/P labels remain the network's choice in these arms.

| Method | Foreground Dice % | Delta vs network, pp [95% CI] | ASSD, mm | HD95, mm | Better / worse / tied |
|---|---:|---:|---:|---:|---:|
| Frozen network | 91.2226 | +0.0000 [+0.0000, +0.0000] | 0.3753 | 1.052 | 0 / 0 / 24 |
| Graph only | 91.2598 | +0.0372 [+0.0040, +0.0723] | 0.3733 | 1.035 | 15 / 9 / 0 |
| Centered atlas + network | 91.2226 | +0.0000 [+0.0000, +0.0000] | 0.3753 | 1.052 | 0 / 0 / 24 |
| Affine atlas + network | 91.2226 | +0.0000 [+0.0000, +0.0000] | 0.3753 | 1.052 | 0 / 0 / 24 |
| Deformable atlas + network | 91.2226 | +0.0000 [+0.0000, +0.0000] | 0.3753 | 1.052 | 0 / 0 / 24 |
| Deformable atlas + network + graph | 91.2598 | +0.0372 [+0.0040, +0.0723] | 0.3733 | 1.035 | 15 / 9 / 0 |

| Arm | Selected alpha | Selected lambda | Foreground voxels corrected / introduced |
|---|---:|---:|---:|
| Graph only | 0 | 0.3 | 298 / 232 |
| Centered atlas + network | 0 | 0 | 0 / 0 |
| Affine atlas + network | 0 | 0 | 0 / 0 |
| Deformable atlas + network | 0 | 0 | 0 / 0 |
| Deformable atlas + network + graph | 0 | 0.3 | 298 / 232 |

## A/P cut localization

All A/P arms preserve original predicted foreground. The model-only plane controls for enforcing a plane, so changes versus the raw network cannot automatically be attributed to atlas information. A/P targets are reference best-fit planes, while Dice retains the original voxel labels.

| Method | Cut MAE, mm | A/P Dice % | Swaps | Raw-correct cuts spoiled |
|---|---:|---:|---:|---:|
| Frozen network | 0.5000 | 89.8793 | 985 | 0/13 |
| Model-only plane | 0.5000 | 89.9270 | 947 | 0/13 |
| Atlas-only plane | 1.5000 | 87.6622 | 2758 | 8/13 |
| Centered atlas + model plane | 0.5000 | 89.9270 | 947 | 0/13 |
| Affine atlas + model plane | 0.7917 | 89.3509 | 1400 | 5/13 |
| Deformable atlas + model plane | 0.5000 | 89.9872 | 901 | 1/13 |

| Atlas plane fusion | Selected alpha | MAE delta vs model plane, mm [95% CI] |
|---|---:|---:|
| Centered atlas + model plane | 0.1 | +0.0000 [+0.0000, +0.0000] |
| Affine atlas + model plane | 3 | +0.2917 [+0.0833, +0.5000] |
| Deformable atlas + model plane | 0.3 | +0.0000 [-0.1250, +0.1250] |

## Does registration add anatomical accuracy?

These atlas-alone results use the same three sources per target at every stage. Source selection is based on image similarity, never target labels.

| Atlas stage | Foreground Dice % | A/P Dice % | Mean relative volume error % |
|---|---:|---:|---:|
| centered | 70.927 | 67.617 | -12.84 |
| affine | 78.309 | 74.704 | -6.47 |
| deformable | 79.348 | 76.020 | -7.38 |

Across all 48 targets, 144/144 affine transforms and 79/144 residual deformations passed the image-only engineering guards. Rejected residuals use the affine result; rejected affines use centered alignment. Thus the deformable prior is a guarded mixture of registration stages. These guards are not proof of anatomical registration correctness.

Mean image NMI (centered / affine / final): 0.0929 / 0.1409 / 0.1626. Image similarity and label accuracy must be assessed separately: matching surrounding tissue does not guarantee the correct hippocampal boundary.

![Predetermined calibration examples](../../../experiments/atlas_pilot_20260924/registration_examples.png)

## Room for improvement: oracle diagnostic

The following hypothetical selector uses the reference to choose, case by case, between the frozen network and the already locked correction. It is not deployable, not a fitted gate, and not a strict bound on all possible methods. It measures error complementarity for these candidates.

| Candidate | Oracle foreground Dice gain, pp | Oracle cut MAE reduction, mm |
|---|---:|---:|
| Deformable atlas + network | 0.0000 | 0.0000 |
| Deformable atlas + network + graph | 0.0551 | 0.0000 |
| Centered atlas + model plane | 0.0000 | 0.0000 |
| Affine atlas + model plane | 0.0000 | 0.0417 |
| Deformable atlas + model plane | 0.0000 | 0.0417 |

No reliability gate was fitted to assessment outcomes. Before escalating to a new model, establish that a label-independent registration-quality or disagreement signal can distinguish useful corrections from damage, using out-of-fold backbone outputs and a fresh evaluation set. If registration itself reduces label accuracy, first improve registration on training-only data rather than assigning the prior more weight.

## Verification and limitations

All 48 new CPU hard predictions exactly match the prior caches and native reference masks. All atlas probabilities, source/input hashes, cohort separation, and locked-settings provenance were checked. Eleven new tests passed (36 passed together with related graph/partition tests), including exact small-graph energy enumeration, transform direction and composition, fractional label interpolation, foreground expansion, A/P support preservation, and plane optimization against direct enumeration.

Binary graph cuts optimize the quantized energy at scale 10,000. A conservative floating-energy suboptimality bound per solve is (number of voxels + number of undirected edges)/10,000. This is a numerical solver guarantee, not an anatomical optimality guarantee. Surface distances use native 1-mm voxels; ASSD averages the two directed surface means and HD95 takes the 95th percentile of their concatenated distances.

Bootstrap intervals use 10,000 paired case resamples, conditional on calibration-selected settings; they omit model-refitting uncertainty and are not corrected for multiple comparisons. Shared backbone training and unverified participant linkage limit generalization. There is no comparison to the 2008 paper's Dice numbers across different datasets/protocols.

## Reproduction and evidence

```bash
rtk proxy .venv/bin/python evaluation/atlas_registration.py --subset train --workers 2
rtk proxy .venv/bin/python evaluation/atlas_model_cache.py --subset train
rtk proxy .venv/bin/python evaluation/atlas_pilot.py --stage calibrate
rtk proxy .venv/bin/python evaluation/atlas_pilot.py --stage assessment
rtk proxy env MPLCONFIGDIR=/tmp/hippo-atlas-mpl .venv/bin/python evaluation/report_atlas_pilot.py
rtk proxy .venv/bin/python -m pytest evaluation/test_atlas_pilot.py -q
```

Requires SimpleITK 2.5.2 in the experiment dependencies folder, the local dataset, canonical checkpoint, and documented existing CPU caches. Code or protocol changes invalidate cached provenance and require a new versioned experiment.

Evidence: [protocol](PROTOCOL.md), [cohort](cohort.json), [implementation notes](IMPLEMENTATION_NOTES.md), [locked settings](locked_settings.json), [calibration cases](calibration_cases.json), [assessment summary](assessment_summary.json), [assessment cases](assessment_cases.json), [registration summary](registration_summary.json), [all-family gate audit](all_family_gate_audit.json), [reporting audit](REPORTING_AUDIT.md), [manifest](manifest.json). Derived arrays and transforms remain in ignored `experiments/atlas_pilot_20260924/`.
