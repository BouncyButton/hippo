# Boundary generalization investigation

## Responsibility and protections
Investigate the causal contributors to hippocampus boundary generalization failure and validate a practical remedy. Sustained cluster diagnostics/training authorized. Preserve ALL MedSAM3 runs/results, originals, useful baselines, augmentation studies, bands/pooled-edge work, and the successful 50-case PCGrad checkpoint. Four eligible remote checkpoint copies were later retired ONLY after byte-identical local archival (see cleanup receipt); no results were lost. Existing dirty/untracked repository work is unrelated and preserved.

## 2026-09-29 initial evidence
- Source of approximately 4k vs 33–40k: docs/thesis/new_constraints/bands/SCIENTIFIC_RATIONALE_AND_FORMULATION.md gives original 52-case baseline FP19512+FN16397=35909 union errors and 3872 AP swaps; total39781. Translation equivariance FP18006+FN15782=33788 plus3531 swaps. Unique voxel totals, not per-case means or pair errors. About690.56 union errors and74.46 swaps/case baseline. These historical counts still need raw-source reconstruction.
- A different boundary definition in the augmentation audit counts both sides of any three-class six-neighbor transition, including AP; cannot substitute for the two-step foreground shell.
- Full-data original-runtime shell audit: augmented Dice31321 shell FP+FN, all-grid31474; AP3630 separately. Edge31608 shell errors. Training counts have208 cases versus52 validation and must be normalized.
- Three-seed augmentation comparison: selected Dice87.8395% to88.6416%, mean2814 fewer wrong voxels across52 cases. Exact-source matching for seeds1/2; seed0 earlier source. Same reused fold, not independent test.
- 50-case PCGrad selects epoch30, Dice85.3531%; no matched control previously. Data count, optimizer updates, warmup update count and LR policy confounded versus10-case history.
- Job676272 and audit676288 completed. No active jobs. stud QoS MaxJobsPU1, MaxSubmitPU2, cpu8.
- Live quota90.76/93.13GiB (2.37 free). Twelve eligible-family selected checkpoints total~0.787GiB, but some are pooled controls and all remain protected pending dependency review. No cleanup required for a bounded pilot; recheck quota at allocation. 50-case PCGrad selected AND resumable checkpoint retained.

## Hypotheses
H1 limited example diversity/augmentation causes part of excess validation boundary error: supported by historical controlled augmentation; quantify in the low-data50 regime.
H2 gradient projection remedies conflict: contradicted at10 cases; 50-case success is uninterpretable without matched summed control.
H3 explicit local constraints cause all gaps: contradicted by gaps in Dice-only; test their incremental effect and interaction with augmentation.
H4 strict face metrics exaggerate apparent global failure: exact face transitions are stricter than voxel classification; quantify with unique voxel, shell-layer and distance metrics.
H5 confidence calibration alone explains hard errors: temperature alone cannot change argmax; separately score BCE and hard geometry.
H6 irreducible annotation/image ambiguity: plausible but NOT established; no repeated labels or independent annotator evidence.

## Next controlled pilot
Four seed0 models on the exact nested50 train/52validation split: Dice, summed Dice+bands+separated-edge, each without or with existing mild_v1. Frozen original source/runtime. Keep LR policy, optimizer,50 updates/epoch,5-epoch warmup,early stopping(min60/patience8/max75),initial seed,checkpoint criterion identical. Constraints keep existing coefficients to isolate augmentation at fixed objective; no claim of optimal calibration. Sum uses original component-gradient summation with projection disabled so historical50 PCGrad has an exact optimizer-method comparator. Audits at15/30/45/60 and selected/final, identity inputs on BOTH splits. Report fixed-update and selected results; LR trajectories may diverge under the shared adaptive policy. Replicate only supported remedies, no blind sweep.

## Submitted pilot676348
Two CPU synthetic tests passed: unique-count decomposition and validity of separated edge rules after affine label transformation. All original source files hash-identical; only a new driver and tests added. Peak estimated new storage under0.8GiB, quota checked before submission. Four arms sequential in one allocation. No old artifacts deleted.
Primary assessment: augmentation must improve validation union-error count and macro Dice, with paired-case estimates; inspect both selected and fixed30/60-epoch values, FP/FN and surface metrics. Run-level replication required before a general remedy claim. Fold0 remains development evidence.

## Count reconstruction and contrary evidence
Original52-case CSV exactly reconstructs35909 union errors and3872 swaps. Union errors present52/52; swaps49/52. Top5 shares15.66% versus35.18%. See COUNT_RECONCILIATION files.
Consolidated convergence report shows10-case mild augmentation can hurt under the75epoch schedule, improves under longer schedules; at600epochs seed2 augmentation Dice85.87% versus79.97% nonaugmented selected. Those600epoch historical runs did NOT early-stop (patience0), so are evidence only, not a policy to repeat. Schedule durations/LR confounded across budgets.
The older self_fit_signal_audit calls local intensity chance results a partial-volume/annotation floor. This is NOT validated: two models share training labels, hard-cache runtime differs from originalCUDA, and no repeat annotations exist. Do not repeat a floor claim.

## Planned next causal check (before seeing pilot endpoints)
If50-case summed training fits and validates well, test whether its advantage over the historical10-case experiment is simply more updates/LR. Repeat each of the original10 cases5times per50-update pass, with exactly the50-case control LR trajectory replayed per pass, same5-pass250-update warmup and early-stop policy. Compare fixed update counts and selected checkpoints on the same52 validation cases. This changes unique training diversity while holding total updates and actual learning rates fixed; it does not isolate which added subjects/anatomical variations matter. Pilot seed0 first, replicate if needed. Do not pretend old75epoch10-case controls had the same budget.

## Interim sum50 signal
At1050updates(pass21) summed-control validationDice85.5949%, exceeding priorPCGrad selected85.3531% at1500updates. Interim, one seed; no superiority claim yet. The GPU2-test gate passed; original runtime/source/data/checkpoint hashes verified in PREFLIGHT. First identity train/validation audit at750updates preserved locally.

## Storage preservation completed
Four exact eligible weights (separated seed1/2 and failed PCGrad seed1/2) archived byte-identically locally, SHA256-verified against completion records and live remote files, then ONLY their remote copies removed.281,649,284 logical bytes archived; roughly0.52GiB charged quota reclaimed. All successful50-case PCGrad(selected+resumable), all three pooled reruns, and all three summed controls hash-checked unchanged before/after. MedSAM3 untouched. Receipt, allowlist, compact case/config records and scientific summary under cleanup/. All sources and other remote metadata remain. Queued diversity job676351 depends afterok676348.

## Fixed1500-update control audit
sum50 atpass30: train/val Dice95.2497/86.0062%; boundary union errors314.22/755.21 percase; correct crossing66.92/41.98%; bandsBCE0.1766/0.5924. Versus historicalPCGrad atsame1500updates(validation85.3531%) this ordinary sum control fits and validates better, although bandsBCE is worse. This demonstrates why BCE confidence and hard geometry must be separated. Selection and LR trajectories still need full paired analysis.

## Completed first pilot arm (sum50)
Selected32, stopped60, validationDice86.2486%, train95.3715%, union88.6480%, shellFP+FN39077. HistoricalPCGrad50 selected30 validation85.3531%, shell40735. Sum improves39/52 caseDice, reduces38/52 shell-error counts; net1658 fewer shell errors. Paired descriptive case interval for Dice change[+0.5605,+1.2455]pp, conditional on these selected models/reusedfold. Sum addsFP while reducing moreFN, and worsens bandsBCE0.5450->0.6116. Exact same data order verified through37epochs; actualLR first differs atpass30 under shared adaptivepolicy. The optimization intervention improves this seed but does not eliminate the train/validation gap. Full report/current metrics: CURRENT_RESULTS.md and RESULTS.json. Other three pilot arms still running.

## Historical augmentation verified from confusion matrices
Selected full208-case models, same52validation cases: seed0 union errors34603->31477(-3126); seed1 35072->32147(-2925); seed2 34909->32368(-2541). AP swaps3846->3629,3614->3636,3583->3927, hence no uniform AP benefit. Original audits retain tinyCUDA drift versus later reevaluation(e.g31477 vs31474), not mixed as one count. Stored HISTORICAL_AUGMENTATION.json. More boundary loss is not required for this demonstrated boundary improvement.

## Additional diagnostic prepared, not submitted yet
reports/boundary_translation_20260929 contains fixed13-view integer-translation inference intervention for the4pilot selected models. Goal: test residual positional sensitivity AFTER augmentation. Hash/runtime matched, noGT in prediction, inversevalid-view averaging, counts corrected/introduced, clipping tracked. Synthetic mapping/coverage normalization and syntax checks passed. It is an inference remedy only, NOT learned-constraint proof. Await available second submission slot. Historical unaugmented13-viewTTA improved Dice0.6837pp, but this does not establish an augmented result.

## Projection magnitude diagnostic
Saved actualPCGrad50 per-update telemetry: medianrelativegradientchange ||projected-sum||/||sum|| is0.368 atpass1,0.211 at5,0.507 at15,0.529 at30,0.825 at45,0.851 at60. Median cosinewithordinarysum falls0.930->0.526(first->last). Tiny scalarauxiliaryweights did not imply a small optimizer perturbation. This describes the testedPCGrad pathway, not proof that everyconflict is harmful. Removingprojection is the controlled intervention; data-onlytestpending. SavedPCGRAD_GRADIENT_CHANGE.json.

## Completed Dice50 control
Selected33/stopped60: train96.0920%, val86.2016%, union88.5550%, shellFP+FN38972. sum50 selectedDicegain+0.0470pp with pairedcaseinterval[-0.1081,+0.2029]pp; shell105MORE errors(+2.019/case,interval[-5.365,+9.347]). Effects mostlyFP+40.40/case,FN-37.81/case; no established overallboundarygain. BandsBCEimproves0.6286->0.6116 despite unchangedhardquality. OrdinaryDice alreadyhas9.89pp trainvalgap; constraints are NOT solecause. Candidateaug arms underway.

## Provenance gate rejected localGTgeometry
Attempted GT-only local geometry(no model inference). Local msd pickle is gzip under.pkl and SHA9ccc983d..., decompressedSHA27204445..., neither equals authoritative remote3311d223.... The local geometry program stopped at the hash assertion before anycase analysis; no geometryoutput exists and no numbersfromthislocaldataareused. Do not relax the hash gate. Added APface/endpoint/labelhash measurements to the still-unsubmitted original-runtime translation audit instead. Originalremote dataset alreadypassed allpilotchecks.

## Fixed1500-update augmentation intervention (interim)
Sum vsSum_aug atpass30: trainDice95.25->91.09%, validation86.01->86.36%; validationunion88.53->89.60%. Shellerrors755.21->679.96/case(3913fewer across52);44/52casesimprove,pairedcaseintervaldelta[-95.65,-55.00]errors/case. ValbandsBCE0.5924->0.4672 andASSD0.4838->0.4395. Trainingfitworsenswhileheld-outouterboundaryimproves, consistentwithregularizationratherthanmoretrainingmemorization. MacroDiceintervalcrosseszero atthisfixedepochbecauseAPbehavior differs; donotoverstateglobalclassimprovement. Final/selectedandDice_augcontrolpending.

## Completed sum_aug50
Selected42/stopped60: trainDice93.3097%,validation87.4341%,union90.0089%,shell34687,cross46.087%,bandsBCE0.4735. Versussum0:macro+1.1854pp[0.6790,1.7136],union+1.3609pp,4390fewer shellerrors(11.234%).49/52boundarycounts andASSDimprove;38/52macroDiceimprove. BothFP(-54.21/case)andFN(-39.48/case)fall; bandsBCEimprovesall52. APswapchange+5.269/case withintervalcrossingzero. This isa substantialboundaryaugmentationeffect withinthismatched50-casepolicy, not an APsolution. TrainvalDicegap9.123->5.876pp. RemainingDice_augpilotrunning (~12 atlastcheck);seed1pairedreplicationconditionalonitsoutcome.

## Fullpilot676348 COMPLETE
All4armsstop60. Dice_augselected38:train93.0287%,val87.3730%,union89.9662%,shell34629,cross46.262%,bandsBCE0.4904. VersusDiceval+1.1714pp[0.6506,1.7263],shell4343fewer(-11.144%),48/52boundarycases andASSDimprove,41/52macroDiceimprove. BothFPandFNfall. Sum_augvsDice_aug+0.0611pp[−0.1711,+0.2929],58MOREshellerrors(intervalcrosseszero). Thusaugmentationbenefitreplicates acrossbothobjectivearmswithinseed0; no convincingextra-boundary-lossbenefit. Conditional seed1Dice/Dice_aug replicationgatePASSED;GATE.jsonrecordedbeforeseed1training. Source/runfilesstagednewroot,butnotyetsubmitteddue2slotlimit.
Secondeligiblearchivalcomplete:seed0separatedandfailed10casePCGrad weights140824642bytesverifiedlocal,onlyremotecopiesremoved. All6eligibleweightsnowlocalarchives; originals/summed/pooled/50casePCGradstateandALLMedSAM3unchanged. Original50casePCGradoldlauncherrequiresrestoringold10casePCGraddependencybeforefuturestartup;currentdriversdonot. Receiptsundercleanup_second.
Diversity676351started automatically afterpilot. Translationaudit submittednext (recordIDfromtool).

## Fixed1500-update diversity intervention
Atpass30,10unique repeated5x versus50unique, exactlymatchedactualLR/warmup/updates: trainingDice98.7267% vs95.2497%; validation80.5830% vs86.0062%. Fiftycasesremove15059shellerrors(54330->39271),all52casesimproveboundarycountandASSD;Diceimproves51/52. BothFP/FNdecrease. Supportsdiversity/repeatedexposureeffectbeyondtrainingduration, oneinitializationandnestedsubset. Final/selectedstillpending.

## Replication submitted
Job676401 submitted for matchedseed1Dice/Dice_aug. Quota91.31/93.13GiB,1.82free. Diversity676351completed;translation676386running. No additionalcleanuprequired; allMedSAM3preserved.

## Diversity completed and matching verified
Selected10repeatspass19:train97.263%,val81.031%,shell54144. Selected50uniquepass32:train95.371%,val86.249%,shell39077. Fiftycasesgain5.217pp andremove15067shellerrors(27.83%). Atfixedpass60gain5.759pp/remove15588. All60passesmatchactualLR,warmup,updates;eachoriginal10appearsexactly5times/pass;0AMPskips. See boundary_diversity_20260929/RESULTS.json andREPORT.md.

## Translation audit completed; replication running
Translation676386 completed all4selectedmodels. Fixed13views improvevalidationDice:Dice+0.4876pp,sum+0.5273pp,Dice_aug+0.3454pp,sum_aug+0.3588pp;remove1631,1655,632,634shellerrors,respectively. All50train+52valinputs/GTunclippedinallviews. AugmentedtrainingDice worsens~0.9–1.0pp,whilevalidationimproves;notuniformrecovery. AuthoritativevalidationGTouterfaces139916 vsAP4108(ratio34.06),outerendpoint155343 vsAP8041. Rawerrorsandfacesdifferunits.
Seed1training676401 RUNNING. Fixed13viewseed1replication676407queued afterok676401;byte-identicalaudit.py,armsDice_aug/Diceonly,nonewtuning.

## Replication startup guard and unchanged retry
Job676401 completedseed1Dice(best38/stop60,val86.0414%,shell39058),thenstoppedBEFOREaugmentedtrainingbecausecachedquota1.06GiB<1.2guard. Refreshedquota91.59/93.13GiB=1.54free;oldtemporarycheckpointalreadyretiredbycompletionpolicy. Noadditionalfilesdeleted. OriginalPREFLIGHTandfailedquotarecordpreservedremotely;sourceunchanged. Retry676417skipscompletedDiceandstartsDice_aug. Impossibleafterokjob676407cancelledandqueuedagainagainst676417.

## Augmentation replication confirmed
Seed1Dicebest38vsaugbest46,bothstop60. ValDice86.0414->87.4333(+1.3919pp);shell39058->34046(-5012,12.83%);50/52caseslowererrors. FP/FNbothdecrease. Seed0+1.1714pp/-4343(11.14%). Equal-seedmean+1.2816pp/-4677.5shellerrorsper52caseeval. Caseordersidenticalall60/seed;0skips. LRfirstdivergespass30seed0,pass36seed1. Bothfixed30/60benefit. APswapsincrease371/439(uncertaintyspanszero),soNOTAPpartitionremedy. REPORT.mdstarted;waiting676418TTAreplicationcompletionandfinalverify.

## Final completion
Allsevenmodelsand6translationauditscompleted; finalverificationPASSED(source/runtime/data/checkpointbindings,caseidentity,earlystopping,21000updates/noskips,6intactarchivehashes). Seed1augTTA87.4337->87.8747(+.4410pp),756shellerrorsremoved; noinput/GTclipping. Allclusterjobsfinished; quota91.86/93.13GiB. FinalREPORT.md,INTERPRETATION.md,VALIDATION.json,causal_summary.png/pdf ready. AllMedSAM3untouched.
