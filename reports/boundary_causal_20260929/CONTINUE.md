# Investigation complete

All authorized work in this investigation is complete. No cluster jobs remain queued or running. Do not resubmit the historical job plans in the chronological research log.

Read REPORT.md for the final synthesis, INTERPRETATION.md for causal reasoning and limits, and VALIDATION.json for the passed final audit. FINAL_CLUSTER_STATUS.json records completed jobs and the single stale-quota retry. Seven models trained with early stopping (21,000 updates, zero AMP skips); six frozen-model13-view audits completed on50train/52validation cases.

## Findings
- More diverse training:50unique versus10repeated underidenticalactualLR/updates removes27.83% selectedvalidation shellerrors; all52casesimprove. One seed/nestedsubset.
- Dice+mild_v1:11.14% and12.83% fewer shellerrors acrossseeds0/1; Dice+1.1714/+1.3919pp. BothFP/FNdecrease. A/Pswapsdonotimprove.
- Additional13viewaverage afteraugmentation removes632/756shellerrors, Dice+0.3454/+0.4410pp.13forwardpasses; allinputs/GTunclipped.
- Existingextra losses show no clear increment atfixedweights. TestedPCGrad underperformsordinarysum.
- Allresultsreusedfold0development; noneproveirreducibleannotationfloororindependentclinicaltestaccuracy.

## Artifact locations
- boundary_causal_20260929: four-arm pilot, analysis, count reconciliation, source, plots, final report, verification, archived checkpoints.
- boundary_diversity_20260929: matched10repeated/50unique intervention and analysis.
- boundary_replication_20260929: two-seed comparison; seed1metadataandbothattemptrecords.
- boundary_translation_20260929: four seed0selected models' fixed13viewaudit.
- boundary_translation_replication_20260929: two seed1selected models' fixed13viewaudit.
Remote experiment roots use the same names with suffix _01 under /mnt/beegfsstudents/home/3160552. Selected/final new model weights remain there. Source snapshotsandcheckpointbindingsareverified.

## Protections and restoration
ALLMedSAM3runs/resultsuntouched. Sixeligibleoldseparated/failed10casePCGradcheckpointsarchivedbyte-identically undercleanup/checkpoints andcleanup_second/checkpoints; sixhashesreverifiedatcompletion. Their remote copies alone were retired; scientificrecords remain. Successful50casePCGrad selectedANDresumable remain remote, as do pooled reruns and sum controls. Noadditionalcleanupwasrequiredafterthe second archival.
The old50casePCGradlauncher checks the old10casePCGradseed0checkpoint atstartup; restore that exact localarchivedfile before reusingTHAToldlauncher. Ournewdriversdidnotneed it. See allowlistsandSHAreceipts.
LocalMSDpicklehashdiffersfromauthoritativeclusterdata; doNOTrelaxthatgate or runcomparativeinferencewithlocalolderTorch. Allmodelinferencehere used originalTorch2.12.1+cu130/MONAI1.6.0/CUDAAMP. LocalPythononly forstatistics/syntheticchecks/plots.

Allshellcommandsstartrtk. Existingunrelateddirtyrepo workisuntouched. No commits, automation, goal, or subagents were created.
