# What the controlled experiments can establish

This investigation treats an error as a distinct incorrectly classified voxel unless a metric explicitly says “face” or “pair.” Foreground/background errors and anterior/posterior swaps are separate outcomes. A/P swaps require both labels to be foreground; “crossing” in the edge objective means foreground/background, not A/P.

## Error counts are real, but their denominators differ

The original 52-case CSV reconstructs 35,909 foreground/background errors and 3,872 A/P swaps, totaling 39,781 incorrectly classified voxels. These are pooled cohort totals, not per-case counts or duplicate edge incidences. Their means are 690.56 and 74.46 per case. The ratio alone does not measure which decision is intrinsically harder: the outer surface and A/P interface have different sizes, and a misplaced surface generates a layer of wrong voxels.

The original-runtime geometry audit on the 52 validation masks counts 139,916 outer foreground/background faces and 4,108 A/P faces: the outer surface has 34.06 times as many faces. Unique endpoint counts are 155,343 and 8,041. This supplies denominator context for the much larger outer-boundary error total. Faces, endpoint voxels and all-grid classification errors remain different units; dividing the original error totals by these face counts would not produce a valid voxel error rate.

Exact face-transition correctness asks for both adjacent endpoints to have precisely the right foreground state. It can therefore be poor even when a predicted surface is close. In the selected 50-case Dice control, mean exact crossing correctness is 42.02%, while the voxel-based surface Dice at one model-grid voxel is 94.18% and mean symmetric surface distance is 0.483 grid voxels. This explains part of the apparent metric discrepancy; it does not erase the actual 38,972 unique shell errors. Model-grid distances are not physical millimeters.

## Generalization failure is not unique to the semantic losses

The 50-case Dice-only control reaches 96.09% training Dice and 86.20% validation Dice. Adding the existing bands and separated-edge losses produces 95.37% and 86.25%, respectively, with 105 more validation shell errors. Its confidence-sensitive bands BCE improves slightly even though hard boundary quality does not clearly improve. A train/validation gap therefore exists without those losses, and better BCE is not sufficient evidence of better hard anatomy.

These local constraints supervise agreement or transitions at the supplied training labels. Satisfying them on those images does not force the model to locate the correct surface on a new image or to give the same aligned prediction after an input translation. The near-perfect training fit and poorer validation geometry are therefore compatible with correctly implemented constraints. The controlled diversity and augmentation interventions test ways of reducing this gap; perfect training truth alone is not such a test.

The auxiliary coefficients were held at their existing values to isolate the interventions. The absence of an incremental benefit applies to this tested configuration; it does not establish that every possible loss weight or boundary formulation is ineffective.

Symmetric PCGrad also fails to solve this tested setup. The matched ordinary gradient sum reaches 86.25% validation Dice versus 85.35% for historical 50-case PCGrad and removes 1,658 shell errors. Their data order, warmup and actual learning rates match through pass 29, when ordinary sum already performs better. Shared adaptive LR policies diverge afterward, so the selected-checkpoint difference includes that downstream response. The measured projection can substantially change the total gradient despite small auxiliary scalar weights; the median relative gradient change rises from 0.368 in the first pass to 0.851 in the last. This rejects the tested projection remedy, not every possible conflict-handling method.

## Training diversity matters beyond the number of optimizer updates

The original 10 training cases were each repeated five times per 50-update pass, then compared with 50 distinct cases. All 60 passes used identical actual learning rates, warmup and update counts; no AMP updates were skipped. Selected validation Dice rises from 81.03% to 86.25%, and shell errors fall from 54,144 to 39,077 (15,067 fewer, or 27.83%). Every validation case improves in shell errors and surface distance. The effect also holds at matched passes 15, 30, 45 and 60; at pass60 the difference is +5.76 Dice points and 15,588 fewer shell errors.

On the same original ten training cases, the 50-case model has lower selected training Dice and more training boundary errors in all ten, while its held-out boundaries improve. This supports a generalization benefit from greater case diversity, rather than simply better training fit or more optimizer steps. At fixed updates, diversity and exposure per individual case necessarily change together. This is one initialization and one nested subset, so it does not identify which additional anatomy matters or establish a universal sample-size curve.

## Augmentation changes held-out behavior

In the matched seed0 factorial pilot, mild_v1 augmentation improves Dice-only validation Dice from 86.20% to 87.37% and reduces shell errors from 38,972 to 34,629 (4,343, or 11.14%). It improves shell counts in 48/52 cases, reduces both FP and FN, and lowers surface distance. The same intervention on the summed objective improves validation Dice from 86.25% to 87.43% and removes 4,390 shell errors (11.23%; 49/52 cases improve). Adding the auxiliary losses to augmented Dice produces only +0.061 Dice points and 58 more shell errors, with paired-case uncertainty spanning either direction.

This supports augmentation as a useful treatment in the tested training pipeline. It does not isolate which mild_v1 component matters. The shared adaptive LR policy responds to each arm's validation trajectory, so this is the effect of adding augmentation under that policy, not a claim that realized learning rates remained identical.

The seed1 Dice-only replication confirms the treatment: validation Dice rises from 86.0414% to 87.4333%, and shell errors fall from 39,058 to 34,046 (5,012 fewer, or 12.83%; 50/52 cases improve). Across the two seeds, the equal-seed mean gain is 1.2816 Dice points and 4,677.5 fewer shell errors per 52-case evaluation. Both FP and FN decrease in both seeds. Fixed-pass30 and pass60 comparisons have the same direction. All 60 case orders match within each seed, with zero skipped AMP updates. Actual LR first diverges at pass30 in seed0 and pass36 in seed1 under the shared adaptive policy.

A/P swaps increase by 371 and 439 voxels in the two augmentation comparisons, with paired-case intervals spanning zero. Macro Dice improves in 41/52 and 31/52 cases, respectively; the outer-boundary improvement is more widespread. This remedy should not be described as a demonstrated improvement to the A/P partition itself.

The existing full-data study supports the direction across three seeds: union errors decrease by 3,126, 2,925 and 2,541. The historical seed1/2 source is exactly matched; seed0 used an earlier source snapshot, so it is weaker supporting evidence. A/P swaps do not improve uniformly. Those historical full-grid counts must not be substituted for the current two-step-shell totals. All studies reuse fold0; none is an untouched final test.

## Residual coordinate sensitivity is directly observable

The frozen seed0 selected models were evaluated under identity and twelve exact integer translations (±1 and ±2 along each axis), with probabilities inverse-mapped and averaged over valid views. All 102 train/validation images retained their nonzero input and foreground support in every view. No label entered prediction or view selection. Repeat identity inference differed from the earlier audit by at most two shell voxels per case, much smaller than the measured cohort effects.

On validation, mean boundary disagreement between a translated view and identity falls from 303 to 168 voxels/case for Dice after augmentation (about 44% lower), and from 296 to 173 for the summed objective. This demonstrates reduced positional sensitivity under the tested augmentation pipeline, without identifying a specific architectural mechanism or augmentation component.

The fixed 13-view ensemble improves seed0 validation Dice by 0.488 points without augmentation and by another 0.345 points after augmentation. It removes 1,631 and 632 shell errors, respectively. For augmented Dice, it corrects 2,732 previously wrong shell voxels while introducing 2,100 errors elsewhere; 39/52 cases improve in shell counts. The summed models show comparable effects. Augmented training Dice becomes worse under the same policy, so this is not uniform recovery of every image or voxel.

The second-seed check confirms the remaining gain: augmented Dice rises from 87.4337% to 87.8747% (+0.4410 points), removing another 756 shell errors (2,765 corrected, 2,009 introduced). The unaugmented seed1 model gains 0.6412 points and removes 1,944 errors. All inputs and labels remain unclipped, and maximum repeated-identity drift is two shell voxels per case. The smaller remaining gain is a replicated inference add-on in this development cohort and costs 13 model forward passes. It is not evidence that an additional semantic constraint was learned. The main training remedy remains augmentation; this ensemble should be considered separately from training changes.

## Limits on causal language

The controlled interventions can establish contributions from the tested training diversity, augmentation policy, projection pathway and input-coordinate sensitivity. They cannot prove that all remaining errors arise from any single cause. Partial-volume effects, limited contrast and annotation ambiguity remain possibilities. No repeated annotations or independent annotator study establishes an irreducible error floor, and near-chance local-intensity probes do not establish one.

Case bootstrap intervals condition on fitted and selected models and this reused validation cohort. They omit checkpoint-selection bias, training-seed variability and possible subject dependence. Seed effects must be reported separately; repeated epochs and repeated use of the same 52 cases are not new independent replications.

## Preservation

All MedSAM3 run artifacts are protected and were untouched. Six specifically eligible old checkpoint files were copied locally and SHA256-verified before only their remote copies were retired. Their scientific records remain preserved. Successful 50-case PCGrad selected and resumable checkpoints, pooled controls and useful original studies remain available. See the two cleanup directories for explicit allowlists, hashes and receipts.
