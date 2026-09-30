# Locked MedSAM3 volume-level replication

User-selected scope: three data folds × three new optimization seeds × three
training objectives = **27 fine-tuning arms in nine matched comparisons**.
This study measures repeatability; it does not promise to prove the absence of chance.

## Cases and pairing

Use existing Dataset101_MSD folds 0, 1 and 2. Select five training volumes per
fold with a separate deterministic selection seed, 20260929 + fold index, then
hold that support set fixed across optimization seeds **17, 83 and 191**.
All support sets are disjoint. Exclude all ten volumes from the original pilot
from selection and evaluation. Exclude the union of all 15 selected support
volumes from every evaluation fold, including folds in which those volumes would
otherwise have been validation cases. This leaves **46, 52 and 47 evaluation
volumes**, respectively: **145 unique volumes**.

The exact cases, data hashes and per-run settings are in `protocols/study.json`
and the nine per-run JSON files. No evaluation score, label geometry or image
content was used to choose cases. Image headers only size the storage budget;
binary file hashes establish the fixed input versions.

The dataset builder's `subject_id` is just the crop filename. The user has no
known mapping to original patients. We therefore cannot verify independence of
left/right crops or claim patient-level generalization. Existing data have also
been used in this project's development, so this is not a pristine external test.
The released medical adapter's complete pretraining provenance is not verified.
The [MedSAM3 paper](https://arxiv.org/html/2511.19046v1) describes its training
datasets; its reported list alone cannot certify the released checkpoint's
absence of overlap with every study case.

## Training and calibration

Retain the pilot's model, original medical LoRA initialization, frozen backbone,
loss definitions, 200 optimizer updates, four slice presentations per update,
learning-rate schedule and inference settings. Each matched comparison trains
baseline, baseline + bands, and baseline + bands + edge independently from the
same initial adapter, with identical RNG resets and the exact same saved sample
schedule. Both constraints are supervised fine-tuning losses. No inference
refinement, validation-driven early stopping or hyperparameter search is used.

The gradient calibration procedure is unchanged, using only each fold's five
training volumes. Its RNG seed is fixed at **20260929** across optimization
seeds, keeping coefficients fixed within a fold while measuring training-seed
variation. Calibration weights are checked for agreement across seeds. No
coefficient is copied from validation results or adjusted during this study.

All three arms' predictions are saved before evaluation labels are opened within
each matched comparison. The study's analysis and training protocol are locked
before any replication result is observed. Complete all planned runs regardless
of intermediate outcomes; do not stop early because an effect looks positive.

## Analysis fixed before results

Primary contrast: **bands+edge minus baseline** in whole-hippocampus 3D union
Dice. Secondary exploratory contrast: bands minus baseline. The edge increment
(bands+edge minus bands) is descriptive.

For each contrast, subtract paired scores on the same case and seed. The primary
estimate gives equal weight to folds, equal weight to seeds within folds and
equal weight to cases within each fold. Report all nine run means, per-fold
effects, positive/negative run counts, mean effect, and seed variation within
each fold. Differences between folds combine support-set and evaluation-cohort
variation; they do not isolate the effect of the training subset alone.

Use 20,000 deterministic crossed fold/seed bootstrap draws (seed 20260930):
resample fold blocks, draw one seed-index vector shared across all sampled folds,
and draw case indices within each sampled fold shared across its selected seeds
and paired arms. Report the 2.5th and 97.5th percentiles as an **exploratory**
interval. Never count repeated case/seed predictions as independent patients.

Also report an exact two-sided fold-block sign-flip sensitivity statistic after
averaging seeds and cases. With only three fold blocks its minimum possible
p-value is **0.25**. Thus this smaller study cannot deliver p<0.05 evidence with
that conservative test, even if all folds improve. Bootstrap tails from three
folds are approximate; an interval excluding zero does not override that
limitation. Results can establish consistency under the tested conditions,
not definitive statistical significance or broad patient-level generalization.

## Execution, retention and checks

One GPU job runs all nine matched comparisons sequentially with a 20-hour limit;
expected runtime is roughly 10–13 hours based on the pilot. The base checkpoint
is broadcast once to node-local storage; temporary model caches stay there.
Every final LoRA adapter is retained. Save all masks, probability volumes,
per-slice records, loss traces, inventories, metrics and hashes, with one plot
per comparison. Estimated logical artifact allowance: **2.616 GiB**. BeeGFS
buddy mirroring charges two copies, giving **5.232 GiB of quota**, plus
a **1 GiB minimum free-quota reserve** (6.232 GiB required before launch). Check quota against the entire remaining
artifact allowance before each comparison; abort rather than exhaust quota.

Every comparison must pass the independent audit: dataset and prediction hashes,
native grids, saved-mask Dice, all 600 component-loss records, matching schedules,
learning rates and initial native losses, unchanged frozen parameters, distinct
updated adapters, and finite calibration gradients. The aggregate requires all
nine audited reports and verifies result hashes; missing or failed runs cannot
silently disappear from the analysis.

Restarting skips only fully completed, hash-verified comparisons. A partial run
causes a stop: preserve it in a clearly labelled failed-attempt directory before
rerunning the same locked configuration. Report failures and reasons; do not
change scientific settings or exclude a difficult seed to obtain a better score.

After all three seeds in a fold pass independent audits, write an atomic fold-completion marker. The user chose manual status checks; no scheduled monitoring or wake-up automation will be created.
