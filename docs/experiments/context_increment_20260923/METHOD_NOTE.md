# Case versus participant independence

Terminology clarification made during extraction, after the unaugmented model's
fixed analysis, without changing the protocol's splits, features, hyperparameter
grid, selection rule, or endpoints:

The protocol and implementation use “subject” in places, following the existing
probe's vocabulary. The actual split unit is a **hippocampal case ID**, not a
verified unique participant. `create_msd_dataset.py` constructs `subject_id`
directly from each image filename; this is not a participant linkage table.

The [MSD challenge paper](https://www.nature.com/articles/s41467-022-30695-9)
describes the hippocampus source as 195 MRI scans from 90 healthy adults and 105
adults with a non-affective psychotic disorder. Consequently, 260 released
training cases must not simply be called 260 independent participants. We have
not established a participant mapping or established that cross-split leakage
actually occurs. We do not infer pairing from adjacent case numbers.

Interpret this analysis as case-level cross-validation and case-level paired
bootstrap intervals. If multiple cases belong to one participant, those intervals
do not account for within-participant dependence. A confirmatory experiment needs
verified participant grouping for backbone folds, probe folds, and resampling.
The original protocol is retained verbatim so its saved checksum stays valid.
