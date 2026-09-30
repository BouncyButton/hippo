# Interpretation: what should inform an LTN cut constraint?

The tested measurements contain repeatable information about the annotated boundary, but no single geometric event is a universal indicator. The best-supported direction is a combination of local area/contour transition and MRI appearance above the hippocampus, with returning profiles as optional evidence.

## Specific observations

1. **A small separate profile disappearing posteriorly is real but uncommon.** It occurs at the fitted cut in 4/208 training and 2/52 validation volumes. The same counts hold at the actual last-anterior plane. Training cases 145 and 160 show the small separate profile at y=29 followed by its disappearance at y=28, resembling validation 185 and 327. This geometric event also occurs at distant locations in 7 training and 3 validation volumes. It is neither sufficiently common nor sufficiently specific to impose on every hippocampus.

2. **Connected return geometry is more common but depends on its definition.** The sagittal anterior-bridge proxy with two-voxel minimum tissue runs occurs at the cut in 27/208 training (13.0%) and 6/52 validation (11.5%) volumes; within two slices, 52/208 and 16/52. It also occurs farther away in 23/208 and 5/52. Allowing one-voxel runs increases coverage to 34/208 and 9/52 at the cut. A missing detection can mean an oblique/curved/thin fold, not an absent uncus. Cases 023 and 042 illustrate local contour notches and small returning configurations; the displayed planes alone do not independently establish anatomical identity.

3. **Axial return geometry is nonspecific.** With two-voxel runs it is present at the cut in 49/208 and 19/52, but somewhere farther away in 130/208 and 33/52. Case 001 illustrates an axial candidate despite a largely continuous coronal outline. Axial geometry may support a candidate established by other information, but this particular rule is a poor standalone locator.

4. **Local broadening is the strongest individual measured cue.** The area increase from the cut plane y to the next anterior plane y+1 ranks the cut above distant candidates with mean within-volume AUC 0.867 training / 0.904 validation. Validation AUC against the immediate neighbours is lower, 0.745: useful local information, not precise identification in every case. A training-fitted threshold detects 91.3% of training and 96.2% of validation cuts, but also flags an average 29.1% of distant validation slices. Broadening cannot be an unconditional axiom.

5. **MRI context adds information.** Intensity asymmetry between the two sides of the band immediately above the mask has training/validation distant AUC 0.860/0.857 and validation local AUC 0.673. Superior boundary changes and darkening of the central superior band also carry weaker signals. These describe image patterns; they do not identify CSF, the crus cerebri, or a named recess. Side symmetry is used because crop laterality is not independently established.

6. **A generic “foldedness” value can fail badly.** In validation 033, the original full ranker chooses a fragmented posterior-tip slice at y=6 instead of y=26. Its near-constant closing-based solidity proxy takes values beyond the training range and overwhelms other evidence. That is a concrete feature-engineering failure, not a finding about normal anatomy. The full model selected by training OOF performance scores 0.933 mm training OOF MAE but 1.250 mm validation MAE, versus 1.159/1.173 mm for position alone.

7. **Combining cues remains promising after diagnosing that failure.** Removing only the three solidity-proxy terms gives 0.938 mm training OOF and 0.865 mm validation MAE; 80.8% of validation cuts are within 1 mm and the maximum error is 4 mm. This is a post-hoc sensitivity analysis prompted by validation, not a newly confirmed generalisation gain. No further feature or hyperparameter search was performed after this check. Even this result uses reference foreground and predicts the annotation rather than an independently verified landmark.

## What to carry forward

Carry forward an uncertain candidate-boundary score supported by:

- local cross-sectional area and superior-contour transitions across several slices;
- raw MRI intensity/edge/context changes near the superior surface;
- sagittal/axial return or disappearing-profile evidence when present;
- a defeasible relative-position prior, tested separately from the image/shape evidence.

Do not interpret the logistic score as a calibrated probability, require the rare morphology to exist, or identify a named structure from mask topology alone. Confidence should include agreement between cues and whether feature values are outside their training support; simply learning a freely adjustable confidence output could let the model avoid the constraint.

An LTN can use a validated boundary estimate to encourage consistent anterior/posterior ordering outside an uncertain boundary band. The band width should be calibrated on training data. A stronger anatomical claim requires independent landmark annotations and evaluation on predicted foreground. Testing an LTN improvement also requires a fresh evaluation set: the present training and validation volumes have both now been explored.

## Review scope

All 260 volumes were measured automatically; 13 selected multi-plane image sheets were generated. During this audit, the summary plot and sheets for 001, 011, 023, 033, 042, 145, 160 and 236 were inspected directly. Earlier work additionally reviewed all 52 validation cases around their boundaries. This is not an expert anatomical adjudication of all 260 volumes.

See [the numerical report](README.md) and [all case results](cases.md). No dataset labels or segmentation-network training code was changed.
