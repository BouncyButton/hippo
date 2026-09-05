# Outer one-cut pilot protocol — 2026-09-02

This protocol advances the 52-case exploratory audit to one matched short
training pilot. It does not authorize A/P-interface supervision or a full
multi-seed campaign.

## Fixed sequence

1. Fresh fold-0, seed-0, five-epoch Dice-only run from scratch.
2. Frozen-checkpoint one-cut gradient calibration on 32 deterministic training
   cases, targeting 10% of Dice logit-gradient RMS with the canonical 50%
   high-gradient cap.
3. Fresh fold-0, seed-0, five-epoch outer one-cut run from scratch, with the
   exact calibrated weight and five-epoch linear warmup.

All steps use MSD, the saved final fold split, 64-cubed crops, AMP, AdamW,
1e-4 learning rate, and the same A100 GPU partition. The one-cut geometry is
radius 3 mm, sample interval 0.5 mm, cut tolerance ±1 mm, margin 0,
temperature 1, and at most 4096 deterministic faces per patient.

## Screening decision

The first decision compares the one-cut pilot to the newly matched Dice-only
arm. The older band pilot remains a secondary reference, not the provenance-
matched control.

Advance to a longer matched run only if:

- the one-cut run completes with finite loss and no skipped-validity anomaly;
- best hard validation Dice is not worse than the new control by more than
  0.001 absolute;
- decoded 1-mm union surface Dice improves on the new control; and
- the direction is not explained by a material FP/FN trade alone: union Dice,
  ASSD, and HD95 are reported together with FP and FN counts.

The surface gate is primary because the intervention targets boundary
geometry. A five-epoch pass is evidence to continue, not a final thesis claim.
