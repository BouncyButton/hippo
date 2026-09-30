# Harmonic A/P partition: exploratory protocol

Written before running this pilot. This is an offline feasibility study, not
an added training loss or a new inference default.

## Question and hypothesis

Can an image-weighted harmonic field place the A/P interface better than the
existing model and better than the same field with uniform connections?
The mechanism is weighted diffusion between trusted cores, not a separation
axiom or a guaranteed topology constraint.

On the six-neighbor graph of foreground voxels, solve

`sum_j w_ij (u_i - u_j) = 0`

on unseeded nodes, with anterior seeds fixed at 0 and posterior seeds at 1.
Use `w_ij = [1e-4 + (1-1e-4) exp(-beta ((I_i-I_j)/s)^2)] / h_ij^2`,
where `s` is the within-support intensity IQR, falling back to its standard
deviation and then 1 for constant images. Positive weights retain every
face connection. The graph follows the voxel volume; it is not HippUnfold.

Two seed recipes, fixed in advance:

- `ends`: seeds in the outer 20% of the foreground's coronal extent, with
  low-Y posterior and high-Y anterior in verified RAS coordinates. This uses
  no model A/P cut.
- `band`: seeds at least approximately 4 mm away on either side of the
  prediction's own best coronal cut. This gives a local repair model but may
  lock in a misplaced cut.

Both restrict seeds to a one-face-erosion interior of the support. No reference
label defines a seed in the development evaluation. Components lacking both
seed types preserve their original predictions; coverage and rejected seed
counts are reported. Predicted foreground remains unchanged, including islands.

## Training-only screen

Use the first 32 fold-0 training cases sorted by SHA256(case identifier), a
deterministic subset chosen without looking at their images or outcomes.
Create synthetic predictions on each reference foreground by moving its
best-fit annotated coronal cut by -2, 0, and +2 slices. This tests recovery from
displacement and preservation of correct cuts on clean support. It is an
optimistic surrogate: it does not reproduce actual network errors, and its
cores may encode information inherited from the reference cut.

Evaluate both seed recipes and beta in {0, 1, 10}. Choose the image-weighted
candidate (beta > 0) minimizing mean best-fit-cut absolute error; break ties by
larger macro A/P Dice and then fixed candidate name. Also report the uniform
candidate for each seed recipe. Freeze the selected image-weighted candidate
before opening the development caches. Failure to beat uniform weighting on
this screen is itself a negative finding, not a reason to expand the grid.

## Existing development-cache check

Use the same 52 fold-0 cases for the early-stopped unaugmented and augmented
CPU model outputs, with native MRI and exact geometry checks. Do not use a
reserved holdout or recompute a model. Compare:

1. Original prediction and its existing hard-plane projection.
2. Selected seed recipe with uniform weights (beta=0).
3. The frozen selected image-weighted candidate.
4. The same candidate with intensities deterministically permuted within the
   predicted foreground (negative control for spatial image information).

For each field, report both its raw 0.5-threshold partition and its best-fitting
coronal plane, because the annotation protocol is predominantly planar.
The primary endpoint is raw-field macro A/P Dice versus the original and
versus uniform weighting; report cut MAE, A/P swaps, helped/harmed cases,
paired case-bootstrap intervals, solver residual, core errors, coverage, and
foreground preservation. Plane results are secondary, not an opportunity to
switch endpoints after seeing results. Seed accuracy uses labels for reporting
only and never filters or repairs a prediction.

Repeated use of this development fold and unavailable participant grouping
preclude independent confirmation or participant-level confidence claims.
The shuffle is a single fixed negative-control realization, not a permutation
significance test. Hard-only caches preclude a confidence-seed comparison.

## Decision

Only consider a learned-affinity pilot if image weights give a consistent
advantage over uniform weights and the original predictions, with tolerable
patient-level harm in both models. Otherwise retain the solver and audit as
research tools, explain which assumption failed, and distinguish rejection of
raw-intensity affinities from rejection of learned semantic affinities.

## Primary references

- Grady, 2006, [Random Walks for Image Segmentation](https://pubmed.ncbi.nlm.nih.gov/17063682/): seeded harmonic segmentation.
- Cerrone, Zeilmann, Hamprecht, 2019, [End-to-End Learned Random Walker for Seeded Image Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Cerrone_End-To-End_Learned_Random_Walker_for_Seeded_Image_Segmentation_CVPR_2019_paper.html): learn graph edge weights through the inference solve.
- Vernaza and Chandraker, CVPR 2017; revised 2018, [Learning random-walk label propagation for weakly-supervised semantic segmentation](https://arxiv.org/abs/1802.00470): joint learning of propagation and semantic prediction.
