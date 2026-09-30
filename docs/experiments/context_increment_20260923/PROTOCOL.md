# Incremental context probe (frozen before extraction)

Question: does local MRI context or a frozen decoder representation improve
anterior/posterior cut localization beyond the same network's predictions,
coordinates, and predicted foreground geometry?

## Fixed experiment

- Dataset101_MSD fold 0: 208 training cases, 52 development cases. Both local
  seed-0 early-stopped SwinUNETR checkpoints (unaugmented and augmented).
- Recompute logits on CPU with the existing 64-cubed, no-resize preprocessing.
  Verify development hard labels against the previous CPU audit cache.
- Candidates are all interior coronal cuts through the largest predicted
  foreground component. A is label 1 on the high-y side. Reference labels only
  supply supervised targets and metrics, never features or inference support.
- Base features: coordinates, distance from the hard prediction's fitted cut,
  conditional A/P log-likelihood of each cut, local probability statistics,
  predicted area, extent, and centroid. Retain the raw prediction and its fitted
  plane as nonlearned controls.
- MRI arm adds fixed 3-D intensity, Gaussian means (sigma 1 and 2 voxels), local
  standard deviations (sigma 1 and 2), and gradient magnitude. Pool these using
  the existing slice/adjacent-slice feature extractor over foreground, superior
  exterior band, and surrounding context. This is a finite descriptor test, not
  a test of every possible contextual feature.
- Decoder arm adds the existing frozen decoder1 feature cache (384 features).
- MRI permutation control uses the same number of columns but independently
  permutes candidate rows of the added features within each case, with a fixed
  identifier-derived seed. Permutation is applied during fitting and evaluation.
- Each arm uses standardized, L2-regularized logistic cut ranking. Every subject
  contributes equal positive and negative total weight. C in {0.01, 0.1, 1} is
  selected by fourfold subject CV (seed 20260923) on training cases only, minimizing
  cut MAE; ties favor stronger regularization. No synthetic target jitter.
- Save selection before opening development feature caches for evaluation.
  Report each selected ranker, plus a deployment gate: use the original hard
  prediction if the ranker's training CV cut MAE fails to beat the original cut.
- Primary comparison: MRI+base versus base on development cut MAE. Secondary:
  decoder+base versus base, macro A/P Dice at fixed predicted foreground, swap
  counts, fraction improved/worsened, and original raw prediction comparisons.
  Report paired bootstrap 95% intervals (10,000 case resamples, seed 20260923).
- A useful signal requires improvement over base and original predictions,
  directional replication across both checkpoints, and a primary paired MAE
  interval excluding zero. This is a screening gate, not a confirmatory claim.

## Limits established in advance

The segmentation networks were trained on all 208 training cases, so their
training predictions and decoder features are in-sample. Inner folds hold out
the probe, not the backbone. These development subjects have also been used in
earlier research and checkpoint selection. Results cannot establish unbiased
generalization. A positive result requires out-of-fold backbone features and a
fresh fold/seed before joint graph training. A negative result only rejects
these fixed small probes. No default training or inference path will change.
