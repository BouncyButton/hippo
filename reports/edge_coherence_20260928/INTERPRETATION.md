# Edge consistency improves boundary contrast but not uniform voxel coherence

Job 673837 completed successfully in 4 minutes 36 seconds. This audit evaluated
the original 27 selected checkpoints: baseline, bands, and bands + edge across
three folds and three seeds, using the 5% data experiments with a 75-epoch cap.
There were 156 distinct validation case files, each evaluated with three seeds.
No model was retrained or reselected.

The answer to the original question is mixed. Bands + edge improves the
probability contrast across the annotated outer boundary consistently. It also
makes neighboring foreground probabilities more similar in the external band.
It does not make the final binary segmentation uniformly more coherent:
agreement just inside the hippocampus deteriorates, and external hard-label
agreement is approximately unchanged relative to baseline.

## Where coherence improved and where it did not

All values below are case means, averaged equally over seeds and folds. The
inner band is hippocampal tissue just inside the GT surface; the outer band
is background just outside it. Same-side pairs share a voxel face and the same
GT foreground/background label. Crossing pairs straddle the GT surface.

| Measurement | Baseline | Bands | Bands + edge |
|---|---:|---:|---:|
| Outer same-side hard-label disagreement | 17.618% | 18.111% | 17.694% |
| Inner same-side hard-label disagreement | 4.733% | 6.207% | 6.743% |
| Correctly oriented hard transitions at true boundary faces | 18.064% | 23.573% | 25.030% |
| Signed probability contrast MSE across all supported faces | 0.16995 | 0.15949 | 0.15656 |
| Outer same-side probability difference MSE | 0.04898 | 0.04582 | 0.04374 |
| Inner same-side probability difference MSE | 0.01473 | 0.01799 | 0.01883 |

The overall signed probability-contrast error decreases by approximately 7.9%
versus baseline and improves in all nine fold–seed comparisons. Correct true
boundary transitions also improve in all nine comparisons, both against baseline
and against bands alone. This is direct evidence of better signed boundary
contrast, beyond an inference from Dice.

Outer probability disagreement falls in eight of nine comparisons against
baseline. However, after decoding the final labels, outer disagreement changes
by only +0.076 percentage points. Its paired conditional 95% interval is
[-0.160, +0.313] points, so there is no clear average improvement relative to
baseline on this metric. Against bands alone, adding edge reduces outer hard
disagreement by 0.416 points, with interval [-0.485, -0.348], improving eight
of nine comparisons.

Inner hard disagreement increases by 2.010 points versus baseline, with interval
[1.904, 2.123], and worsens in all nine comparisons. It also worsens in eight
of nine comparisons against bands. Inner probability disagreement increases
too. These measurements do not support improved coherence throughout the tissue
near the outer surface.

## Correctness explains why agreement alone is misleading

Both outer endpoints are correctly classified in 44.94% of baseline pairs and
58.54% of bands + edge pairs. Thus the external band is much more correct even
though its hard-label disagreement is nearly unchanged. A baseline can agree
with itself while consistently predicting excess foreground.

Inside, the percentage of pairs with both endpoints correct falls from 90.68%
to 87.03%. This is consistent with the additional missed foreground. The earlier
summary's average whole-volume FN count per 52-case evaluation rises from 9,855
to 13,548, while FP falls from 97,390 to 65,327. FN did not remain constant.

An arithmetic decomposition of the measured contrast loss confirms the tradeoff.
Contributions to the total mean loss change as follows:

| Face category | Baseline contribution | Bands + edge contribution |
|---|---:|---:|
| Inner same-side | 0.005054 | 0.006458 |
| Outer same-side | 0.023196 | 0.020715 |
| True crossing | 0.141704 | 0.129382 |

Each contribution uses the corresponding face count divided by the case's total
supported face count before averaging cases. Most of the total reduction comes
from true crossing faces; it outweighs the worsening inner contribution.

## Foreground extent and fragmentation

Matching the baseline's foreground volume to bands + edge, by selecting its
highest foreground probabilities without using GT to set the volume, increases
baseline union Dice from 75.31% to 79.18%. Bands + edge still reaches 79.95% and
has more correct true boundary transitions (25.03% versus 23.19%). Thus a change
in foreground extent can reproduce much of the overlap gain, but not all of
the spatial accuracy. This diagnostic changes the baseline's decoding rule;
it is not a trained control or a causal decomposition of the edge loss.

There is no evidence of globally reduced fragmentation. The mean number of
six-connected foreground components rises from 3.37 to 4.74. Voxels outside the
largest component are approximately unchanged (11.80 versus 12.41 per case;
paired difference interval [-1.81, 2.94]). These are mostly small fragments,
not evidence that the main hippocampal structure has split into several large
pieces. Some GT masks themselves contain two components, so component counts
are descriptive rather than a complete anatomical correctness test.

## Supported research claim

The edge term improves signed foreground contrast and localization of the
outer boundary, and adds an external-band consistency benefit relative to
bands alone. It does not produce a uniform improvement in hard voxel coherence
or global connectedness. Higher Dice therefore cannot be used as proof of those
stronger claims.

This is an exploratory analysis of development validation data and selected
checkpoints. Bootstrap intervals pair cases across methods, average the three
seeds first, and resample within folds; they do not cover checkpoint selection,
new training sets, or new folds. Case IDs are distinct across folds, but patient
independence was not established. Small repeat-inference numerical differences
from the earlier audit were recorded: maximum case macro-Dice change was 0.0402
percentage points, with mean absolute change of 0.000806 points across evaluated
train and validation records. No cause for that numerical variation was isolated.

The [full report](results/REPORT.md) contains every fold–seed comparison,
[SUMMARY.json](results/SUMMARY.json) includes training results and all paired
intervals, and [the figure](results/coherence.png) shows the nine individual
trajectories. Per-case records and checkpoint provenance are retained alongside
them. Source: completed inference job 673837, exit 0.
