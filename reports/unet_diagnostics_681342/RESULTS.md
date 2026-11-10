# U-Net validation saturation and residual error locations

Analysis of job **681342**: all nine U-Net runs, with ten training cases per fold and 64³ inputs. Learning curves use the original CUDA validation records. Spatial analysis evaluates the selected best checkpoint of every run on all 52 validation cases per fold: **468 case/run evaluations, covering 156 distinct cases**.

## When did gains slow?

The sharp early rise slowed around **epochs 180–240**. By epoch **242.7** (2,427 updates, 20% of the update allowance), mean best-so-far Dice was **0.8081**, leaving only **0.0089** improvement to the eventual selected mean of **0.8170**. By epoch **424.7**, only **0.0036** remained. By **606.7**, only **0.0015** remained. These correspond to 0.89, 0.36 and 0.15 Dice percentage points.

| Epoch | Updates | Mean best Dice so far | Mean remaining gain |
| --- | ---: | ---: | ---: |
| 121.4 | 1,214 | 0.7881 | 0.0290 |
| 182.1 | 1,821 | 0.8019 | 0.0151 |
| 242.7 | 2,427 | 0.8081 | 0.0089 |
| 424.7 | 4,247 | 0.8134 | 0.0036 |
| 606.7 | 6,067 | 0.8155 | 0.0015 |
| 667.4 | 6,674 | 0.8161 | 0.0009 |
| 1213.4 | 12,134 | 0.8170 | 0.0000 |

Near-plateau timing using a retrospective remaining-gain threshold:

| Absolute Dice gain still available | First aggregate milestone | Median individual-run milestone | Individual-run range |
| --- | ---: | ---: | ---: |
| ≤1.0 percentage points | 242.7 epochs | 242.7 epochs | 242.7–364.1 epochs |
| ≤0.5 percentage points | 424.7 epochs | 424.7 epochs | 364.1–485.4 epochs |
| ≤0.2 percentage points | 606.7 epochs | 606.7 epochs | 424.7–910.1 epochs |
| ≤0.1 percentage points | 667.4 epochs | 667.4 epochs | 485.4–910.1 epochs |

This is a post hoc descriptive definition, not a preregistered early-stopping test. Validation was performed every approximately 60.7 patient-pass epochs, so an exact change point cannot be located. The warmup/cosine LR schedule also changes along the curve. A late selected checkpoint does not imply large late gains: for fold 0 seed 0, the best was at epoch 970.8, but the remaining gain after epoch 424.7 was only 0.00222 Dice. The mean augmented training loss continued falling from 0.0623 at epoch 424.7 to 0.0413 at the end while validation improvements were small.

![Learning curves](learning_curves.png)

## Where are the errors?

Across all evaluated models/cases, **84.2%** of mislabeled voxels were within **one** grid voxel of the ground-truth outer contour, and **95.6%** were within **two**. The outer-contour hypothesis is supported by the observed error locations.

| Error category | Voxels | Share of all label errors |
| --- | ---: | ---: |
| Extra foreground: ground-truth background predicted as hippocampus | 254,434 | 48.4% |
| Missed foreground: hippocampus predicted as background | 230,210 | 43.8% |
| Anterior/posterior label swaps within the foreground | 41,127 | 7.8% |

The first two categories jointly account for **92.2%** of errors. The remaining **7.8%** are internal-label errors, and **95.8%** of those swaps lie within two grid voxels of the true anterior/posterior interface.

The two-voxel outer band and interface band overlap. Of all errors, 80.3% are near only the outer contour, 15.3% near both boundaries, 0.9% near only the internal interface, and 3.5% near neither. Distance proximity alone does not uniquely assign an error to one boundary.

## Does thin anatomy explain the concentration?

The two-voxel inner shell contains **90.5%** of ground-truth foreground, so a high near-surface error share partly reflects the hippocampus being thin. After accounting for region size, its label-error rate is **19.1%**, compared with **3.46%** in the deeper foreground—**5.51× higher**. The deeper-region label errors include swaps; outright missed deep foreground is only 0.126% of deep foreground voxels.

Mean whole-foreground Dice (merging anterior and posterior) is **0.8439**, versus **0.8170** macro anterior/posterior Dice in the replay. This shows that outer segmentation dominates the remaining errors, with a smaller but measurable internal-label component. The mean symmetric surface distance is **0.65 grid voxels**; the mean of each case's 95th-percentile surface distance is **1.61 grid voxels**.

![Spatial error summary](spatial_error_summary.png)

## Illustrative label maps

Examples are the median and worst macro-Dice cases within fold 0 seed 0, selected by the declared rule rather than by visual appearance. Each panel is centered on the ground-truth foreground extent along an input-grid axis; orientation and physical spacing are not inferred. These are derived label maps, without MRI intensities. The aggregate statistics above use every case and all nine runs.

![Median case](example_median.png)

![Worst case](example_worst.png)

## Reproducibility and limits

The spatial audit used CPU FP32 with no gradient/optimizer steps or weight changes. Original training source, dataset and checkpoint hashes were checked. The largest replay deviation from the original CUDA bfloat16 validation macro Dice was **0.00011776** (approximately **0.0118 percentage points**); replay mean Dice is 0.817011 versus original selected mean 0.817027.

Distances are Euclidean distances between voxel centers in the padded/cropped input grid, measured from foreground surface voxels under six-connectivity. Here “outer contour” means the binary hippocampus/background boundary, including any boundaries around background cavities; it is distinguished from the anterior/posterior label interface. Surface voxels have distance zero. Distances are in grid voxels, with no physical-spacing assumption. Surface summaries use surface-voxel counts, not physical surface-area weights.

Spatial voxel counts pool 468 repeated case/run evaluations; Dice and surface metrics average per evaluation. These are descriptive validation findings, not independent test results or significance tests. Only best and final model states were retained, so this audit locates residual errors at the selected checkpoint; it cannot show their evolution at every earlier milestone or establish why gains plateaued.

Three unit checks pass for exact FP/FN/swap partitioning, contour/interface distances and overlap, and retrospective plateau-threshold calculation. The inference and analysis code is [`analyze_unet_budget_diagnostics.py`](../../scripts/analyze_unet_budget_diagnostics.py). Rendering code is [`render_unet_budget_diagnostics.py`](../../scripts/render_unet_budget_diagnostics.py).

Numeric artifacts: [plateau summary](plateau_summary.json), [spatial summary](spatial_summary.json), [all per-case/run measurements](spatial_cases.csv). PDF figures are saved alongside their PNG versions.
