# Superseded MedSAM3 inference-refinement pilots (28 September 2026)

Archived at the user's request before deleting generated experiment artifacts.
These experiments applied bands/edge as new mask-free **post-inference
refinements**, which did not answer the intended training-loss question.

Zero-shot job 673230, hippocampus_017: baseline Dice 0.0086180980;
+bands 0.0079565302; +bands+edge 0.0079634845.

Five-shot job 673282 used training volumes 011, 056, 146, 319, 363 (prefix
hippocampus_), seed 42, final LoRA after 200 AdamW updates, four random slices
per update, LR 5e-5 with 20-update warmup/cosine decay to 5e-6. The base model
remained frozen. Whole-hippocampus binary 3D Dice on separate validation volumes:

| Case | Zero-shot | Five-shot | + bands | + bands + edge |
|---|---:|---:|---:|---:|
| hippocampus_017 | 0.008618 | 0.868163 | 0.853706 | 0.855359 |
| hippocampus_019 | 0.000514 | 0.836421 | 0.839829 | 0.840205 |
| hippocampus_033 | 0.001736 | 0.881514 | 0.859003 | 0.860000 |
| hippocampus_035 | 0.003947 | 0.873124 | 0.873990 | 0.875187 |
| hippocampus_037 | 0.008629 | 0.836256 | 0.812246 | 0.812439 |
| Mean | 0.004689 | 0.859096 | 0.847755 | 0.848638 |

Both jobs completed; saved-mask Dice and grid/hash checks passed. The original
case's repeated zero-shot output was identical. Few-shot adaptation helped,
while post-inference constraints lowered mean Dice. These results do not test
supervised constraints during fine-tuning. The old predictions, trained adapter,
plots, logs, result archives and detailed run records were authorized for
removal. Original datasets, downloaded SAM3/medical weights, reusable source
code and shared runtime are retained for the corrected training experiment.
