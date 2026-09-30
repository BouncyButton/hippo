# Frozen-baseline image-conditioned boundary head, 2026-09-22

## Question and design

Can a separately supervised head find correct hippocampal outer-boundary information where an early-stopped SwinUNETR mask is wrong? This is a **head-only diagnostic, not a test of the proposed end-to-end boundary-feedback model**. It does not smooth the predicted surfaces, alter the segmentation output, or train the baseline. It is distinct from the withdrawn four-arm contour pilot.

The frozen model is fold-0 `swin_early_stopping_665420/models/MSD_fold0/model_best.pt`, selected at epoch 21. Its decoder features (24 channels) and the normalized T1 volume feed a small 3-D convolutional head. The head emits a categorical distribution over 64 z positions for each of the lower and upper edges of every `(x,y)` column, plus a column-presence logit. Ground-truth presence and edge positions come from the three-class label union. Edge loss is masked on empty reference columns. The head loss combines presence BCE and soft Dice, edge negative log-likelihood, expected-position error, and an edge-crossing penalty. Presence threshold is fixed at 0.5.

The head trained on 166 fold-0 outer-training patients. The remaining 42 selected its checkpoint by minimum head loss; epoch 18 was selected and training stopped after epoch 24. The 52 outer-validation patients were evaluated after selection. The backbone had originally been trained on **all 208** outer-training patients, including the 42 used for head selection. Thus the 42 are a head-validation cohort conditional on a backbone that has seen them, while the 52 are unseen by both training stages. Fold-0 validation has also been studied in earlier project work, so this evaluation is exploratory rather than independent confirmation.

## Result on 52 outer-validation patients

| Measure | Frozen baseline | Head | Interpretation |
| --- | ---: | ---: | --- |
| Mean union Dice | 0.89636 | 0.89563 | Head minus baseline: -0.00073; patient bootstrap 95% interval [-0.00144, +0.00002] |
| Mean edge MAE, conditional on each method detecting the reference column | 0.50354 | 0.49101 | Head lower by 0.01253 voxel; paired patient bootstrap interval [-0.01625, -0.00902]. Supports differ between methods. |
| Presence precision | 0.9448 | 0.9527 | Head makes fewer false-positive columns. |
| Presence recall | 0.9451 | 0.9331 | Head misses more true columns. |

On **4,029 reference-occupied columns where both methods detected foreground but predicted different edges**, the head had smaller combined lower-plus-upper error in 2,112, the baseline in 1,825, and they tied in 92. Among non-ties, the head won **53.6%**; a patient-grouped bootstrap interval was **51.2–56.1%**. This directly compares the same columns and supports a small complementary edge signal.

On **687 columns where the methods disagreed about presence**, the head was correct in 298 and the baseline in 389. The head's win share was **43.4%** (patient-grouped bootstrap interval **38.7–48.4%**). It produced 258 fewer false-positive columns but 349 more false-negative columns. The head's union Dice was better for 20 patients and worse for 32. A boundary head with better conditional edges therefore did **not** improve the standalone union mask.

The union-Dice interval uses 2,000 patient resamples from the experiment script. The paired edge-MAE and disagreement-win intervals use 5,000 patient resamples, each with random seed 0. These intervals describe this cohort and do not remove the prior fold-0 exploration caveat.

## Interpretation and next decision

The image-conditioned head contributes some edge information beyond the frozen mask, but its presence decisions currently erase more true columns. This result **does not establish whether boundary supervision can improve segmentation**: the backbone was frozen, its three-class segmentation Dice received no training gradient, and the head's label losses did not feed through a differentiable fit into the mask logits. The soft Dice used here supervised only the two-dimensional presence map. Joint training could change both the head and the mask, so this diagnostic must not be used to accept or reject the full hypothesis.

The next experiment that addresses the hypothesis needs a matched early-stopping baseline and a jointly trained model. Both must optimize the same three-class segmentation loss; the joint model must additionally compare its predicted presence and boundary functions with label-derived targets, route function errors into the segmentation logits, and use a separate ground-truth voxel loss on responsible boundary mistakes. A head-only auxiliary arm can isolate whether the explicit feedback path adds value. TV fitting is an optional later ablation, not a substitute for this test.

## Artifacts and reproduction

- [summary.json](summary.json): cohort aggregates and patient-bootstrap Dice interval.
- [case_metrics.csv](case_metrics.csv): 42 inner-validation and 52 outer-validation case rows.
- [history.jsonl](history.jsonl): per-epoch training and selection losses.
- [config.json](config.json): patient split, fixed hyperparameters, and source/data/checkpoint hashes.
- [experiment source](../../../thesis/new_constraints/independent_surface_head_experiment.py): model, losses, metrics, and selection procedure.

Slurm smoke job `666062` verified tensor shapes, finite loss, and head-only gradients. Full job `666068` completed with exit code `0:0` in 3 minutes 53 seconds. The selected 272 KB head checkpoint remains on the cluster at `/home/3160552/hippopotamus_runs/independent_surface_head_20260922_01/results/head_best.pt` (SHA-256 `260a725be46481b00c96c8f2a154761bfc0fc5570c24253491140c59d69163cb`). No probability volumes were saved.
