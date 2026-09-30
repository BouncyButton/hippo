# Supplement: isolate the head update with a matched backbone

Frozen after inspecting aggregate results of job 666735, before the repeat.
The original output records and code package remain on the cluster unchanged.

Job 666735 completed 32 real AdamW steps, with matching J/H/V forward losses
and no skipped steps. Natural J/H backbone updates were not identical:
42,725–50,216 parameters per update differed by more than 1e-6; the largest
absolute difference ranged from 0.0001941 to 0.0001987. The learning rate was
0.0001. Therefore the natural H/J output difference cannot be attributed solely
to head changes. The source of the numerical discrepancy is not isolated by
the saved norms; AMP accumulation and the reset optimizer's sensitivity near
zero gradients are possible mechanisms, not established explanations.

Repeat the same locked batches, seeds and J/H/V/R updates, adding two controls:

- H_matched: after H's real update, replace only its backbone state with J's
  post-update backbone. Assert exact equality. Keep H's head state unchanged.
  This is a hybrid diagnostic, not the native H optimizer trajectory.
- J_bias_matched: shift J's scalar presence bias to match H_matched's all-ray
  mean presence logit on the update images. No labels determine the bias.

All original outcomes remain reported. The new comparison H_matched versus J
isolates the effect of the different head update at a common backbone state.
Comparison with J_bias_matched checks whether a mean shift reproduces it. The
control does not match FP burdens; longer-training selection would still need
inner calibration and the previously specified FP/ASSD guards.

Do not interpret the repeated images as more cases or independent replication.
No updated weights are saved and no longer continuation is launched by this
supplement. Numerical artifacts remain on the cluster; only non-identifying
aggregate outcomes are retrieved following the transfer-review restriction.
