# Verification, 2026-09-05

The original Family-B source was verified file-for-file against both official
controls before submission and again after staging the separate follow-up.
Seed 0 (650078) is RUNNING on gnode02/A100 MIG 4g.40gb; seed 1 (650080) is
PENDING because the account allows only one running job. The last check was
18m16s into seed 0, before its first reported epoch. This is successful launch
verification, not evidence of a completed epoch or improved segmentation.

The development source digest agrees locally and on the cluster:
`8b30d57574dda9c89c52a96b63ad986d19ff2d97fd18e1a5e3e86d91825fe7b1`.
Its archive SHA is recorded in DEVELOPMENT_SNAPSHOT.json. The development
snapshot is read-only and separate from the active experiment source.

Across the targeted test runs, **219 distinct tests passed**, with one intentional
skip (augmentation with the held A/P-cut route). These include six new tests of
augmentation alignment/replay/RNG isolation, fixed overlap, exact 66-pair expected
gradient equality, corrective saturated-logit gradients, the coefficient policy,
and gradient-energy attribution. Two additional tiny-model training/resume cases
exercise augmentation alone and augmentation plus common-support teacher.

Test groups: existing equivariance/bands/onecut/telemetry; supervised Dice/CE;
teacher; A/P posterior; follow-up trainer integration; official-audit routing;
translation follow-up. Interpreter: /Users/filippofocaccia/anaconda3/bin/python3.
An initial command named an obsolete test_new_constraints.py; no tests ran in
that attempt. It was corrected to the actual per-module test files.

These local tests use synthetic tensors/tiny models and do not constitute a CPU
audit of the official baseline. That requested audit previously ran on cluster
CUDA (650074). The new teacher calibration has **not run on GPU yet**, and the
revised source has not completed a real SwinUNETR training smoke. New training
launchers and calibration are prepared, not submitted.

The known unrelated closure-analysis collection failure (missing
equivariance.evaluate_closure) remains outside the targeted passing suite.
Scoped git diff --check passed. Shell launchers were checked with bash -n.
