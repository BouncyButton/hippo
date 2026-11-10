# Submission

Job **689899** was submitted on 8 October 2026 at 12:41:16 Europe/Rome and started at 12:41:17 on `gnode01`. Scheduler readback confirmed RUNNING, one A100 4g.40gb GPU slice, four CPUs, 24 GiB RAM, and a 12-hour limit. No dependency or additional campaign was submitted.

The twelve runs use the six screenshot rows crossed with depth balancing off/on, native MRI-only auxiliary branch, fold 0, seed 0, 5% only (10 training / 52 validation cases).

See `submission.json` for the scheduler receipt and `protocol.json` for the immutable experiment specification. CUDA preflight and scientific training status are recorded in the remote campaign root.

CUDA preflight passed on all ten real training cases under MONAI 1.6.0 / PyTorch 2.12.1+cu130. It verified finite gradients and optimizer updates in the native network, MRI-only encoder, and fusion, with all four constraints active. The worker then launched `5pct_dice_s0_native_mri_context`, the first of twelve runs. The startup snapshot records this state; no scientific result was available yet at handoff.
