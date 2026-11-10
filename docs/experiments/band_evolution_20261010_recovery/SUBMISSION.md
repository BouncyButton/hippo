# Recovery of the independent-target and band sweep

**Final status:** all 189 arms completed successfully on 11 October 2026 at 01:18 Europe/Rome. See [final results and verification](final_results/FINAL_SUMMARY.md). The submission-time checks below are retained as provenance.

Job **692240** was submitted on 10 October 2026 with no scheduler dependency.
It replaces failed job 691444, which stopped in CUDA preflight before training.

The repair makes device transfer recurse through lists and tuples of sample tensors.
The original helper handled tensors and dictionaries only. Eight scientific tests
passed, including a new meta-device transfer regression, exact control equivalence,
geometry/gradient checks, restart and retention. All 189 objectives passed the
local real-model checks, with geometry checked on all ten training cases and a
verified SwinUNETR parameter update.

Protocol comparison confirms that only the root name, source hashes, protocol
identity and recovery provenance changed. The experiment matrix, split, seed,
warmup, optimizer, losses and calibration settings are unchanged. The original
failed source, protocol, logs and warmup remain preserved.

Resources: one MIG 4g.40gb GPU, four CPUs, 24 GiB RAM, 23:50:00 time limit.
Quota at submission: 3.78 GiB free. The existing bounded checkpoint-retention policy
is unchanged. Every result and prediction is kept.

CUDA preflight passed for all 189 objectives, including a real CUDA parameter
update. Calibration passed for all 189 arms on all ten training cases. Slurm
confirmed RUNNING on gnode01 and the worker advanced to the first `bands` run.
No scientific run had completed at that check. See runtime_verification.json.

Protocol identity: `8132b9215d1bbb6ad0ead278fcb3cf40c6e9fd05a2db0cdc7c54502eb08099d5`.
Payload SHA256: `1951057778d42536a5fe80020bff81b7d006560bf3b5c628153155895bdc8f6d`.
