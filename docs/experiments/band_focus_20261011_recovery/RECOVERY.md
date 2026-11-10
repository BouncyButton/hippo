# Recovery of the 193-arm focused-band campaign

The original job **694088** failed in `worker.verify` before GPU preflight, calibration, or scientific training. Six multiscale entries were constructed with nested Python tuples but serialized to JSON lists. Python's direct equality check rejected the reconstructed matrix even though all numerical settings matched. The original remote root and its failure audit were preserved.

The repair constructs those scale/weight pairs as JSON-native lists. A new regression test verifies direct equality after a JSON round trip, and submission now checks the same matrix invariant before requesting an allocation. The startup check remains strict. Both local and staged remote checks confirm **all 193 scientific configurations equal the original frozen matrix**.

Only the campaign root, recovery metadata, source hashes, and resulting protocol identity differ. No objective, split, training schedule, calibration policy, or experiment order was changed. The recovery source is `scripts/band_focus/`; the original source remains in the original payload and failed remote root.

## Verification

- Twelve scientific tests passed, including the JSON round-trip regression, exact control loss/gradient parity, anatomical edge cases, and restart equivalence.
- All 193 objectives and all ten training-case geometries passed the ordinary-SwinUNETR CPU smoke check. The spatial/A/P combined objective produced finite gradients and an actual optimizer update.
- All 191 staged payload manifest entries matched their SHA-256 hashes.
- Staged worker matrix equals both the recovery protocol and the original protocol.
- GPU and complete training-case calibration checks remain mandatory before scientific training; their observed results are recorded in the submission/startup records.

## Identity

- Failed job: `694088`
- Recovery root: `/mnt/beegfsstudents/home/3160552/band_focus_fold0_seed0_5pct_20261011_02`
- Recovery protocol identity: `13e1b44d2dd58874eb61682411fdb7ba4827fc63f8ba03c5ea1d6e7b0c004bfb`
- Recovery archive SHA-256: `36a43d6d86027123e35c246fd83c9ede3b7a4982dd1f06e93712a8f6e8530470`
- Original failure audit: `../band_focus_20261011/failure_audit.json`
- Unchanged experiment matrix: [README.md](README.md)

The original storage cleanup was sufficient; no additional runs were removed for this recovery.
