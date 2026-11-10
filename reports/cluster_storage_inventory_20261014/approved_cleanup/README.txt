Cleanup complete. See SUMMARY.txt for the final totals and quota.

Experiment retirement archive — 14 October 2026

Scope: the six experiment groups explicitly selected in this chat (16 directories).
The user's retention choice is: preserve results and predictions; delete checkpoint weights.

results_catalog.json maps experiments to their result archives and lists original result paths.
results_archives/ contains portable .tar.gz files with the original experiment-relative paths.
Each archive includes ARCHIVE_MANIFEST.json with original file sizes and SHA-256 checksums.
Checkpoint weights (.pt, .pth, .ckpt, .safetensors) are explicitly excluded and listed in the manifests.
Configurations, source snapshots, data splits, calibration, histories, plots, logs and saved arrays are retained.

results_verification.json records byte-for-byte archive verification and JSON/NPZ readability checks.
geofwi3d_voxel_metrics.json preserves exact FN/FP counts reconstructed from existing per-volume voxel counts.
The verified evaluation supplement contains 100 hippocampus checkpoint evaluations, voxel-error metrics and hard predictions.
Historical validation scores remain preserved separately from additional evaluations.

The native-branch archive includes its original external atlas symlink. Its target remains outside the deletion scope.
Source repositories, datasets, Conda environments, ADNI baseline checkpoints and prior durable local reports are outside the deletion scope.

Deletion is permitted only after complete archive verification, completed required evaluations, verified independence of any active jobs,
and a fresh comparison of original metadata and archived file hashes. The deletion receipt and final quota snapshot
record the eventual outcome; absence of a successful deletion receipt means cleanup is not yet confirmed.
