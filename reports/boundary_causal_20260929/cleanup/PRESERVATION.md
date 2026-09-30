# Narrow archive plan before remote cleanup

All MedSAM3 runs/results are excluded and protected. Also protect all original pooled/bands/augmentation controls, the pooled reruns INSIDE separated_edge, all three summed PCGrad controls, seed0 separated/PCGrad diagnostic anchors, the successful50-case PCGrad selected and final resumable checkpoints, and all non-weight files.

Only four selected weights are eligible: separated seed1/2 and failed symmetric PCGrad seed1/2. First make a byte-identical LOCAL archive and compare SHA256 against original completion records and live remote files. Only after verification remove the exact four remote files. No experiment directory is removed. These weights can be restored locally; no scientific result is lost.

Scientific value preserved: splitting edge means yielded negligible overlap gain(77.7751% pooled vs77.8598% separated) and no broad coherence gain. Selected separated epochs73/67/73; stopped75/71/75. PCGrad harmed train fit and validation in all3seeds: mean train/val55.6544/53.7484 vs85.8249/77.7043 sum. PCGrad selected61/58/62; stopped69/66/70. Seed0 oversizing explained apparently perfect inner correctness. Preserve all configurations, provenance, per-case metrics and per-update gradient diagnostics in existing reports, with the four runs' compact config/completion/selected metrics copied here.

Dependency check: current676348 reads the50-case reference and source manifests, neither of the four targeted weights. Queued diversity test reads seed0 sum config and new50-case sum weights. Calibration dependencies use seed0, not these four selected weights. Both source trees and all calibration files remain. No active job uses these four weights. Future seed1/2 inference can restore the local archives if needed.

BeeGFS quota charges appear to include mirroring: current pilot consumed~0.63GiB for~0.28GiB logical files. Update storage budgeting to twice logical sizes. The local archive prevents destructive scientific loss while freeing~0.52GiB charged quota.
