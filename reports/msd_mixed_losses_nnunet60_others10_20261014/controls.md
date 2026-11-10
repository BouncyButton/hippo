# MSD mixed-loss sweep controls

From `/Users/filippofocaccia/Desktop/hippo`:

```sh
rtk proxy python3 -m scripts.cross_dataset_campaign.manage_msd_mixed_losses status
rtk proxy python3 -m scripts.cross_dataset_campaign.manage_msd_mixed_losses pause
rtk proxy python3 -m scripts.cross_dataset_campaign.manage_msd_mixed_losses resume
rtk proxy python3 -m scripts.cross_dataset_campaign.manage_msd_mixed_losses collect
rtk proxy python3 -m scripts.cross_dataset_campaign.manage_msd_mixed_losses report
```

Pause saves optimizer/RNG/sampling state and cancels the queued continuation. Resume restarts this campaign only. Frozen scientific settings must not be edited beneath an active run; pause first and prepare a separately versioned modification.

All 56 runs: MSD fold0 seed0, 10 training/52 validation crops, no augmentation, optimization cap31,976,450,667,090 FLOPs. nnUNet uses native foreground Dice+CE and60% auxiliary targets (p95 cap75%). The other three models use established MONAI Dice-only including background and10% targets (p95 cap50%). Targets are sequential per auxiliary relative to the accumulated loss gradient.

Each completed experiment is evaluated, archived locally, verified by hashes/readability, then cleaned on BeeGFS. Leave this Mac awake/connected for archive delivery. If unavailable, the allocation releases after a five-minute archive wait and the collector can resume automatically when back online, unless manually paused.

After reopening the Mac, use `collect` to ensure the archive service is running. This does not clear a manual pause. Use `resume` only when you intend to clear a manual pause as well.

The campaign uses the four-model GPU/cache optimizations validated by `msd_runtime_benchmark_v2_20261014`. The optional preloaded startup backend was not adopted; its diagnostic was abandoned to meet the submission deadline, archived, and cleaned.

The abandoned MSD60 campaign, ADNI/seismic follow-on and full replicated sweep remain stopped/held.
