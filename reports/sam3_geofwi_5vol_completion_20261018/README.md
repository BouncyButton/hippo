# SAM3 five-volume completion batch

Job **706255** queues the 22 variants not run by SAM3 job 703589. It is the account's single queued job, dependent on running job **706250** with `afterany`, and requests two isolated `4g.40gb` GPU workers for at most 24 hours. The original two completed 10-update runs are excluded; their verified results remain in `reports/sam3_geofwi_5vol_update_sweep_20261016`.

The frozen five training volumes, 96 validation volumes, SAM3 initialization, LoRA training source, optimizer schedule, fold 0, seed 0, and no-augmentation setting are unchanged. Missing arms: native and bands20 at 25/50/100/200 updates; bands10 and bands40 at 50/100; bands20 plus edge at 0.5/1/2/5% at 50/100, plus 1/2% at 200. These percentages are calibrated auxiliary gradient targets against the native SAM3 loss, which already includes Dice.

Preflight verified the original source and data hashes, checkpoint, cohort, prior two archived/cleaned results, quota, and scheduler request. At submission, the personal BeeGFS quota was 29.19/93.13 GiB used (63.94 GiB free), including artifacts of the other running job. The estimated mirrored peak is 8 GiB plus a 2 GiB reserve, with a 10 GiB minimum free-space gate.

Each completed variant must be fully evaluated and archived to this local `reports/` folder with hash verification before its generated checkpoint is removed from BeeGFS. The collector runs as user LaunchAgent `com.hippo.sam3-five-volume-completion`, with idle-sleep prevention and restart on unsuccessful exit. The cluster worker waits up to one hour for an archive acknowledgment. Submission and preflight receipts are saved beside this document.

On 18 October 2026 the user requested unqueueing this campaign to prioritize MSD seed 5 SwinUNETR/U-Net constraints. Pending job 706255 was cancelled before starting, and control.json was set to pause=true. Do not automatically resubmit. See user_cancellation.json.
