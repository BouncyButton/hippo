# Stopped by user on 2026-10-14

Do not resume or submit this campaign. The MSD 56-run sweep was abandoned; the ADNI/seismic/MedSAM3 follow-on is held. The active replacement is `msd_transformer_ce_bands_target10_20_20261014`. Prior instructions below are historical and do not authorize restart.

Pause safely and cancel the queued continuation:

    rtk proxy python3 -m scripts.cross_dataset_campaign.manage_followon pause

Resume the saved optimizer and sampling state:

    rtk proxy python3 -m scripts.cross_dataset_campaign.manage_followon resume

Status / archive service / results:

    rtk proxy python3 -m scripts.cross_dataset_campaign.manage_followon status
    rtk proxy python3 -m scripts.cross_dataset_campaign.manage_followon collect
    rtk proxy python3 -m scripts.cross_dataset_campaign.manage_followon report

Run from /Users/filippofocaccia/Desktop/hippo. These controls cover both the current MSD predecessor and the queued ADNI/seismic follow-on. Pause before changing experiments. Frozen submitted scientific configurations must not be edited in place. Completed results remain archived; interrupted runs retain recovery files.

Each allocation lasts at most 24 hours and queues one successor. The shared chain finishes MSD first. If the durable archive service is offline, the GPU is released after five minutes and automatically resumed after the collector returns. Keep the Mac awake and connected for uninterrupted archiving and SAM3 checkpoint delivery.
