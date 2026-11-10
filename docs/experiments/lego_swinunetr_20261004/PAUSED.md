# Paused LEGO baseline

**Subsequent retirement, 4 October 2026:** the user requested removal of all LEGO dataset code. The local scripts and cluster campaign source were deleted; checkpoints, metrics and provenance records were preserved. The resume procedure below describes the former state and is no longer executable without rebuilding the removed pipeline.

Job 682573 was cancelled at the user's request on 4 October 2026 at 16:33:43 Europe/Rome. No replacement job or monitor was created.

The verified resume checkpoint contains epoch 46; automatic resume begins with epoch 47. Best validation crop Dice was 0.824726 at epoch 40. This is the fixed-ROI checkpoint selection metric, not full-volume or test Dice. Dense evaluation was not reached.

Both best.pt and latest.pt, including optimizer/scaler state for latest.pt, remain in the original cluster campaign. Local copies are in reports/lego_swinunetr_20261004/results/ and match the SHA-256 hashes in reports/lego_swinunetr_20261004/PAUSED.json.

Removed the entire cluster Dataset202_LEGOCT directory (40,657,126,093 bytes, 40.66 GB / 37.86 GiB). Preserved catalogue, preparation and acquisition metadata in the campaign's retained_dataset_metadata directory and copied it locally alongside PAUSED.json. The frozen scripts, manifest, protocol and environment were retained.

To resume later, redownload the original Zenodo dataset (https://zenodo.org/records/15730445), restore all original CT and prepared-label arrays at the paths in the frozen manifest, and verify their array hashes against the retained metadata. Keep the existing frozen manifest and protocol unchanged. Submit the retained campaign only when the user's queue requirement is satisfied. Training restores latest.pt automatically; data-loader worker augmentation random states may differ after restart.
