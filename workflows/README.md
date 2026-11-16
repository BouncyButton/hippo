# Final experiment workflows

## Training from scratch on MSD, ADNI and MNI

Use Python 3.11+ for the launcher and a Linux training environment with the bundle's profiled PyTorch, MONAI and CUDA dependencies. The root `requirements.txt` includes the scratch package versions. Released adapters use their own upstream SAM3 checkout, LoRA configuration and compatible training environment. Dataset converters accept local source images. Rebuilding or reserializing a dataframe can change its byte digest. The scratch worker checks the exact prepared data used by the archived protocol.

`scratch_bundles.json` identifies the five frozen source bundles by SHA-256. `scratch_protocols.csv` lists their final protocol digests, training losses, learning rates, schedules, compute limits and training-only calibration settings. Point `--bundle` to the extracted bundle named for the dataset and seed. Only the selected protocol directory (protocol, split and architecture JSON), the prepared pickle and, for MSD/ADNI, the bundle's initialization reference are needed. The launcher checks those inputs against the final protocol hashes and runs the training code in this checkout.

```sh
python workflows/scratch.py --dataset MSD --model nnunet --arm dice_bands_sdf_depth \
  --fold 0 --seed 0 --bundle "$BUNDLE" --output outputs/replay --check-only
```

Remove `--check-only` to train in a fresh per-cell output directory. Use `--data PATH` if the matching pickle is mounted separately. MNI's 97 reused cells are identified in the original bundle alongside the 407 queued cells. The training implementations are under `scripts/cross_dataset_campaign/`, `scripts/mni_constraints/` and `scripts/unetr_subfields/`. In particular, the MNI trainer handles aligned native depth weights and its recorded augmentation policy.

## HFH scratch and MedSAM3: train from this checkout

These workflows need the study data, a compatible Linux CUDA environment, and (only for MedSAM3) the two pretrained checkpoints. The folds, objectives, calibration records and schedules are in [protocols](protocols/). Training never needs a previous run, a private source bundle, Drive or Slurm. [native_archives.csv](native_archives.csv) records historical archive identities for comparison.

Use Python 3.11 or 3.12 and the root requirements. Install a PyTorch 2.12.1 CUDA build compatible with your NVIDIA driver. The recorded MedSAM3 runtime used CUDA 13.0 and native BF16. Select a single GPU; allow at least 40 GB VRAM and 12 GiB of available host RAM for the adapter. Full scratch runs also require native BF16. Use a new output directory on a disk large enough for model and recovery checkpoints.

| Dataset | Input to the launcher |
|---|---|
| HFH | Directory of verified image/label NPZ crops. Build it from the official Train Analyze images and matching Labels directory. Both scratch and MedSAM3 use the same arrays. |
| MNI | Directory of verified native image/label NPZ crops. Build it from the study dataframe. |
| ADNI | The study's gzip-compressed `adni_hippocampus_full.pkl`. The launcher checks its digest before unpickling and resolves the split by participant ID. |
| MSD | The study's `imagesTr/` and `labelsTr/` NIfTI directories, with the original filenames and contents. The frozen split file is supplied by the checkout. |

```sh
python workflows/prepare_crops.py --dataset HFH --source "$HFH_RAW/Train" --output data/hfh_crops
python workflows/prepare_crops.py --dataset MNI --source "$MNI_PICKLE" --output data/mni_crops
python workflows/portable.py --dataset HFH --model nnunet --list
python workflows/portable.py --dataset HFH --model nnunet --arm dice_bands \
  --fold 0 --seed 0 --data data/hfh_crops --output outputs/hfh_nnunet_f0s0
```

The other HFH scratch model names are `basic_unet16`, `swinunetr`, and `unetr_vit_b16`. The CPU plumbing check below writes a real optimizer step, checkpoint, and evaluation. It does not reproduce a paper score. Some constrained workers additionally require a byte-identical model initialization or native prefix; a different PyTorch build or hardware can fail that check.

```sh
python workflows/portable.py --dataset HFH --model basic_unet16 --arm dice \
  --fold 0 --seed 0 --data data/hfh_crops --output outputs/hfh_cpu_smoke --device cpu --smoke-steps 1
```

For MedSAM3, install the Python packages and public upstream source once. The installer fetches commit `84f43118b5b67b50abad81d7c946681ebb97dcda` from [MedSAM3](https://github.com/Joey-S-Liu/MedSAM3/tree/84f43118b5b67b50abad81d7c946681ebb97dcda) and verifies the source files, LoRA configuration and tokenizer. It prints the cache path. An offline installation can supply the verified checkout with `--upstream PATH`. Provide the SAM3 base checkpoint and medical LoRA checkpoint; the launcher verifies both by SHA-256.

```sh
python -m pip install -r requirements.txt -r workflows/requirements-medsam3.txt
python workflows/setup_medsam3.py
python workflows/portable.py --dataset MNI --model medsam3 --arm dice_bands_edge_depth \
  --fold 0 --seed 0 --data data/mni_crops --base-weights "$SAM3_PT" \
  --lora-weights "$MEDICAL_LORA_PT" --output outputs/mni_adapter_f0s0
```

Use `--list` for the valid objective/fold/seed combinations. Use `--show-protocol` with an objective, fold and seed to inspect the resolved settings without data. Use `--check-only` with the real input arguments to verify data, weights, split membership, and training schedules without allocating a GPU or creating an output directory. Each training invocation starts a fresh run and refuses to overwrite an existing output directory.

The MSD primary worker runs its equivalence check and fold calibration before training. The historical BAND and BAND+EDGE workflow trains the original three-arm block together, including its matched control. Use `--arm original_dice` for that block or `--arm dice` for the primary batch control. Running any original-block selector produces all three results. On MNI, native and constrained runs use the recorded learning rates and slab sequence, including earlier milestone horizons. HFH nnU-Net preserves the timing and coefficient choices per objective and seed. The [validation design](../README.md#validation-design) describes how validation informed selection.

HFH, MNI and ADNI use localized hippocampus crops. Where a reference mask defines the crop, the reported task measures segmentation conditional on localization. Replaying a frozen workflow does not turn its validation folds into an independent test set.
