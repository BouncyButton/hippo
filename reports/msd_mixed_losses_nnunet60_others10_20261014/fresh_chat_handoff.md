# Fresh-chat handoff: finish optimization, then submit the authorized MSD sweep

Prepared on 2026-10-14 at the user's explicit request to continue in a fresh chat. The originating chat will stop experiment work after creating the new chat. Treat status below as last observed and verify live state before mutations. The user has already authorized the 56-run submission after optimization; no new permission is needed for that scope.

## Immediate objective and latest conversation

The user requested a new MSD fold 0 / seed 0 / 5% campaign with all 14 established conditions and four models, then asked to optimize GPU runtime before submission. We have completed and validated several optimizations. **The 56-run scientific sweep has NOT been submitted.** Its existing local/remote stage is paused for optimization.

The user's last substantive question was: “is there anything else to optimise or have you already submitted the job?” The response already delivered was: “The 56-run sweep has not been submitted yet. The GPU optimizations have passed exact checks for all four models. One worthwhile optimization remains: reusing loaded libraries while keeping each run in a fresh process. That could reduce startup overhead across 56 runs. I’ll make that the final benchmark, then submit with the verified optimizations. Budget about 10–15 more minutes, including final checks.”

The user now requested a fresh chat instead of further compaction. Continue promptly. **Do not expand optimization scope again.** Finish or reject the prepared startup benchmark, freeze verified code, preflight, submit the scientific job, launch collector, restore monitoring, verify launch and report its ID.

## Environment and instructions

- Repository: `/Users/filippofocaccia/Desktop/hippo`. Read AGENTS.md and `/Users/filippofocaccia/.codex/RTK.md`. Every shell command must start with `rtk`, usually `rtk proxy ...`.
- Very large preexisting dirty/untracked tree. Do not reset/revert unrelated files, and do not commit data or generated reports.
- SSH alias `bocconi-cluster`; user 3160552, UID 1395. Remote root `/mnt/beegfsstudents/home/3160552`; `/home/3160552` is also BeeGFS.
- Network shell operations have used `sandbox_permissions: require_escalated`. Previously accepted command prefixes include rtk proxy ssh/rsync/python3 and Python 3.12 below.
- Scientific remote Python: `/home/3160552/.conda/envs/hippocampus/bin/python`, Torch 2.12.1+cu130, MONAI 1.6.0. Default remote python3 is 3.13; local default is 3.9.
- Local archive extraction MUST use `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` because the collector uses `tarfile.extractall(filter='data')`.
- Local numerical test environment: `rtk proxy env PYTHONPATH=/private/tmp/hippo_adni_profile_deps:.:scripts:baselines/nnUNet MPLCONFIGDIR=/private/tmp/hippo-campaign-mpl .venv/bin/python ...`. This supplies MONAI 1.6.0 over the venv default; local Torch is 2.2.2.
- GPU allocation: partition stud, `--gres=gpu:4g.40gb:1`, four CPUs, 40G RAM.
- Keep user informed with concise meaningful updates. No subagents unless newly authorized. Never infer a scientific change from a performance optimization request.

## Authorized scientific recipe

Campaign name: `msd_mixed_losses_nnunet60_others10_20261014`.

- 56 fresh runs: four models × 14 existing screenshot conditions. All MSD fold 0 / seed 0, saved 5% split of 10 training and 52 validation crops. **No augmentation.**
- Optimization cap: **31,976,450,667,090 FLOPs**. Calibration is separately accounted.
- nnUNet: established native foreground Dice+CE, deep supervision, SGD; sequential auxiliary gradient target 60%, p95 ceiling 75%.
- SwinUNETR, UNETR ViT-B16 and 3D UNet basic_unet16: established MONAI Dice-only including background, AdamW; target 10%, p95 ceiling 50%.
- Preserve original loss graphs, SDF timing, 50-update prefix/warmup rules, optimizer, learning-rate schedule, initialization and sample order. Fresh four baselines first, then four bands runs, then remaining combinations.
- The exact 14 arms already exist in the campaign code: baseline; bands; signed distance; bands+signed distance; bands+original edge; bands+signed distance+original edge; depth CE; bands+depth CE; original edge+depth CE; signed distance+depth CE; bands+original edge+depth CE; bands+signed distance+depth CE; original edge+signed distance+depth CE; all four constraints.
- Current verified schedules: nnUNet baseline 90 updates, constrained 89; SwinUNETR 206; UNETR 245; 3D UNet 562–563. Do not alter them to accelerate runtime.

## Storage and execution authorization

At most one RUNNING and one QUEUED allocation across the account. Max job duration 24h; controller must automatically queue continuations with optimizer/RNG/sampling preserved. Respect manual pauses. Per experiment finish Dice/FN/FP and A/P swap evaluation, archive metrics/curves/config/split/calibration/source/logs outside BeeGFS in local reports, verify hashes and readability, then promptly delete generated checkpoints/predictions/temporary files and log deletion. Preserve datasets, original annotations, source, durable archives, unrelated files and active/recovery artifacts. Failed/interrupted recovery is retained until resumed or explicitly abandoned; completed diagnostic dependencies may be retired after successful replacement verification.

Check **personal quota** with `beegfs quota list-usage --uids current --pool pool_home`, not filesystem df. Last observed before benchmark v2: 26.28/93.13 GiB used, 66.85 GiB free. V2 subsequently deleted 5.113 GiB logical. Fresh quota required before submission. Scientific estimate uses 5 GiB logical peak, 10 GiB mirrored plus 2 GiB reserve.

**Never resume these:** `msd_four_models_ce_target60_20261014` was explicitly abandoned; ADNI/seismic/MedSAM3 `adni_seismic_ce60_followon_20261014` is held/cancelled; original replicated 1863-run sweep is held. Do not add seeds/folds, augmentation, seismic SAM3 or other experiments.

## Last known live state

- Benchmark v1 job **701213** failed only because the benchmark fixture normalized the same in-memory data twice. It has aged out of `squeue -j`; its generated checkpoint files still await cleanup (~0.978 GiB logical).
- Benchmark v2 job **701219** completed, all four models passed exact checks. Metadata/source archived and verified, all 16 generated checkpoint files deleted, **5.113188482820988 GiB logical** released.
- Queue was empty after v2. Verify now.
- Startup benchmark code uploaded but **NOT submitted**. Intended name `msd_runtime_benchmark_startup_20261014`.
- Scientific 56-run job **NOT submitted**. Existing stage control pause=true, reason GPU optimization.
- No known live unified exec sessions or collector for the new campaign. Previous collector call was interrupted before it launched.

## Verified optimizations

`scripts/cross_dataset_campaign/msd_runtime_fast.py` implements `RuntimeAcceleration`:

1. Cache the 62 preprocessed MSD volumes/labels on GPU (~186 MiB).
2. Cache immutable SDF/depth targets by tensor storage, offset, shape, stride, dtype/version plus affine/settings. Strong references prevent pointer reuse; in-place label mutation invalidates the cache.
3. Keep validation batch size **1** and identical GPU math, defer metric readback until the end to remove per-case CPU/GPU synchronization.
4. Skip rewriting an unchanged `checkpoint_best.pt` only when the same immutable best payload and unchanged file signature prove it redundant. Latest recovery checkpoints retain their original cadence.

No precision, optimizer, forward/backward graph, sampling, checkpoint cadence or scientific budget changes. Larger evaluation batches 2/4/8 changed numerical metrics and were rejected. No torch.compile, flash attention or optimizer fusion adopted.

CPU tests in `test_msd_runtime_fast.py`: four passed (exact metrics/RNG, SDF/depth cache/view/mutation behavior, duplicate best-write skipping while latest always persists, augmented protocol rejection).

GPU `benchmark_msd_runtime.py` v2 tested actual saved data, all four models and six diagnostic updates with all four auxiliary terms. Exact losses, gradient hashes, model/optimizer hashes, RNG, calibration, prepared image/label tensors, all 52 evaluation metrics and 3+3 resume equivalence passed. Checkpoint round-trip passed. Diagnostic per-step timing includes hashing/readback overhead, so do not claim full training speedup from it.

Measured component speedup factors (old time/new time):

| Model | Evaluation | Repeated checkpoint work |
|---|---:|---:|
| nnUNet | 1.2374 | 1.4489 |
| SwinUNETR | 1.0383 | 1.3717 |
| UNETR | 1.0848 | 1.2928 |
| 3D UNet | 1.2620 | 1.3519 |

Thus evaluation used about 4–21% less time and repeated checkpoint work 23–31% less. Cached annotation preparation fell from ~120 ms to ~0.1 ms. These are component measurements, not a measured end-to-end sweep speedup.

Evidence: `reports/msd_runtime_benchmark_v2_20261014/` contains runs/{model}.json, runs/summary.json (passed), exact benchmark source, parent_frozen, archive_manifest.json, archive_acknowledgment.json, cleanup.json (complete), Slurm logs and submission receipt.

V1 evidence is in `reports/msd_runtime_benchmark_20261014/`, including original source and first_equivalence_audit. Fixture reused a dataframe twice; MONAI normalization mutated it. V2 reloads fresh pickle separately per variant and asserts every prepared tensor identical. Actual production runs already load fresh data, so no scientific data bug was found.

## Immediate small fix and cleanup

`collect_msd_runtime_benchmark.py` archives metadata/source/frozen parent, verifies local readability/hashes and remote hashes, then deletes only generated runs/**/*.pt listed in the inventory. V2 cleanup succeeded. V1 retry failed **before deletion** because both embedded remote checks use `squeue -h -j701213`, which errors once an old job ages out.

Fix BOTH queue checks to list account jobs (`squeue -h -u 3160552 -o %i`) and test whether the target job is among returned IDs. Do not hide unrelated scheduler errors. Then run:

`rtk proxy /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m scripts.cross_dataset_campaign.collect_msd_runtime_benchmark --name msd_runtime_benchmark_20261014 --superseded`

This requires the corrected v2 summary passed (it is), preserves the failed diagnostic metadata, and retires now-unneeded diagnostic checkpoints. Do not rerun v2's completed collector blindly: rebuilding its manifest could lose the original deleted inventory. Finish collections **before replacing the old local scientific stage**, because the collector copies that frozen parent.

## Final startup optimization: prepared, not tested

Files already uploaded into existing remote held stage/source:

- `preload_msd_runtime.py`: sets PYTORCH_NVML_BASED_CUDA_CHECK=1 before Torch import, imports scientific trainer, asserts CUDA is not initialized.
- `runtime_worker.py`: `TrainerProcess` uses Python multiprocessing forkserver preloaded with the CPU-only modules. Fresh child asserts no CUDA initialized/bad fork, sets cwd/log/argv, runs the actual trainer. Adapter exposes poll/returncode/pid/send_signal/wait for the controller. Restricts module and interpreter. Not yet wired into controller.
- `benchmark_msd_startup.py`: runs forkserver then ordinary subprocess baselines for all four models, three diagnostic updates each, actual saved data/recipes. Hashes complete latest checkpoint payload (model, optimizer, RNG, history, configuration) and compares validation/training/calibration exactly. Measures per-process elapsed time including startup; first fork includes preload, later ones show amortized gain. Parent checks CUDA never initializes. Uses existing frozen main without new acceleration flag so worker isolation is tested separately.
- `submit_msd_runtime_benchmark.py` supports `--startup`, checks account idle, quota >=24 GiB free, held scientific control and frozen hashes; 30-minute benchmark allocation.

After cleanup, submit:

`rtk proxy python3 -m scripts.cross_dataset_campaign.submit_msd_runtime_benchmark --name msd_runtime_benchmark_startup_20261014 --startup`

If forkserver poisons CUDA or deadlocks, reject this optional optimization promptly and proceed with the already-verified GPU/cache changes. Do not spend another round on it or leave an idle benchmark for 30 minutes. If all four exact comparisons pass AND meaningful startup gain exists, collect/archive with the Python 3.12 collector above (new benchmark name, no --superseded), then integrate only this verified backend.

Integration should be an OPTIONAL controller launch factory (`plan['trainer_process_backend'] == 'preloaded_forkserver'`), otherwise normal subprocess.Popen. **Never globally monkeypatch subprocess.Popen**, which would interfere with scheduler/SSH processes. Controller scientific Python must match trainer Python. Preserve signals and pause/resume semantics. Freeze startup benchmark summary/source hashes as evidence and validate them in prepare/verify. A new server is created each allocation. Current controller has no worker integration yet.

## Scientific code state and final submission steps

Workspace files exist in `scripts/cross_dataset_campaign/`:

- `msd_mixed_losses.py`: correct mixed native loss/optimizer graph. Recent workspace changes install RuntimeAcceleration when protocol flag enabled, accept --device, write performance.json and performance_invocations.jsonl in finally, restore hooks. These integration edits are **not yet in the remote scientific freeze**.
- `msd_mixed_losses_job.py`: prepare/verify/controller binding. Recent workspace prepare requires passing v2 summary + current benchmark source hashes, writes protocol runtime_acceleration config (batch1), and freezes runtime_benchmark.json and runtime_benchmark_manifest.json. Verification binds evidence/hash. No startup backend yet. prepare refuses to overwrite the existing stage.
- `preflight_msd_mixed_losses.py`: numerical checks all 14 objectives, native NN deep supervision, optimizers/calibration, original profile source hashes, schedules/splits and cluster pickle hash. Local verify_data=False passed. Prior remote preflight was interrupted; **fresh remote preflight is required**.
- `report_msd_mixed_losses.py`: verify archives/config/runtime recipes/profile/cap/schedule/init, extract final Dice/train/FNFP/AP swaps/calibration/cleanup, compare fresh same-model/same-loss baselines.
- `manage_msd_mixed_losses.py`: status/pause/resume/collect/report, delegates remote frozen job. collect selects safe local Python 3.12 launcher.

Parent scientific source comes from `reports/msd_four_models_ce_target60_20261014/frozen`; NN CE profile there, other model profiles from `experiments/cross_dataset_full_20261014/protocols/MSD_<model>_f0_target10/profile.json`. Eighteen profile source hashes were verified unchanged. Runtime optimizations do not alter FLOP-counted graphs (annotation preprocessing was already excluded).

After final benchmark decision:

1. Preserve the old **unsubmitted** `experiments/msd_mixed_losses_nnunet60_others10_20261014` under that campaign's local reports as a uniquely named pre_optimization_source directory. Do not overwrite prior evidence.
2. Run `python3 -m scripts.cross_dataset_campaign.msd_mixed_losses_job prepare` via rtk. Verify all56 recipes/cap/updates/new source manifest/evidence. Only add startup backend if verified.
3. Upload the fresh whole stage with `rtk proxy rsync -a --exclude=__pycache__ experiments/NAME/ bocconi-cluster:/mnt/beegfsstudents/home/3160552/NAME/` (substitute actual NAME; no --delete). Wait for transfer completion. Fresh authorized control may unpause now that optimization is finished.
4. Run remote preflight with scientific conda Python from stage/source and PYTHONPATH=.:scripts:baselines/nnUNet: module `scripts.cross_dataset_campaign.msd_mixed_losses_job preflight --stage /mnt/beegfsstudents/home/3160552/NAME`. It writes manifest-bound preflight.json.
5. Submit using existing job CLI/controller, with fresh quota + account queue checks and preflight binding. For uncertain SSH outcome, inspect receipts/queue before retrying. Save receipt in local experiments and reports.
6. Start `rtk proxy python3 -m scripts.cross_dataset_campaign.manage_msd_mixed_losses collect` using network escalation. Verify detached collector actually lives. Launcher uses Python 3.12 + caffeinate.
7. Verify real allocation/running recipe and own successor queue, no augmentation, all scientific settings correct. Keep at most one running/one queued. Controller writes latest every five updates/validation and charges ledger before each update; unchanged cadence is essential.
8. Resume the existing paused heartbeat for this new campaign, retarget it to the new chat if supported so updates arrive there, and preserve held/abandoned exclusions. Notify first four baselines, complete bands comparison, material results/failure/completion. At the end verify all archives/reports/cleanup and pause automation.
9. Report job ID, recipe, measured component gains, honest ETA, quota and pause/resume control. Before optimization rough estimate was 4–6h; recorded train/eval sum ~2.328h excluded startup/archive and underestimated heavy depth arms. Avoid inventing overall speedup; use actual new timing to refine estimate.

Controller already handles 24h continuations (23h40 grace, USR1 signal), archive wait 300 seconds then releases GPU if local Mac unavailable, and local collector resumes only if not manually paused. Do not use manage_followon for this campaign.

## Heartbeat to reuse, not duplicate

Automation ID `report-msd-four-model-60-constraints`, currently PAUSED, name `Report MSD mixed-loss constraint sweep`, RRULE `FREQ=MINUTELY;INTERVAL=20`, original target thread `01a13993-be9b-7f90-8038-1c34737a74cd`. File: `/Users/filippofocaccia/.codex/automations/report-msd-four-model-60-constraints/automation.toml`.

Current prompt says scientific sweep held/unsubmitted during GPU optimization and must not auto-submit. Once actual submission is verified, update full fields through automation_update, preserving notification preference and making monitor stay quiet on unchanged/non-actionable state. Retarget to the fresh chat after user-requested continuation; do not allow two concurrent controllers/collectors.

## Prior results (context only; do not rerun)

Prior completed four-run pilot: Swin CE baseline70.2286205, bands10=70.8830424,20=70.4994073,60=69.1612618; UNETR CE baseline43.2020980,bands10=45.5455478,20=45.8906197,60=33.5080500. Pilot fully archived/cleaned. Other old matched CE results: NN64.96→74.29 at60,3D81.18→83.05. UNETR60 union foreground Dice50.95→57.30 while A/P swaps11787→43399, consistent with union boundary vs class-label competition, not proof of an architecture-wide mechanism. Historical augmented/Dice-only baselines are not matched controls for these CE pilots.

The user explicitly chose native mixed-loss pipelines for the new study after discussing that within-model gains are interpretable, whereas architecture-only rankings would be confounded by different baseline recipes. Do not re-open that decision.

Primary docs consulted if explanation needs citations: PyTorch tuning guide (avoid CPU/GPU synchronization), multiprocessing notes (CUDA-safe fresh processes), CUDA environment variable docs (NVML-based availability check). No further web research needed to complete routine implementation.
