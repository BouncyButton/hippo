"""Package a full notebook and an external scheduler without refreezing science."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

from scripts.colab_msd.prepare import notebook as original_notebook
from scripts.colab_msd.runtime import digest, read, write

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015'
EXPORT = ROOT/'reports/msd_colab_max_workers_3fold_3seed_noaug_20261015'
CONTROL_ZIP = 'msd_max_workers_control.zip'
NOTEBOOK = 'MSD_max_workers_3fold_3seed_no_augmentation.ipynb'


def notebook(code_sha, data_sha, control_sha, scientific_sha):
    book = deepcopy(original_notebook(code_sha, data_sha))
    cells = book['cells']
    cells[0]['source'] = cells[0]['source'].replace('MSD replication —', 'MSD replication with maximum workers —').replace(
        '**504 fresh runs:**', '**504 total runs, preserving completed and interrupted work:**')
    cells[0]['source'] += textwrap.dedent('''

    **Switching from the existing notebook:** interrupt its run cell once and wait until checkpoint
    uploads finish. Keep the same A100 type, environment and results directory. Upload only
    `msd_max_workers_control.zip` beside the two original ZIPs; open this notebook and run its cells in order.
    The original science bundle is unchanged. Verified results are skipped, recovery has launch priority,
    and an unchanged preflight is reused. Reuse the current runtime when possible. If Colab assigns a
    fresh A100 runtime, verified recovery is restored from Drive. Never delete the old runtime before
    its clean stop and recovery uploads finish; do not leave two sweeps running against these results.

    **Keep worker slots filled:** the scheduler immediately targets the CPU-supported maximum and
    fills slots whenever eligible work and memory/storage headroom are available. There are no solo
    memory-profiling runs, gradual ramp-up or throughput-triggered reductions. Memory is observed
    during ordinary concurrent runs. Scientific loss calibration still runs exactly as prescribed.
    Concurrent execution is locally tested; its actual Colab throughput must be measured.

    **Storage update:** when `/mnt/local-scratch` is a separate mounted disk, new run checkpoints
    and recovery staging use it automatically. Existing run directories stay exactly where they are;
    interrupted work has launch priority. No dataset upload or migration of active files is needed.
    Google Drive archives, per-update charge writes, checkpoint format and cadence stay unchanged.

    **Faster archive verification:** each saved archive file is read from Drive once per check.
    Hashes, readability, identity and cleanup proofs are then checked locally with the original verifier.
    Results and memory profiles are parsed from that same verified copy, which is discarded afterward.
    Verification starts fresh on every restart; no old cache is trusted. Folder counts and the current
    filename remain visible while Drive is slow. The results cell uses the same verification method.
    ''')
    cells[1]['source'] = '## 1. Mount Drive and stage the unchanged science bundle plus scheduler\n'
    cells[2]['source'] = textwrap.dedent(f'''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import fcntl, hashlib, json, os, shutil, signal, subprocess, sys, zipfile

    DRIVE_ROOT = Path('/content/drive/MyDrive/hippo_msd_3fold_3seed')
    INPUT = DRIVE_ROOT/'input'
    RESULTS = DRIVE_ROOT/'results_noaug_v1'  # Preserve this exact existing results folder.
    BUNDLE = Path('/content/hippo_msd_bundle')
    CONTROL = Path('/content/hippo_msd_max_workers')
    SCIENTIFIC_SHA = '{scientific_sha}'
    EXPECTED = {{'msd_data_verified.zip':'{data_sha}', 'msd_replication_code.zip':'{code_sha}'}}
    CONTROL_SHA = '{control_sha}'
    def sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
        return h.hexdigest()

    BUNDLE.mkdir(exist_ok=True)
    (BUNDLE/'work').mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    def unpack_verified(src, expected, destination):
        assert sha256(src) == expected, f'Upload hash mismatch: {{src.name}}'
        local = Path('/content')/src.name
        shutil.copyfile(src, local)
        assert sha256(local) == expected
        with zipfile.ZipFile(local) as archive:
            for info in archive.infolist():
                member = Path(info.filename)
                assert not member.is_absolute() and '..' not in member.parts
                assert not ((info.external_attr >> 16) & 0o170000) == 0o120000
            archive.extractall(destination)

    with (BUNDLE/'work/sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The old sweep is running. Interrupt it once and wait for recovery before switching.')
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                words = proc.read_bytes().decode(errors='replace').split('\\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if str(BUNDLE/'entry.py') in words or (str(BUNDLE) in words and any(w.endswith('/worker.py') for w in words)):
                raise RuntimeError('An existing trainer is still running. Wait for its clean stop before setup.')
        manifest = BUNDLE/'bundle_manifest.json'
        if manifest.exists():
            assert sha256(manifest) == SCIENTIFIC_SHA, 'Existing bundle differs; stop and preserve it for inspection.'
            for name, expected in json.loads(manifest.read_text()).items():
                assert sha256(BUNDLE/name) == expected, f'Existing bundle file differs: {{name}}'
            print('Verified existing scientific bundle in place; work/checkpoints preserved.')
        else:
            assert not any((BUNDLE/'work').glob('*/budget_ledger.json')), 'Existing work without a manifest needs inspection.'
            for name, expected in EXPECTED.items():
                unpack_verified(INPUT/name, expected, BUNDLE)
            assert sha256(manifest) == SCIENTIFIC_SHA
            print('Staged original bundle; existing Drive results/recovery preserved.')
        CONTROL.mkdir(exist_ok=True)
        unpack_verified(INPUT/'{CONTROL_ZIP}', CONTROL_SHA, CONTROL)
    print('No augmentation. Input data stays on the runtime; durable results stay in', RESULTS)
    ''').strip()+'\n'
    cells[3]['source'] += '\nConcurrency reserves 8 GiB of Drive headroom per active worker and 12 GiB of local SSD per worker.\n'
    setup = cells[6]['source']
    setup = setup.replace("args = [str(PYTHON), '-u', str(BUNDLE/'runtime.py'), command, '--bundle', str(BUNDLE), '--durable', str(RESULTS), *map(str, extra)]",
        "args = ([str(PYTHON), '-u', str(CONTROL/'scheduler.py'), command, '--bundle', str(BUNDLE), '--durable', str(RESULTS), *map(str, extra)]\n"
        "            if command in ('run', 'report') else [str(PYTHON), '-u', str(BUNDLE/'runtime.py'), command, '--bundle', str(BUNDLE), '--durable', str(RESULTS), *map(str, extra)])")
    cells[6]['source'] = setup
    cells[9]['source'] = textwrap.dedent('''
    ## 5. Run or resume with maximum workers

    `MAX_WORKERS = 0` chooses a ceiling from available CPU cores (two threads per experiment, with
    two cores reserved for orchestration). That ceiling becomes the target immediately: for example,
    a 12-core runtime targets five simultaneous workers. A positive value caps this target.
    The scheduler fills eligible slots without waiting for solo memory measurements and refills
    them after completed runs are archived. GPU activity and throughput are reported but do not
    reduce the worker target. A full target is not a guarantee of maximum completed runs per hour.

    Unmeasured model/condition pairs receive provisional allowances of at least 12 GiB GPU memory
    and 8 GiB system RAM per worker; known same-model peaks and live observations can raise them.
    Measured pairs reuse full-run profiles with margins. Every admission accounts for other workers'
    reserved future growth plus GPU headroom of at least 8 GiB/15% and 4 GiB system RAM.
    These allowances are estimates, not proven peak bounds. Out-of-memory or other worker failure
    stops new launches and saves other workers; failed charged work is preserved for an audit.

    Interrupted tasks have launch priority, with other eligible work filling spare slots.
    Fresh baselines have priority over new constraints; each new constraint waits for its own matched
    baseline archive. Other folds/seeds can continue concurrently while a baseline finishes.
    Each task still has its own original seeds, batch size 1, BF16, calibration, losses, optimizer,
    update schedule, checkpoints and 31,976,450,667,090 optimization-FLOP cap. No augmentation.
    Training can remain partly idle when Drive is the bottleneck; do not equate VRAM occupancy with speed.

    New run checkpoints and recovery tar staging use `/mnt/local-scratch` when it is mounted separately.
    The `STORAGE` line prints the actual location. Existing interrupted directories remain in place;
    all 504 task IDs and their archive/recovery paths stay unchanged. The normal recovery snapshot is
    still verified on Drive, and every update keeps its original durable accounting write/readback.
    Storage headroom is checked on the actual destination. Do not manually delete work-folder symlinks.
    `WAIT` messages explain any unfilled slots: memory, storage or matched-baseline dependencies.
    Startup verification, archive I/O, stopping and the final few tasks can also leave slots unfilled.

    Before training, `STARTUP` messages show the verification phase, checked/total/verified folder counts,
    current archive and filename. Each finished folder is reported. Every 30 seconds a heartbeat shows
    `PROGRESS` if work advanced or `STILL WAITING` otherwise, with time since the last progress event.
    The GPU may remain unused until verification finishes; a blocked Drive read cannot be bypassed.
    The same status is saved locally at `/content/hippo_msd_bundle/parallel_status/startup.json`.
    Run/recovery directories are listed once to avoid probing hundreds of absent Drive folders.
    Each archive file is copied once to a temporary local snapshot; the unchanged verifier checks all
    hashes, byte counts, readability, task identity, source binding, acknowledgment and cleanup locally.
    Metrics and memory profiles use the same verified bytes. The local snapshot is removed immediately
    after use; Drive files are never modified by verification. Every restart checks fresh Drive bytes.

    During training, every minute the notebook prints completed count, active workers, GPU activity, memory and charged
    update counts. Ledger-write/readback time, checkpoint time and memory peaks are archived per run.
    Existing completed archives are never rewritten. No extra scientific runs or tuning trials are added.

    **To stop:** interrupt once and wait for every worker's recovery upload. A worker failure stops new
    launches and asks the others to save. Hard VM loss still requires an audit if charged updates are
    newer than durable checkpoints; accounting is never reset to hide lost work.

    `SESSION_HOURS` counts from starting this cell. Set it below your **remaining** runtime allowance,
    including time already spent in the previous notebook. The default is 8 more hours for this handover;
    on a newly allocated Pro+ runtime you may use 22. Colab can terminate earlier. Existing STOP files are respected.
    ''').strip()+'\n'
    cells[10]['source'] = textwrap.dedent('''
    MAX_NEW_RUNS = 504  # Upper bound on remaining tasks launched this session, not extra repeats.
    MAX_WORKERS = 0    # Fill the CPU-supported maximum immediately; no solo profiling.
    SESSION_HOURS = 8.0  # Remaining time for this session; fresh runtime may use 22.0.
    campaign('run', '--max-runs', MAX_NEW_RUNS, '--hours', SESSION_HOURS, '--workers', MAX_WORKERS)
    ''').strip()+'\n'
    cells[13]['source'] += '\nScheduler provenance, preservation inventory and minute-by-minute timing are under `results_noaug_v1/parallel/`.\n'
    book['metadata']['colab']['name'] = NOTEBOOK
    for cell in cells:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    return book


def prepare():
    EXPORT.mkdir(parents=True, exist_ok=True)
    control = EXPORT/'control'
    control.mkdir(exist_ok=True)
    before = digest(ORIGINAL/'bundle/bundle_manifest.json')
    summary = read(ORIGINAL/'export_summary.json')
    if before != summary['bundle_manifest_sha256']:
        raise ValueError('Original science bundle changed')
    files = ('scheduler.py', 'worker.py', 'local_storage.py', 'archive_verification.py')
    for name in files:
        shutil.copyfile(Path(__file__).parent/name, control/name)
    write(control/'control_manifest.json', dict(scientific_bundle_sha256=before,
        files={name:digest(control/name) for name in files},
        compatible_profile_worker_sha256=[digest(control/'worker.py'),
            '07986a0e0adffae283abd578c7c29f250a27b8c7373546acb8f73ec46cc0ae1c'],
        scope='Maximum worker occupancy without solo profiling or throughput backoff; single-read verification and NVMe; frozen science unchanged'))
    archive = EXPORT/CONTROL_ZIP
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(control.iterdir()):
            z.write(path, path.name)
    book = notebook(summary['code_zip_sha256'], summary['data_zip_sha256'], digest(archive), before)
    write(EXPORT/NOTEBOOK, book)
    write(EXPORT/'export_summary.json', dict(status='locally_tested_gpu_throughput_not_yet_measured',
        total_runs=504, extra_scientific_runs=0, augmentation='none', scientific_bundle_sha256=before,
        control_zip_sha256=digest(archive), original_code_zip_sha256=summary['code_zip_sha256'],
        data_zip_sha256=summary['data_zip_sha256'], results_directory='results_noaug_v1',
        same_preflight_reused=True, workers='Fill CPU-supported target immediately, subject to memory/storage and matched-baseline dependencies',
        solo_memory_profiling=False, throughput_backoff=False,
        scientific_loss_calibration_changed=False, unknown_gpu_allowance_gib=12, unknown_ram_allowance_gib=8,
        storage='New runs use a separately mounted /mnt/local-scratch; existing directories remain in place',
        checkpoint_format_changed=False, checkpoint_cadence_changed=False,
        durable_accounting_changed=False, compatible_prior_memory_profiles=True,
        archive_verification='One fresh Drive read per archive file; unchanged verifier and parsing on local snapshot',
        archive_checks_bypassed=False, cross_session_archive_cache=False, report_uses_local_verification=True))
    assert digest(ORIGINAL/'bundle/bundle_manifest.json') == before
    print(EXPORT)


if __name__ == '__main__':
    prepare()
