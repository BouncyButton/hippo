"""Package a full notebook and an external scheduler without refreezing science."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

from scripts.colab_adni.prepare import notebook as original_notebook
from scripts.colab_adni.runtime import digest, read, write

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'reports/adni_colab_3fold_3seed_noaug_20261015'
PREVIOUS = [ROOT/'reports/adni_colab_visible_adaptive_3fold_3seed_noaug_20261015',
            ROOT/'reports/adni_colab_worker_refill_3fold_3seed_noaug_20261015',
            ROOT/'reports/adni_colab_max_workers_3fold_3seed_noaug_20261015']
EXPORT = ROOT/'reports/adni_colab_validated_acceleration_3fold_3seed_noaug_20261015'
EVIDENCE = ROOT/'reports/adni_colab_acceleration_validation_20261015/returned_a100_validation.zip'


def notebook(code_sha, data_sha, control_sha, scientific_sha):
    book = deepcopy(original_notebook(code_sha, data_sha))
    cells = book['cells']
    cells[0]['source'] = cells[0]['source'].replace('ADNI replication —', 'ADNI maximum-slot parallel replication —').replace(
        '**504 fresh runs:**', '**504 total runs, preserving completed and interrupted work:**')
    cells[0]['source'] = cells[0]['source'].replace(
        'The single-fold/seed cluster pilot is provenance; all 504 Colab runs are fresh.',
        'The cluster pilot remains provenance. Completed Colab runs in the existing results folder are reused.')
    start = cells[0]['source'].index('**Before connecting a GPU:**')
    end = cells[0]['source'].index('Run cells in order.', start)
    cells[0]['source'] = (cells[0]['source'][:start] +
        'Keep your existing `adni_data_verified.zip`, `adni_replication_code.zip` and results on Drive. '
        'No dataset re-upload is needed. Use the same **A100-SXM4-80GB** and software environment '
        'that passed your validation; the evidence gate checks the exact versions.\n\n' +
        cells[0]['source'][end:])
    cells[0]['source'] += textwrap.dedent('''

    **Switching from the existing notebook:** interrupt its run cell once and wait until checkpoint
    uploads finish. Keep the same A100 type, environment and results directory. Upload only
    `adni_validated_acceleration_control_v5.zip` beside the two original ZIPs; open this notebook and run its cells in order.
    The original science bundle is unchanged. Your returned A100 equivalence checks passed for all four
    models. This package enables exactly that tested acceleration code. Verified results are skipped, recovery resumes first,
    and the existing preflight is required and reused, with a separate read-only acceleration evidence gate. Reuse the current runtime when possible. If Colab assigns a
    fresh A100 runtime, verified recovery is restored from Drive. Never delete the old runtime before
    its clean stop and recovery uploads finish; do not leave two sweeps running against these results.
    **Do not switch while the old run is active.** This package does not stop it for you.
    No dataset upload is needed; leave the existing data/code ZIPs and results exactly where they are.
    Setup verifies an existing local bundle in place; on a fresh VM it stages the same existing ZIPs.
    Local process locks and a persistent Drive ownership marker reject visible competing controllers.
    After an unclean VM loss, a leftover ownership marker needs inspection; it is never expired or
    deleted automatically. Drive mounts are not a distributed lock service: use one Colab runtime
    for this results folder, and never launch controllers simultaneously from separate VMs.

    `MAX_WORKERS=0` targets the CPU-supported maximum immediately: two threads per worker and
    two CPU cores reserved for orchestration. Slots fill as soon as memory, storage and baseline
    dependencies permit, with no solo memory runs or throughput ramp. The target does not fall
    when GPU utilization or measured throughput changes. Actual occupancy can still be lower.
    New checkpoints and recovery staging use a separately mounted `/mnt/local-scratch` when
    present; existing run directories/checkpoints remain where they are. All durable charges and
    recovery uploads remain unchanged. Exact GPU equivalence is verified on your A100; total sweep
    speedup and maximum-worker throughput have not yet been measured.
    ''')
    cells[1]['source'] = '## 1. Mount Drive and stage the unchanged science bundle plus scheduler\n'
    cells[2]['source'] = textwrap.dedent(f'''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import fcntl, hashlib, json, os, shutil, signal, subprocess, sys, zipfile

    DRIVE_ROOT = Path('/content/drive/MyDrive/hippo_adni_3fold_3seed')
    INPUT = DRIVE_ROOT/'input'
    RESULTS = DRIVE_ROOT/'results_noaug_v1'  # Preserve this exact existing results folder.
    BUNDLE = Path('/content/hippo_adni_bundle')
    CONTROL = Path('/content/hippo_adni_adaptive')
    SCIENTIFIC_SHA = '{scientific_sha}'
    EXPECTED = {{'adni_data_verified.zip':'{data_sha}', 'adni_replication_code.zip':'{code_sha}'}}
    CONTROL_SHA = '{control_sha}'
    def sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
        return h.hexdigest()

    BUNDLE.mkdir(exist_ok=True)
    (BUNDLE/'work').mkdir(exist_ok=True)
    with visible_phase('open existing Drive results folder'):
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
            with visible_phase('hash-check frozen local science files'):
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
        unpack_verified(INPUT/'adni_validated_acceleration_control_v5.zip', CONTROL_SHA, CONTROL)
    print('No augmentation. All input data stays on local SSD; results stay in', RESULTS)
    ''').strip()+'\n'
    phase_helper = textwrap.dedent(r'''
    from contextlib import contextmanager
    import json, threading, time
    from pathlib import Path
    @contextmanager
    def visible_phase(label):
        print('PHASE:', label, flush=True)
        stop = threading.Event()
        start = time.monotonic()
        path = Path('/content/hippo_adni_setup_status.json')
        def heartbeat():
            while not stop.wait(30):
                value = dict(phase=label, elapsed_seconds=time.monotonic()-start,
                             status='still waiting; this heartbeat is not evidence of progress')
                path.write_text(json.dumps(value))
                print('HEARTBEAT:', label, 'still waiting', round(value['elapsed_seconds']), 'seconds', flush=True)
        path.write_text(json.dumps(dict(phase=label, status='started')))
        thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
        try:
            yield
        finally:
            stop.set(); thread.join()
            path.write_text(json.dumps(dict(phase=label, status='phase ended', elapsed_seconds=time.monotonic()-start)))
    ''').strip()+'\n'
    stage = cells[2]['source']
    stage = stage.replace("drive.mount('/content/drive')", "with visible_phase('mount Google Drive'):\n    drive.mount('/content/drive')")
    stage = stage.replace("    assert sha256(src) == expected, f'Upload hash mismatch: {src.name}'\n    local =", "    local =")
    stage = stage.replace("    shutil.copyfile(src, local)\n    assert sha256(local) == expected", "    with visible_phase('stage and hash-check ' + src.name):\n        shutil.copyfile(src, local)\n        assert sha256(local) == expected, f'Upload hash mismatch: {src.name}'")
    # One read from Drive then local hashing; the pinned ZIP hash still verifies every byte.
    stage = stage.replace("            for name, expected", "            for name, expected")
    stage = stage.replace("    if manifest.exists():", "    print('PHASE: verify existing local bundle or stage original uploads', flush=True)\n    if manifest.exists():")
    cells[2]['source'] = phase_helper + stage
    cells[4]['source'] = "with visible_phase('authenticate and check Drive quota'):\n" + textwrap.indent(cells[4]['source'], '    ')
    cells[3]['source'] += '\nConcurrency reserves 8 GiB of Drive headroom per active worker and 12 GiB of local SSD per worker.\n'
    setup = cells[6]['source']
    setup = setup.replace("args = [str(PYTHON), '-u', str(BUNDLE/'runtime.py'), command, '--bundle', str(BUNDLE), '--durable', str(RESULTS), *map(str, extra)]",
        "args = ([str(PYTHON), '-u', str(CONTROL/'scheduler.py'), command, '--bundle', str(BUNDLE), '--durable', str(RESULTS), *map(str, extra)]\n"
        "            if command in ('run', 'report') else [str(PYTHON), '-u', str(BUNDLE/'runtime.py'), command, '--bundle', str(BUNDLE), '--durable', str(RESULTS), *map(str, extra)])")
    setup = setup.replace('import urllib.request', "import urllib.request\n        print('PHASE: bootstrap pip from PyPA', flush=True)")
    setup = setup.replace("    run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')",
        "    with visible_phase('campaign ' + command):\n        run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')" )
    # The scheduler has its own detailed heartbeat; avoid a second generic run heartbeat.
    setup = setup.replace("    with visible_phase('campaign ' + command):\n        run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')",
        "    if command == 'run':\n        run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')\n    else:\n        with visible_phase('campaign ' + command):\n            run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')")
    setup = setup.replace("    with log_path.open('a', encoding='utf-8') as log:",
        "    from contextlib import nullcontext\n    context = nullcontext() if str(CONTROL/'scheduler.py') in args else visible_phase('subprocess setup or verification')\n    with context, log_path.open('a', encoding='utf-8') as log:")
    setup = setup.replace("    if command == 'run':\n        run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')\n    else:\n        with visible_phase('campaign ' + command):\n            run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')",
        "    if command == 'report':\n        args += ['--max-runs', '0']\n    run_setup_logged(args, Path('/content')/f'hippo_adni_{command}.log')")
    setup = setup.replace('if result and not interrupted:', 'if result:')
    setup = setup.replace("    if command == 'report':",
        "    if command == 'preflight':\n        args = [str(PYTHON), '-u', str(CONTROL/'acceleration_gate.py'), '--bundle', str(BUNDLE), '--durable', str(RESULTS)]\n    if command == 'report':")
    cells[6]['source'] = setup
    cells[7]['source'] = textwrap.dedent('''
    ## 4. Reuse the existing preflight and verify your A100 acceleration evidence

    This cell reads the original passed preflight and environment binding, verifies the unchanged
    scientific bundle, and checks the exact original/candidate equivalence evidence you returned
    for all four models. It does not rerun GPU diagnostics or add training/benchmarks.
    The binary ADNI loss graphs, saved subject splits and all scientific settings remain unchanged.
    A GPU/software mismatch blocks execution without changing receipts or accounting.
    ''').strip()+'\n'
    cells[9]['source'] = textwrap.dedent('''
    ## 5. Run or resume with maximum admissible workers

    `MAX_WORKERS = 0` immediately targets `(available_CPU_cores - 2) // 2` workers, at least one.
    A positive value sets a lower ceiling. GPU utilization and throughput are observations only:
    no gradual ramp, mandatory solo memory run, or automatic target reduction is applied.
    Eligible slots refill after each completed worker's archive is verified and cleanup is recorded.
    Unknown conditions receive provisional allowances of at least 12 GiB VRAM and 8 GiB RAM,
    enlarged by same-model ADNI measurements and live worker peaks. These are conservative
    reservations, not validated upper bounds. Known conditions reserve twice the recorded GPU
    peak plus 2 GiB, and 1.5 times the RAM peak plus 1 GiB (minimum 4 GiB each).
    Every active worker's future growth is reserved, with an additional 8 GiB/15% GPU margin,
    4 GiB system RAM, 8 GiB Drive space per worker, and 12 GiB per worker on each occupied disk.
    Compatible, hash-verified ADNI observer records are reused across controller versions. Old
    unaccelerated records gain 1 GiB of cache allowance and keep the provisional 12/8 GiB floor
    until a complete accelerated scientific run supplies its own measured peaks. Diagnostic
    memory peaks are never used as full-run profiles.
    No other dataset's memory measurements are used. Every unfilled slot has a visible reason:
    resource headroom, matched-baseline dependency, archiving/recovery, remaining tasks or stopping.

    `/mnt/local-scratch` must be a real separate mount to be selected. Only missing run directories
    are created there, with their original logical paths retained. Existing work is never migrated.
    Recovery hardlinks and tar staging use the physical checkpoint filesystem. If the separate
    disk is unavailable, the controller clearly falls back to the runtime disk. Ownership markers,
    the shared local lock and the Drive controller lock remain mandatory.

    Startup and the report cell list saved folders once, read each required archive file once into
    a fresh disposable local snapshot, and run all unchanged hash, size, readability, identity,
    source-binding, acknowledgment and cleanup checks locally. Results and memory observations
    come from those verified bytes. The snapshots are removed, and a failed verification leaves
    the previous report intact. Baseline proof bytes are retained only in memory for this session
    and rechecked locally before the frozen initialization restoration gate.

    The exact candidate you validated on the A100 is installed after the original binary recipe
    is configured: preprocessed images/labels remain on GPU; immutable SDF/depth annotations
    are reused; evaluation keeps identical batch size, operations and case order but defers host
    metric readback; unchanged best-checkpoint writes are skipped. Latest checkpoints, durable
    ledger writes/readback and recovery cadence remain intact. The frozen scientific source and
    protocols are untouched; separate execution receipts record these external hooks honestly.
    All four model diagnostics matched original losses, gradients, calibration, optimizer/model
    state, binary metrics, RNG and checkpoint replay exactly. This evidence covers fold 0/seed 0
    with all four auxiliary terms; it is not a full-sweep benchmark. No tolerances were relaxed.
    The original preflight is still required. The additional evidence gate binds the exact candidate,
    frozen bundle, A100-SXM4-80GB and package/CUDA/cuDNN versions from your report. A mismatch
    blocks execution and preserves your data; it never resets the binding or substitutes new trials.

    Existing interrupted tasks are admitted first, then fresh baselines and constraints whose own
    baseline is already verified. Other eligible tasks can fill spare slots while resumed work runs.
    Before each constrained run, its archived baseline initialization is verified and restored locally
    for the frozen ADNI model builder. Each subject's left/right crops stay in the same saved split.
    Each task still has its own original seeds, batch size 1, BF16, calibration, losses, optimizer,
    update schedule, checkpoints and 31,976,450,667,090 optimization-FLOP cap. No augmentation.
    Training can remain partly idle when Drive is the bottleneck; do not equate VRAM occupancy with speed.

    Every 30 seconds a local observer prints the current phase, checked/verified archive counts,
    completed count, active run IDs, successful and charged update counts, worker count, GPU activity,
    VRAM and CPU/RAM. A heartbeat explicitly says when there is **no new progress**. Status continues
    during a blocked Drive request; it does not mean Drive is healthy. Worker phases distinguish
    initialization, loading, calibration, training, evaluation, checkpointing and waiting for Drive.
    Timing covers data loading, evaluation, ledger writes/readback, checkpoints and recovery uploads.
    These are wall-clock timings; overlapping upload time is not additive and no GPU synchronizations
    or profiling trials are inserted. GPU utilization measures activity; VRAM measures capacity.
    Local live files are `/content/hippo_adni_bundle/parallel_status/<session>/status.json`
    and `timing.jsonl`; worker statuses are in the parent folder. Per-run measurements are archived,
    and session telemetry is copied to Drive at a clean stop. A hard VM loss can lose local telemetry,
    but never relaxes the original durable scientific-accounting requirements.
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
    MAX_WORKERS = 0    # Immediately target CPU-supported maximum; resource checks gate occupancy.
    SESSION_HOURS = 8.0  # Remaining time for this session; fresh runtime may use 22.0.
    campaign('run', '--max-runs', MAX_NEW_RUNS, '--hours', SESSION_HOURS, '--workers', MAX_WORKERS)
    ''').strip()+'\n'
    cells[13]['source'] += '\nScheduler provenance, preservation inventory and session timing are under `results_noaug_v1/parallel/`.\n'
    cells.extend([
        dict(cell_type='markdown', metadata={}, source=textwrap.dedent("""
        ## 8. Export validation evidence without extra runs

        After a normal run cell finishes or stops cleanly, run this section and send the downloaded
        `adni_colab_validation_v5.zip` back in chat. It contains only existing local controller logs,
        worker progress, memory/timing records, environment binding and storage placement details.
        It launches no jobs, calls no model, reads no Drive files and does not alter results or accounting.
        This validates observed scheduling/storage behavior; it does not certify new GPU acceleration.
        If training failed, you can still run this section to collect the local diagnostic evidence.
        """).strip()+'\n'),
        dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=textwrap.dedent("""
        VALIDATION_ZIP = Path('/content/adni_colab_validation_v5.zip')
        subprocess.run([str(PYTHON), str(CONTROL/'validation.py'), '--bundle', str(BUNDLE),
                        '--control', str(CONTROL), '--output', str(VALIDATION_ZIP),
                        '--log', '/content/hippo_adni_run.log'], check=True)
        from google.colab import files
        files.download(str(VALIDATION_ZIP))
        """).strip()+'\n')])
    book['metadata']['colab']['name'] = 'ADNI_VALIDATED_ACCELERATION_v5_3fold_3seed_NO_AUGMENTATION.ipynb'
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
    names = ('scheduler.py', 'worker.py', 'telemetry.py', 'local_storage.py',
             'archive_verification.py', 'validation.py', 'acceleration_candidate.py', 'acceleration_gate.py')
    for name in names:
        shutil.copyfile(Path(__file__).parent/name, control/name)
    shutil.copyfile(EVIDENCE, control/'acceleration_evidence.zip')
    with zipfile.ZipFile(EVIDENCE) as evidence:
        report = json.loads(evidence.read('summary.json'))
    acceleration = dict(evidence_file='acceleration_evidence.zip', evidence_sha256=digest(EVIDENCE),
        candidate_sha256=report['candidate_sha256'], validator_sha256=report['validator_sha256'], models=report['models'])
    compatible, workers = [], {digest(control/'worker.py')}
    for previous_export in PREVIOUS:
        previous_manifest = previous_export/'control/control_manifest.json'
        if not previous_manifest.exists():
            continue
        previous = read(previous_manifest)
        if previous['scientific_bundle_sha256'] != before:
            raise ValueError('Prior observer belongs to another scientific bundle')
        for name, expected in previous['files'].items():
            if digest(previous_manifest.parent/name) != expected:
                raise ValueError('Previous control package changed; cannot reuse its memory profiles')
        compatible.append(digest(previous_manifest))
        # Historical unaccelerated profiles are hints with extra cache allowance,
        # never labeled as measurements of the accelerated full run.
        workers.add(previous['files']['worker.py'])
    write(control/'control_manifest.json', dict(scientific_bundle_sha256=before,
        compatible_profile_control_sha256=compatible,
        compatible_profile_worker_sha256=sorted(workers),
        files={name:digest(control/name) for name in (*names, 'acceleration_evidence.zip')},
        acceleration=acceleration,
        scope='Validated external acceleration and scheduling/storage/observation; frozen science and 504 tasks unchanged'))
    from scripts.colab_adni_parallel.acceleration_gate import verify
    receipt = verify(control, ORIGINAL/'bundle', report['binding'], dict(status='passed', **report['binding']))
    write(EXPORT/'accepted_acceleration_evidence.json', receipt)
    archive = EXPORT/'adni_validated_acceleration_control_v5.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(control.iterdir()):
            z.write(path, path.name)
    book = notebook(summary['code_zip_sha256'], summary['data_zip_sha256'], digest(archive), before)
    write(EXPORT/'ADNI_VALIDATED_ACCELERATION_v5_3fold_3seed_NO_AUGMENTATION.ipynb', book)
    write(EXPORT/'export_summary.json', dict(status='A100_equivalence_passed_packaged_pending_final_local_validation',
        total_runs=504, extra_scientific_runs=0, augmentation='none', scientific_bundle_sha256=before,
        control_zip_sha256=digest(archive), original_code_zip_sha256=summary['code_zip_sha256'],
        data_zip_sha256=summary['data_zip_sha256'], results_directory='results_noaug_v1',
        same_preflight_reused=True, acceleration_evidence_sha256=digest(EVIDENCE),
        gpu_equivalence='exact four-model fold0/seed0 checks passed on A100-SXM4-80GB; no full throughput benchmark',
        workers='immediate CPU-supported target; occupancy gated by memory/storage/dependencies',
        solo_memory_profiling=False, throughput_ramp=False, new_work_storage='separate mounted local-scratch if available',
        validation_export='existing local observations only; no new scientific runs'))
    assert digest(ORIGINAL/'bundle/bundle_manifest.json') == before
    print(EXPORT)


if __name__ == '__main__':
    prepare()
