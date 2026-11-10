"""Publish a complete parallel notebook without regenerating scientific files."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import zipfile

from scripts.colab_medsam3.parallel import FROZEN_MANIFEST
from scripts.colab_medsam3.stage_parallel import stage_parallel

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015'
EXPORT = ROOT/'reports/medsam3_msd_colab_108_parallel_20261015'
SOURCE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def notebook(uploads, package_sha):
    cells = []
    def cell(kind, value):
        value = textwrap.dedent(value).strip()+'\n'
        item = dict(cell_type=kind, metadata={}, source=value, id=f'parallel_{len(cells):02d}')
        if kind == 'code':
            ast.parse(value)
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '''
    # MSD MedSAM3 — resume the same 108 runs with adaptive concurrency

    **No augmentation. Folds 0, 1, 3; seeds 17, 83, 191. Original 200 updates per run,
    accumulation 2, native loss, medical pretrained LoRA, 10% target / 50% calibration p95 cap.**
    This is the same study, not an additional 108 runs. Bands and bands+edge are still excluded.
    The 27 historical results remain reference only. This study is update matched, not FLOP capped.

    The scientific bundle is byte-for-byte unchanged. This notebook keeps the exact existing
    **results_matched200_noaug_108_v1** folder, completed archives, calibration, preflight and checkpoints.
    Each worker runs the original engine in a separate process with its own task, RNG and optimizer.
    Only scheduling changes. Production CUDA numerical variability remains; concurrency is not a
    guarantee of bitwise agreement with serial execution.

    **Before switching:** interrupt the old Run cell once and wait until its clean-stop upload
    finishes. Keep this VM connected. Do not delete the runtime or its local work folder.
    Then open this notebook on that same runtime and run the cells below in order. If opening a
    notebook gives you a different VM, first ensure the old cell has finished stopping: recovery
    will then come from Drive. Use only one VM against this results folder.

    Upload only **medsam3_msd_108_parallel_code.zip** to the existing `input` folder.
    Keep the existing `medsam3_msd_data.zip` and `sam3.pt`; no new dataset or checkpoint upload is needed.
    ''')
    cell('markdown', '## 1. Connect to the same Drive results')
    cell('code', '''
    from google.colab import drive, auth
    drive.mount('/content/drive')
    auth.authenticate_user()
    from pathlib import Path
    import hashlib, json, os, signal, subprocess, sys, time
    NOTEBOOK_STARTED = time.monotonic()
    HOME_DIR = Path('/content/drive/MyDrive/hippo_medsam3_msd_extension')
    INPUT = HOME_DIR/'input'
    RESULTS = HOME_DIR/'results_matched200_noaug_108_v1'
    BUNDLE = Path('/content/hippo_medsam3_msd_108_bundle')
    CONTROLLER = Path('/content/hippo_medsam3_parallel_v1')
    RESULTS.mkdir(parents=True, exist_ok=True)
    print('Preserving and resuming:', RESULTS)
    ''')
    cell('markdown', '''
    ## 2. Verify and stage without replacing work
    The existing scheduler lock must be free. Unchanged files are reused; differing files cause
    a stop instead of replacement. Existing `work/`, recovery, archives and source bindings stay
    intact. On a new VM only, this restores code and data from the existing uploads.
    ''')
    cell('code', inspect.getsource(stage_parallel)+f'''\n
stage_parallel(INPUT/'medsam3_msd_108_parallel_code.zip', INPUT/'medsam3_msd_data.zip', INPUT/'sam3.pt',
               BUNDLE, CONTROLLER, RESULTS, {package_sha!r}, {FROZEN_MANIFEST!r},
               {uploads['medsam3_msd_data.zip']!r}, {uploads['sam3.pt']!r})
''')
    cell('markdown', '''
    ## 3. Reuse the original environment and define notebook commands
    An existing matching environment is reused without installation. A new VM installs the same
    frozen requirements. Changing GPU/software after scientific work is blocked by the original
    environment binding. Select the same A100 runtime type and High RAM.
    ''')
    cell('code', '''
    VENV = Path('/content/hippo_medsam3_venv')
    PYTHON = VENV/'bin/python'
    def setup(args):
        result = subprocess.run(list(map(str,args)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with (RESULTS/'parallel_setup.log').open('a') as log: log.write(result.stdout)
        print('\\n'.join(result.stdout.splitlines()[-35:]))
        if result.returncode: raise RuntimeError('Environment setup failed; see parallel_setup.log')
    if not PYTHON.exists():
        setup([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV])
    probe = subprocess.run([str(PYTHON),'-m','pip','--version'], capture_output=True, text=True)
    if probe.returncode:
        import urllib.request
        bootstrap = Path('/content/medsam3_get_pip.py')
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', bootstrap)
        setup([PYTHON,bootstrap])
    dependency_probe = """
    import importlib.metadata as m, pathlib, sys
    for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
        if not line.strip(): continue
        parts = line.strip().split('==')
        try: version = m.version(parts[0])
        except m.PackageNotFoundError: sys.exit(1)
        if len(parts) == 2 and version != parts[1]: sys.exit(1)
    """
    ready = subprocess.run([str(PYTHON),'-c',dependency_probe,str(BUNDLE/'requirements-colab.txt')]).returncode == 0
    if not ready:
        setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'])
    else:
        print('Reusing the existing dependency versions.')
    def execute(args, log_name, parallel=False):
        process = subprocess.Popen(list(map(str,args)), start_new_session=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        interrupted = False
        with (RESULTS/log_name).open('a') as log:
            while True:
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True); log.write(line); log.flush()
                    code = process.wait()
                    break
                except KeyboardInterrupt:
                    interrupted = True
                    print('Clean stop requested. Keep this VM connected until every worker verifies its Drive recovery.')
                    if process.poll() is None:
                        if parallel: process.send_signal(signal.SIGUSR1)
                        else: os.killpg(process.pid, signal.SIGINT)
        if code: raise RuntimeError(f'Command failed. See {RESULTS/log_name}. Existing work is retained.')
        return not interrupted
    def campaign(command, *extra):
        return execute([PYTHON,BUNDLE/'runtime.py',command,'--bundle',BUNDLE,'--durable',RESULTS,*extra], command+'.log')
    campaign('verify')
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    cell('markdown', '''
    ## 4. Reuse the passed preflight
    The original command verifies the same bundle and environment. A matching passed preflight
    returns immediately without repeating calibration or GPU diagnostics. An incomplete preflight
    resumes only its missing work. Scientific runs cannot bypass this check.
    ''')
    cell('code', "campaign('preflight')")
    cell('markdown', '''
    ## 5. Run or resume with automatic worker selection
    `MAX_WORKERS = 0` means automatic; it does **not** mean two workers. The supervisor starts with
    one pending experiment, measures progress, then tries additional workers within GPU, CPU, RAM
    and storage reserves. On an 80 GB A100 it can consider more than two workers; the actual useful
    number is measured on this runtime. No extra benchmark experiments are created.

    It compares stable training/evaluation throughput windows and stops increasing when a measured
    gain is too small. Mixed phases use GPU activity and headroom to decide whether to try another
    worker. Startup is staggered; reduced limits drain running work instead of discarding updates.
    Resource estimates are conservative, not a guarantee against OOM. Memory pressure stops workers
    cleanly and records a lower ceiling for the next resume. A failed update is retained for recovery
    audit instead of replayed silently.

    Saved unfinished baselines resume first; all nine baselines complete before new constraint runs.
    Every 30 seconds, output shows **GPU usage, worker count, update/200 and evaluated volumes**.
    Full logs stay in `BUNDLE/work/<experiment>/train.log`. Status and tuning decisions are also
    written to Drive (`parallel_status.json`, `parallel_sessions/`).

    Interrupt this cell **once** to stop all workers. Wait until it returns: local checkpoints are
    saved every update, periodic Drive recovery continues, and a clean stop verifies every final
    recovery upload. Partly completed evaluation and logs are also saved on a clean stop. Completed
    archives are verified and skipped; completed bulky recovery files are retired by the original code.
    A manual `RESULTS/STOP` is respected and never removed automatically.

    The 22-hour setting includes time spent in this notebook's setup cells. It does not know time
    spent in a previous notebook: lower it accordingly if this VM is already old. Colab can stop
    earlier; a hard VM loss with newer charged updates than the saved recovery requires an audit.
    Leave this VM running until the cell has confirmed clean stops; do not start another notebook
    or VM writing the same results concurrently. On resume, run this cell again on the same VM;
    on a new VM, run all setup cells first.
    ''')
    cell('code', '''
    MAX_WORKERS = 0  # Automatic: resource and throughput limited, not fixed at two.
    MAX_NEW_RUNS = 108  # Upper bound on starts; already completed experiments are skipped.
    SESSION_HOURS = 22.0
    remaining_hours = SESSION_HOURS - (time.monotonic()-NOTEBOOK_STARTED)/3600
    if remaining_hours <= 0.1:
        raise RuntimeError('Notebook session budget reached. Resume in a fresh runtime after clean shutdown.')
    execute([PYTHON, CONTROLLER/'parallel.py', '--bundle', BUNDLE, '--durable', RESULTS,
             '--max-workers', str(MAX_WORKERS), '--max-runs', str(MAX_NEW_RUNS),
             '--hours', str(remaining_hours)], 'parallel_run.log', parallel=True)
    ''')
    cell('markdown', '''
    ## 6. Results with fresh matched controls
    Same original comparison: validation Dice and gains against the fresh baseline for each fold
    and seed. Historical bands/bands+edge runs remain reference only. No conditions are added.
    ''')
    cell('code', '''
    campaign('report')
    import pandas as pd
    comparison = json.loads((RESULTS/'comparison.json').read_text())
    display(pd.DataFrame(comparison['aggregate']))
    display(pd.DataFrame(comparison['runs']))
    ''')
    return dict(cells=cells, metadata=dict(colab=dict(name='MedSAM3_MSD_108_adaptive_parallel.ipynb'),
        kernelspec=dict(display_name='Python 3', language='python', name='python3'), accelerator='GPU'), nbformat=4, nbformat_minor=5)


def main():
    EXPORT.mkdir(parents=True, exist_ok=True)
    transfer = json.loads((ORIGINAL/'transfer_manifest.json').read_text())
    original_zip = ORIGINAL/'medsam3_msd_108_code.zip'
    if digest(original_zip) != transfer['uploads'][original_zip.name]:
        raise ValueError('Original code ZIP changed')
    payload = {}
    with zipfile.ZipFile(original_zip) as archive:
        for name in archive.namelist():
            payload['bundle/'+name] = archive.read(name)
    payload['controller/parallel.py'] = (SOURCE/'parallel.py').read_bytes()
    members = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
    payload['package_manifest.json'] = (json.dumps(members, indent=2)+'\n').encode()
    target = EXPORT/'medsam3_msd_108_parallel_code.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(payload.items()):
            archive.writestr(name, value)
    sha = digest(target)
    nb = EXPORT/'MedSAM3_MSD_108_adaptive_parallel.ipynb'
    nb.write_text(json.dumps(notebook(transfer['uploads'], sha), indent=1)+'\n')
    (EXPORT/'transfer_manifest.json').write_text(json.dumps(dict(
        code_zip=target.name, code_sha256=sha, notebook=nb.name, notebook_sha256=digest(nb),
        original_bundle_sha256=FROZEN_MANIFEST, original_scientific_files_identical=True,
        same_results_folder='results_matched200_noaug_108_v1',
        total_experiments=108, added_experiments=0, augmentation='none', updates=200,
        max_workers_default='automatic', data_uploads_unchanged={k:v for k,v in transfer['uploads'].items()
            if k in ('medsam3_msd_data.zip', 'sam3.pt')}), indent=2)+'\n')
    (EXPORT/'UPLOAD_AND_RESUME.txt').write_text(
        '1. Interrupt the OLD Run cell once and WAIT for its clean-stop recovery upload. Keep the VM.\n'
        '2. Upload medsam3_msd_108_parallel_code.zip to the existing Drive hippo_medsam3_msd_extension/input folder.\n'
        '3. Open MedSAM3_MSD_108_adaptive_parallel.ipynb on that same runtime and run its cells in order.\n'
        '4. Use the existing results_matched200_noaug_108_v1 folder. Do not delete work, recovery, STOP or results.\n'
        '5. MAX_WORKERS=0 automatically tries additional workers within measured resource/throughput limits.\n'
        'No new data/checkpoint upload, no extra experiments, no augmentation, no scientific source changes.\n'
        'GPU speedup has not been measured locally; it is evaluated from real pending work on Colab.\n')
    print(json.dumps(dict(notebook=str(nb), code=str(target), code_sha256=sha, original_bundle_sha256=FROZEN_MANIFEST), indent=2))


if __name__ == '__main__':
    main()
