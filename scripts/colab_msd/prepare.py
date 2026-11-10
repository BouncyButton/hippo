"""Export only the authorized MSD 3-fold/3-seed replication and a Colab notebook."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

from scripts.colab_msd.runtime import digest, read, write, verify_bundle
from scripts.cross_dataset_campaign.common import ARMS, MODELS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015'
PARENT = ROOT/'reports/msd_mixed_losses_nnunet60_others10_20261014/frozen'
NAME = 'msd_colab_3fold_3seed_noaug_20261015'


def notebook(code_sha, data_sha):
    cells = []
    def md(source):
        cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(source).strip()+'\n'))
    def code(source):
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                          source=textwrap.dedent(source).strip()+'\n'))
    md('''
    # MSD replication — 3 folds × 3 seeds, no augmentation

    **504 fresh runs:** four models × 14 conditions × folds 0/1/2 × seeds 0/1/2.
    Each fold retains its saved 5% training subset (10 volumes) and 52 validation volumes.
    The completed cluster pilot is provenance, not silently counted as a Colab replication.

    - **No augmentation:** no random flips, rotations, intensity jitter, or test-time augmentation.
      The existing deterministic normalization/padding stays unchanged.
    - nnUNet: native foreground Dice+CE, original deep supervision and SGD; 60% auxiliary targets, 75% p95 ceiling.
    - SwinUNETR, UNETR and 3D U-Net: established Dice including background and AdamW; 10% targets, 50% p95 ceiling.
    - Preserve BF16 precision, batch size 1, initialization, LR schedule, calibration, SDF timing and the
      31,976,450,667,090 optimization-FLOP cap per run. Calibration and preflight compute are separate.

    **Before connecting a GPU:** upload `msd_data_verified.zip` and `msd_replication_code.zip` to
    `My Drive/hippo_msd_3fold_3seed/input/`. Keep both zipped. No SSH keys or cluster access are needed.
    Then choose **Runtime → Change runtime type → GPU**. Use a native-BF16 GPU such as H100,
    A100, L4 or RTX PRO 6000 Blackwell;
    a T4 cannot preserve this recipe. One notebook runtime at a time should use this results folder.

    Run cells in order. All results/recovery go directly to Drive; your Mac is not an archive server.
    Colab availability, session lifetime and background execution depend on your plan. Closing a laptop
    is not a guarantee of uninterrupted Colab execution. See the [official FAQ](https://research.google.com/colaboratory/faq.html).
    ''')
    md('## 1. Mount Drive and verify both uploads')
    code(f'''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import hashlib, json, os, shutil, signal, subprocess, sys, zipfile

    DRIVE_ROOT = Path('/content/drive/MyDrive/hippo_msd_3fold_3seed')
    INPUT = DRIVE_ROOT / 'input'
    RESULTS = DRIVE_ROOT / 'results_noaug_v1'
    BUNDLE = Path('/content/hippo_msd_bundle')
    EXPECTED = {{'msd_data_verified.zip': '{data_sha}', 'msd_replication_code.zip': '{code_sha}'}}
    def sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
        return h.hexdigest()
    BUNDLE.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED.items():
        src = INPUT/name
        if not src.exists():
            raise FileNotFoundError(f'Upload {{name}} to My Drive/hippo_msd_3fold_3seed/input/')
        assert sha256(src) == expected, f'Upload hash mismatch: {{name}}'
        local = Path('/content')/name
        shutil.copyfile(src, local)
        assert sha256(local) == expected
        with zipfile.ZipFile(local) as archive:
            for item in archive.infolist():
                member = Path(item.filename)
                assert not member.is_absolute() and '..' not in member.parts
                assert not ((item.external_attr >> 16) & 0o170000) == 0o120000
            archive.extractall(BUNDLE)
        print('Verified and staged:', name)
    # Archives remain on Drive as the durable dataset/source record.
    probe = RESULTS/'write_probe.json'
    probe.write_text('{{"status": "ok"}}')
    assert json.loads(probe.read_text())['status'] == 'ok'
    probe.unlink()
    print('No augmentation; data will be read from the runtime SSD, not Drive.')
    ''')
    md('''
    ## 2. Check Drive headroom
    Reserve at least 8 GiB for current recovery snapshots and metadata. Completed model checkpoints
    are retired only after their evaluated results have been archived and verified.
    ''')
    code('''
    from google.colab import auth
    auth.authenticate_user()
    import google.auth
    from googleapiclient.discovery import build
    creds, _ = google.auth.default()
    quota = build('drive', 'v3', credentials=creds, cache_discovery=False).about().get(fields='storageQuota').execute()['storageQuota']
    if 'limit' in quota:
        free = int(quota['limit']) - int(quota['usage'])
        print(f'Drive quota headroom: {free/1024**3:.2f} GiB')
        assert free >= 8*1024**3, 'Free at least 8 GiB before training.'
    else:
        print('Drive reports no fixed quota limit.')
    assert shutil.disk_usage('/content').free >= 20*1024**3, 'At least 20 GiB runtime SSD headroom required.'
    ''')
    md('''
    ## 3. Create an isolated training environment
    The notebook kernel is left intact. Package pins match the cluster recipe, including actual
    dynamic-network-architectures 0.4.4. Installation is reused while this runtime survives.
    Missing pip is bootstrapped from PyPA. Setup and campaign output is printed and saved locally.
    CUDA initialization and numerical equivalence are checked in the next cell.
    ''')
    code('''
    VENV = Path('/content/hippo_msd_venv')
    PYTHON = VENV/'bin/python'
    def run_setup_logged(args, log_path):
        log_path = Path(log_path)
        tail = []
        with log_path.open('a', encoding='utf-8') as log:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1, start_new_session=True)
            def drain():
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line); log.flush()
                    tail.append(line)
                    del tail[:-40]
            interrupted = False
            try:
                drain()
                result = proc.wait()
            except KeyboardInterrupt:
                interrupted = True
                print('Requesting a clean stop. Wait for the process and recovery upload to finish.')
                os.killpg(proc.pid, signal.SIGINT)
                drain()
                result = proc.wait()
            finally:
                proc.stdout.close()
            if result and not interrupted:
                raise RuntimeError(f'Command exited {result}. Full log: {log_path}\\n' + ''.join(tail))

    def ensure_training_pip(python, bootstrap, log_path):
        probe = subprocess.run([str(python), '-m', 'pip', '--version'], capture_output=True, text=True)
        if probe.returncode:
            import urllib.request
            urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', str(bootstrap))
            run_setup_logged([str(python), str(bootstrap)], log_path)

    if not PYTHON.exists():
        run_setup_logged([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(VENV)], '/content/hippo_msd_setup.log')
    ensure_training_pip(PYTHON, Path('/content/hippo_msd_get_pip.py'), '/content/hippo_msd_setup.log')
    run_setup_logged([str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(BUNDLE/'requirements-colab.txt')], '/content/hippo_msd_setup.log')
    def campaign(command, *extra):
        args = [str(PYTHON), '-u', str(BUNDLE/'runtime.py'), command, '--bundle', str(BUNDLE), '--durable', str(RESULTS), *map(str, extra)]
        run_setup_logged(args, Path('/content')/f'hippo_msd_{command}.log')
    campaign('verify')
    subprocess.run(['nvidia-smi'], check=True)
    ''')
    md('''
    ## 4. Required GPU preflight
    Checks all 12 initializations.
    Original CPU truncated-normal tensors are included for portability between Python builds;
    the random draws and complete model hashes must still match the verified cluster recipe.
    Then compares the original and accelerated code on this GPU: losses, gradients, optimizer/model
    state, calibration, evaluation, RNG and checkpoint resume.
    Its diagnostic updates do not count as scientific runs. A failure blocks training.
    This can take several minutes; it is reused only with the same code, GPU and software.
    ''')
    code("campaign('preflight')")
    md('''
    ## 5. Run or resume the sweep
    Starts with the four fresh baselines for each fold/seed, followed by the same 13 constraint conditions.
    Already verified cells are skipped. It trains one experiment at a time, using cached GPU inputs,
    cached immutable annotation fields, deferred metric readback and fewer redundant checkpoint writes.

    Recovery copies use a background writer; only a small compute ledger is synchronously sent to Drive
    before each update. Metadata, metrics and cleanup receipts are persisted after every experiment.
    Drive headroom is checked before every experiment; trash also counts against the quota.
    `SESSION_HOURS` is a conservative stop target, not a Colab runtime guarantee.

    To stop cleanly, interrupt this cell **once** and let the checkpoint upload finish. Repeated interrupts,
    deleting the runtime, or hard termination can leave charged updates newer than the durable checkpoint;
    the notebook refuses to hide those missing updates and will require a recovery audit.
    ''')
    code('''
    MAX_NEW_RUNS = 504
    SESSION_HOURS = 22.0  # Pro+: clean-stop target, subject to available runtime and compute units.
    # If you deliberately created RESULTS/'STOP', remove it only when you want to resume.
    campaign('run', '--max-runs', MAX_NEW_RUNS, '--hours', SESSION_HOURS)
    ''')
    md('''
    ## 6. Results so far
    Each gain is paired with the same model/fold/seed baseline. A final mean is shown only after all
    nine cells of that model/condition are present: average the three seeds within each fold, then
    average the three fold means. The displayed SD is across fold means, not nine independent subjects.
    Architecture remains confounded with baseline loss/optimizer; this is validation, not an independent test set.
    ''')
    code('''
    campaign('report')
    result = json.loads((RESULTS/'comparison.json').read_text())
    import pandas as pd
    print(f"Complete: {result['completed']}/{result['total']}")
    display(pd.DataFrame(result['aggregate']))
    display(pd.DataFrame(result['runs']))
    ''')
    md('''
    ## 7. Curves and GPU use
    Per-run `metrics.csv`, `performance.json`, `calibration.json`, full diagnostics, configuration and
    integer FN/FP/A–P swaps live under `results_noaug_v1/runs/`. Inspect training and validation together.
    Use the runtime's GPU utilization display or `nvidia-smi`; high utilization alone does not establish
    a faster or scientifically equivalent experiment. Do not raise batch size or change precision.
    ''')
    code('''
    import matplotlib.pyplot as plt
    finished = result['runs']
    if finished:
        chosen = finished[-1]['id']
        curve = pd.read_csv(RESULTS/'runs'/chosen/'metrics.csv')
        plt.plot(curve['compute_fraction'], 100*curve['macro_hard_dice'], marker='.')
        plt.xlabel('Fraction of optimization FLOP budget')
        plt.ylabel('Validation Dice (%)')
        plt.title(chosen)
        plt.grid(alpha=.25)
        plt.show()
    subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv'], check=True)
    ''')
    md('''
    ## Resume later
    Open this same notebook, use the same GPU type, and rerun setup/preflight/run cells. Keep the same
    results folder and uploaded archives. No laptop collector or cluster scheduler is involved.
    Do not run two copies against the same output directory. A changed GPU/software environment is
    rejected instead of mixing its results into the nine-cell comparisons.

    Google Drive's quota API is documented [here](https://developers.google.com/workspace/drive/api/reference/rest/v3/about).
    Colab resources and background execution remain subject to the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
    Disconnect and delete the GPU runtime when finished; the verified Drive results remain.
    ''')
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(
        colab=dict(name='MSD_3fold_3seed_no_augmentation.ipynb', provenance=[]),
        kernelspec=dict(display_name='Python 3', name='python3'), language_info=dict(name='python'), accelerator='GPU'),
        cells=[dict(id=f'cell-{i:02d}', **c) for i,c in enumerate(cells)])


def prepare():
    OUT.mkdir(exist_ok=True)
    stage = OUT/'bundle'
    stage.mkdir(exist_ok=True)
    for name, sha in read(PARENT/'manifest.json').items():
        if digest(PARENT/name) != sha:
            raise ValueError('Pilot frozen source changed: '+name)
        if name.startswith('source/'):
            dest = stage/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT/name, dest)
    hashes = read(OUT/'initialization_hashes.json')
    plan = dict(name=NAME, augmentation='none', folds=[0,1,2], seeds=[0,1,2],
        models=list(MODELS), arms=list(ARMS), num_runs=504,
        parent_manifest_sha256=digest(PARENT/'manifest.json'),
        policy='504 fresh Colab runs; completed pilot retained as provenance, no automatic reuse across hardware', tasks=[])
    for model in MODELS:
        old = PARENT/'protocols'/model
        original = read(old/'protocol.json')
        if hashes[model]['0'] != original['baseline_reference']['initialization_sha256']:
            raise ValueError('Seed 0 does not match pilot')
        for seed in (0,1,2):
            folder = stage/'protocols'/f'{model}_s{seed}'; folder.mkdir(parents=True, exist_ok=True)
            for name in ('splits_10patients.json','nnunet_architecture.json','profile.json'):
                shutil.copyfile(old/name, folder/name)
            p = deepcopy(original)
            p.update(name=NAME, data='/content/hippo_msd_bundle/data/msd_hippocampus_full.pkl',
                folds=[0,1,2], seeds=[seed], parent_protocol_sha256=digest(old/'protocol.json'))
            p['baseline_reference'] = dict(initialization_sha256=hashes[model][str(seed)], seed=seed,
                policy='Fresh matched baseline for each fold and seed; initialization verified in original cluster environment',
                pilot_reference=original['baseline_reference'])
            write(folder/'protocol.json', p)
    for fold in (0,1,2):
        for seed in (0,1,2):
            for arm in ARMS:
                for model in MODELS:
                    plan['tasks'].append(dict(id=f'MSD_{model}_{arm}_f{fold}s{seed}', model=model, arm=arm,
                        fold=fold, seed=seed, kind='conventional', dataset='MSD', protocol=f'{model}_s{seed}'))
    write(stage/'plan.json', plan)
    for name in ('runtime.py','entry.py'):
        shutil.copyfile(ROOT/'scripts/colab_msd'/name, stage/name)
    shutil.copyfile(OUT/'initialization_verification.json', stage/'initialization_verification.json')
    shutil.copyfile(OUT/'initialization_reference.json', stage/'initialization_reference.json')
    for name in ('runtime_benchmark.json', 'runtime_benchmark_manifest.json', 'source_version.json'):
        shutil.copyfile(PARENT/name, stage/name)
    shutil.copyfile(PARENT/'manifest.json', stage/'pilot_manifest.json')
    packages = {'torch':'2.12.1', 'torchvision':'0.27.1', 'monai':'1.6.0', 'numpy':'2.4.6',
        'pandas':'3.0.3', 'scipy':'1.17.1', 'scikit-image':'0.26.0', 'nibabel':'5.4.2',
        'einops':'0.8.2', 'dynamic-network-architectures':'0.4.4', 'batchgenerators':'0.25.3',
        'batchgeneratorsv2':'0.3.5', 'acvl-utils':'0.2.6', 'timm':'1.0.22',
        'matplotlib':'3.11.1', 'scikit-learn':'1.9.0'}
    (stage/'requirements-colab.txt').write_text(''.join(f'{k}=={v}\n' for k,v in packages.items()))
    data_zip = OUT/'msd_data_verified.zip'
    with zipfile.ZipFile(data_zip) as archive:
        archive.extractall(stage)
    data = read(stage/'dataset_manifest.json')
    assert digest(stage/data['file']) == data['sha256'] == original['pkl_sha256']
    names = {str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name != 'bundle_manifest.json' and '__pycache__' not in p.parts}
    write(stage/'bundle_manifest.json', names)
    verify_bundle(stage)
    path = OUT/'msd_replication_code.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*names, 'bundle_manifest.json']:
            if name.startswith('data/') or name == 'dataset_manifest.json':
                continue
            archive.write(stage/name, name)
    book = notebook(digest(path), digest(data_zip))
    write(OUT/'MSD_3fold_3seed_no_augmentation.ipynb', book)
    write(OUT/'export_summary.json', dict(status='prepared_for_colab_gpu_preflight', runs=504,
        augmentation='none', folds=[0,1,2], seeds=[0,1,2], data_sha256=data['sha256'],
        bundle_manifest_sha256=digest(stage/'bundle_manifest.json'),
        code_zip_sha256=digest(path), data_zip_sha256=digest(data_zip),
        notebook='MSD_3fold_3seed_no_augmentation.ipynb',
        not_executed='Colab GPU preflight and training require an authenticated user runtime'))
    print(json.dumps(read(OUT/'export_summary.json'), indent=2))


if __name__ == '__main__':
    prepare()
