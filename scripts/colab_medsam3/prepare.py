"""Build the self-contained MedSAM3 extension transfer package and notebook."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import textwrap
import zipfile

from scripts.colab_medsam3.runtime import ARMS, EXISTING, FOLDS, SEEDS, digest, read, write

ROOT = Path(__file__).resolve().parents[2]
SOURCE_EXPORT = ROOT/'reports/medsam3_msd_colab_3fold_3seed_noaug_20261015'
EXPORT = ROOT/'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015'


def prepare_preflight_upgrade(bundle, results, expected_manifest):
    """Preserve a known failed preflight before staging the corrected package."""
    import hashlib
    import json
    from pathlib import Path
    import shutil
    import tempfile
    bundle, results = Path(bundle), Path(results)
    manifest = bundle/'bundle_manifest.json'
    if not manifest.exists(): return
    previous = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if previous == expected_manifest: return
    if previous != '92b387c59ce409c5700e6e9b84f4bcf27069d9cb43f29f0fd8ce67ceb84d543d':
        raise RuntimeError('Unrecognized existing bundle; preserve its work before changing it.')
    work = [bundle/'work', results/'runs', results/'recovery']
    if any(p.is_file() for folder in work for p in folder.rglob('*')):
        raise RuntimeError('Scientific work or recovery exists. This preflight-only upgrade cannot proceed.')
    # Preserve all old preflight evidence, including an interrupted calibration.
    backup_root = results/'preflight_migrations'; backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix='original_preflight_', dir=backup_root))
    for name in ('engine.py', 'runtime.py', 'bundle_manifest.json'):
        shutil.copyfile(bundle/name, backup/name)
    for name in ('environment_binding.json', 'source_binding.json', 'preflight.json'):
        if (results/name).exists(): shutil.copyfile(results/name, backup/name)
    if (results/'calibration').exists():
        (results/'calibration').rename(backup/'calibration')
    (backup/'migration.json').write_text(json.dumps(dict(
        previous_manifest=previous, next_manifest=expected_manifest,
        reason='Correct strict GPU equivalence audit before any scientific training'), indent=2))
    print('Preserved previous preflight evidence:', backup)


def notebook(uploads, manifest_sha):
    cells=[]
    def cell(kind, source):
        item=dict(id=f'cell_{len(cells):02d}',cell_type=kind,metadata={},source=textwrap.dedent(source).strip()+'\n')
        if kind=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''
    # MSD MedSAM3: matched controls and constraints, three folds × three seeds

    **No augmentation. Original fine-tuning recipe: 200 optimizer updates, accumulation 2,
    native SAM3 loss, medical pretrained LoRA, 10% auxiliary-gradient target and 50% p95 cap.**
    This notebook is **not capped at 31.98 TFLOPs** and is not compute-matched to the four-model MSD/ADNI sweeps.

    It retains folds **0, 1, 3** and seeds **17, 83, 191**, with five original support volumes per fold.
    **108 runs** cover the 11 missing conditions (99 runs) plus fresh baseline controls (9 runs).
    Baselines run first. Bands and bands+edge are not repeated. The 27 historical cluster results are retained for reference only; they are
    never used as matched controls or counted as completed runs in this study.

    All coefficients, including bands and edge, are calibrated sequentially on this runtime's
    training support, once per fold and shared across seeds. The original 10% median target is
    limited by the original 50% p95 rule; the median may be lower when the p95 cap binds.
    SDF uses full-volume foreground-class signed distances clipped at 8 mm; depth uses the existing
    normalized anatomical depth CE on the native 1-mm RAS grid. All active auxiliary terms use the
    original 20-update ramp. There is no new SDF delay, validation tuning or early stopping.

    Use an A100 or H100 with at least 40 GB GPU RAM and a High-RAM runtime. GPU preflight is mandatory.
    All primary comparisons use fresh Colab controls in the same GPU/software environment.
    These selected folds support exploratory comparisons, not unbiased full CV. Production CUDA
    kernels retain the original numerical variability; equal seeds do not guarantee bitwise replay.
    ''')
    cell('markdown','''
    ## 1. Upload and connect Drive
    Create `MyDrive/hippo_medsam3_msd_extension/input` and upload these three files:
    `medsam3_msd_108_code.zip`, `medsam3_msd_data.zip`, and `sam3.pt`.
    **If the data ZIP and SAM3 checkpoint are already there, upload only the new code ZIP.**
    The MedSAM3 medical adapter is included in the code ZIP. Keep the original files on Drive.
    This uses a different folder from the running MSD and ADNI notebooks.
    ''')
    cell('code','''
    from google.colab import drive, auth
    drive.mount('/content/drive')
    auth.authenticate_user()
    from pathlib import Path
    import hashlib, json, os, shutil, signal, subprocess, sys, zipfile
    HOME_DIR = Path('/content/drive/MyDrive/hippo_medsam3_msd_extension')
    INPUT = HOME_DIR/'input'
    RESULTS = HOME_DIR/'results_matched200_noaug_108_v1'
    BUNDLE = Path('/content/hippo_medsam3_msd_108_bundle')
    PREVIOUS_RESULTS = HOME_DIR/'results_original200_noaug_v1'
    PREVIOUS_BUNDLE = Path('/content/hippo_medsam3_msd_bundle')
    RESULTS.mkdir(parents=True, exist_ok=True)
    ''')
    cell('markdown','''
    ## 2. Verify uploads and stage on the VM SSD
    Copies the 3.45 GB frozen SAM3 checkpoint once. The dataset is the original NIfTI collection,
    not the resized PKL used by the four-model sweep. Upload and extraction hashes are checked.
    This uses separate bundle and results folders and leaves the earlier study intact.
    On the same VM it reuses the verified local base checkpoint through a hard link; otherwise
    it copies the existing checkpoint from Drive. No new data transfer is required.
    ''')
    cell('code',textwrap.dedent(f'''
    EXPECTED_UPLOADS = {uploads!r}
    EXPECTED_MANIFEST = {manifest_sha!r}
    def sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
        return h.hexdigest()
    def checked_copy(source, target, expected):
        if target.exists() and sha256(target) == expected: return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.copying')
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected: raise ValueError('Copy failed verification: '+str(target))
        temporary.replace(target)
    def safe_extract(path, destination):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                name = Path(item.filename)
                if name.is_absolute() or '..' in name.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe ZIP member')
            archive.extractall(destination)
    for name, expected in EXPECTED_UPLOADS.items():
        path = INPUT/name
        if not path.exists(): raise FileNotFoundError('Upload '+name+' into '+str(INPUT))
        if sha256(path) != expected: raise ValueError('Upload hash mismatch: '+name)
        print('Verified', name, flush=True)
    if shutil.disk_usage('/content').free < 16*1024**3:
        raise RuntimeError('Need 16 GiB free on the runtime disk before staging')
    if (BUNDLE/'bundle_manifest.json').exists() and sha256(BUNDLE/'bundle_manifest.json') != EXPECTED_MANIFEST:
        raise RuntimeError('Existing matched bundle differs; preserve its work before replacing it.')
    BUNDLE.mkdir(exist_ok=True)
    safe_extract(INPUT/'medsam3_msd_108_code.zip', BUNDLE)
    safe_extract(INPUT/'medsam3_msd_data.zip', BUNDLE)
    base = BUNDLE/'weights/sam3.pt'
    previous_base = PREVIOUS_BUNDLE/'weights/sam3.pt'
    if not base.exists() and previous_base.exists() and sha256(previous_base) == EXPECTED_UPLOADS['sam3.pt']:
        os.link(previous_base, base)
    checked_copy(INPUT/'sam3.pt', base, EXPECTED_UPLOADS['sam3.pt'])
    if sha256(BUNDLE/'bundle_manifest.json') != EXPECTED_MANIFEST: raise ValueError('Bundle manifest differs')
    (RESULTS/'source_binding.json').write_text(json.dumps(dict(uploads=EXPECTED_UPLOADS, manifest=EXPECTED_MANIFEST), indent=2))
    print('Staged on the local SSD. No training has started.')
    '''))
    cell('markdown','''
    ## 3. Create the environment
    Uses the original dependency versions. Installation output is captured and displayed on error.
    The pip bootstrap handles Colab runtimes without ensurepip. No cluster connection is needed.
    ''')
    cell('code','''
    VENV = Path('/content/hippo_medsam3_venv')
    PYTHON = VENV/'bin/python'
    def setup(args, log_name):
        result = subprocess.run(list(map(str,args)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (RESULTS/log_name).write_text(result.stdout)
        print('\\n'.join(result.stdout.splitlines()[-35:]))
        if result.returncode: raise RuntimeError(f'Command failed; full output: {RESULTS/log_name}')
    if not PYTHON.exists():
        setup([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV], 'venv_setup.log')
    probe = subprocess.run([str(PYTHON),'-m','pip','--version'], capture_output=True, text=True)
    if probe.returncode:
        import urllib.request
        bootstrap = Path('/content/medsam3_get_pip.py')
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', bootstrap)
        setup([PYTHON,bootstrap], 'pip_bootstrap.log')
    setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'], 'dependencies.log')
    def campaign(command, *extra):
        args = [str(PYTHON),str(BUNDLE/'runtime.py'),command,'--bundle',str(BUNDLE),'--durable',str(RESULTS),*map(str,extra)]
        process = subprocess.Popen(args, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log_path = RESULTS/(command+'.log')
        try:
            with log_path.open('a') as log:
                for line in process.stdout:
                    print(line,end='',flush=True); log.write(line);log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            print('Requesting clean stop. Wait for recovery upload to finish.')
            os.killpg(process.pid, signal.SIGINT);process.wait();return
        if code: raise RuntimeError(f'{command} failed. Details are printed above and saved in {log_path}')
    campaign('verify')
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    cell('markdown','''
    ## 4. Preserve and reuse completed preflight work
    This reads the previous failed-preflight folder without moving or deleting it. Complete saved
    measurements and passed audits are reused only after checking the exact previous package,
    GPU/software, unchanged code, data, medical adapter and all audit checks. It imports no weights
    or training results. Fold 0 and fold 1 can therefore reuse their 20 measurements each on the
    same A100 runtime. A different environment runs all preflight work anew. A manual STOP is preserved.
    This cell does not approve the new campaign; run the required preflight below.
    ''')
    cell('code', "campaign('reuse-preflight', '--previous-results', PREVIOUS_RESULTS, '--previous-bundle', PREVIOUS_BUNDLE)")
    cell('markdown','''
    ## 5. Required GPU preflight and shared calibration
    Verifies exact pretrained parameter hashes, original data/calibration context, native/bands/edge
    loss and gradient equivalence, unchanged label-free inference, and exact optimizer/RNG resume.
    Strict numerical comparisons run in a separate process with deterministic GPU algorithms,
    including an original-versus-original repeatability control. Original BF16 precision is retained.
    Only in that audit, the empty point-prompt sampler uses its exact zero derivative while retaining
    native forward execution. Nonempty sampling remains subject to the original strict checks.
    Calibration and training retain the original runtime settings; strict diagnostic settings do not
    carry over. Exact checkpoint replay is tested under the diagnostic settings, while production GPU
    kernels can have the same run-to-run numerical variation observed in the original implementation.
    Then calibrates every term from 20 training slabs per fold; no historical coefficients are reused. No scientific training updates
    are charged during these diagnostics. It may take substantially longer than a simple smoke test.
    Completed fold receipts are reused only for the same bundle, GPU and software.
    ''')
    cell('code',"campaign('preflight')")
    cell('markdown','''
    ## 6. Run or resume the 108 matched experiments
    One experiment at a time; already archived runs in this study are skipped. The local checkpoint is saved every update;
    a coherent recovery snapshot is uploaded to Drive asynchronously about every three minutes.
    Each update is recorded before execution. A hard VM loss with newer charged updates than the
    durable checkpoint blocks automatic replay and retains recovery for an audit.

    Interrupt once for a clean stop and wait for the upload. `SESSION_HOURS` is a conservative
    stop target, not a runtime guarantee. A `STOP` file in RESULTS also requests a clean stop;
    remove it deliberately when ready to resume. Do not run two copies against this results folder.

    Dice, FN/FP, A/P swaps, training curves, configuration, calibration and source identities are
    archived and hash-verified after each run. Its adapter/optimizer recovery files are then retired.
    Evaluation uses the final 200-update checkpoint, with no validation-based selection.
    ''')
    cell('code',"MAX_NEW_RUNS = 108\nSESSION_HOURS = 22.0\ncampaign('run', '--max-runs', MAX_NEW_RUNS, '--hours', SESSION_HOURS)")
    cell('markdown','''
    ## 7. Compare with fresh matched controls
    Dice and gains are averaged across seeds within each fold, then equally across the three folds.
    Gains require the fresh baseline with the same fold and seed. Missing baselines are never filled
    from the historical cluster study. The 27 old results remain in `historical_reference` for context.
    Bands and bands+edge have no fresh matched runs here; comparisons involving those historical
    arms remain exploratory and are not mixed into the primary gains.
    The seed-46 pilot is excluded. Training curves and final train/validation metrics are retained;
    no new per-epoch validation schedule is introduced.
    ''')
    cell('code',"campaign('report')\nimport pandas as pd\ncomparison = json.loads((RESULTS/'comparison.json').read_text())\ndisplay(pd.DataFrame(comparison['aggregate']))\ndisplay(pd.DataFrame(comparison['runs']))")
    return dict(cells=cells,metadata=dict(colab=dict(name='MedSAM3_MSD_108_runs_3fold_3seed.ipynb'),
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),nbformat=4,nbformat_minor=5)


def main():
    bundle=EXPORT/'bundle';bundle.mkdir(parents=True,exist_ok=True)
    # Local verification uses the same immutable inputs; do not duplicate the 3.45 GB checkpoint.
    (bundle/'weights').mkdir(exist_ok=True)
    base = bundle/'weights/sam3.pt'
    if not base.exists(): os.link(SOURCE_EXPORT/'assets/sam3.pt', base)
    shutil.copytree(SOURCE_EXPORT/'bundle/data', bundle/'data', dirs_exist_ok=True)
    originals=SOURCE_EXPORT/'original/source'
    selected=['evaluation/medsam3_zero_shot.py','scripts/train_medsam3_few_shot.py',
        'scripts/train_medsam3_constraint_few_shot.py','scripts/train_medsam3_multiclass.py',
        'scripts/medsam3_supervised_constraints.py','scripts/medsam3_calibration_cache.py',
        'thesis/new_constraints/bands/outer_boundary.py','thesis/new_constraints/edge_consistency.py',
        'thesis/new_constraints/constraint_result.py']
    for name in selected:
        target=bundle/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(originals/name,target)
    target=bundle/'source/scripts/depth_supervision/geometry.py';target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SOURCE_EXPORT/'bundle/source/scripts/depth_supervision/geometry.py',target)
    for folder in [bundle/'source',*(p for p in (bundle/'source').rglob('*') if p.is_dir())]:
        (folder/'__init__.py').touch()
    shutil.copytree(SOURCE_EXPORT/'upstream',bundle/'upstream',dirs_exist_ok=True)
    (bundle/'weights').mkdir(exist_ok=True);shutil.copyfile(SOURCE_EXPORT/'assets/best_lora_weights.pt',bundle/'weights/best_lora_weights.pt')
    for name in ['runtime.py','engine.py','audit_numerics.py','reuse_preflight.py']:
        shutil.copyfile(ROOT/'scripts/colab_medsam3'/name,bundle/name)
    old=read(SOURCE_EXPORT/'preserved_results.json');agg=old['records']['analysis/aggregate.json']['summary']
    imported=[]
    for run in agg['runs']:
        for arm in EXISTING:
            v=run['summary']['validation'][arm];t=run['summary']['train'][arm]
            imported.append(dict(id=f"MSD_MedSAM3_{arm}_f{run['fold']}s{run['seed']}",
                fold=run['fold'],seed=run['seed'],arm=arm,validation_dice=v['ap_mean_dice'],
                training_dice=t['ap_mean_dice'],foreground_fn=v['foreground_fn'],foreground_fp=v['foreground_fp'],
                ap_swaps=v['ap_swaps'],audit_status=run['status'],origin='Original cluster study',updates=200))
    write(bundle/'existing_results.json',dict(runs=imported,provenance=old))
    for fold in FOLDS:
        cals=[SOURCE_EXPORT/'original/results'/f'fold{fold}_seed{s}'/'calibration.json' for s in SEEDS]
        if len({digest(p) for p in cals})!=1:raise ValueError('Original calibration changed across seeds')
        target=bundle/'calibration'/f'fold{fold}.json';target.parent.mkdir(exist_ok=True);shutil.copyfile(cals[0],target)
        for seed in SEEDS:
            original=read(SOURCE_EXPORT/'original/results'/f'fold{fold}_seed{seed}'/'protocol.json')
            original.update(augmentation='none',budget_policy='original_200_updates_not_flop_capped')
            write(bundle/'protocols'/f'fold{fold}_seed{seed}.json',original)
    plan=dict(name='medsam3_msd_108_constraints_original200',
        calibration_policy='fresh_runtime_sequential',historical_results_role='reference_only',folds=list(FOLDS),seeds=list(SEEDS),arms=list(ARMS),
        augmentation='none',updates=200,optimization_flop_cap=None,
        base_sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',
        upstream_commit='84f43118b5b67b50abad81d7c946681ebb97dcda',
        tasks=[dict(id=f'MSD_MedSAM3_{a}_f{f}s{s}',fold=f,seed=s,arm=a)
               for a in ('baseline', *(a for a in ARMS if a not in EXISTING)) for f in FOLDS for s in SEEDS])
    write(bundle/'plan.json',plan);shutil.copyfile(SOURCE_EXPORT/'data_manifest.json',bundle/'data_manifest.json')
    requirements='''torch==2.12.1
torchvision==0.27.1
numpy==2.4.6
scipy==1.17.1
nibabel==5.4.2
Pillow==12.3.0
timm==1.0.22
einops==0.8.2
PyYAML==6.0.3
hydra-core==1.3.7
omegaconf==2.3.1
iopath==0.1.10
decord==0.6.0
pycocotools==2.0.11
ftfy==6.3.1
regex==2026.6.28
huggingface-hub==1.22.0
transformers==5.13.0
accelerate==1.14.0
wandb==0.28.1
torchmetrics==1.9.0
matplotlib==3.11.1
scikit-image==0.26.0
scikit-learn==1.9.0
pandas==3.0.3
psutil==7.2.2
tqdm==4.68.3
google-api-python-client
google-auth-httplib2
'''
    (bundle/'requirements-colab.txt').write_text(requirements)
    compatibility = bundle/'compatibility'; compatibility.mkdir(exist_ok=True)
    for name in ('bundle_manifest.json', 'engine.py', 'runtime.py'):
        shutil.copyfile(SOURCE_EXPORT/'bundle'/name, compatibility/name)
    if digest(compatibility/'bundle_manifest.json') != '26f4f23962fb7dd1931f70c2e672e8f76d8c5f4cd983ce980b2dc041c994f20a':
        raise ValueError('Unknown predecessor package')
    manifest={str(p.relative_to(bundle)):digest(p) for p in sorted(bundle.rglob('*')) if p.is_file()
              and p != bundle/'bundle_manifest.json' and '__pycache__' not in p.parts and 'data' not in p.relative_to(bundle).parts[:1]
              and p.relative_to(bundle).as_posix()!='weights/sam3.pt'}
    write(bundle/'bundle_manifest.json',manifest)
    code=EXPORT/'medsam3_msd_108_code.zip'
    with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*manifest,'bundle_manifest.json']:archive.write(bundle/name,name)
    uploads={p.name:digest(p) for p in [code,SOURCE_EXPORT/'medsam3_msd_data.zip',SOURCE_EXPORT/'assets/sam3.pt']}
    path=EXPORT/'MedSAM3_MSD_108_runs_3fold_3seed.ipynb'
    write(path,notebook(uploads,digest(bundle/'bundle_manifest.json')))
    write(EXPORT/'transfer_manifest.json',dict(uploads=uploads,bundle_manifest_sha256=digest(bundle/'bundle_manifest.json'),
        notebook_sha256=digest(path),new_runs=108,reused_runs=0,historical_reference_runs=27,optimization_flop_cap=None,updates=200,augmentation='none'))
    print(json.dumps(read(EXPORT/'transfer_manifest.json'),indent=2))


if __name__=='__main__':main()
