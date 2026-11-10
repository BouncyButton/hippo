"""Build the self-contained MNI pilot source/data ZIP and Colab notebook."""
import ast
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

import numpy as np
import pandas as pd

from scripts.colab_mni_medsam3.data import select_support
from scripts.cross_dataset_campaign.common import digest, read, write

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/medsam3_mni_colab_pilot_20261108'
PARENT = ROOT/'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/frozen'
MSD = ROOT/'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle'


def build():
    bundle = EXPORT/'bundle'
    if bundle.exists():
        raise ValueError('Output already exists; preserve the frozen bundle')
    bundle.mkdir(parents=True)
    for name, expected in read(PARENT/'manifest.json').items():
        if name.startswith('source/') and digest(PARENT/name) != expected:
            raise ValueError('Reference source changed: '+name)
    shutil.copytree(PARENT/'source', bundle/'source')
    shutil.copytree(Path(__file__).parent, bundle/'source/scripts/colab_mni_medsam3',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(MSD/'requirements-colab.txt', bundle/'requirements-colab.txt')
    source = ROOT/'datasets/Dataset102_MNI'
    folds = read(source/'splits_final.json')
    selection = select_support(folds, 1)
    p = read(PARENT/'parent_protocol.json')
    p.update(dataset='MNI', name='MNI_MedSAM3_n1_f0s0_native200', fold=0, seed=0,
        classes=['background', 'CA1_3', 'subiculum', 'CA4_DG'],
        prompts={'1':'hippocampal CA1 CA2 CA3 subfields', '2':'subiculum',
                 '3':'hippocampal CA4 dentate gyrus'},
        data='/content/hippo_mni_medsam3/bundle/data',
        upstream='/content/hippo_mni_medsam3/shared/upstream',
        lora_weights='/content/hippo_mni_medsam3/shared/weights/best_lora_weights.pt',
        train_subjects=1, train_crops=2, **selection,
        support_selection='First subject in the historical MNI training-fold RandomState(0) order; both L/R crops; fixed independently of outcomes.',
        primary_metric='Final-update per-volume mean Dice across CA1_3, subiculum and CA4_DG',
        decision_rule='Inspect one fold0 seed0 pilot before setting a target or scheduling any extension.',
        inference='Whole-volume reconstruction using image and three fixed text prompts only; no label-derived prompts.',
        geometry_units='Native MNI cropped array voxels; no axis reorientation',
        requested_arms=['dice'],
        initialization='medical_pretrained_LoRA',
        dataset_sha256=digest(source/'mni_hippocampus_full.pkl'))
    for key in ('origin_protocol_sha256','reference_protocol_sha256','reference_protocol','runtime_optimization'):
        p.pop(key, None)
    write(bundle/'protocol.json', p)
    write(bundle/'saved_folds.json', folds)
    df = pd.read_pickle(source/'mni_hippocampus_full.pkl', compression='gzip')
    inventory = []
    (bundle/'data').mkdir()
    cases = set()
    for row in df.itertuples():
        case = f'hippocampus_mni_{row.subject_id}_{row.direction}'
        if case in cases:
            raise ValueError('Duplicate MNI case')
        cases.add(case)
        image = np.asarray(row.image_data)
        label = np.asarray(row.data)
        if image.shape != (40,56,48) or label.shape != image.shape or not np.isfinite(image).all():
            raise ValueError('Invalid native MNI image')
        if set(np.unique(label)) != {0,1,2,3}:
            raise ValueError('Missing/invalid MNI subfields')
        np.savez_compressed(bundle/'data'/f'{case}.npz', image=image, label=label.astype(np.uint8))
        inventory.append(dict(case=case, subject=row.subject_id, side=row.direction,
            shape=list(image.shape), class_voxels={str(c):int((label==c).sum()) for c in range(4)}))
    if cases != set(folds[0]['train']+folds[0]['val']):
        raise ValueError('Data and saved splits differ')
    write(bundle/'data_audit.json', dict(cases=inventory, selection=selection,
        source_pkl_sha256=p['dataset_sha256'], saved_folds_sha256=digest(source/'splits_final.json'),
        validation_subjects=5, validation_crops=10, status='passed'))
    write(bundle/'source_provenance.json', dict(parent_manifest_sha256=digest(PARENT/'manifest.json'),
        parent_protocol_sha256=digest(PARENT/'parent_protocol.json'),
        adaptation='NPZ MNI reader; four-class supports; three fixed subfield prompts. Original native objective, medical LoRA, optimizer, 200 updates and sampling preserved.',
        gpu_validation='Not run locally; actual finite-loss and gradient preflight executes before training in Colab.'))
    write(bundle/'manifest.json', {str(f.relative_to(bundle)):digest(f) for f in sorted(bundle.rglob('*'))
        if f.is_file() and '__pycache__' not in f.parts})
    package = EXPORT/'medsam3_mni_pilot.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(bundle.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts:
                z.write(file, 'bundle/'+str(file.relative_to(bundle)))
    notebook(package, bundle)
    write(EXPORT/'build_verification.json', dict(status='passed', package_sha256=digest(package),
        manifest_sha256=digest(bundle/'manifest.json'), selection=selection,
        runs=1, updates=200, augmentation='none', gpu_test=False))
    print(json.dumps(dict(export=str(EXPORT), package_bytes=package.stat().st_size,
                         support=selection['train_cases']), indent=2))


def notebook(package, bundle):
    cells = []
    def cell(kind, source):
        source = textwrap.dedent(source).strip()+'\n'
        if kind == 'code':
            ast.parse(source)
        item = dict(cell_type=kind, metadata={}, id=f'mni_{len(cells):02d}', source=source)
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '''
    # MNI MedSAM3: minimum-data pilot
    Sign in as **filippofocaccia2@gmail.com**. Select an **A100 GPU** (80 GB preferred), then run all cells.

    **Exactly one run:** fold 0, optimization seed 0, one subject / two L+R crops,
    ten validation crops from five separate subjects. Native MedSAM3 loss, medical LoRA,
    200 updates, accumulation 2, AdamW, learning rate 5e-5 → 5e-6, 20 warmup updates, no augmentation.
    Three prompts predict CA1–3, subiculum and CA4–DG. No constraint ablation is scheduled.

    This is an exploratory development pilot. Inspect results before defining “works” or increasing
    the sample count; one pilot cannot establish the minimum or validate three seeds/folds.
    Subfield prompt vocabulary is a new MNI adaptation and has not yet been validated on a GPU.

    Reuses the existing MSD package and base weights. Results go to a separate MNI folder.
    Recovery is saved every 25 updates and on a clean stop; an abrupt VM loss can replay up to 24 updates.
    Interrupt the Run cell once and wait for the recovery message before disconnecting.
    Run only one Colab VM for this pilot. Completed results are verified before its temporary checkpoints are removed.
    ''')
    cell('code', '''
    from google.colab import drive, auth
    drive.mount('/content/drive')
    auth.authenticate_user()
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default()
    service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    email = service.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']
    if email.lower() != 'filippofocaccia2@gmail.com':
        raise RuntimeError('Authenticate with filippofocaccia2@gmail.com; got '+email)
    from pathlib import Path
    import hashlib, json, os, shutil, signal, subprocess, sys, zipfile
    ROOT = Path('/content/hippo_mni_medsam3')
    BUNDLE = ROOT/'bundle'
    SHARED = ROOT/'shared'
    HOME = Path('/content/drive/MyDrive/hippo_medsam3_mni_pilot')
    OLD = Path('/content/drive/MyDrive/hippo_medsam3_msd_extension/input')
    RESULTS = HOME/'results_n1_f0s0_native200_v1'
    OUTPUT = ROOT/'work'
    RESULTS.mkdir(parents=True, exist_ok=True)
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
        return h.hexdigest()
    def copy_verified(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name+'.uploading')
        shutil.copyfile(source, temp)
        if sha(source) != sha(temp): raise IOError('Copy mismatch: '+str(source))
        temp.replace(target)
    print('Account:', email, '\\nResults:', RESULTS)
    ''')
    cell('code', f'''
    PACKAGE_SHA = {digest(package)!r}
    MANIFEST_SHA = {digest(bundle/'manifest.json')!r}
    OLD_PACKAGE_SHA = '4fef5c8191d394594f33c701bbf5e4c733a20cbfd5e8e2f2162a4efa42b3948b'
    package = HOME/'medsam3_mni_pilot.zip'
    old_package = OLD/'medsam3_msd_108_fast_eval_code.zip'
    for file, expected in [(package, PACKAGE_SHA), (old_package, OLD_PACKAGE_SHA)]:
        if sha(file) != expected: raise ValueError('Input ZIP differs: '+str(file))
    def extract(archive_path, select, target):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                name = Path(info.filename)
                if name.is_absolute() or '..' in name.parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe ZIP member')
                relative = select(name)
                if relative is None or info.is_dir(): continue
                destination = target/relative
                value = archive.read(info)
                if destination.exists():
                    if sha(destination) != hashlib.sha256(value).hexdigest():
                        raise ValueError('Existing input differs; retained: '+str(destination))
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(value)
    extract(package, lambda p: p if p.parts[0]=='bundle' else None, ROOT)
    extract(old_package, lambda p: Path(*p.parts[1:]) if len(p.parts)>2 and p.parts[0]=='bundle'
            and p.parts[1] in ('upstream','compatibility','weights') and p.name!='sam3.pt' else None, SHARED)
    if sha(BUNDLE/'manifest.json') != MANIFEST_SHA: raise ValueError('Manifest mismatch')
    for name, expected in json.loads((BUNDLE/'manifest.json').read_text()).items():
        if sha(BUNDLE/name) != expected: raise ValueError('Bundle changed: '+name)
    protocol = json.loads((BUNDLE/'protocol.json').read_text())
    BASE = ROOT/'sam3.pt'
    if not BASE.exists(): copy_verified(OLD/'sam3.pt', BASE)
    if sha(BASE) != protocol['base_sha256']: raise ValueError('Base weights differ')
    if sha(SHARED/'weights/best_lora_weights.pt') != protocol['lora_sha256']: raise ValueError('Medical adapter differs')
    print('Verified source, data and weights. Training:', protocol['train_cases'])
    ''')
    cell('code', '''
    VENV = ROOT/'venv'
    PYTHON = VENV/'bin/python'
    def setup(args):
        result = subprocess.run(list(map(str,args)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with (RESULTS/'setup.log').open('a') as log: log.write(result.stdout)
        print('\\n'.join(result.stdout.splitlines()[-25:]))
        if result.returncode: raise RuntimeError('Setup failed; see setup.log')
    if not PYTHON.exists(): setup([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV])
    if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
        import urllib.request
        bootstrap = ROOT/'get-pip.py'
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py',bootstrap)
        setup([PYTHON,bootstrap])
    setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'])
    env = dict(os.environ, PYTHONPATH=str(SHARED/'compatibility')+':'+str(BUNDLE/'source')+':'+str(BUNDLE/'source/scripts'),
        MNI_DURABLE=str(RESULTS), MNI_MANIFEST_SHA=MANIFEST_SHA, PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled')
    probe = subprocess.run([str(PYTHON),'-c',
        "import torch,json; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); p=torch.cuda.get_device_properties(0); assert p.total_memory>=38*1024**3; print(json.dumps(dict(gpu=p.name,torch=torch.__version__,cuda=torch.version.cuda)))"],
        env=env, capture_output=True,text=True,check=True)
    environment = json.loads(probe.stdout.strip().splitlines()[-1])
    binding = dict(manifest=MANIFEST_SHA, **environment)
    binding_path = RESULTS/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError('Resume environment changed; use the same GPU/software for this pilot')
    binding_path.write_text(json.dumps(binding,indent=2))
    # Archive the complete source, protocol, folds and data hashes before training.
    for name in ['protocol.json','saved_folds.json','data_audit.json','source_provenance.json','manifest.json']:
        copy_verified(BUNDLE/name, RESULTS/'provenance'/name)
    copy_verified(HOME/'medsam3_mni_pilot.zip', RESULTS/'provenance/medsam3_mni_pilot.zip')
    print(environment)
    ''')
    cell('markdown', '''
    ## Run the single pilot
    A finite-loss and nonzero-gradient check runs before any optimizer updates.
    The original trainer logs every five updates, then evaluates both training crops and all ten validation crops.
    The label `dice` is the inherited baseline identifier: the optimized objective is SAM3's native loss, not Dice-only.
    ''')
    cell('code', '''
    import fcntl
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_lock = (ROOT/'pilot.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not (RESULTS/'completion.json').exists():
            if (RESULTS/'checkpoint_latest.pt').exists() and not (OUTPUT/'checkpoint_latest.pt').exists():
                for file in RESULTS.iterdir():
                    if file.suffix in ('.json','.pt') and file.name not in ('binding.json',):
                        copy_verified(file, OUTPUT/file.name)
                # History is authoritative inside the checkpoint; trainer restores it on resume.
            command = [str(PYTHON),'-m','scripts.colab_mni_medsam3.train',
                '--protocol',str(BUNDLE/'protocol.json'),'--output',str(OUTPUT),
                '--arm','dice','--base-weights',str(BASE)]
            process = subprocess.Popen(command, cwd=BUNDLE/'source', env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            with (RESULTS/'train.log').open('a') as log:
                while True:
                    try:
                        for line in process.stdout:
                            print(line,end='',flush=True); log.write(line); log.flush()
                        code = process.wait()
                        break
                    except KeyboardInterrupt:
                        print('Saving recovery. Wait for the process to finish before disconnecting.')
                        if process.poll() is None: process.send_signal(signal.SIGUSR1)
            if code == 75: raise RuntimeError('Stopped cleanly; recovery saved. Rerun this cell to continue.')
            if code: raise RuntimeError('Pilot failed; preserve recovery and inspect train.log')
        done = json.loads((RESULTS/'completion.json').read_text())
        print('Pilot complete:', done)
    finally:
        fcntl.flock(run_lock, fcntl.LOCK_UN)
        run_lock.close()
    ''')
    cell('code', '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    evaluation = json.loads((RESULTS/'evaluation.json').read_text())
    history = json.loads((RESULTS/'training_history.json').read_text())
    expected = {(s,c) for s,cases in [('training_support',protocol['train_cases']),('validation',protocol['evaluation_cases'])] for c in cases}
    if {(r['split'],r['case_name']) for r in evaluation['records']} != expected or len(history)!=200:
        raise ValueError('Incomplete evaluation/history; retain checkpoints')
    if sha(RESULTS/'evaluation.json') != done['evaluation_sha256']:
        raise ValueError('Evaluation checksum mismatch')
    rows=[]
    for record in evaluation['records']:
        matrix=np.asarray(record['confusion'])
        tp=matrix[1:,1:].sum(); fp=matrix[0,1:].sum(); fn=matrix[1:,0].sum()
        row=dict(split=record['split'],case=record['case_name'],
                 mean_subfield_dice=record['macro_hard_dice'],whole_hippocampus_dice=2*tp/(2*tp+fp+fn),
                 foreground_fp=int(fp),foreground_fn=int(fn),
                 subfield_confusions=int(matrix[1:,1:].sum()-np.trace(matrix[1:,1:])))
        for name, metrics in record['per_class'].items():
            if name=='background': continue
            for metric in ('dice','fp','fn'): row[name+'_'+metric]=metrics[metric]
        rows.append(row)
    table=pd.DataFrame(rows); table.to_csv(RESULTS/'per_case_metrics.csv',index=False)
    summary=table.groupby('split').mean(numeric_only=True)
    summary.to_csv(RESULTS/'summary.csv')
    display(summary[[c for c in summary if 'dice' in c]]*100)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot([h['update'] for h in history],[h['native'] for h in history]); axes[0].set_title('Native training loss')
    axes[1].plot([h['update'] for h in history],[h['gradient_norm'] for h in history]); axes[1].set_title('Gradient norm')
    summary[[c+'_dice' for c in protocol['classes'][1:]]].T.plot.bar(ax=axes[2],ylim=(0,1),rot=25)
    axes[2].set_title('Final subfield Dice');fig.tight_layout();fig.savefig(RESULTS/'pilot_results.png',dpi=160);plt.show()
    # Check readable durable metadata before removing only this completed pilot's checkpoints.
    metadata={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.png','.log') and p.name not in ('archive_manifest.json','cleanup.json')}
    manifest_path=RESULTS/'archive_manifest.json'
    manifest_path.write_text(json.dumps(metadata,indent=2))
    for name,expected_sha in json.loads(manifest_path.read_text()).items():
        if sha(RESULTS/name)!=expected_sha: raise ValueError('Archive verification failed')
        if name.endswith('.json'): json.loads((RESULTS/name).read_text())
    deleted=[]
    for checkpoint in [OUTPUT/'checkpoint_latest.pt',RESULTS/'checkpoint_latest.pt']:
        if checkpoint.exists():
            if sha(checkpoint)!=done['checkpoint_sha256']: raise ValueError('Checkpoint changed; retained')
            deleted.append(dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)))
            checkpoint.unlink()
    cleanup_path=RESULTS/'cleanup.json'
    if not cleanup_path.exists(): cleanup_path.write_text(json.dumps(dict(status='complete',deleted=deleted),indent=2))
    print('Results archived and verified:', RESULTS)
    print('Inspect this pilot before choosing a target or scheduling more subjects/seeds/folds.')
    ''')
    result = dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        accelerator='GPU',colab=dict(name='MedSAM3_MNI_one_subject_pilot.ipynb',gpuType='A100',gpuClass='premium',machine_shape='hm'),
        kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    write(EXPORT/'MedSAM3_MNI_one_subject_pilot.ipynb', result)


if __name__ == '__main__':
    build()
