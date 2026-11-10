"""Freeze the 126-run campaign and its Colab notebook for the requested Drive account."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import textwrap
import zipfile

from scripts.colab_sam3_geofwi import runtime as rt
from scripts.colab_sam3_geofwi.replication import ARMS, FOLDS, SEEDS, tasks

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/sam3_geofwi_f013_s012_126_20261019'
OLD = ROOT/'reports/sam3_geofwi_colab_two_runs_20261018'
NOTEBOOK = 'SAM3_GeoFWI3D_126_folds013_seeds012.ipynb'
RESULTS = '1EDy575RclOhGBxhuylaHszLA67wT80qU'


def build():
    sys.path.insert(0,str(ROOT/'scripts'))
    from geofwi3d_replication import fold_splits
    bundle = EXPORT/'bundle'; bundle.mkdir(parents=True,exist_ok=True)
    old_manifest = rt.read(OLD/'bundle/bundle_manifest.json')
    for name, sha in old_manifest.items():
        if name.startswith(('source/','upstream/')) or name in ('runtime.py','engine.py','parallel.py','fast_eval.py','training_loop.py','requirements-colab.txt'):
            if rt.digest(OLD/'bundle'/name) != sha: raise ValueError('Frozen predecessor differs: '+name)
            target = bundle/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(OLD/'bundle'/name,target)
    for name in ('replication.py','replication_store.py','missing_constraints.py'):
        shutil.copy2(ROOT/'scripts/colab_sam3_geofwi'/name,bundle/name)
    shutil.copy2(ROOT/'scripts/colab_sam3_geofwi/diagnose.py',bundle/'sam3_edge_diagnose_v1.py')
    shutil.copy2(ROOT/'scripts/colab_msd_cloud/store.py',bundle/'store.py')
    shutil.copy2(ROOT/'scripts/colab_drive_transport/request_gate.py',bundle/'request_gate.py')
    shutil.copy2(ROOT/'scripts/geofwi3d_replication.py',bundle/'source/fold_split_reference.py')
    original = rt.read(OLD/'warm_edge_results/protocol.json')
    splits = fold_splits(original['manifest']); selected = set()
    for fold in FOLDS:
        support = splits[fold]['training_subsets']['5pct'][:5]
        validation = splits[fold]['validation']
        assert len(support)==5 and len(validation)==96 and not set(support)&set(validation)
        selected.update(support+validation)
        for seed in SEEDS:
            p = deepcopy(original)
            for key in ('arm','fixed_bands_weight','edge_calibration_after_updates','edge_target','edge_p95_cap',
                        'extension_sha256','probe_source_sha256','data','upstream','lora_weights','lora_sha256'):
                p.pop(key,None)
            p.update(fold=fold,seed=seed,train_cases=support,evaluation_cases=validation,
                requested_arms=list(ARMS),name='sam3_geofwi_f013_s012_126_v1',
                gradient_ratio_targets={n:.1 for n in ('bands','edge','sdf','depth')},
                gradient_ratio_p95_caps={n:.5 for n in ('bands','edge','sdf','depth')},
                constraint_calibration_after_updates=5,
                support_selection='First five of the established fold-specific 5pct support; fixed across seeds and arms',
                calibration_policy='Fresh bands per fold/seed at initialization; other selected terms sequentially after five prefix updates; 10% median / 50% p95, fixed coefficients.',
                historical_results_role='Reference only; all 126 runs are fresh, including fold0/seed0 controls')
            p['manifest']['splits'] = splits[fold]
            rt.write(bundle/f'protocols/f{fold}s{seed}.json',p)
    data_manifest = {}
    for case in sorted(selected):
        row = original['manifest']['cases'][case]
        for kind in ('image','label'):
            path = ROOT/'datasets/Dataset203_GeoFWI3D'/row[kind]
            if rt.digest(path) != row[kind+'_sha256']: raise ValueError('Data differs: '+str(path))
            data_manifest[row[kind]] = row[kind+'_sha256']
    rt.write(bundle/'data_manifest.json',data_manifest)
    rt.write(bundle/'plan.json',dict(tasks=tasks(),num_runs=126,folds=FOLDS,seeds=SEEDS,arms=ARMS,
        volumes=len(selected),source_manifest_sha256=rt.digest(OLD/'bundle/bundle_manifest.json')))
    manifest = {str(p.relative_to(bundle)):rt.digest(p) for p in sorted(bundle.rglob('*'))
                if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts}
    rt.write(bundle/'bundle_manifest.json',manifest)
    archives = []
    code = EXPORT/'sam3_126_code.zip'
    with zipfile.ZipFile(code,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'bundle_manifest.json']: z.write(bundle/name,name)
    archives.append(code)
    names = sorted(data_manifest)
    for part in range(12):
        archive = EXPORT/f'sam3_290vol_data12_{part+1:02d}.zip'
        if not archive.exists():
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                for name in names[part::12]: z.write(ROOT/'datasets/Dataset203_GeoFWI3D'/name,'data/'+name)
        with zipfile.ZipFile(archive) as z:
            assert set(z.namelist()) == {'data/'+n for n in names[part::12]}
        archives.append(archive)
    uploads = {p.name:dict(sha256=rt.digest(p),bytes=p.stat().st_size) for p in archives}
    rt.write(EXPORT/'uploads.json',uploads)
    print(json.dumps(dict(volumes=len(selected),runs=126,uploads=uploads),indent=2))


def notebook():
    uploads = rt.read(EXPORT/'uploads.json')
    ids = rt.read(EXPORT/'drive_files.json')
    assets = [dict(name=n,sha256=v['sha256'],id=ids[n]) for n,v in uploads.items()]
    weights = dict(name='sam3.pt',sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',id='1tAGRr_09MO1h2oVe8htEdisLhBP6rKlM')
    cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,metadata={},source=textwrap.dedent(source).strip()+'\n',id=f'cell{len(cells):02d}')
        if kind=='code':
            ast.parse(cell['source']);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','''
    # SAM3: 3 folds × 3 seeds × 14 recipes

    Sign in as **focacciafilippo@gmail.com**, select an **A100/H100 GPU with at least 40 GB VRAM and High-RAM**, then **Run all**.
    GeoFWI3D: folds **0, 1, 3**, seeds **0, 1, 2**; **126 fresh runs**, five training volumes and **25 updates** per run, no augmentation.
    Each model receives one final evaluation on its fold's five support volumes and **96 held-out volumes**.
    Support is fixed across seeds and recipes within each fold; validation folds are disjoint.

    Recipes: native baseline; bands; SDF; bands+SDF; bands+edge; bands+edge+SDF; depth; bands+depth;
    edge+depth; SDF+depth; bands+edge+depth; bands+SDF+depth; edge+SDF+depth; bands+edge+SDF+depth.
    Legacy `dice` labels mean the original **full native SAM3 loss**, not Dice-only training.

    Bands is freshly calibrated once per fold/seed on 20 training slabs. For other selected terms,
    the first five updates use native+bands when bands is present, otherwise native loss.
    Edge → SDF → depth are then calibrated sequentially on those same training slabs,
    using a **10% median target / 50% p95 calibration ceiling**. Coefficients stay fixed;
    the original 20-update constraint ramp and 200-update learning-rate horizon remain unchanged.
    These targets are not persistent gradient bounds. Gradients are monitored at updates 10 and 25.
    All results are retained; earlier exploratory runs are historical references, not substitutes for these 126 runs.

    **Use only one Colab runtime for this campaign.** Runs are sequential, with a fresh process per model.
    This campaign will require multiple sessions. Rerun this same notebook to resume and skip verified completed runs.
    Use the same GPU/software on later sessions. Do not launch a second copy while this runtime is active.
    ''')
    bootstrap=(EXPORT/'bundle/store.py').read_text()
    gate=(EXPORT/'bundle/request_gate.py').read_text()
    transport=(EXPORT/'bundle/replication_store.py').read_text()
    add('code',f'''
from google.colab import auth
auth.authenticate_user()
import os, sys, json, time, shutil, subprocess, signal
from pathlib import Path
STARTED=time.monotonic()
BUNDLE=Path('/content/hippo_sam3_126_v1')
BUNDLE.mkdir(exist_ok=True)
BOOTSTRAP=Path('/content/sam3_126_bootstrap'); BOOTSTRAP.mkdir(exist_ok=True)
SOURCES={{'store.py':{bootstrap!r},'request_gate.py':{gate!r},'replication_store.py':{transport!r}}}
for name,source in SOURCES.items():(BOOTSTRAP/name).write_text(source)
sys.path.insert(0,str(BOOTSTRAP))
from replication_store import Cloud
from store import digest, extract_zip
import store, request_gate
store.pace_write=lambda:request_gate.pace(spacing=.5)
cloud=Cloud({RESULTS!r})
about=cloud.check_account()
quota=about['storageQuota']
print('Verified account:',about['user']['emailAddress'])
print('Drive storage used GiB:',round(int(quota['usage'])/1024**3,2))
if 'limit' in quota:print('Drive free GiB:',round((int(quota['limit'])-int(quota['usage']))/1024**3,2))
if shutil.disk_usage(BUNDLE).free < 18*1024**3:raise RuntimeError('Need 18 GiB free runtime SSD')
ASSETS={assets!r}
WEIGHTS={weights!r}
for asset in ASSETS:
    target=BUNDLE.parent/asset['name']
    cloud.download(asset['id'],target,asset['sha256'])
    extract_zip(target,BUNDLE)
    print('Staged and verified:',asset['name'])
cloud.download(WEIGHTS['id'],BUNDLE/'weights/sam3.pt',WEIGHTS['sha256'])
print('All data, source and weights are local. Drive is not mounted.')
''')
    add('markdown','''
    ## Pinned environment

    Reuses matching packages, otherwise installs the same versions as the successful exploration.
    Source and all data hashes are checked before training. The first run records the GPU/software binding.
    ''')
    add('code','''
    VENV=Path('/content/hippo_sam3_126_venv'); PYTHON=VENV/'bin/python'
    def setup(args):
        result=subprocess.run(list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with (BUNDLE/'setup.log').open('a') as f:f.write(result.stdout)
        print('\\n'.join(result.stdout.splitlines()[-25:]))
        if result.returncode:raise RuntimeError('Environment setup failed; see local setup.log')
    if not PYTHON.exists():setup([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV])
    if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
        import urllib.request
        path=Path('/content/sam3_get_pip.py')
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py',path)
        setup([PYTHON,path])
    probe="import importlib.metadata as m,sys; from pathlib import Path; lines=Path(sys.argv[1]).read_text().splitlines(); ok=all(m.version(s.split('==')[0])==s.split('==')[1] if '==' in s else bool(m.version(s)) for s in lines if s.strip()); sys.exit(0 if ok else 1)"
    if subprocess.run([str(PYTHON),'-c',probe,str(BUNDLE/'requirements-colab.txt')],capture_output=True).returncode:
        setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'])
    os.environ.update(PYTHONUNBUFFERED='1',WANDB_MODE='disabled',MPLCONFIGDIR=str(BUNDLE/'matplotlib_cache'))
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    add('markdown','''
    ## Run or resume

    Training, per-slice data access, logs and evaluation use local SSD. Drive receives paced API requests,
    compact result archives, small acknowledged update ledgers and occasional recovery archives.
    A real server write/read check must pass first; transient quota errors use bounded exponential backoff.
    Finished checkpoints are removed only after the complete result archive is verified on Google's server.
    Previous recovery archives are retired after a replacement is acknowledged.

    **Interrupt once, then wait for the clean-stop message before disconnecting.** Recovery is saved at
    training boundaries and approximately every three minutes, with small evaluation progress snapshots.
    A sudden VM loss can leave charged updates beyond the latest saved checkpoint: the runner stops for
    an audit instead of silently repeating them. If Drive refuses writes, keep the VM connected; local files
    remain available. No notebook can guarantee exemption from Google's account/server quota limits.

    `SESSION_HOURS` is a clean-stop budget, not a promise about Colab runtime availability.
    ''')
    add('code','''
    SESSION_HOURS=11.0
    remaining=SESSION_HOURS-(time.monotonic()-STARTED)/3600
    if remaining<=.1:raise RuntimeError('Session budget exhausted; resume in a fresh matching runtime')
    process=subprocess.Popen([str(PYTHON),str(BUNDLE/'replication.py'),'--bundle',str(BUNDLE),
        '--hours',str(remaining)],start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (BUNDLE/'session.log').open('a') as log:
        while True:
            try:
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait();break
            except KeyboardInterrupt:
                print('Clean stop requested. Keep this VM connected until recovery finishes.')
                if process.poll() is None:process.send_signal(signal.SIGUSR1)
    if code:raise RuntimeError('Campaign stopped on an error; local/recovery files retained. See output above.')
    ''')
    add('code','''
    import pandas as pd
    path=BUNDLE/'durable/comparison.json'
    if path.exists():
        result=json.loads(path.read_text()); print(result['completed'],'/',result['total'],'completed')
        table=pd.DataFrame(result['rows'])
        if not table.empty:
            table['validation_dice_percent']=100*table['validation_dice']
            display(table[['fold','seed','arm','validation_dice_percent','foreground_fn','foreground_fp']].sort_values(['fold','seed','arm']))
            display(table.pivot_table(index='arm',columns=['fold','seed'],values='validation_dice_percent'))
    else:print('No completed runs yet.')
    print('Drive results: https://drive.google.com/drive/folders/1EDy575RclOhGBxhuylaHszLA67wT80qU')
    ''')
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        accelerator='GPU',colab=dict(name=NOTEBOOK)))
    rt.write(EXPORT/NOTEBOOK,nb)
    print(EXPORT/NOTEBOOK)


if __name__=='__main__':
    if '--notebook' in sys.argv:notebook()
    else:build()
