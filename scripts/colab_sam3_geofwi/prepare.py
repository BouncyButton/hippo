"""Build a portable notebook and immutable code/data packages for SAM3 Colab."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parents[2]
EXPORT=ROOT/'reports/sam3_geofwi_colab_exploration_20261018'
PARENT=ROOT/'reports/sam3_geofwi_5vol_update_sweep_20261016'
MED=ROOT/'reports/medsam3_msd_colab_matched_3fold_3seed_noaug_20261015/bundle'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def notebook(uploads,manifest_sha):
    cells=[]
    def cell(kind,source):
        value=dict(cell_type=kind,metadata={},source=textwrap.dedent(source).strip()+'\n')
        if kind=='code':value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell('markdown','''
    # SAM3 GeoFWI3D — efficient five-volume exploration

    Run the frozen **24-condition** update/constraint comparison on an **A100 (40/80 GB), High RAM** Colab runtime.
    Five training volumes, all 96 validation volumes, fold 0 / seed 0, no augmentation, original native SAM3 loss,
    fresh zero-output LoRA, accumulation 2, and the same fixed 200-update learning-rate horizon.
    These are update-matched experiments, **not** the four-model FLOP-capped baselines.

    | Conditions | Stopping updates |
    |---|---|
    | Native and bands 20% | 10, 25, 50, 100, 200 |
    | Bands 10% and 40% | 50, 100 |
    | Bands 20% + edge 0.5%, 1%, 2%, 5% | 50, 100 |
    | Bands 20% + edge 1%, 2% | 200 |

    Percentages are sequential auxiliary **gradient targets** against the native loss (which includes Dice),
    with a p95 ceiling of five times each target. They are not literal loss coefficients.
    Colab runs fresh native controls: the two historical cluster results are retained as reference only.

    ## Upload once
    Put `sam3_geofwi_code.zip` and `sam3_geofwi_101vol_data.zip` in
    `MyDrive/hippo_sam3_geofwi_exploration/input`. Reuse the exact `sam3.pt` already used by MedSAM3 MSD;
    the setup cell checks the original hash. No medical adapter is loaded.
    If using your **Second account**, mount Drive as **filippofocaccia2@gmail.com**.

    Run cells in order. Use one VM per RESULTS folder. A new VM with the same GPU/software resumes
    verified recovery; already completed runs are skipped. This notebook does not submit or modify cluster jobs.
    ''')
    cell('code','''
    from google.colab import drive, auth
    drive.mount('/content/drive')
    auth.authenticate_user()
    from pathlib import Path
    import hashlib, json, os, signal, subprocess, sys, time
    NOTEBOOK_STARTED=time.monotonic()
    HOME_DIR=Path('/content/drive/MyDrive/hippo_sam3_geofwi_exploration')
    INPUT=HOME_DIR/'input'
    RESULTS=HOME_DIR/'results_5vol_24conditions_v1'
    BUNDLE=Path('/content/hippo_sam3_geofwi_5vol_bundle')
    BASE_WEIGHTS=Path('/content/drive/MyDrive/hippo_medsam3_msd_extension/input/sam3.pt')
    # Change BASE_WEIGHTS if your existing sam3.pt is elsewhere; it is never overwritten.
    RESULTS.mkdir(parents=True,exist_ok=True)
    print('Results and recovery:',RESULTS)
    ''')
    cell('markdown','''
    ## Stage verified inputs on the local SSD
    Files already present with matching hashes are reused. Different files cause a stop; active work is preserved.
    The data ZIP contains only the exact five training and 96 validation volumes, not the full dataset.
    ''')
    cell('code','EXPECTED_UPLOADS='+repr(uploads)+'\nEXPECTED_MANIFEST='+repr(manifest_sha)+'\n'+textwrap.dedent('''
    import fcntl, shutil, zipfile
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    def safe(name):
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe ZIP path')
        return p
    def install_zip(path,expected,allowed):
        if sha(path)!=expected:raise ValueError('ZIP hash mismatch: '+str(path))
        with zipfile.ZipFile(path) as archive:
            if len(archive.namelist())!=len(set(archive.namelist())):raise ValueError('Duplicate ZIP members')
            if set(archive.namelist())!=set(allowed):raise ValueError('ZIP inventory mismatch')
            for info in archive.infolist():
                safe(info.filename)
                if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlink in ZIP')
            for name,expected_sha in allowed.items():
                target=BUNDLE/safe(name)
                if target.exists():
                    if target.is_symlink() or sha(target)!=expected_sha:raise ValueError('Existing file differs: '+str(target))
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_name(target.name+'.staging')
                with archive.open(name) as source,tmp.open('wb') as dest:shutil.copyfileobj(source,dest)
                if sha(tmp)!=expected_sha:raise ValueError('Extracted member differs')
                tmp.replace(target)
    (BUNDLE/'work').mkdir(parents=True,exist_ok=True)
    with (BUNDLE/'work/sweep.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        package=INPUT/'sam3_geofwi_code.zip'
        if sha(package)!=EXPECTED_UPLOADS[package.name]:raise ValueError('Code ZIP changed')
        with zipfile.ZipFile(package) as archive:
            raw=archive.read('bundle_manifest.json')
            if hashlib.sha256(raw).hexdigest()!=EXPECTED_MANIFEST:raise ValueError('Scientific manifest differs')
            files=json.loads(raw);files['bundle_manifest.json']=EXPECTED_MANIFEST
        install_zip(package,EXPECTED_UPLOADS[package.name],files)
        data_files={'data/'+n:h for n,h in json.loads((BUNDLE/'data_manifest.json').read_text()).items()}
        if any(not (BUNDLE/n).exists() or sha(BUNDLE/n)!=h for n,h in data_files.items()):
            install_zip(INPUT/'sam3_geofwi_101vol_data.zip',EXPECTED_UPLOADS['sam3_geofwi_101vol_data.zip'],data_files)
        base=BUNDLE/'weights/sam3.pt';base.parent.mkdir(exist_ok=True)
        if not base.exists():
            if shutil.disk_usage(BUNDLE).free<16*1024**3:raise RuntimeError('Need16 GiB free runtime SSD before staging')
            if sha(BASE_WEIGHTS)!=EXPECTED_UPLOADS['sam3.pt']:raise ValueError('Existing sam3.pt differs')
            tmp=base.with_suffix('.staging');shutil.copyfile(BASE_WEIGHTS,tmp)
            if sha(tmp)!=EXPECTED_UPLOADS['sam3.pt']:raise ValueError('Checkpoint copy differs')
            tmp.replace(base)
        elif sha(base)!=EXPECTED_UPLOADS['sam3.pt']:raise ValueError('Local SAM3 checkpoint differs')
        provenance=RESULTS/'source'
        provenance.mkdir(exist_ok=True)
        dest=provenance/package.name
        if not dest.exists():shutil.copyfile(package,dest)
        if sha(dest)!=EXPECTED_UPLOADS[package.name]:raise ValueError('Durable source package differs')
    print('Verified frozen source,101 volumes and SAM3 weights on local SSD.')
    '''))
    cell('markdown','''
    ## Environment and commands
    Reuses matching installed versions; otherwise installs the same pinned dependencies as the MSD notebook.
    An A100/H100-class BF16 GPU with at least 40 GB VRAM is required. Colab speed and CUDA equivalence
    are checked on your actual GPU; local packaging tests cannot establish runtime speedups.
    ''')
    cell('code','''
    VENV=Path('/content/hippo_sam3_geofwi_venv');PYTHON=VENV/'bin/python'
    def setup(args):
        p=subprocess.run(list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with (RESULTS/'setup.log').open('a') as log:log.write(p.stdout)
        print('\\n'.join(p.stdout.splitlines()[-30:]))
        if p.returncode:raise RuntimeError('Setup failed; see setup.log')
    if not PYTHON.exists():setup([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV])
    if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
        import urllib.request
        path=Path('/content/sam3_get_pip.py')
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py',path)
        setup([PYTHON,path])
    probe="import importlib.metadata as m,sys; from pathlib import Path; lines=Path(sys.argv[1]).read_text().splitlines(); ok=all(m.version(s.split('==')[0])==s.split('==')[1] if '==' in s else bool(m.version(s)) for s in lines if s.strip()); sys.exit(0 if ok else 1)"
    if subprocess.run([str(PYTHON),'-c',probe,str(BUNDLE/'requirements-colab.txt')],capture_output=True).returncode:
        setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'])
    else:print('Reusing matching dependency versions')
    def execute(args,log_name,parallel=False):
        p=subprocess.Popen(list(map(str,args)),start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        with (RESULTS/log_name).open('a') as log:
            while True:
                try:
                    for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    code=p.wait();break
                except KeyboardInterrupt:
                    print('Clean stop requested. Keep this VM connected until the cell returns and recovery is verified.')
                    if p.poll() is None:
                        if parallel:p.send_signal(signal.SIGUSR1)
                        else:os.killpg(p.pid,signal.SIGINT)
        if code:raise RuntimeError('Command failed; see '+str(RESULTS/log_name))
    def campaign(command):
        execute([PYTHON,BUNDLE/'runtime.py',command,'--bundle',BUNDLE,'--durable',RESULTS],command+'.log')
    campaign('verify')
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    cell('markdown','''
    ## Shared GPU calibration — once per matching runtime
    Runs the original native/bands/edge backward checks, then measures the three-loss gradient Gram matrix
    on exactly20 slabs from the five training volumes. Each variant derives its own sequential median/p95
    coefficients using the unchanged calibrator; these raw measurements do not depend on stopping updates
    or requested target. No historical cluster coefficients are imported. Partial calibration resumes.
    A passed preflight is reused only for identical source, data, base weights, GPU and software.
    ''')
    cell('code',"campaign('preflight')")
    cell('markdown','''
    ## Run or resume
    `MAX_WORKERS=0` adapts to available VRAM, host RAM, CPU and observed throughput. Startup is staggered;
    a lower limit lets existing workers finish. Each worker owns its model, optimizer, RNG and run folder.
    All five native stopping-point controls finish before new constraint runs. Every30 seconds the cell reports
    GPU utilization, workers, completed runs, updates, evaluation volume and slice.

    Deterministic CPU transforms are cached with a1 GiB bound per worker and released for evaluation.
    This binary problem has one prompt: no two-class image-sharing speedup is assumed. Evaluation-only
    text reuse requires identical labels, strict log-probability agreement, unchanged RNG and at least2%
    observed slice-time improvement on the first full volume; the first slice of every later volume is checked.
    Failed checks use the original evaluator. Every slice and all101 volumes remain in the evaluation.

    Local checkpoints are atomic after every update; coherent Drive snapshots upload asynchronously about
    every3 minutes. **Interrupt once and wait** for clean-stop recovery to finish. Completed volumes and a
    partial current volume are retained on clean stop. A hard VM loss with charged updates newer than saved
    recovery stops for an audit rather than silently replaying updates. Finished metrics, curves and logs are
    verified on Drive before their checkpoints are removed. Keep this Colab tab/runtime connected.

    Only one VM may write this RESULTS folder. The local lock prevents overlapping cells on one VM; it is
    not a cross-VM lock. A manual RESULTS/STOP file is respected. The22-hour budget includes setup in this
    notebook; reduce it if the VM was already in use. Colab may interrupt earlier.
    ''')
    cell('code','''
    MAX_WORKERS=0
    MAX_NEW_RUNS=24
    SESSION_HOURS=22.0
    FAST_EVALUATION=True
    remaining=SESSION_HOURS-(time.monotonic()-NOTEBOOK_STARTED)/3600
    if remaining<=0.1:raise RuntimeError('Session budget exhausted; resume in a fresh matching runtime')
    args=[PYTHON,BUNDLE/'parallel.py','--bundle',BUNDLE,'--durable',RESULTS,
          '--max-workers',str(MAX_WORKERS),'--max-runs',str(MAX_NEW_RUNS),'--hours',str(remaining)]
    if FAST_EVALUATION:args.append('--fast-evaluation')
    execute(args,'parallel_run.log',parallel=True)
    ''')
    cell('markdown','''
    ## Compare completed results
    Gains use fresh Colab native and bands20 controls at the same stopping update.
    The JSON also includes paired volume wins/losses, p10 and worst Dice change, FN/FP, and historical results.
    Incomplete controls leave gains blank. This is a single-fold/seed exploration.
    ''')
    cell('code','''
    campaign('report')
    import pandas as pd
    result=json.loads((RESULTS/'comparison.json').read_text())
    table=pd.DataFrame(result['runs'])
    if not table.empty:
        table['validation_dice_percent']=100*table['validation_dice']
        columns=['updates','scientific_arm','bands_target','edge_target','validation_dice_percent',
                 'gain_vs_native_pp','gain_vs_bands20_pp','foreground_fn','foreground_fp']
        display(table[columns].sort_values(['updates','scientific_arm','bands_target','edge_target']))
        table.to_csv(RESULTS/'comparison.csv',index=False)
    print(result['new_completed'],'/',result['new_total'],'complete')
    ''')
    return dict(nbformat=4,nbformat_minor=5,cells=[dict(id=f'cell{i:02d}',**c) for i,c in enumerate(cells)],
        metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
                      language_info=dict(name='python',version='3.12'),accelerator='GPU',colab=dict(name='SAM3_GeoFWI3D_efficient_exploration.ipynb')))


def main():
    frozen=PARENT/'frozen';bundle=EXPORT/'bundle';bundle.mkdir(parents=True,exist_ok=True)
    original_manifest=json.loads((frozen/'manifest.json').read_text())
    for name,sha in original_manifest.items():
        if digest(frozen/name)!=sha:raise ValueError('Frozen predecessor changed: '+name)
    shutil.copytree(frozen/'source',bundle/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(frozen/'protocols',bundle/'protocols',dirs_exist_ok=True)
    shutil.copytree(MED/'upstream',bundle/'upstream',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    hashes=json.loads((frozen/'source_version.json').read_text())['upstream_provenance']['python_source_sha256']
    for name,sha in hashes.items():
        if digest(bundle/'upstream'/name)!=sha:raise ValueError('Upstream differs: '+name)
    for name in ('runtime.py','engine.py','parallel.py','fast_eval.py','training_loop.py'):
        shutil.copy2(ROOT/'scripts/colab_sam3_geofwi'/name,bundle/name)
    shutil.copy2(MED/'requirements-colab.txt',bundle/'requirements-colab.txt')
    reference=bundle/'reference';reference.mkdir(exist_ok=True)
    for name in ('plan.json','manifest.json','source_version.json','original_five_volume_protocol.json'):
        shutil.copy2(frozen/name,reference/name)
    shutil.copy2(PARENT/'comparison.json',reference/'comparison.json')
    original=json.loads((frozen/'plan.json').read_text()); tasks=deepcopy(original['tasks'])
    for task in tasks:
        task['scientific_arm']=task['arm']
        if task['arm']=='dice':task['arm']='baseline'
    tasks.sort(key=lambda t:(t['arm']!='baseline',t['updates'],t['id']))
    plan=dict(name='sam3_geofwi_colab_5vol_24conditions_v1',tasks=tasks,num_runs=24,
        base_sha256=original['base_sha256'],source_manifest_sha256=digest(frozen/'manifest.json'),
        calibration_policy='fresh_shared_native_bands_edge_gram',historical_results_role='reference_only')
    write(bundle/'plan.json',plan)
    p=json.loads((frozen/'original_five_volume_protocol.json').read_text());data_manifest={}
    for case in p['train_cases']+p['evaluation_cases']:
        for kind in ('image','label'):
            row=p['manifest']['cases'][case];name=row[kind];sha=row[kind+'_sha256']
            source=ROOT/'datasets/Dataset203_GeoFWI3D'/name
            if digest(source)!=sha:raise ValueError('Frozen data differs: '+name)
            data_manifest[name]=sha
    write(bundle/'data_manifest.json',data_manifest)
    data=EXPORT/'sam3_geofwi_101vol_data.zip'
    with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_STORED) as z:
        for name in sorted(data_manifest):z.write(ROOT/'datasets/Dataset203_GeoFWI3D'/name,'data/'+name)
    manifest={str(f.relative_to(bundle)):digest(f) for f in sorted(bundle.rglob('*'))
        if f.is_file() and f.name!='bundle_manifest.json' and '__pycache__' not in f.parts}
    write(bundle/'bundle_manifest.json',manifest)
    code=EXPORT/'sam3_geofwi_code.zip'
    with zipfile.ZipFile(code,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'bundle_manifest.json']:z.write(bundle/name,name)
    uploads={code.name:digest(code),data.name:digest(data),'sam3.pt':plan['base_sha256']}
    write(EXPORT/'uploads.json',uploads)
    write(EXPORT/'SAM3_GeoFWI3D_efficient_exploration.ipynb',notebook(uploads,digest(bundle/'bundle_manifest.json')))
    print(json.dumps(dict(export=str(EXPORT),code_bytes=code.stat().st_size,data_bytes=data.stat().st_size,
                         runs=24,volumes=101,manifest_sha256=digest(bundle/'bundle_manifest.json')),indent=2))


if __name__=='__main__':main()
