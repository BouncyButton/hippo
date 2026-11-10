"""Package frozen native-grid data, four baseline recipes and a Colab notebook."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import textwrap
import zipfile

from scripts.colab_geofwi_seed_search.runtime import CAP, MODELS, digest, read, write

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/geofwi3d_colab_seed_search_20261018'
FULL=ROOT/'experiments/cross_dataset_full_20261014/protocols'
CE=ROOT/'reports/adni_seismic_ce60_followon_20261014/frozen'


def notebook(uploads,manifest):
    cells=[]
    def cell(kind,text):
        c=dict(cell_type=kind,metadata={},source=textwrap.dedent(text).strip()+'\n',id=f'cell-{len(cells):02}')
        if kind=='code':c.update(outputs=[],execution_count=None)
        cells.append(c)
    cell('markdown','''
    # GeoFWI3D — find three qualifying baseline seeds across three folds

    **Completion means 3 seeds × 3 folds × 4 models = 36 verified passing results.**
    Each seed must achieve **strictly greater than 50%** final-budget validation foreground Dice in
    **every one of its 12 model/fold cells**. Exactly 50% fails. No rounding or averaging is used for selection.
    Seeds are tried in numerical order, starting at 0. On the first completed failing cell, reject the entire
    seed, retain its evidence, skip its remaining cells and continue. Technical errors pause the search;
    they do not reject a seed. Only the first three qualifying seeds are selected.

    - GeoFWI3D native 96³ seismic images, physical salt masks, frozen folds 0/1/2.
    - 19 training and 96 validation volumes per fold; the same subsets for every model/seed.
    - **No augmentation**, batch 1, BF16 autocast with FP32 parameters/optimizer, training from scratch.
    - **182,178.12654316 GFLOPs per run**; final budget checkpoint is primary; best validation is secondary.
    - nnU-Net: native foreground Dice+CE, deep supervision and SGD, **151 updates**.
    - SwinUNETR: Dice and AdamW, **362 updates**; UNETR: **414**; 3D U-Net width 16: **952**.
    - Original learning-rate trajectories stay fixed. Preflight, validation, I/O and preprocessing are outside
      the optimization cap and reported separately. There are no constraints or auxiliary calibration.

    **This is a validation-selected seed cohort.** Keep all attempted/rejected scores when reporting it;
    it is not an unbiased random-seed estimate or an independent test. Success is not guaranteed at this cap.
    The notebook never marks a session limit, technical failure, or insufficient passing seeds as complete.

    Select an **A100 GPU and High RAM**, then run all cells. Inputs are already uploaded to
    `MyDrive/hippo_geofwi3d_baseline_seed_search/input` on **filippofocaccia2@gmail.com**.
    No SSH credentials, cluster access or pretrained model downloads are required.
    Run only one copy of this notebook against this results folder.

    **Revision 2 (18 October):** fixes AdamW checkpoint device placement in preflight and training resume.
    If the previous version stopped in preflight, reload this notebook and run all cells from the top.
    The code stages into a fresh runtime directory; the existing data and Drive records are preserved.
    ''')
    cell('code','''
    from google.colab import drive, auth
    drive.mount('/content/drive')
    auth.authenticate_user()
    from googleapiclient.discovery import build
    import google.auth
    credentials, _ = google.auth.default()
    service = build('drive','v3',credentials=credentials,cache_discovery=False)
    about = service.about().get(fields='user(emailAddress),storageQuota').execute()
    assert about['user']['emailAddress'].lower() == 'filippofocaccia2@gmail.com', 'Mount/authenticate the requested account.'
    quota=about['storageQuota']
    if 'limit' in quota:
        assert int(quota['limit'])-int(quota['usage']) >= 8*1024**3, 'Keep at least 8 GiB free in Drive, including trash.'
    from pathlib import Path
    import hashlib,json,os,shutil,signal,subprocess,sys,time,zipfile
    HOME_DIR=Path('/content/drive/MyDrive/hippo_geofwi3d_baseline_seed_search')
    INPUT=HOME_DIR/'input'
    RESULTS=HOME_DIR/'results_mixed_noaug_cap182178_v1'
    BUNDLE=Path('/content/hippo_geofwi_seed_bundle_r2')
    RESULTS.mkdir(parents=True,exist_ok=True)
    BUNDLE.mkdir(exist_ok=True)
    assert shutil.disk_usage('/content').free >= 20*1024**3, 'Need at least 20 GiB runtime disk space.'
    print('Account:',about['user']['emailAddress'],'Results:',RESULTS)
    ''')
    cell('markdown','''
    ## Verify and stage the uploaded data/code
    All archive and per-file SHA256 hashes are checked. The 319 unique images/labels are read from
    the runtime SSD during training. Existing matching files are reused; changed files stop execution.
    ''')
    cell('code','EXPECTED_UPLOADS='+repr(uploads)+'\nEXPECTED_MANIFEST='+repr(manifest)+'\n'+textwrap.dedent('''
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        return h.hexdigest()
    for name,expected in EXPECTED_UPLOADS.items():
        package=INPUT/name
        if not package.exists():raise FileNotFoundError(str(package))
        assert sha(package)==expected, 'Upload checksum mismatch: '+name
        local=Path('/content')/name
        if not local.exists() or sha(local)!=expected:
            shutil.copyfile(package,local)
        assert sha(local)==expected
        with zipfile.ZipFile(local) as z:
            assert len(z.namelist())==len(set(z.namelist())), 'Duplicate archive members'
            for info in z.infolist():
                p=Path(info.filename)
                assert not p.is_absolute() and '..' not in p.parts
                assert (info.external_attr>>16)&0o170000 != 0o120000
            for info in z.infolist():
                target=BUNDLE/info.filename
                if target.exists():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_name(target.name+'.extracting')
                with z.open(info) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
                temp.replace(target)
        print('Staged',name)
    assert sha(BUNDLE/'bundle_manifest.json')==EXPECTED_MANIFEST
    for listing in ['bundle_manifest.json','data_manifest.json']:
        for name,expected in json.loads((BUNDLE/listing).read_text()).items():
            assert sha(BUNDLE/name)==expected, 'Extracted file mismatch: '+name
    source=RESULTS/'provenance'/EXPECTED_MANIFEST;source.mkdir(parents=True,exist_ok=True)
    for name in ['geofwi3d_seed_search_code.zip']:
        dst=source/name
        if not dst.exists():shutil.copyfile(INPUT/name,dst)
        assert sha(dst)==EXPECTED_UPLOADS[name]
    print('All inputs verified. No augmentation.')
    '''))
    cell('markdown','''
    ## Install the frozen environment
    Uses the package versions from the existing Colab training environment, in a separate venv.
    Setup output is streamed and saved in Drive. Keep the same GPU type/software on resume.
    ''')
    cell('code','''
    VENV=Path('/content/hippo_geofwi_seed_venv');PYTHON=VENV/'bin/python'
    def command(args,log_name):
        with (RESULTS/log_name).open('a') as log:
            proc=subprocess.Popen(list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,start_new_session=True)
            tail=[]
            def drain():
                for line in proc.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                    tail.append(line);del tail[:-40]
            try:drain();code=proc.wait()
            except KeyboardInterrupt:
                # Signal the orchestrator only; it stops the active trainer at a safe boundary.
                proc.send_signal(signal.SIGINT)
                print('Saving recovery to Drive. Please wait; do not interrupt again.')
                drain();code=proc.wait()
            finally:proc.stdout.close()
            if code:raise RuntimeError('Process failed; recovery retained.\\n'+''.join(tail))
    if not PYTHON.exists():
        command([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV],'setup.log')
    if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
        command([PYTHON,'-m','ensurepip','--upgrade'],'setup.log')
    command([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'],'setup.log')
    def campaign(action,*extra):
        command([PYTHON,'-u',BUNDLE/'runtime.py',action,'--bundle',BUNDLE,'--durable',RESULTS,*extra],action+'.log')
    campaign('verify')
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    cell('markdown','''
    ## Required GPU preflight
    All four models must pass a real native-96³ BF16 forward/backward, finite-gradient check,
    binary confusion-count evaluation and exact model/optimizer/RNG checkpoint replay before screening.
    These diagnostic updates are saved as preflight evidence, never scientific results.
    ''')
    cell('code',"campaign('preflight')")
    cell('markdown','''
    ## Run or resume the search
    The job continues until three seeds qualify. A session timer pauses safely; rerun this cell or
    the notebook in a new runtime to continue. Completed and rejected seeds are reconstructed from
    hash-verified archives. Already completed experiments are never rerun automatically.

    Interrupt **once** to request a graceful checkpoint and wait for the Drive upload. A hard VM loss
    can occur between the latest recovery snapshot and later charged updates; in that case the notebook
    stops for a compute/recovery audit instead of silently losing updates, exceeding the cap, or discarding
    that seed. No automatic restart can guarantee recovery of work that Colab destroyed before upload.
    ''')
    cell('code','''
    SESSION_HOURS=10.0  # Conservative clean-stop target; increase only within your session allowance.
    MAX_NEW_RUNS_THIS_SESSION=10000  # A session guard, never a scientific completion rule.
    # Creating RESULTS/'STOP' requests a stop. Remove it only when intentionally resuming.
    campaign('run','--hours',str(SESSION_HOURS),'--max-runs',str(MAX_NEW_RUNS_THIS_SESSION))
    ''')
    cell('markdown','''
    ## Inspect results and the completion marker
    `search_status.json` records qualified/rejected seeds and reasons. `all_attempted_results.csv`
    preserves every completed cell. `qualified_baselines.csv` contains the selected cohort.
    `COMPLETE.json` exists only after all **36** selected cells have final validation Dice **>50%**.
    Per-run histories, final/best metrics, FN/FP counts, configuration, seed/fold, initialization hashes,
    FLOPs ledger, source provenance and logs are preserved. Checkpoints are removed only after verified
    archival. The input data and source archives remain in Drive for future paired constraint experiments.
    ''')
    cell('code','''
    campaign('report')
    import pandas as pd
    status=json.loads((RESULTS/'search_status.json').read_text())
    print('Status:',status['status'],'Qualified seeds:',status['qualified_seeds'])
    display(pd.DataFrame(status['decisions']))
    if status['runs']:
        table=pd.DataFrame(status['runs'])
        table['validation_dice_percent']=100*table['final_dice']
        display(table.pivot(index=['seed','fold'],columns='model',values='validation_dice_percent'))
    if (RESULTS/'COMPLETE.json').exists():
        print('COMPLETE:',json.loads((RESULTS/'COMPLETE.json').read_text()))
    else:print('Not complete: resume the run cell when ready.')
    ''')
    return dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        colab=dict(name='GeoFWI3D_find_3_solid_seeds.ipynb',provenance=[]),
        kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python'),accelerator='GPU'))


def prepare():
    stage=OUT/'bundle';stage.mkdir(parents=True,exist_ok=True)
    with tarfile.open(OUT/'geofwi3d_data.tar') as tar:
        for member in tar:
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts or not member.isfile():raise ValueError('Unsafe data member')
            target=stage/p
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    folds=[read(stage/f'data/fold_{f}_manifest.json') for f in range(3)]
    splits=[dict(train=m['splits']['training_subsets']['5pct'],val=m['splits']['validation']) for m in folds]
    write(stage/'splits.json',splits)
    ids={c for s in splits for c in s['train']+s['val']}
    assert len(ids)==319
    hashes={}
    for f,m in enumerate(folds):
        assert splits[f]==read(FULL/f'GeoFWI3D_swinunetr_f{f}_target10/splits_10patients.json')[f]
        assert len(set(splits[f]['train']))==19 and len(set(splits[f]['val']))==96
        assert not set(splits[f]['train'])&set(splits[f]['val'])
        assert not {m['cases'][c]['seed'] for c in splits[f]['train']}&{m['cases'][c]['seed'] for c in splits[f]['val']}
        for c in ids:
            for key in ('image','label'):
                name='data/'+m['cases'][c][key];expected=m['cases'][c][key+'_sha256']
                if name not in hashes:
                    assert digest(stage/name)==expected,name
                    hashes[name]=expected
        hashes[f'data/fold_{f}_manifest.json']=digest(stage/f'data/fold_{f}_manifest.json')
    assert len({c for s in splits for c in s['val']})==288
    write(stage/'data_manifest.json',hashes)
    source=CE/'source'
    for f in source.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts:
            target=stage/'source'/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(f,target)
    # Scoped, recorded runtime repair: preserve frozen recipes and the parent source archive.
    trainer=stage/'source/scripts/cross_dataset_campaign/capped_core.py'
    before=trainer.read_text();before_sha=digest(trainer)
    old='cp = torch.load(latest, map_location=device, weights_only=False)'
    new='cp = torch.load(latest, map_location="cpu", weights_only=False)'
    if before.count(old)!=1:raise ValueError('Frozen checkpoint loader changed; inspect before patching')
    trainer.write_text(before.replace(old,new))
    runtime_patches=[dict(path=str(trainer.relative_to(stage/'source')),
        before_sha256=before_sha,after_sha256=digest(trainer),old=old,new=new,
        reason='Load on CPU; optimizer.load_state_dict places moments on the parameter device and keeps noncapturable AdamW step on CPU')]
    updates={}
    provenance={}
    for model in MODELS:
        src=CE/'protocols'/f'GeoFWI3D_{model}' if model=='nnunet' else FULL/f'GeoFWI3D_{model}_f0_target10'
        p=deepcopy(read(src/'protocol.json'));alias=p['trainer_alias']
        p['models']={alias:dict(parameters=p['models'][alias]['parameters'],
                               objectives={'dice':p['models'][alias]['objectives']['dice']})}
        p.update(name='geofwi3d_colab_seed_search_20261018',weight_ce=int(model=='nnunet'),weight_dice=1,
                 constraints=False,requested_arm_order=['dice'],sdf_policy={'dice':'not_applicable'},
                 training_loss='Native foreground Dice+CE with deep supervision' if model=='nnunet' else 'Dice including background',
                 geometry='Native seismic 96x96x96 grid; voxel units',
                 label_budget='19 volumes per fold, frozen 5% training subset; 96 validation volumes',
                 constraint_geometry='Not applicable: baseline only',calibration_policy='Not applicable: baseline only',
                 checkpoint_retention='Archive verified scalar records, then retire completed checkpoints',
                 data='generated per seed/fold',pkl_sha256='generated per seed/fold',
                 folds=[0,1,2],seeds=[],initialization_policy='Fresh CPU seeded initialization; identical hash required across folds; fixed Colab environment',
                 parent_protocol_sha256=digest(src/'protocol.json'))
        p.pop('baseline_reference',None)
        schedule=p['models'][alias]['objectives']['dice']
        assert schedule['max_steps']==CAP//schedule['flops_per_update']
        assert p['budget_flops']==CAP
        updates[model]=schedule['max_steps']
        folder=stage/'templates'/model;folder.mkdir(parents=True,exist_ok=True)
        write(folder/'protocol.json',p)
        shutil.copyfile(src/'nnunet_architecture.json',folder/'nnunet_architecture.json')
        profile=read(src/'profile.json')
        write(folder/'profile_baseline.json',dict(parent_profile_sha256=digest(src/'profile.json'),
            parameters=p['models'][alias]['parameters'],baseline_schedule=schedule,
            cost=profile.get('costs',{}).get('dice'),accounting=p['accounting']))
        provenance[model]=dict(protocol=str(src.relative_to(ROOT)),sha256=digest(src/'protocol.json'),
                              profile_sha256=digest(src/'profile.json'))
    for name in ('runtime.py','worker.py'):
        shutil.copyfile(ROOT/'scripts/colab_geofwi_seed_search'/name,stage/name)
    shutil.copyfile(ROOT/'scripts/colab_msd/runtime.py',stage/'storage_runtime.py')
    pins=(ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015/bundle/requirements-colab.txt').read_text()
    (stage/'requirements-colab.txt').write_text(pins+'google-api-python-client>=2,<3\ngoogle-auth>=2,<3\n')
    write(stage/'plan.json',dict(dataset='GeoFWI3D',models=list(MODELS),folds=[0,1,2],augmentation='none',
        seed_order='0,1,2,... until three qualify',threshold=.5,comparison='strictly_greater',required_seeds=3,
        required_passing_cells=36,budget_flops=CAP,updates=updates,
        seed_rejection='First completed final-budget cell <=0.5 rejects the seed globally; preserve evidence',
        technical_errors='Pause with recovery; never reject seed or mark complete',
        primary_metric='Mean per-volume foreground hard Dice at final budget',
        selection_bias='Validation-selected cohort; all attempted scores retained; no unbiased random-seed claim'))
    write(stage/'source_provenance.json',dict(parent_frozen_source=str(source.relative_to(ROOT)),
        parent_manifest_sha256=digest(CE/'manifest.json'),protocols=provenance,
        runtime_patches=runtime_patches,
        new_wrapper_source='scripts/colab_geofwi_seed_search',
        git_head=subprocess.check_output(['rtk','git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    files={str(p.relative_to(stage)):digest(p) for p in stage.rglob('*')
           if p.is_file() and not p.is_relative_to(stage/'data') and p.name!='bundle_manifest.json'
           and '__pycache__' not in p.parts and not p.is_relative_to(stage/'derived_protocols')}
    write(stage/'bundle_manifest.json',files)
    code=OUT/'geofwi3d_seed_search_code.zip'
    with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*files,'bundle_manifest.json']):z.write(stage/name,name)
    uploads={code.name:digest(code)}
    names=sorted(hashes);parts=[];group=[];size=0
    for name in names:
        n=(stage/name).stat().st_size
        if group and size+n>80*1024**2:parts.append(group);group=[];size=0
        group.append(name);size+=n
    if group:parts.append(group)
    for i,group in enumerate(parts):
        path=OUT/f'geofwi3d_data_part{i+1:02}.zip'
        # NIfTI files are already compressed; stored ZIP keeps packaging fast.
        reusable=False
        if path.exists():
            with zipfile.ZipFile(path) as z:reusable=set(z.namelist())==set(group)
        if not reusable:
            with zipfile.ZipFile(path,'w',zipfile.ZIP_STORED) as z:
                for name in group:z.write(stage/name,name)
        with zipfile.ZipFile(path) as z:
            assert set(z.namelist())==set(group)
        uploads[path.name]=digest(path)
    nb=notebook(uploads,digest(stage/'bundle_manifest.json'))
    write(OUT/'GeoFWI3D_find_3_solid_seeds.ipynb',nb)
    write(OUT/'package.json',dict(uploads=uploads,data_cases=len(ids),folds=splits,updates=updates,
        bundle_manifest_sha256=digest(stage/'bundle_manifest.json'),notebook='GeoFWI3D_find_3_solid_seeds.ipynb'))
    print(json.dumps(dict(updates=updates,uploads=uploads,data_cases=len(ids)),indent=2))


if __name__=='__main__':prepare()
