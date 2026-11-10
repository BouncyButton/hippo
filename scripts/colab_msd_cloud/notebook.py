"""Complete recovery notebook; no manual patch cells or mounted-Drive writes."""
import textwrap


def make_notebook(zip_sha, manifest_sha, report):
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=textwrap.dedent(s).strip()+'\n'))
    md(f'''# MSD recovery — folds 0/1/2, seeds 0/2/20, no augmentation

    This notebook preserves **{report['retained']} completed experiments** whose scientific results were verified or recovered from the old cloud files. **{len(report['rerun_completed'])} reported completions lack required recoverable scientific files** and are scheduled again. Seven interrupted tasks are audited automatically: a surviving valid checkpoint is resumed; only work without one starts again. The original Drive folders, seed-1 results and old recovery artifacts remain untouched.

    **Use this notebook only after the old runtime has stopped.** Your original VM was confirmed terminated. Select the same **A100-SXM4-80GB** environment; a software/GPU mismatch blocks training. Run sections 1–4 in order. No data re-upload and no separate diagnostic cells are needed.

    Results are saved through the Drive API in `hippo_msd_3fold_3seed/results_seeds0220_server_verified_v2`, on **focacciafilippo@gmail.com**. There are no critical writes to the mounted Drive filesystem. A server write/readback test runs before preflight or training.

    The frozen scientific source, initialization, splits, losses, calibration, optimizers, precision, evaluation acceleration and **31,976,450,667,090 optimization FLOPs per trajectory** remain unchanged. **No augmentation.** Charge-before-update accounting remains active. Lost-attempt/replayed compute is recorded separately as recovery overhead; a truncated old ledger is reported as unknown, never zero. The seed-20 selection remains exploratory because it was chosen after inspecting validation results.
    ''')
    md('''## 1. Authenticate and stage verified files
    Use your existing data and original code ZIPs. The corrected ZIP can be uploaded to the existing `input` folder. If it is unavailable there because Drive is blocking uploads, this cell asks you to select the downloaded ZIP directly into the Colab VM. It verifies the same checksum either way.
    ''')
    code(f'''
    from pathlib import Path
    import hashlib, json, os, shutil, signal, subprocess, sys, threading, time, zipfile
    from google.colab import auth, files
    auth.authenticate_user()
    import google.auth, google_auth_httplib2, httplib2
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    credentials=google.auth.default()[0]
    drive_api=build('drive','v3',http=google_auth_httplib2.AuthorizedHttp(credentials,http=httplib2.Http(timeout=120)),cache_discovery=False)
    account=drive_api.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']
    assert account.lower()=='focacciafilippo@gmail.com', 'Authenticate focacciafilippo@gmail.com.'
    PACKAGE=Path('/content/hippo_msd_cloud_resume')
    ORIGINAL_BUNDLE=Path('/content/hippo_msd_bundle')
    BUNDLE=Path('/content/hippo_msd_seed20_bundle')
    LOCAL=Path('/mnt/local-scratch/hippo_msd_cloud_results') if Path('/mnt/local-scratch').is_mount() else Path('/content/hippo_msd_cloud_results')
    ZIP_NAME='msd_seeds0220_server_verified_resume.zip'
    ZIP_SHA='{zip_sha}'
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try: words=proc.read_bytes().decode(errors='replace').split('\\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        assert not any(w.endswith(('/worker.py','/phase.py','/campaign.py')) and 'hippo_msd' in w for w in words), 'An MSD process is active; do not stage files over it.'
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
        return h.hexdigest()
    def download(file_id,path,expected):
        path=Path(path)
        if path.exists() and sha(path)==expected: return path
        print('DOWNLOADING',path.name,flush=True)
        with path.with_suffix('.download').open('wb') as f:
            loader=MediaIoBaseDownload(f,drive_api.files().get_media(fileId=file_id),chunksize=8*1024**2)
            done=False;last=time.monotonic()
            while not done:
                status,done=loader.next_chunk(num_retries=5)
                if time.monotonic()-last>20: print(path.name,round(100*status.progress()),'%',flush=True);last=time.monotonic()
        temp=path.with_suffix('.download');assert sha(temp)==expected,'Download checksum differs: '+path.name
        temp.replace(path);return path
    local_zip=Path('/content')/ZIP_NAME
    if not local_zip.exists() or sha(local_zip)!=ZIP_SHA:
        matches=drive_api.files().list(q="'1GSyC3eLWvTFJZBzysAUq19A0UxranWrl' in parents and trashed=false and name='"+ZIP_NAME+"'",fields='files(id,name)',pageSize=100).execute()['files']
        assert len(matches)<=1,'Duplicate recovery ZIPs in Drive; choose the audited file.'
        if matches: download(matches[0]['id'],local_zip,ZIP_SHA)
        else:
            print('Select the corrected '+ZIP_NAME+' from your computer. This bypasses Drive upload limits for staging only.',flush=True)
            uploaded=files.upload()
            assert ZIP_NAME in uploaded,'Select the corrected ZIP.'
            local_zip.write_bytes(uploaded[ZIP_NAME]);del uploaded
    assert sha(local_zip)==ZIP_SHA,'Wrong or incomplete recovery package'
    def extract(path,target):
        with zipfile.ZipFile(path) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            for i in z.infolist():
                p=Path(i.filename)
                assert not p.is_absolute() and '..' not in p.parts and ((i.external_attr>>16)&0o170000)!=0o120000
            assert z.testzip() is None
            z.extractall(target)
    extract(local_zip,PACKAGE)
    assert sha(PACKAGE/'package_manifest.json')=='{manifest_sha}'
    for name,expected in json.loads((PACKAGE/'package_manifest.json').read_text()).items():
        assert sha(PACKAGE/name)==expected,'Changed package file: '+name
    original=download('1X-dCqatZE-bskaviAQB-9rBlCDVcPNmI','/content/msd_replication_code.zip','b0a6dc1a5b94ce8d7ae2506e1583401313cc60fae6735dd4c09f6c73f5e6c51b')
    data=download('1dO3yEKeoz5DF9e3xzvWQfWUDAQ_V9xzb','/content/msd_data_verified.zip','ff045aea37338769009e6322ed55fe08b654681e643ce9cb78782b80e11921f4')
    extract(original,ORIGINAL_BUNDLE);extract(data,ORIGINAL_BUNDLE)
    shutil.copytree(PACKAGE/'seed20_bundle',BUNDLE,dirs_exist_ok=True)
    (BUNDLE/'data').mkdir(exist_ok=True)
    if not (BUNDLE/'data/msd_hippocampus_full.pkl').exists():
        (BUNDLE/'data/msd_hippocampus_full.pkl').symlink_to(ORIGINAL_BUNDLE/'data/msd_hippocampus_full.pkl')
    for root in (ORIGINAL_BUNDLE,BUNDLE):
        (root/'work').mkdir(exist_ok=True)
        for name,expected in json.loads((root/'bundle_manifest.json').read_text()).items():assert sha(root/name)==expected,'Scientific file differs: '+name
    print('VERIFIED: unchanged science, preserved results package and local dataset.',flush=True)
    ''')
    md('''## 2. Restore results and recovery automatically
    Saving must pass a server checksum plus independent download check. If Drive still refuses writes, this cell stops before any new training. Do not remove receipts or reset ledgers. The notebook restores checkpoints and records any lost charged tail before resuming it.
    ''')
    code('''
    VENV=Path('/content/hippo_msd_venv');PYTHON=VENV/'bin/python'
    def run_logged(args,name):
        log_path=Path('/content')/('msd_cloud_'+name+'.log')
        with log_path.open('a') as log:
            p=subprocess.Popen(list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            def drain():
                for line in p.stdout: print(line,end='',flush=True);log.write(line);log.flush()
            try: drain();rc=p.wait()
            except KeyboardInterrupt:
                print('Requesting clean stop. Wait for all verified uploads; do not interrupt twice.',flush=True)
                p.send_signal(signal.SIGINT);drain();p.wait();raise
            finally:p.stdout.close()
            if rc:raise RuntimeError('Stopped safely; inspect '+str(log_path)+'. Existing results and local recovery are retained.')
    if not PYTHON.exists():run_logged([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV],'setup')
    if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
        import urllib.request
        bootstrap=Path('/content/hippo_msd_get_pip.py')
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py',bootstrap)
        run_logged([PYTHON,bootstrap],'setup')
    run_logged([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',ORIGINAL_BUNDLE/'requirements-colab.txt'],'setup')
    DRIVER=PACKAGE/'cloud/driver.py'
    ARGS=['--package',PACKAGE,'--local',LOCAL,'--parent-bundle',ORIGINAL_BUNDLE,'--bundle',BUNDLE]
    def campaign(command,*extra):run_logged([PYTHON,'-u',DRIVER,command,*ARGS,*extra],command)
    recovery_args=[]
    stores=drive_api.files().list(q="'1ipls98K9dmRQ5rZwfbepw1JyWVL3Yq0P' in parents and trashed=false and name='results_seeds0220_server_verified_v2'",fields='files(id)').execute()['files']
    assert len(stores)<=1,'Duplicate server result folders'
    if stores:
        leases=drive_api.files().list(q="'"+stores[0]['id']+"' in parents and trashed=false and name='ACTIVE_CONTROLLER'",fields='files(id,appProperties)').execute()['files']
        if leases:
            assert len(leases)==1,'Multiple controller leases; preserve and inspect'
            print('A previous controller lease exists. Check Colab Manage sessions before proceeding.')
            answer=input('Only if its old runtime has terminated, type TERMINATED. Otherwise stop here: ')
            assert answer.strip()=='TERMINATED','Previous work is protected; no lease was changed.'
            recovery_args=['--terminated-session',leases[0]['appProperties']['token']]
    campaign('prepare',*recovery_args)
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    md('''## 3. Verify the original GPU/software binding
    Existing passed preflights are reused only with the exact frozen bundle and environment. No benchmarks are counted as scientific training. The storage layer changed; model and evaluation computations did not.
    ''')
    code('''
    COMMON=['--parent-bundle',ORIGINAL_BUNDLE,'--parent-results',LOCAL/'02','--bundle',BUNDLE,'--durable',LOCAL/'20']
    run_logged([PYTHON,'-u',PACKAGE/'preflight.py',*COMMON],'preflight')
    ''')
    md('''## 4. Resume the missing work
    The scheduler fills eligible worker slots without solo profiling, subject to CPU/GPU/RAM headroom and matched-baseline dependencies. GPU data caches, geometry caches, accelerated evaluation, batch-1 evaluation and NVMe checkpoints are retained.

    Local checkpoint cadence and periodic recovery timing remain unchanged. Recovery uploads are losslessly compressed and serialized; an identical snapshot is not uploaded twice. A finished run saves its full result archive before deleting any local checkpoint, instead of uploading another redundant multi-GB recovery copy. Small charge ledgers are updated through the API before every optimization update. Storage errors stop new work.

    `SESSION_HOURS=22` is a clean-stop target for a fresh runtime, not a Colab lifetime guarantee. A hard VM termination can still lose the newest uncommitted work; its charged cost remains in the server ledger and is recorded separately if replay is necessary. Interrupt once for a clean stop and wait for uploads.
    ''')
    code('''
    MAX_NEW_RUNS=504  # Upper bound on missing tasks, never 504 extra repeats.
    MAX_WORKERS=0    # Maximum CPU-supported occupancy, with memory headroom checks.
    SESSION_HOURS=22.0
    campaign('run','--max-runs',MAX_NEW_RUNS,'--workers',MAX_WORKERS,'--hours',SESSION_HOURS)
    ''')
    md('''## 5. Results and recovery accounting
    Comparisons pair each model/condition with its matched fold/seed baseline. Average seeds within each fold, then average folds; one final mean requires all nine cells. Seed 1 stays in the original archive and is excluded from this post-selection report. Recovery overhead is additional compute and must not be reported as part of the per-trajectory cap.
    ''')
    code('''
    campaign('report','--hours',2)
    import pandas as pd
    result=json.loads((LOCAL/'20/comparison.json').read_text())
    print('Completed:',result['completed'],'/',result['total'])
    display(pd.DataFrame(result['aggregate']))
    display(pd.DataFrame(result['runs']))
    recovery=json.loads((PACKAGE/'recovery_plan.json').read_text())
    print('Original recovery audit:',recovery['counts'])
    print(recovery['unknown_original_charges'])
    accounting=json.loads((LOCAL/'recovery_accounting.json').read_text())
    display({k:v for k,v in accounting.items() if k!='recovery_events'})
    events=[json.loads(p.read_text()) for p in (LOCAL/'recovery_events').glob('*.json')]
    if events:display(pd.DataFrame(events)[['task','action','lost_charged_attempts','lost_training_flops','lost_calibration_flops']])
    config=json.loads((LOCAL/'cloud_config.json').read_text())
    print('Drive results: https://drive.google.com/drive/folders/'+config['folder_id'])
    ''')
    return dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        colab=dict(name='MSD_SEEDS_0_2_20_SERVER_VERIFIED_RESUME.ipynb'),
        kernelspec=dict(name='python3',display_name='Python 3'),accelerator='GPU'))
