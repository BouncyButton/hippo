"""One self-contained notebook, three inputs, no historical folder scan."""
import textwrap


def notebook(code_sha,manifest_sha,data_sha):
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=textwrap.dedent(s).strip()+'\n'))
    md('''# MSD — only the 324 missing experiments

    **Account: filippofocaccia2@gmail.com.** Four models, folds **0/1/2**, seeds **0/2/20**, **no augmentation**. The fixed manifest excludes the 180 completed results already audited on 18 October. They are neither scanned nor reverified, and cannot be launched by this notebook. Of the 324 tasks, three replace incomplete result archives and seven were interrupted. The surviving UNETR checkpoint was copied into this account and will be validated before resuming it.

    This is self-contained: the code ZIP contains both original scientific bundles and all controls; the data ZIP contains the exact dataset. No original-account folder or notebook is required. Results use `hippo_msd_missing_324_20261018/results_noaug_v1` on this account. The historical summary is included only for matched comparisons and fixed exclusions; full old archives stay in their original location.

    Use the original **A100-SXM4-80GB** runtime/software. Run sections **1–4** in order. The optimization cap stays **31,976,450,667,090 FLOPs per trajectory**; calibration and lost/replayed work are accounted for separately. Some truncated old ledgers leave historical spending as a lower bound. Seed20 was selected after inspecting validation results, so the combined study remains exploratory.

    **19 October repair, revision 3: fewer checkpoint uploads and reserved recovery capacity.**
    If the failed runtime is still connected, keep it: rerun sections 1–4 in that same runtime.
    Setup preserves work directories, repairs only an append-only shutdown log, finalizes completed
    work without training again, and restores verified server results. It never changes a scientific
    result to make a hash pass. A 17 MiB upload/readback check must pass before any new GPU work.
    Transient ledger timeouts now reconcile the same server file and retry the exact same
    bytes with bounded backoff. No optimizer update proceeds without acknowledgment.
    If a stopped local checkpoint trails the charged ledger, its exact state is reused
    and the lost tail is recorded on Drive as recovery overhead before resuming.
    Setup then protects each unfinished local checkpoint on Drive before any new training.
    Wait for `PROTECTED UNFINISHED CHECKPOINT` messages and `STORAGE POLICY v3` in section 2.
    Keep the current runtime; refreshing this notebook page does not delete its local files.
    Do not select **Disconnect and delete runtime** or clear the result folders.
    ''')
    md('''## 1. Authenticate and load the code and data
    Both ZIPs can be read from the prepared Drive folder. If a ZIP is missing there, this cell asks you to select that downloaded file directly from your computer. No Drive mount is used for training or saving.
    ''')
    code(f'''
    from pathlib import Path
    import hashlib,json,os,shutil,signal,subprocess,sys,time,zipfile
    from google.colab import auth,files
    auth.authenticate_user()
    import google.auth,google_auth_httplib2
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload,build_http
    credentials=google.auth.default()[0]
    transport=build_http();transport.timeout=120
    api=build('drive','v3',http=google_auth_httplib2.AuthorizedHttp(credentials,http=transport),cache_discovery=False)
    email=api.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']
    assert email.lower()=='filippofocaccia2@gmail.com','Authenticate filippofocaccia2@gmail.com.'
    FOLDER='1fErhcTI519LV2wpPtbzLjiHd5dzxRUQu'
    PACKAGE=Path('/content/hippo_msd_missing_package')
    ORIGINAL=PACKAGE/'original_bundle';BUNDLE=PACKAGE/'seed20_bundle'
    LOCAL=(Path('/mnt/local-scratch') if Path('/mnt/local-scratch').is_mount() else Path('/content'))/'hippo_msd_missing_results'
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:words=proc.read_bytes().decode(errors='replace').split('\\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(w.endswith(('/worker.py','/phase.py','/campaign.py')) and 'hippo_msd' in w for w in words),'An MSD process is still active; stop before staging.'
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    def obtain(name,expected):
        path=Path('/content')/name
        if path.exists() and sha(path)==expected:return path
        found=api.files().list(q="'"+FOLDER+"' in parents and trashed=false and name='"+name+"'",fields='files(id)',pageSize=100).execute()['files']
        assert len(found)<=1,'Duplicate input ZIP: '+name
        if found:
            print('DOWNLOADING',name,flush=True)
            tmp=path.with_suffix('.download')
            with tmp.open('wb') as f:
                loader=MediaIoBaseDownload(f,api.files().get_media(fileId=found[0]['id']),chunksize=8*1024**2)
                done=False;last=time.monotonic()
                while not done:
                    status,done=loader.next_chunk(num_retries=5)
                    if time.monotonic()-last>20:print(name,round(status.progress()*100),'%',flush=True);last=time.monotonic()
            assert sha(tmp)==expected,'Incomplete or wrong download: '+name
            tmp.replace(path)
        else:
            print('Select '+name+' from your computer.',flush=True)
            uploaded=files.upload();assert name in uploaded,'Select '+name
            path.write_bytes(uploaded[name]);del uploaded
        assert sha(path)==expected,'Wrong ZIP: '+name
        return path
    code_zip=obtain('msd_missing_324_code.zip','{code_sha}')
    data_zip=obtain('msd_missing_data.zip','{data_sha}')
    def extract(path,destination):
        with zipfile.ZipFile(path) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            for item in z.infolist():
                p=Path(item.filename)
                assert not p.is_absolute() and '..' not in p.parts and ((item.external_attr>>16)&0o170000)!=0o120000
            assert z.testzip() is None
            z.extractall(destination)
    extract(code_zip,PACKAGE)
    assert sha(PACKAGE/'package_manifest.json')=='{manifest_sha}'
    for name,expected in json.loads((PACKAGE/'package_manifest.json').read_text()).items():
        assert sha(PACKAGE/name)==expected,'Changed packaged file: '+name
    initial=json.loads((ORIGINAL/'initialization_reference.json').read_text())
    extra=json.loads((BUNDLE/'seed20_initialization_reference.json').read_text())
    for model,values in extra['models'].items():initial['models'][model].update(values)
    combined=(json.dumps(initial,indent=2,allow_nan=False)+'\\n').encode()
    expected=json.loads((BUNDLE/'bundle_manifest.json').read_text())['initialization_reference.json']
    assert hashlib.sha256(combined).hexdigest()==expected,'Initialization reconstruction differs'
    (BUNDLE/'initialization_reference.json').write_bytes(combined)
    del initial,extra,combined
    extract(data_zip,ORIGINAL)
    (BUNDLE/'data').mkdir(exist_ok=True)
    if not (BUNDLE/'data/msd_hippocampus_full.pkl').exists():
        (BUNDLE/'data/msd_hippocampus_full.pkl').symlink_to(ORIGINAL/'data/msd_hippocampus_full.pkl')
    for root in (ORIGINAL,BUNDLE):(root/'work').mkdir(exist_ok=True)
    scope=json.loads((PACKAGE/'missing_scope.json').read_text())
    print('READY INPUTS:',len(scope['missing_tasks']),'missing tasks;',len(scope['excluded_completed_tasks']),'excluded; no historical folders read.',flush=True)
    ''')
    md('''## 2. Check saving and restore only interrupted work
    Small-file and multi-chunk checkpoint writes, checksum verification and independent downloads must succeed before training. First setup imports no old run folders. Later sessions inspect only results generated by these 324 tasks, to avoid repeating them. No automatic approval can reclaim an active controller: confirm termination only when prompted and its previous VM has ended.
    ''')
    code('''
    VENV=Path('/content/hippo_msd_missing_venv');PYTHON=VENV/'bin/python'
    def run_logged(args,name):
        log_path=Path('/content')/('msd_missing_'+name+'.log')
        with log_path.open('a') as log:
            proc=subprocess.Popen(list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            def drain():
                for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            try:drain();rc=proc.wait()
            except KeyboardInterrupt:
                print('Clean stop requested. Wait for verified checkpoint uploads; do not interrupt again.',flush=True)
                proc.send_signal(signal.SIGINT);drain();proc.wait();raise
            finally:proc.stdout.close()
            if rc:raise RuntimeError('Stopped; results and recovery are retained. Log: '+str(log_path))
    if not PYTHON.exists():run_logged([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV],'setup')
    if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
        import urllib.request
        bootstrap=Path('/content/msd_missing_get_pip.py')
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py',bootstrap)
        run_logged([PYTHON,bootstrap],'setup')
    run_logged([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',ORIGINAL/'requirements-colab.txt'],'setup')
    DRIVER=PACKAGE/'cloud/missing.py'
    ARGS=['--package',PACKAGE,'--local',LOCAL,'--parent-bundle',ORIGINAL,'--bundle',BUNDLE]
    def campaign(command,*extra):run_logged([PYTHON,'-u',DRIVER,command,*ARGS,*extra],command)
    extra=[]
    stores=api.files().list(q="'"+FOLDER+"' in parents and trashed=false and name='results_noaug_v1'",fields='files(id)').execute()['files']
    assert len(stores)<=1,'Duplicate result stores'
    if stores:
        leases=api.files().list(q="'"+stores[0]['id']+"' in parents and trashed=false and name='ACTIVE_CONTROLLER'",fields='files(id,appProperties)').execute()['files']
        if leases:
            assert len(leases)==1,'Multiple controller leases'
            assert input('Check Manage sessions. Only if the previous runtime TERMINATED, type TERMINATED: ').strip()=='TERMINATED'
            extra=['--terminated-session',leases[0]['appProperties']['token']]
    campaign('prepare',*extra)
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    md('''## 3. Required scientific preflight
    This checks the local code/data and original GPU/software binding. It does **not** verify old runs. Previously passed diagnostic receipts are reused only when they match the exact unchanged bundle and environment.
    ''')
    code("campaign('preflight')")
    md('''## 4. Run only missing experiments
    The scheduler fills all worker slots supported by the runtime CPU and memory headroom; there are no solo calibration runs or throughput-based worker downgrades. NVMe checkpoints, data/geometry caches and accelerated evaluation are retained. Model mathematics and update budgets are unchanged.

    Full recovery snapshots are losslessly compressed, serialized and uploaded about every **15 minutes** when needed, including a 15-minute delay for the first routine upload. This reduces the scheduled upload frequency by 80%; short runs finish by saving their compact scientific result instead. **Local checkpoint frequency stays unchanged.** Clean pauses force a server save, and setup saves unfinished local work before resuming. Server-acknowledged charge ledgers still precede every update. A sudden VM loss can require replaying work since the last verified server checkpoint (roughly 15 minutes plus upload/queue time); all such charged work remains in the accounting.

    Writes are paced across workers. Identical already-acknowledged charges are not uploaded twice;
    each changed charge still reaches Drive before its optimizer update. Worker completion uses
    copied log bytes; the final full log and shutdown diagnostics are archived after worker exit.
    Neither the historical exclusions nor any checkpoint/FLOP safety check is relaxed.

    Bulk uploads have a **320 GiB rolling 24-hour guard**, with **32 GiB reserved for recovery and result saves**. All old reservations, including failed/uncertain attempts, are retained; the old 200 GiB counter is never reset. This bounded increase provides headroom to recover from the old limit while uploads become less frequent. Routine snapshots cannot spend the reserve: reaching their allowance requests a clean stop and forces verified checkpoint saves from that reserve. The hard ceiling still blocks excess uploads. This is a notebook policy, not a Google quota measurement; other notebooks and provider errors can still affect saving. Results are never considered saved just because a mounted-Drive cache can read them back.

    For a fresh runtime, 22 hours is a clean-stop target, not a guaranteed Colab lifetime. Interrupt once and wait for uploads. On a new session, rerun sections 1–4: completed new tasks are skipped and coherent checkpoints resume. A hard termination can lose the newest uncommitted tail; its charged cost is retained as recovery overhead.
    ''')
    code('''
    MAX_NEW_RUNS=324  # Upper bound on missing tasks; never repeat the excluded 180.
    MAX_WORKERS=0    # Fill the maximum CPU/memory-supported worker slots.
    SESSION_HOURS=22.0
    campaign('run','--max-runs',MAX_NEW_RUNS,'--workers',MAX_WORKERS,'--hours',SESSION_HOURS)
    ''')
    md('''## 5. Results
    New results are paired with the included historical summary where a matched baseline was already completed. The historical records come from the fixed prior audit; they were not reverified by this notebook. Seed20 selection is post hoc and exploratory. The full old archives remain separate.
    ''')
    code('''
    campaign('report','--hours',2)
    import pandas as pd
    result=json.loads((LOCAL/'20/comparison.json').read_text())
    print('New completed tasks:',result['new_completed'],'/324')
    print('Combined results:',result['completed'],'/504; 180 from the fixed prior audit')
    display(pd.DataFrame(result['aggregate']))
    display(pd.DataFrame(result['runs']))
    accounting=json.loads((LOCAL/'recovery_accounting.json').read_text())
    display({k:v for k,v in accounting.items() if k!='recovery_events'})
    print('Historical reference is separate:',len(json.loads((PACKAGE/'historical_reference.json').read_text())),'rows')
    config=json.loads((LOCAL/'cloud_config.json').read_text())
    print('Results: https://drive.google.com/drive/folders/'+config['folder_id'])
    ''')
    return dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        colab=dict(name='MSD_MISSING_324_NO_AUGMENTATION.ipynb'),
        kernelspec=dict(name='python3',display_name='Python 3'),accelerator='GPU'))
