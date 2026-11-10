"""Build the complete ADNI seeds 0/1/4 notebook and its pinned code ZIP."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

from scripts.colab_adni_seed014 import derivation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/adni_colab_seeds014_noaug_20261018'
PARENT = ROOT/'reports/adni_colab_3fold_3seed_noaug_20261015/bundle'
CONTROL = ROOT/'reports/adni_colab_validated_acceleration_3fold_3seed_noaug_20261015/control'
SOURCE = Path(__file__).parent
ZIP_NAME = 'adni_seeds014_resume_noaug_code.zip'
BOOK_NAME = 'ADNI_SEEDS_0_1_4_RESUME_NO_AUGMENTATION.ipynb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replace(source, old, new, count=None):
    if old not in source or (count is not None and source.count(old) != count):
        raise ValueError('Unexpected original source: '+old)
    return source.replace(old, new)


def prepare():
    OUT.mkdir(exist_ok=True)
    package = OUT/'package'; package.mkdir(exist_ok=True)
    stage = package/'seed4_bundle'; stage.mkdir(exist_ok=True)
    original = read(PARENT/'bundle_manifest.json')
    assert sha(PARENT/'bundle_manifest.json') == derivation.PARENT_SHA
    for name, digest in original.items():
        assert sha(PARENT/name) == digest, name
        if not name.startswith('data/'):
            dest = stage/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT/name, dest)
    canonical = read(OUT/'seed4_initialization_reference.json')
    cluster = read(ROOT/'reports/adni_msd_four_models_seeds456_baselines_20261015/initialization_reference.json')
    cluster_hashes = {r['model']: r['model_sha256'] for r in cluster['models'] if r['dataset']=='ADNI' and r['seed']==4}
    reference = read(PARENT/'initialization_reference.json')
    plan = read(PARENT/'plan.json')
    for model in plan['models']:
        assert canonical['models'][model]['4']['model_sha256'] == cluster_hashes[model]
        reference['models'][model]['4'] = canonical['models'][model]['4']
        for fold in plan['folds']:
            old = PARENT/'protocols'/f'{model}_f{fold}_s0'
            folder = stage/'protocols'/f'{model}_f{fold}_s4'
            shutil.copytree(old, folder, dirs_exist_ok=True)
            p = read(old/'protocol.json')
            p['seeds'] = [4]
            p['baseline_run_id'] = f'ADNI_{model}_dice_f{fold}s4'
            p['baseline_reference'].update(seed=4, initialization_sha256=cluster_hashes[model])
            write(folder/'protocol.json', p)
    plan['tasks'] = [dict(t, seed=4, id=t['id'][:-1]+'4', protocol=t['protocol'][:-1]+'4')
                     for t in plan['tasks'] if t['seed']==0]
    plan.update(name='adni_seed4_extension_20261018', seeds=[4], num_runs=168,
                policy='Seed4 Colab extension; retain existing seeds0/1; seed2 remains archived. Post-selection exploratory analysis.')
    write(stage/'plan.json', plan)
    write(stage/'initialization_reference.json', reference)
    shutil.copyfile(OUT/'seed4_initialization_reference.json', stage/'seed4_initialization_reference.json')
    write(stage/'seed4_cluster_hashes.json', cluster_hashes)
    shutil.copyfile(SOURCE/'derivation.py', stage/'derivation.py')
    code = (PARENT/derivation.CLI_FILE).read_text()
    (stage/derivation.CLI_FILE).write_text(replace(code, derivation.CLI_OLD, derivation.CLI_NEW, 1))
    runtime = (PARENT/'runtime.py').read_text()
    runtime = replace(runtime, '504', '168')
    runtime = replace(runtime, "plan['seeds'] != [0,1,2]", "plan['seeds'] != [4]", 1)
    runtime = replace(runtime, 'for f in (0, 1, 2) for s in (0, 1, 2)', 'for f in (0, 1, 2) for s in (4,)', 1)
    runtime = replace(runtime, 'for seed in (0, 1, 2):', 'for seed in (4,):', 1)
    runtime = replace(runtime, "if len([x for x in r if x['fold']==f]) == 3", "if len([x for x in r if x['fold']==f]) == 1", 1)
    runtime = replace(runtime, 'completed=len(r), expected=9', 'completed=len(r), expected=3', 1)
    runtime = replace(runtime, 'if len(r)==9', 'if len(r)==3', 1)
    runtime = replace(runtime, 'each averaging three seeds', 'each containing the seed4 run', 1)
    (stage/'runtime.py').write_text(runtime)
    gpu = (PARENT/'gpu_preflight.py').read_text()
    gpu = replace(gpu, '_f0_s0', '_f0_s4')
    gpu = replace(gpu, 'exp.base.seed_everything(0)', 'exp.base.seed_everything(4)', 1)
    (stage/'gpu_preflight.py').write_text(gpu)
    manifest = {str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*'))
                if p.is_file() and p.name != 'bundle_manifest.json' and '__pycache__' not in p.parts}
    manifest['data/adni_hippocampus_full.pkl'] = original['data/adni_hippocampus_full.pkl']
    write(stage/'bundle_manifest.json', manifest)
    # Verify every unchanged byte and every allowed seed-only difference locally.
    (stage/'data').mkdir(exist_ok=True)
    data = stage/'data/adni_hippocampus_full.pkl'
    if not data.exists():
        data.symlink_to(PARENT/'data/adni_hippocampus_full.pkl')
    proof = derivation.verify(stage, PARENT)
    write(OUT/'derivation_verification.json', proof)
    # Numerical backend/recovery stay byte-identical; only the progress denominator
    # is generalized in the copied scheduler. A separate gate binds seed4.
    shutil.copytree(CONTROL, package/'control01', dirs_exist_ok=True)
    shutil.copytree(CONTROL, package/'control04', dirs_exist_ok=True)
    for directory in (package/'control01', package/'control04'):
        scheduler=(directory/'scheduler.py').read_text()
        scheduler=replace(scheduler, "{len(ids)}/504 verified runs", '{len(ids)}/{len(plan["tasks"])} verified runs', 1)
        (directory/'scheduler.py').write_text(scheduler)
        manifest_control=read(directory/'control_manifest.json')
        manifest_control['files']['scheduler.py']=sha(directory/'scheduler.py')
        manifest_control['compatible_profile_control_sha256'].append(sha(CONTROL/'control_manifest.json'))
        write(directory/'control_manifest.json',manifest_control)
    shutil.copyfile(SOURCE/'acceleration_gate.py', package/'control04/acceleration_gate.py')
    control_manifest = read(package/'control04/control_manifest.json')
    control_manifest['scientific_bundle_sha256'] = sha(stage/'bundle_manifest.json')
    control_manifest['compatible_profile_control_sha256'] = []
    control_manifest['files']['acceleration_gate.py'] = sha(package/'control04/acceleration_gate.py')
    control_manifest['scope'] = 'Seed4 only; unchanged validated backend, verified CLI/seed-only derivation, independent seed4 preflight.'
    write(package/'control04/control_manifest.json', control_manifest)
    for name in ('campaign.py', 'phase.py', 'preflight.py'):
        shutil.copyfile(SOURCE/name, package/name)
    write(package/'selection.json', dict(seeds=[0,1,4], folds=[0,1,2], total=504,
        reused_seeds=[0,1], seed4_tasks=168, augmentation='none',
        original_results='results_noaug_v1', new_results='results_seeds014_noaug_v1',
        reason='User requested seed2 replacement after observing its performance; preserve original results and label post-selection analysis.'))
    files = {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
             if p.is_file() and not p.is_symlink() and p.name != 'package_manifest.json' and '__pycache__' not in p.parts}
    write(package/'package_manifest.json', files)
    archive = OUT/ZIP_NAME
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in [*files, 'package_manifest.json']:
            z.write(package/name, name)
    summary = read(PARENT.parent/'export_summary.json')
    write(OUT/BOOK_NAME, notebook(sha(archive), summary['code_zip_sha256'], summary['data_zip_sha256'], sha(stage/'bundle_manifest.json')))
    write(OUT/'export_summary.json', dict(status='packaged_pending_local_tests_and_Colab_seed4_preflight',
        notebook=BOOK_NAME, code_zip=ZIP_NAME, code_zip_sha256=sha(archive),
        seed4_bundle_sha256=sha(stage/'bundle_manifest.json'), parent_bundle_sha256=derivation.PARENT_SHA,
        total_target_runs=504, retained_seeds=[0,1], new_seed=4, seed4_tasks=168,
        augmentation='none', gpu_tested=False, source_derivation=proof,
        original_results_untouched=True, no_dataset_upload=True, cluster_jobs_submitted=0))
    print(OUT)


def notebook(package_sha, code_sha, data_sha, seed4_sha):
    cells=[]
    def add(kind, text):
        cell=dict(cell_type=kind, metadata={}, source=textwrap.dedent(text).strip()+'\n', id=f'cell-{len(cells):02d}')
        if kind=='code':
            ast.parse(cell['source']);cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''
    # ADNI — folds 0/1/2 × seeds 0/1/4, no augmentation

    Reuse every fully verified seed 0/1 result and resume interrupted seed 0/1 work first.
    Run missing seed 0/1 combinations, then all missing seed 4 combinations. No seed 2 task is launched.
    There are 504 results in the combined target; if seeds 0/1 are complete, only 168 new runs are needed.
    Original seed 2 results and recovery remain untouched in the original folder.

    **Before switching:** stop the old run cell once and wait for verified recovery uploads. Do not run
    this notebook simultaneously with the old notebook. It does not stop or delete the old runtime.
    Upload only `adni_seeds014_resume_noaug_code.zip` to the existing
    `My Drive/hippo_adni_3fold_3seed/input/`. Keep the existing data ZIP, original code ZIP and results.
    Mount the Google account containing those ADNI files. No dataset upload is needed.

    Use the original validated A100-SXM4-80GB and software versions. All saved subject splits,
    5% training data, initialization procedure, losses, calibration, BF16, batch size 1, optimizers,
    learning-rate schedules, update counts and 31,976,450,667,090 optimization-FLOP cap remain unchanged.
    nnUNet retains native Dice+CE/SGD and 60% targets; the other models retain Dice/AdamW and 10% targets.
    **Absolutely no augmentation.** Seed 4 initialization has been exported on CPU and matched against
    the completed cluster seed-4 controls. Those cluster results are provenance, not imported Colab runs.

    This is a **post-selection exploratory comparison**: seed 2 was replaced after observing scores.
    Keep the original seeds-0/1/2 report for transparent reporting. Do not describe this as an unbiased seed sample.
    ''')
    add('markdown', '## 1. Mount Drive and stage the pinned code; preserve existing work')
    add('code', f'''
    from pathlib import Path
    from contextlib import contextmanager, ExitStack
    import fcntl, hashlib, json, os, shutil, signal, subprocess, sys, threading, time, zipfile
    @contextmanager
    def phase(label):
        print('PHASE:', label, flush=True)
        stop=threading.Event(); started=time.monotonic()
        def pulse():
            while not stop.wait(30):
                value=dict(phase=label, elapsed=time.monotonic()-started, status='still waiting; heartbeat is not progress')
                Path('/content/adni_seeds014_setup_status.json').write_text(json.dumps(value))
                print('HEARTBEAT:', value, flush=True)
        thread=threading.Thread(target=pulse,daemon=True);thread.start()
        try: yield
        finally: stop.set();thread.join()
    from google.colab import drive
    with phase('mount Drive'): drive.mount('/content/drive')
    DRIVE_ROOT=Path('/content/drive/MyDrive/hippo_adni_3fold_3seed')
    INPUT=DRIVE_ROOT/'input'
    ORIGINAL_RESULTS=DRIVE_ROOT/'results_noaug_v1'
    RESULTS=DRIVE_ROOT/'results_seeds014_noaug_v1'
    ORIGINAL_BUNDLE=Path('/content/hippo_adni_bundle')
    BUNDLE=Path('/content/hippo_adni_seed4_bundle')
    PACKAGE=Path('/content/hippo_adni_seeds014')
    def sha256(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    def stage_zip(name, expected, destination):
        local=Path('/content')/name
        with phase('copy '+name+' from Drive to local SSD'):
            shutil.copyfile(INPUT/name,local)
        assert sha256(local)==expected, 'ZIP hash mismatch: '+name
        with zipfile.ZipFile(local) as z:
            assert z.testzip() is None
            for i in z.infolist():
                p=Path(i.filename)
                assert not p.is_absolute() and '..' not in p.parts and ((i.external_attr>>16)&0o170000)!=0o120000
            z.extractall(destination)
    def verify_files(root, manifest):
        for name, expected in json.loads(manifest.read_text()).items():
            assert sha256(root/name)==expected, 'Changed file: '+name
    with phase('check original results and ownership'):
        assert (ORIGINAL_RESULTS/'environment_binding.json').is_file(), 'Mount the account containing your existing ADNI campaign.'
        for folder in (ORIGINAL_RESULTS,RESULTS):
            assert not (folder/'.adni_controller_lock').exists(), 'Existing controller ownership requires inspection; do not delete it.'
        assert not (RESULTS/'.adni_seeds014_controller_lock').exists(), 'Existing seeds014 ownership requires inspection.'
    with ExitStack() as locks:
        for root in (ORIGINAL_BUNDLE,BUNDLE):
            (root/'work').mkdir(parents=True,exist_ok=True)
            lock=locks.enter_context((root/'work/sweep.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
            try: words=cmdline.read_bytes().decode(errors='replace').split('\\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            for root in (ORIGINAL_BUNDLE,BUNDLE):
                assert not (str(root/'entry.py') in words or (str(root) in words and any(w.endswith(('/worker.py','/phase.py')) for w in words))), 'An ADNI worker is still active.'
        with phase('verify or stage original scientific bundle'):
            marker=ORIGINAL_BUNDLE/'bundle_manifest.json'
            if marker.exists():
                assert sha256(marker)=='{derivation.PARENT_SHA}'
                verify_files(ORIGINAL_BUNDLE,marker)
            else:
                assert not any((ORIGINAL_BUNDLE/'work').glob('*/budget_ledger.json')), 'Preserve and inspect existing work without manifest.'
                stage_zip('adni_replication_code.zip','{code_sha}',ORIGINAL_BUNDLE)
                stage_zip('adni_data_verified.zip','{data_sha}',ORIGINAL_BUNDLE)
                assert sha256(marker)=='{derivation.PARENT_SHA}'
                verify_files(ORIGINAL_BUNDLE,marker)
        stage_zip('{ZIP_NAME}','{package_sha}',PACKAGE)
        verify_files(PACKAGE,PACKAGE/'package_manifest.json')
        with phase('stage separate seed4 source; reuse local dataset'):
            marker=BUNDLE/'bundle_manifest.json'
            if marker.exists():
                assert sha256(marker)=='{seed4_sha}'
                verify_files(BUNDLE,marker)
            else:
                assert not any((BUNDLE/'work').glob('*/budget_ledger.json')), 'Preserve seed4 work without manifest.'
                shutil.copytree(PACKAGE/'seed4_bundle',BUNDLE,dirs_exist_ok=True)
                (BUNDLE/'data').mkdir(exist_ok=True)
                data=BUNDLE/'data/adni_hippocampus_full.pkl'
                if not data.exists():data.symlink_to(ORIGINAL_BUNDLE/'data/adni_hippocampus_full.pkl')
                verify_files(BUNDLE,marker)
    RESULTS.mkdir(parents=True,exist_ok=True)
    print('Seeds 0/1 results stay in:',ORIGINAL_RESULTS)
    print('Seed4 and combined comparison:',RESULTS)
    ''')
    add('markdown', '## 2. Authenticate quota checks and prepare the same training environment')
    add('code', '''
    from google.colab import auth
    with phase('authenticate Drive quota API'):auth.authenticate_user()
    VENV=Path('/content/hippo_adni_venv');PYTHON=VENV/'bin/python'
    def run_logged(args, name):
        path=Path('/content')/('adni_seeds014_'+name+'.log')
        with path.open('a') as log:
            p=subprocess.Popen(list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            def drain():
                for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            interrupted=False
            try:drain();code=p.wait()
            except KeyboardInterrupt:
                interrupted=True
                print('Requesting clean stop; wait for all verified recovery uploads.',flush=True)
                if name in ('run','report'):p.send_signal(signal.SIGINT)
                else:os.killpg(p.pid,signal.SIGINT)
                drain();code=p.wait()
            finally:p.stdout.close()
            if interrupted:raise KeyboardInterrupt('Stopped cleanly; rerun later to resume.')
            if code:raise RuntimeError(f'Exit {code}; inspect {path}. Recovery preserved.')
    with phase('prepare training environment'):
        if not PYTHON.exists():run_logged([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV],'setup')
        if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
            import urllib.request
            bootstrap=Path('/content/hippo_adni_get_pip.py')
            urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py',bootstrap)
            run_logged([PYTHON,bootstrap],'setup')
        run_logged([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',ORIGINAL_BUNDLE/'requirements-colab.txt'],'setup')
    COMMON=['--parent-bundle',ORIGINAL_BUNDLE,'--parent-results',ORIGINAL_RESULTS,'--bundle',BUNDLE,'--durable',RESULTS]
    def campaign(command,*args):
        run_logged([PYTHON,'-u',PACKAGE/'campaign.py',command,*COMMON,*args],command)
    subprocess.run(['nvidia-smi'],check=True)
    ''')
    add('markdown', '''
    ## 3. Required preflight

    The original passed preflight is reused only in its exact environment. Seed4 has a separate
    preflight: its four CPU initializations, original binary loss/calibration/optimizer/evaluation
    and exact checkpoint resume are checked. Diagnostics are separate from scientific run accounting.
    The acceleration implementation is unchanged from the A100 validation; the gate verifies that
    only the seed selection and CLI allowance changed. No evidence is relabeled as a seed4 benchmark.
    A failure blocks training. Neither existing environment bindings nor ledgers are reset.
    ''')
    add('code', '''
    with phase('verify original environment and run/reuse seed4 preflight'):
        run_logged([PYTHON,'-u',PACKAGE/'preflight.py',*COMMON],'preflight')
    ''')
    add('markdown', '''
    ## 4. Resume missing seeds0/1, then run missing seed4

    Full archive verification runs before any launches. Checked/verified counts and live phases
    appear every30seconds, with local status independent of Drive. The scheduler fills the CPU-supported
    worker target subject to memory/storage/baseline dependencies; no profiling trials are added.
    Both phases share one session deadline and launch limit. Every run keeps the original charge-before-update
    ledger, recovery uploads, FLOP cap and archive-before-cleanup checks. Seed2 is never launched or removed.
    A phase failure prevents the next phase. Original and new STOP files are respected, never cleared.

    Interrupt once to stop cleanly and wait for all uploads. Use the same notebook and folders to resume.
    Set SESSION_HOURS below the time remaining in your current Colab runtime;22hours is only a conservative
    target for a newly allocated runtime, not a guarantee. The default8hours allows a handover.
    ''')
    add('code', '''
    MAX_NEW_RUNS=504  # Maximum missing tasks, never repeats verified results.
    MAX_WORKERS=0    # CPU-supported target; actual occupancy depends on resource checks.
    SESSION_HOURS=8.0
    campaign('run','--max-runs',MAX_NEW_RUNS,'--workers',MAX_WORKERS,'--hours',SESSION_HOURS)
    ''')
    add('markdown', '''
    ## 5. Combined comparison for seeds0/1/4

    This cell re-verifies archives but starts no training. All nine model/condition cells are required
    for a final mean: average seeds within each fold, then average the three folds. Gains use the exact
    matched model/fold/seed baseline. The original comparison.json is preserved; combined results live
    in results_seeds014_noaug_v1/comparison.json. Seed2 is excluded from this view but remains archived.
    Architecture/loss/optimizer confounding and the post-selection seed change must remain explicit.
    ''')
    add('code', '''
    campaign('report','--hours',2)
    import pandas as pd
    result=json.loads((RESULTS/'comparison.json').read_text())
    print(f"Verified requested results: {result['completed']}/{result['total']}")
    display(pd.DataFrame(result['aggregate']))
    display(pd.DataFrame(result['runs']))
    ''')
    add('markdown', '''
    Local live status: /content/hippo_adni_bundle/parallel_status and
    /content/hippo_adni_seed4_bundle/parallel_status. Notebook logs are /content/adni_seeds014_*.log.
    Original results/recovery remain in results_noaug_v1; new seed4 results/recovery and combined tables
    are in results_seeds014_noaug_v1. Existing completed runs are never migrated or rewritten.
    No further dataset upload is needed after a VM reset: rerun setup to stage the existing Drive ZIPs.
    Do not delete ownership locks after an unclean termination; inspect recovery/accounting first.
    ''')
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        colab=dict(name=BOOK_NAME,provenance=[]),kernelspec=dict(name='python3',display_name='Python3'),
        language_info=dict(name='python'),accelerator='GPU'))


if __name__ == '__main__':
    prepare()
