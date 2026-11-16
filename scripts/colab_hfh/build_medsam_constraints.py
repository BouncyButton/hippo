"""Build a verified MedSAM-only overlay and update the existing notebook."""
import os
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'


def main():
    files = {str(p.relative_to(ROOT)): p.read_bytes()
             for p in sorted((ROOT/'scripts/colab_hfh').glob('medsam_constraints_*.py'))}
    reference = 'scripts/cross_dataset_campaign/adni_foundation_full_runtime.py'
    files[reference] = (ROOT/reference).read_bytes()
    recovery = 'scripts/colab_hfh/medsam_recovery.py'
    files[recovery] = (ROOT/recovery).read_bytes()
    manifest = {p: hashlib.sha256(data).hexdigest() for p, data in files.items()}
    for data in files.values(): ast.parse(data.decode())
    asset = REPORT/'hfh_medsam_constraints_v1.zip'
    with zipfile.ZipFile(asset, 'w', zipfile.ZIP_DEFLATED) as z:
        for relative, data in files.items(): z.writestr('source/'+relative, data)
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
    checksum = hashlib.sha256(asset.read_bytes()).hexdigest()
    launcher = '''# Paired MedSAM3 BAND -> strict gate -> full constraint matrix.
import hashlib, json, os, signal, subprocess, time, zipfile
from pathlib import Path
ROOT=Path('/content/hippo_hfh')
BUNDLE=ROOT/'bundle'
PYTHON=ROOT/'venv/bin/python'
CONSTRAINT_SESSION_HOURS=20
ASSET_SHA256='__SHA__'
private=Path('/content/.hfh_storage_auth')
package=ROOT/'hfh_medsam_constraints_v1.zip'
if not package.exists() or hashlib.sha256(package.read_bytes()).hexdigest()!=ASSET_SHA256:
    if os.environ.get('HFH_DRIVE_TRANSPORT')=='colab_adc':
        storage.download('hfh_storage:hfh_medsam_constraints_v1.zip',package)
    else:
        subprocess.run([str(private/'rclone'),'--config',str(private/'rclone.conf'),
            'copyto','hfh_storage:hfh_medsam_constraints_v1.zip',str(package),'--checksum'],check=True,timeout=300)
if hashlib.sha256(package.read_bytes()).hexdigest()!=ASSET_SHA256:
    raise ValueError('Constraint overlay checksum differs')
import psutil
active=[]
for p in psutil.process_iter(['cmdline']):
    if 'scripts.colab_hfh.medsam_constraints_campaign' in (p.info['cmdline'] or []): active.append(p)
if len(active)>1: raise RuntimeError('Duplicate constraint controllers; inspect before continuing')
if active and '--strict' not in active[0].cmdline():
    raise RuntimeError('An older numerical campaign is active; wait for its clean stop')
log_path=ROOT/'medsam_constraints_controller.log'
if active:
    process=active[0]
    print('Attached to running controller',process.pid,flush=True)
else:
    with zipfile.ZipFile(package) as z:
        manifest=json.loads(z.read('manifest.json'))
        for relative, expected in manifest.items():
            path=Path(relative)
            if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe overlay path')
            data=z.read('source/'+relative)
            if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('Overlay source mismatch')
            target=BUNDLE/'source'/relative
            if relative=='scripts/cross_dataset_campaign/adni_foundation_full_runtime.py' and target.exists() and target.read_bytes()!=data:
                raise ValueError('Existing reference calibration engine differs; preserve it for review')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    work=ROOT/'medsam_constraints_strict_v1';work.mkdir(exist_ok=True)
    (work/'STOP').unlink(missing_ok=True)
    launch_env=dict(os.environ,**globals().get('env',{}))
    launch_env['PYTHONPATH']=str(BUNDLE/'source')
    launch_env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    launch_env['HFH_REQUIRE_COMPLETED_F0S0']='1'
    with log_path.open('a') as log:
        child=subprocess.Popen([str(PYTHON),'-m','scripts.colab_hfh.medsam_constraints_campaign',
            '--root',str(ROOT),'--hours',str(CONSTRAINT_SESSION_HOURS),'--strict'],env=launch_env,
            cwd=BUNDLE/'source',stdout=log,stderr=subprocess.STDOUT)
    process=psutil.Process(child.pid)
    print('Started paired constraint controller',process.pid,flush=True)
position=0
try:
    while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
        if log_path.exists():
            with log_path.open() as log:
                log.seek(position); chunk=log.read();position=log.tell()
            if chunk: print(chunk,end='',flush=True)
        time.sleep(10)
except (psutil.NoSuchProcess,ProcessLookupError):
    pass
except KeyboardInterrupt:
    (ROOT/'medsam_constraints_strict_v1/STOP').touch()
    print('Clean stop requested; wait for recovery. Rerun this cell to attach or resume.',flush=True)
if log_path.exists():
    with log_path.open() as log:
        log.seek(position); print(log.read(),end='',flush=True)
status=ROOT/'medsam_constraints_strict_v1/campaign_status.json'
if status.exists(): print('Campaign status:',json.loads(status.read_text())['status'])
'''.replace('__SHA__', checksum)
    ast.parse(launcher)
    target = REPORT/'HFH_MedSAM3_200_3fold_3seed.ipynb'
    backup = REPORT/'before_constraints'/target.name
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists(): shutil.copyfile(target, backup)
    notebook = json.loads(target.read_text())
    from scripts.colab_hfh.medsam_constraints_notebook import update_storage
    update_storage(notebook)
    notebook['cells'][0]['source'] = '''# HFH MedSAM3 — reproducible controls, BAND and conditional matrix

The previous nine native controls and 17 BAND trials are preserved as exploratory results. No recipe passed all nine pairs. A zero-update replay exposed differing native gradients despite identical inputs, weights and RNG. Strict algorithms plus exact-zero derivatives for SAM3's **empty point grids** produced identical repeated gradient hashes; native forward and nonempty operations remain unchanged. This is a separate numerical campaign with fresh controls, not a continuation mixing old and new controls.

First run **nine fresh native controls**, seeds 0,1,2 × folds 0,1,2. If a control falls below50% validation Dice, stop for native-recipe diagnosis. Then test three declared BAND recipes: target0.01/ramp20, target0.03/p95cap0.15/ramp20, and target0.01 delayed until update101 with ramp20. The delayed variant retains the full zero-weight auxiliary graph during its native-only prefix. Every trial uses the same initialization, samples, LR trajectory and **200 updates**. A BAND recipe qualifies only if **every one of the nine pairs improves validation Dice**, with every Dice≥50%. Reject a recipe after its first completed nonpositive pair.

After qualification, automatically start **12 remaining established combinations ×9 pairs =108 additional runs**. With the nine fresh native controls and chosen nine BAND results this is the established14-arm,126-cell matrix. Constraint/calibration overhead is additional: this is update matched, not a measured equal-FLOPs claim. Training/evaluation/recovery are strict; unsupported nondeterministic operations fail visibly. Prior interrupted recovery remains preserved separately.

All comparisons are exploratory validation-selected results. A positive average cannot qualify a recipe. Full-matrix arms are reported regardless of effect direction. This session uses A100/BF16 with strict deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8. New results are in **results_medsam3_constraints_strict_v1** on the configured storage account; compute remains on the configured compute account. Repeated first-batch gradients passed the GPU audit; complete-run reproducibility still requires verification.

On the existing prepared runtime, run **cell 4**. A controller is attached if already running. Each session is capped at 20 hours and resumes verified archives/checkpoints; the full matrix may span multiple Colab sessions. Do not rerun setup while a worker is active.
'''
    notebook['cells'][4]['source'] = launcher
    notebook['cells'][5]['source'] = '''from pathlib import Path
import json, pandas as pd
path=Path('/content/hippo_hfh/medsam_constraints_strict_v1/campaign_status.json')
state=json.loads(path.read_text())
print(state['status']);print('Qualified recipe:',state.get('winner'))
if state.get('controls'):
    print('Fresh native controls (historical deltas are descriptive only)')
    display(pd.DataFrame(state['controls'])[['seed','fold','training_dice','validation_dice']])
if state['results']:
    display(pd.DataFrame(state['results'])[['recipe','arm','seed','fold','training_dice','validation_dice','baseline_dice','delta_dice_pp']])
print(state['warning'])
'''
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source=cell['source'];ast.parse(''.join(source) if isinstance(source,list) else source)
            cell['execution_count']=None;cell['outputs']=[]
    target.write_text(json.dumps(notebook, indent=1)+'\n')
    (REPORT/'medsam_constraints_launcher.py').write_text(launcher)
    (REPORT/'medsam_constraints_package.json').write_text(json.dumps(dict(asset=str(asset),sha256=checksum,
        manifest=manifest,notebook=str(target),launcher=str(REPORT/'medsam_constraints_launcher.py')),indent=2)+'\n')
    print(json.dumps(dict(asset=str(asset),sha256=checksum,bytes=asset.stat().st_size)))


if __name__ == '__main__': main()
