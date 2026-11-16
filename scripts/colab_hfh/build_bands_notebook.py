"""Extend the existing four-model notebook in place with matched BAND runs."""
import ast
import base64
import json
from pathlib import Path
import zlib
import textwrap
import hashlib
import zipfile
from scripts.colab_hfh.prepare_bands import ROOT,EXPORT,build


def main():
    payload=build()
    path=EXPORT/'HFH_baseline_shared_seed_search.ipynb'
    notebook=json.loads(path.read_text())
    backup=EXPORT/'baseline_before_bands_notebook.json'
    if not backup.exists():backup.write_text(json.dumps(notebook,indent=1)+'\n')
    control=json.loads(backup.read_text())
    assets={'bands_v1_payload.json':json.dumps(payload,indent=2)+'\n',
            'bands_campaign.py':(ROOT/'scripts/colab_hfh/bands_campaign.py').read_text()}
    for name in ('bands_worker.py','bands_archive.py'):
        assets['bundle/source/scripts/colab_hfh/'+name]=(ROOT/'scripts/colab_hfh'/name).read_text()
    asset_path=EXPORT/'hfh_bands_v1_assets.zip'
    with zipfile.ZipFile(asset_path,'w',zipfile.ZIP_DEFLATED) as zipped:
        for name,source in assets.items():
            info=zipfile.ZipInfo(name,date_time=(2026,11,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            zipped.writestr(info,source)
    asset_sha=hashlib.sha256(asset_path.read_bytes()).hexdigest()
    notebook['cells'][0]['source']='''# HFH — four-model controls and matched BAND comparison

**Controls completed:** seeds **0, 1, 2**, folds **0, 1, 2**, all four models: **36/36 final Dice ≥50% on L4 BF16**. Baseline results remain available below and in `results_recipe_diagnostics_v3`.

This same notebook now runs **36 native-loss + BAND conditions**, one for every matched model/fold/seed. It retains every improvement and regression, with no seed rejection or validation-driven coefficient search. Improvement is measured against the final matched control, not guaranteed or required for a run to be recorded.

Compute account: the configured compute account. Inputs, results and archives: the configured storage account. Use an available NVIDIA GPU; the current controls use BF16, so paired BAND runs must preserve that precision. A different precision requires matching controls first (`RUN_MODE="controls"`). The A100 is not required: this campaign uses the existing L4.

| Model | Initial LR | Updates | Native-only prefix | BAND ramp | Control mean / minimum Dice |
|---|---:|---:|---:|---:|---:|
| nnU-Net | 0.1 | 90 | 9 | 9 | 74.58% / 69.67% |
| UNETR | 0.0003 | 245 | 50 | 50 | 57.67% / 52.21% |
| BasicUNet | 0.0003 | 563 | 50 | 50 | 57.85% / 54.20% |
| SwinUNETR | 0.001 | 206 | 50 | 50 | 70.58% / 63.99% |

The model, initialization hash, optimizer, native loss, batch size, data order, five training subjects (ten crops), five validation subjects (ten crops), no augmentation, learning-rate schedule and final-checkpoint criterion are matched. Every optimization run stays below **31,976,450,667,090 FLOPs**; BAND calibration cost is reported separately, following the established optimization-only budget.

BAND uses the established balanced inner/outer binary cross entropy on **two six-connected morphological voxel steps** around the training mask. These are voxel steps, not a uniform millimetre band on this anisotropic grid. Weight calibration uses only the ten training crops after the native-only prefix. The established gradient-ratio target / p95 cap is **0.6 / 0.75 for nnU-Net**, **0.1 / 0.5 for the other models**. No focal, edge, SDF or depth term is introduced.

Cell 4 defaults to `RUN_MODE="bands"`; it attaches to an active BAND controller or starts one. Training/checkpoints use local SSD; each completed run is uploaded as one verified archive to **results_bands_v1**. Interrupting the cell requests a stop after the current run is safely archived. Cell 5 shows the completed controls and paired BAND deltas in percentage points. All results are exploratory validation results, not an unbiased test estimate.
'''
    launch='''import psutil,hashlib,zipfile
active=[]
for item in psutil.process_iter(['cmdline']):
    try:
        if str(ROOT/'bands_campaign.py') in (item.info['cmdline'] or []): active.append(item)
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
if len(active)>1: raise RuntimeError('Multiple BAND controllers; inspect before continuing')
def alive(p):
    try:return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
log_path=ROOT/'bands_controller.log'
if active:
    process=active[0]
    print('Attached to BAND controller',process.pid,flush=True)
else:
    package=ROOT/'hfh_bands_v1_assets.zip'
    if not package.exists() or hashlib.sha256(package.read_bytes()).hexdigest()!=BAND_ASSET_SHA256:
        private=Path('/content/.hfh_storage_auth')
        subprocess.run([str(private/'rclone'),'--config',str(private/'rclone.conf'),
            'copyto','hfh_storage:hfh_bands_v1_assets.zip',str(package),'--checksum'],check=True,timeout=300)
    if hashlib.sha256(package.read_bytes()).hexdigest()!=BAND_ASSET_SHA256:raise ValueError('BAND package checksum mismatch')
    with zipfile.ZipFile(package) as zipped:
        for item in zipped.infolist():
            relative=Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe BAND package path')
            destination=ROOT/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(zipped.read(item))
    (ROOT/'bands_v1/STOP').unlink(missing_ok=True)
    with log_path.open('a') as log:
        child=subprocess.Popen([str(PYTHON),str(ROOT/'bands_campaign.py'),'--root',str(ROOT),'--hours',str(min(SESSION_HOURS,6))],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    process=psutil.Process(child.pid)
with log_path.open() as log:
    try:
        while alive(process):
            content=log.read()
            if content:print(content,end='',flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        (ROOT/'bands_v1/STOP').touch()
        print('Stop requested after current run and verified archive.',flush=True)
        while alive(process):time.sleep(2)
    print(log.read(),end='',flush=True)
state_path=ROOT/'bands_v1/campaign_status.json'
print(json.loads(state_path.read_text()) if state_path.exists() else 'Inspect startup log above')
'''
    control_packed=base64.b64encode(zlib.compress(control['cells'][4]['source'].encode(),9)).decode()
    encoded=lambda value:"'''\n"+'\n'.join(textwrap.wrap(value,120))+"\n'''"
    notebook['cells'][4]['source']="RUN_MODE = 'bands'  # 'controls' resumes the existing baseline exploration\nimport base64,zlib\n"+'BAND_ASSET_SHA256 = '+repr(asset_sha)+'\nCONTROL_RUNNER = zlib.decompress(base64.b64decode('+encoded(control_packed)+')).decode()\nif RUN_MODE == "controls":\n    exec(CONTROL_RUNNER)\nelif RUN_MODE == "bands":\n'+''.join('    '+line+'\n' for line in launch.splitlines())+'else:\n    raise ValueError("RUN_MODE must be controls or bands")\n'
    control_summary=control['cells'][5]['source'].replace("status=json.loads((ROOT/('consistent_recipes_status'+WORK_SUFFIX+'.json')).read_text())", "status=json.loads(control_status.read_text())")
    notebook['cells'][5]['source']='''import json
import pandas as pd
control_status=ROOT/('consistent_recipes_status'+globals().get('WORK_SUFFIX','')+'.json')
if control_status.exists():
'''+textwrap.indent(control_summary,'    ')+'''else:
    reference_path=ROOT/'bands_v1_payload.json'
    if reference_path.exists():
        references=json.loads(reference_path.read_text())['references']
        controls=pd.DataFrame([item['result'] for item in references.values()])
        print('Frozen matched controls:',len(controls),'/36')
        display((100*controls.pivot(index=['fold','seed'],columns='model',values='validation_dice')).round(2))
    else: print('Run cell 4 to load the matched controls.')

band_status=ROOT/'bands_v1/campaign_status.json'
if band_status.exists():
    band=json.loads(band_status.read_text())
    print('BAND status:',band['status'],'completed',len(band['results']),'/36')
    paired=pd.DataFrame(band['results'])
    if not paired.empty:
        display(paired[['model','fold','seed','baseline_dice','band_dice','delta_dice_pp','weight','updates']])
        display(paired.groupby('model').agg(completed=('band_dice','size'),baseline=('baseline_dice','mean'),bands=('band_dice','mean'),delta_pp=('delta_dice_pp','mean'),improved_runs=('improved','sum')))
    print('All outcomes retained. Positive delta means BAND improved final validation Dice.')
else: print('BAND has not started in this runtime. Run cell 4 in bands mode.')
'''
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source']);cell['outputs']=[];cell['execution_count']=None
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    (EXPORT/'bands_deploy_assets.json').write_text(json.dumps(assets))
    saved=EXPORT/'recipe_analysis/baseline_final_saved.ipynb'
    if saved.exists():
        old=json.loads(saved.read_text())
        for i in (1,2,3,5):
            notebook['cells'][i]['outputs']=old['cells'][i].get('outputs',[])
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    print(path,'updated in place; separate verified package',asset_path.stat().st_size,'bytes')


if __name__=='__main__':main()
