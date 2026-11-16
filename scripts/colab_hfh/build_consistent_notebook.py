"""Embed the fixed-seed, fixed-fold recipe campaign in the baseline notebook."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT/'reports/hfh_colab_exploration_20261111/HFH_baseline_shared_seed_search.ipynb'


def main():
    notebook = json.loads(NOTEBOOK.read_text())
    notebook['cells'][0]['source'] = '''# HFH — consistent recipes, three seeds × three folds

Compute: the configured compute account. Inputs and results: the configured storage account.
Use an available NVIDIA GPU and High-RAM. Native BF16 is preferred; older GPUs use a separate FP32 campaign. Actual GPU and precision are recorded; no automatic batch-size or image-resolution reduction.

We now tune the training recipe with **fixed seeds 0, 1, 2 and fixed patient folds 0, 1, 2**. Each model must reach **at least 50% final validation Dice on all nine combinations**. Success means all four models pass the same nine combinations (36 runs). A failed final score rejects that model's candidate recipe; runtime/storage errors pause execution instead of rejecting a seed. All attempts remain available, including failures. These are validation-selected exploratory results, not an unbiased test estimate.

Only the initial learning rate is explored. Optimizers, losses, batch size, no augmentation, data splits, LR horizon and per-run FLOPs cap stay fixed. Candidate rates: nnU-Net SGD **0.1, 0.05, 0.2**; UNETR AdamW **0.0003, 0.001, 0.0001**; BasicUNet AdamW **0.0003, 0.001, 0.0001**; SwinUNETR AdamW **0.0001, 0.0003, 0.001**. The first recipe passing all nine cells is retained per model. If none passes, execution pauses for diagnosis.

Training allowances remain **nnU-Net 90, UNETR 245, BasicUNet 563, SwinUNETR 206 updates**, each within **31,976,450,667,090 optimization FLOPs**. Each fold uses five fixed training subjects (ten hippocampus crops) and five disjoint validation subjects (ten crops). Crops are 64×32×64 at 1×2×1 mm, padded to 64³. Validation removes padding and averages per-crop foreground Dice.

Training and recovery writes use local VM SSD. Each completed diagnostic run is packaged into one small archive, uploaded directly to Drive and checksum-verified. Completed archives restore on a fresh runtime; an unfinished charged attempt without local recovery requires an audit rather than silent replay. Keep this runtime connected during an active run. Results are in **results_recipe_diagnostics_v3** (or its separate FP32 folder), preserving v1/v2 results.

Run cells in order. Cell 4 attaches to an existing campaign or starts one worker, then streams progress. Interrupting it requests a stop after the current run is archived. The session duration is an upper limit, not a required delay. Cell 5 reports the current qualifying recipes and every attempted cell.
'''
    files = {name:(ROOT/'scripts/colab_hfh'/name).read_text()
             for name in ('recipe_probe.py','consistent_recipes.py','recipe_transport.py')}
    notebook['cells'][4]['source'] = '''# Attach to the existing worker, or restore verified results and continue.
import psutil
active=[]
for item in psutil.process_iter(['cmdline']):
    try:
        if str(ROOT/'consistent_recipes.py') in (item.info['cmdline'] or []): active.append(item)
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
if len(active)>1: raise RuntimeError('Multiple campaign controllers detected; inspect before continuing')
''' + 'embedded_scripts = '+repr(files)+'''
log_path=ROOT/'consistent_recipes.log'
def still_running(process):
    try: return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess: return False
if active:
    process=active[0]
    print('Attached to existing campaign PID',process.pid,flush=True)
else:
    for name,source in embedded_scripts.items(): (ROOT/name).write_text(source)
    (ROOT/'STOP_RECIPE_SEARCH').unlink(missing_ok=True)
    with log_path.open('a') as log:
        child=subprocess.Popen([str(PYTHON),str(ROOT/'consistent_recipes.py'),'--root',str(ROOT),
            '--hours',str(min(SESSION_HOURS,6))],cwd=ROOT,env=env,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    process=psutil.Process(child.pid)
with log_path.open() as log:
    try:
        while still_running(process):
            content=log.read()
            if content: print(content,end='',flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        (ROOT/'STOP_RECIPE_SEARCH').touch()
        print('Stop requested. Current run will finish and verify its archive; keep the runtime connected.',flush=True)
        while still_running(process): time.sleep(2)
    print(log.read(),end='',flush=True)
state_path=ROOT/('consistent_recipes_status'+WORK_SUFFIX+'.json')
print(json.loads(state_path.read_text()) if state_path.exists() else 'Inspect the log above for startup errors.')
'''
    notebook['cells'][5]['source'] = '''import pandas as pd
status=json.loads((ROOT/('consistent_recipes_status'+WORK_SUFFIX+'.json')).read_text())
if status['status']=='all_four_models_passed_same_nine_cells':
    print('PASS: ALL 36 RUNS ≥50%')
else: print('Status:',status['status'])
print('Seeds:',status['seeds'])
print('Folds:',status['folds'])
names={'nnunet':'nnU-Net','unetr_vit_b16':'UNETR','basic_unet16':'BasicUNet','swinunetr':'SwinUNETR'}
for model,recipe in status['selected_recipes'].items():
    print(names[model], '9/9 passed')
    print(f"  mean {100*recipe['mean_dice']:.2f}%")
    print(f"  worst {100*recipe['minimum_dice']:.2f}%")
recipes=pd.DataFrame(status['selected_recipes']).T
if not recipes.empty:
    display(recipes[['initial_lr','cells','mean_dice','minimum_dice']])
table=pd.DataFrame(status['attempts'])
if not table.empty:
    selected=table.apply(lambda r:r['model'] in status['selected_recipes'] and
        r['lr']==status['selected_recipes'][r['model']]['initial_lr'],axis=1)
    if selected.any():
        display((100*table[selected].pivot(index=['fold','seed'],columns='model',values='validation_dice')).round(2))
    if (~selected).any(): display(table[~selected][['model','lr','fold','seed','validation_dice','updates']])
print('All four models must pass all nine fixed seed/fold combinations. Report failed recipes too.')
'''
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(cell['source'])
            cell['outputs']=[];cell['execution_count']=None
    NOTEBOOK.write_text(json.dumps(notebook,indent=1)+'\n')
    print(NOTEBOOK)


if __name__=='__main__':
    main()
