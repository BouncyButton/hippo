"""Freeze a single augmentation experiment from the verified rescue bundle."""
import ast
from pathlib import Path
import shutil
import zipfile
from scripts.colab_geofwi_seed_search.prepare_rescue import replace_once
from scripts.colab_geofwi_seed_search import prepare as base

ROOT=base.ROOT
PARENT=ROOT/'reports/geofwi3d_unetr_rescue_20261019'
OUT=ROOT/'reports/geofwi3d_unetr_xy_aug_20261019'
read,write,digest=base.read,base.write,base.digest


def prepare():
    stage=OUT/'bundle';stage.mkdir(parents=True,exist_ok=True)
    source=PARENT/'bundle'
    for name,sha in read(source/'bundle_manifest.json').items():
        assert digest(source/name)==sha,name
        dst=stage/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dst)
    if not (stage/'data').exists():(stage/'data').symlink_to(source/'data',target_is_directory=True)
    shutil.copyfile(ROOT/'scripts/colab_geofwi_seed_search/xy_augmentation.py',stage/'xy_augmentation.py')
    def edit(name,old,new):
        path=stage/name;path.write_text(replace_once(path.read_text(),old,new))
    edit('runtime.py',"plan['augmentation'] != 'none'","plan['augmentation'] != 'xy_d4_v1'")
    edit('source/scripts/cross_dataset_campaign/common.py',"if protocol.get('augmentation') != 'none':\n        raise ValueError('Only explicitly unaugmented runs are allowed')", "if protocol.get('augmentation') != 'xy_d4_v1':\n        raise ValueError('Only the authorized XY D4 experiment is allowed')")
    edit('worker.py',"augmentation='none',weight_dice=1", "augmentation='xy_d4_v1',augmentation_source_sha256=io.digest(Path(__file__).parent/'xy_augmentation.py'),weight_dice=1")
    edit('worker.py',"p['augmentation']!='none'", "p['augmentation']!='xy_d4_v1'")
    edit('worker.py',"    def step():\n        model.train();", "    from xy_augmentation import augment\n    def step(update):\n        xx,yy=augment(x,y,0,update)\n        model.train();")
    edit('worker.py','logits=model(x)','logits=model(xx)')
    edit('worker.py',"training_loss(logits,y,'dice'", "training_loss(logits,yy,'dice'")
    edit('worker.py','first=step()','first=step(0)')
    edit('worker.py','expected_loss=step();','expected_loss=step(1);')
    edit('worker.py','actual_loss=step()','actual_loss=step(1)')
    edit('worker.py','exact_model_optimizer_rng_replay=True,diagnostic_optimizer_updates=3,', 'exact_model_optimizer_rng_replay=True,xy_d4_applied=True,diagnostic_optimizer_updates=3,')
    trainer='source/scripts/cross_dataset_campaign/capped_core.py'
    edit(trainer,'    configuration = {', "    from xy_augmentation import augment\n    configuration = {\n        'augmentation_source_sha256': base.digest(Path(__file__).resolve().parents[3] / 'xy_augmentation.py'),")
    edit(trainer,'        current_affine = original_affine\n', '        current_affine = original_affine\n        if protocol["augmentation"] == "xy_d4_v1":\n            image, label = augment(image, label, args.seed, charged)\n')
    # Constraints are not used; image and label receive the identical XY permutation.
    runtime=(stage/'rescue_runtime.py').read_text()
    runtime=runtime.replace('TOTAL_CAP=CAP*3//2','TOTAL_CAP=CAP').replace("ARMS=('standard_lr','lower_lr')","ARMS=('xy_d4',)").replace("LRS={'standard_lr':1e-4,'lower_lr':3e-5}","LRS={'xy_d4':1e-4}")
    runtime=runtime.replace("'GeoFWI3D_unetr_rescue_'+arm", "'GeoFWI3D_unetr_aug_'+arm")
    runtime=runtime.replace('full*cost+half*cost>TOTAL_CAP','full*cost>TOTAL_CAP')
    runtime=runtime.replace("recipe['augmentation']!='none'", "recipe['augmentation']!='xy_d4_v1' or config['augmentation']!='xy_d4_v1'")
    runtime=runtime.replace('for i in (3,4,5)','for i in (4,5)').replace('sum(scores)/3','sum(scores)/2')
    runtime=runtime.replace("reason='Lower predeclared screening score'", "reason='Mean validation Dice at 40% and 50% budget below 0.40'")
    runtime=runtime.replace("    value=dict(status='complete' if final else (status or 'in_progress'),", "    pruned=bool(screens and (durable/'screening'/task('xy_d4')['id']/'pruned.json').exists())\n    value=dict(status='complete' if final else ('pruned' if pruned else (status or 'in_progress')),")
    runtime=runtime.replace("if selection:\n        arm=selection['winner'];", "if selection and selection['winner']:\n        arm=selection['winner'];")
    runtime=runtime.replace("scope='Two-recipe development experiment; not three-seed baseline completion'", "scope='Single XY augmentation experiment; not three-seed baseline completion'")
    runtime=runtime.replace("'rescue_status.json'","'augmentation_status.json'").replace("if final:io.write(durable/'RESCUE_COMPLETE.json',value)","if final or pruned:io.write(durable/'EXPERIMENT_FINISHED.json',value)")
    start=runtime.index('def choose(rows):');end=runtime.index('\ndef prune(',start)
    runtime=runtime[:start]+'''def gate(row):
    score=row['selection_score']
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid gate score')
    return dict(winner='xy_d4' if score>=.4 else None, threshold=.4,
        score=score, decision='continue' if score>=.4 else 'prune',
        rule='Mean validation Dice at updates 165 and 207 must be >= 0.40')

'''+runtime[end:]
    start=runtime.index('def run(root,durable,hours):');end=runtime.index('\ndef main():',start)
    runtime=runtime[:start]+'''def run(root,durable,hours):
    value=io.binding(root,durable)
    if io.read(durable/'preflight.json')!=dict(status='passed',**value):raise ValueError('Run preflight first')
    work=root/'work';work.mkdir(exist_ok=True)
    with (work/'augmentation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.time()+hours*3600;cost,half,full=settings(root);arm='xy_d4'
        try:
            if publish(root,durable)['status'] in ('complete','pruned'):return
            if read_screen(root,durable,arm) is None:
                if not execute(root,durable,arm,half,deadline):
                    publish(root,durable,'paused');return
            decision=gate(read_screen(root,durable,arm));path=durable/'selection.json'
            if path.exists() and io.read(path)!=decision:raise ValueError('Gate changed')
            io.write(path,decision);print('HALF-BUDGET GATE',decision,flush=True)
            if decision['decision']=='prune':
                prune(root,durable,arm);publish(root,durable);return
            if not io.completed(durable,task(arm)):
                if not execute(root,durable,arm,full,deadline):
                    publish(root,durable,'paused');return
            publish(root,durable)
        except Exception as exc:
            io.write(durable/'augmentation_error.json',dict(error=str(exc),time=time.time()))
            publish(root,durable,'technical_error');raise

'''+runtime[end:]
    runtime=runtime.replace('"""Two half-budget arms, one exact continuation; total optimization <= 1.5 caps."""','"""Single XY D4 arm with a 40% Dice futility gate; total optimization <= one cap."""')
    (stage/'augmentation_runtime.py').write_text(runtime)
    (stage/'rescue_runtime.py').unlink()
    p=read(stage/'templates/unetr_vit_b16/protocol.json')
    p.update(name='geofwi3d_unetr_xy_aug_20261019',augmentation='xy_d4_v1',augmentation_policy='Uniform eight XY symmetries; preserve Z; no interpolation/intensity changes; train only')
    write(stage/'templates/unetr_vit_b16/protocol.json',p)
    plan=read(PARENT/'next_experiment_plan.json');plan.update(status='authorized',dataset='GeoFWI3D',models=['unetr_vit_b16'],folds=[0],augmentation='xy_d4_v1',threshold=.5,budget_flops=base.CAP,maximum_optimization_flops_total=base.CAP)
    write(stage/'plan.json',plan)
    provenance=read(stage/'source_provenance.json');provenance.update(augmentation_parent_manifest_sha256=digest(source/'bundle_manifest.json'),augmentation_source_sha256=digest(stage/'xy_augmentation.py'),accounting='Model/loss/backward/optimizer unchanged; XY transforms are integer-index data permutations outside optimization FLOPs')
    write(stage/'source_provenance.json',provenance)
    files={str(f.relative_to(stage)):digest(f) for f in stage.rglob('*') if f.is_file() and not f.is_relative_to(stage/'data') and '__pycache__' not in f.parts and f.name!='bundle_manifest.json'}
    write(stage/'bundle_manifest.json',files)
    for name in files:
        if name.endswith('.py'):ast.parse((stage/name).read_text())
    code=OUT/'geofwi3d_unetr_xy_aug_code_v1.zip'
    with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*files,'bundle_manifest.json']):z.write(stage/name,name)
    package=read(PARENT/'package.json');uploads={k:v for k,v in package['uploads'].items() if k.startswith('geofwi3d_data_part')};uploads[code.name]=digest(code)
    nb=read(PARENT/'GeoFWI3D_UNETR_rescue_f0s0.ipynb')
    for cell in nb['cells']:
        s=cell['source'];s=''.join(s) if isinstance(s,list) else s
        s=s.replace('results_unetr_rescue_f0s0_v1','results_unetr_xy_aug_f0s0_v1').replace('hippo_geofwi_unetr_rescue_v1','hippo_geofwi_unetr_xy_aug_v1').replace('rescue_runtime.py','augmentation_runtime.py').replace('geofwi3d_unetr_rescue_code_v1.zip',code.name).replace('All inputs verified. No augmentation.','All inputs verified. Training augmentation: XY D4 only.')
        cell['source']=s
    s=nb['cells'][3]['source'];s=s[s.index('def sha(path):'):]
    nb['cells'][3]['source']=f'EXPECTED_UPLOADS={uploads!r}\nEXPECTED_MANIFEST={digest(stage/"bundle_manifest.json")!r}\n'+s
    nb['cells'][0]['source']='''# GeoFWI3D — UNETR horizontal augmentation pilot
One fresh seed-0 / fold-0 run. **Only augmentation changes** from the archived standard-LR rescue:
uniform rotations/reflections in XY, paired image/mask, depth preserved. No interpolation, crops,
intensity changes, validation augmentation or test-time augmentation.

UNETR ViT-B/16 (92,802,274 parameters), 19 train / 96 validation volumes, native 96³,
batch 1, BF16, foreground-only squared Dice + CE (1:1), dropout 0.1, peak LR 1e-4,
AdamW decay 1e-5, unchanged 477-update LR horizon and initialization.

**Stop at 207 updates unless mean validation Dice at steps 165 and 207 is >=40%.**
If the gate passes, resume the same optimizer/RNG state through 414 updates.
Maximum optimization: 181,964.977 GFLOPs, below the original 182,178.127 GFLOP cap.
One GPU preflight uses three separately reported diagnostic updates. Validation/setup are outside the optimization cap.
Success requires **final-budget Dice >50%**. Pruned runs are not full-budget failures or successful baselines.
No automatic extra seeds. Prior results remain separate; the archived no-augmentation run is the control.
Account: **focacciafilippo@gmail.com**; use **A100-SXM4-80GB + High RAM**. Run one copy.
'''
    nb['cells'][6]['source']='## Required GPU checks\nNative 96³ augmented forward/backward, finite gradients, original initialization hash and exact optimizer/model/RNG checkpoint replay. Three diagnostic updates, excluded from scientific results.\n'
    nb['cells'][8]['source']='## Run or resume\nTrain to 207 updates; continue to 414 only if mean validation Dice at updates 165 and 207 is at least 40%. An interruption preserves recovery. Completed or pruned work is never restarted.\n'
    nb['cells'][10]['source']='## Results\n`augmentation_status.json` records the gate and charged compute. `EXPERIMENT_FINISHED.json` means completed or pruned; only a full-budget result strictly above 50% passes. This is not completion of the three-seed/three-fold baseline.\n'
    nb['cells'][11]['source']=nb['cells'][11]['source'].replace('rescue_status.json','augmentation_status.json')
    nb['cells'][11]['source']=nb['cells'][11]['source'].replace("else:print('Not complete; inspect status/logs before resuming.')", "elif status['status']=='pruned':print('Experiment finished at the futility gate; no automatic continuation.')\nelse:print('Paused or incomplete; inspect status/logs before resuming.')")
    name='GeoFWI3D_UNETR_XY_augmentation_f0s0.ipynb';nb['metadata']['colab']['name']=name;nb['metadata']['colab']['gpuType']='A100'
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source']);cell['outputs']=[];cell['execution_count']=None
    write(OUT/name,nb);write(OUT/'package.json',dict(notebook=name,uploads=uploads,manifest_sha256=digest(stage/'bundle_manifest.json')))
    print(OUT/name)

if __name__=='__main__':prepare()
