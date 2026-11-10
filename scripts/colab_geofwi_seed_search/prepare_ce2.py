"""Freeze Dice+2CE and a conditional matched BANDS follow-up."""
import ast
import shutil
import zipfile
from copy import deepcopy
from scripts.colab_geofwi_seed_search.prepare_rescue import replace_once
from scripts.colab_geofwi_seed_search import prepare as base
ROOT=base.ROOT
PARENT=ROOT/'reports/geofwi3d_unetr_xy_aug_20261019'
OUT=ROOT/'reports/geofwi3d_unetr_ce2_bands_20261019'
read,write,digest=base.read,base.write,base.digest


def prepare():
    stage=OUT/'bundle';stage.mkdir(parents=True,exist_ok=True);source=PARENT/'bundle'
    for name,sha in read(source/'bundle_manifest.json').items():
        assert digest(source/name)==sha,name
        dst=stage/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dst)
    if not (stage/'data').exists():(stage/'data').symlink_to(source/'data',target_is_directory=True)
    for src,dst in [('ce2_objective.py','rescue_objective.py'),('ce2_runtime.py','ce2_runtime.py')]:
        shutil.copyfile(ROOT/'scripts/colab_geofwi_seed_search'/src,stage/dst)
    (stage/'augmentation_runtime.py').unlink()
    shutil.copyfile(OUT/'profile.json',stage/'rescue_profile.json')
    profile=read(stage/'rescue_profile.json');assert profile['recipe_sha256']==digest(stage/'rescue_objective.py')
    def edit(name,old,new):
        path=stage/name;path.write_text(replace_once(path.read_text(),old,new))
    worker='worker.py'
    edit(worker,"exp.configure(p,'dice')","exp.configure(p,p['objective'])")
    edit(worker,'    exp.trainer.training_loss=rescue.training_loss',
        "    exp.trainer.training_loss=rescue.training_loss\n    exp.trainer.calibration_case=rescue.calibration_case\n    original_calibrate=exp.trainer.calibrate\n    def calibrate(*args,**kwargs):\n        report=original_calibrate(*args,**kwargs)\n        report.update(base_gradient='Foreground squared Dice + 2CE',legacy_dice_rms_means='Dice+2CE logit-gradient RMS')\n        return report\n    exp.trainer.calibrate=calibrate")
    edit(worker,"objective='dice',fold=args.fold","objective=p['objective'],fold=args.fold")
    edit(worker,"model=p['model_identity'],\n        augmentation", "model=p['model_identity'],objective=p['objective'],calibration_base='Dice+2CE',\n        augmentation")
    edit(worker,"training_loss(logits,yy,'dice',{},.2,criterion=native)","training_loss(logits,yy,p['objective'],{'bands':.04} if p['objective']=='dice_bands' else {},1.,criterion=native)")
    edit(worker,"['objectives']['dice']['flops_per_update']","['objectives'][p['objective']]['flops_per_update']")
    trainer='source/scripts/cross_dataset_campaign/capped_core.py'
    edit(trainer,'    if ledger["training_flops"] > budget:', '    if ledger["training_flops"] + ledger["calibration_flops"] > budget:')
    edit(trainer,'            ledger["calibration_flops"] += cal_cost',
        '            if ledger["calibration_attempts"] or ledger["training_flops"] + cal_cost > budget:\n                raise RuntimeError("Calibration retry requires audit; no duplicate compute authorized")\n            ledger["calibration_flops"] += cal_cost')
    edit(trainer,'if ledger["training_flops"] + cost > budget:', 'if ledger["training_flops"] + ledger["calibration_flops"] + cost > budget:')
    edit(trainer,'budget - ledger["training_flops"] >= schedule["flops_per_update"]', 'budget - total >= schedule["flops_per_update"]')
    edit(trainer,'"unused_flops": budget - ledger["training_flops"],', '"unused_flops": budget - total, "cap_includes_calibration": True,')
    edit(trainer,'        "budget_scope": SCOPE,\n', '        "budget_scope": SCOPE, "cap_includes_calibration": True,\n')
    storage='source/scripts/cross_dataset_campaign/storage.py'
    edit(storage,"config['augmentation']!='none' or done['consumed_training_flops']>config['budget_flops']", "config['augmentation']!=task['augmentation'] or done['consumed_total_flops']>config['budget_flops']")
    edit('runtime.py',"schedule = p['models'][p['trainer_alias']]['objectives']['dice']", "schedule = p['models'][p['trainer_alias']]['objectives'][t['arm']]")
    edit('runtime.py',"or done['budget_flops']!=CAP or config['smoke_test']", "or done['budget_flops']!=CAP or config['smoke_test'] or done['status']!='complete'\n            or done['objective']!=t['arm'] or config['objective']!=t['arm']\n            or done['calibration_flops']!=schedule['calibration_flops']\n            or done['consumed_total_flops']!=done['consumed_training_flops']+done['calibration_flops']\n            or done['consumed_total_flops']>CAP")
    edit('runtime.py',"updates=done['updates'], training_flops=done['consumed_training_flops'],", "updates=done['updates'], training_flops=done['consumed_training_flops'],\n                calibration_flops=done['calibration_flops'],total_flops=done['consumed_total_flops'],")
    p=read(stage/'templates/unetr_vit_b16/protocol.json');baseline=profile['costs']['dice']['update_flops']
    schedules={}
    for arm in ('dice','dice_bands'):
        measured=profile['costs'][arm];prefix=50 if arm=='dice_bands' else 0
        # One host ramp division and coefficient-times-ramp multiplication.
        cost=measured['update_flops']+(2 if prefix else 0);cal=measured['calibration_flops']
        n=prefix+(base.CAP-prefix*baseline-cal)//cost
        training=prefix*baseline+(n-prefix)*cost
        s=deepcopy(p['models']['unet']['objectives']['dice'])
        s.update(prefix_updates=prefix,sdf_prefix_updates=prefix,baseline_flops_per_update=baseline,
            early_flops_per_update=cost,flops_per_update=cost,calibration_flops=cal,
            max_steps=n,consumed_training_flops=training,consumed_flops=training,
            consumed_total_flops=training+cal,unused_flops=base.CAP-training-cal,patient_pass_epochs=n/19)
        schedules[arm]=s
    p['models']['unet']['objectives']=schedules
    p.update(name='geofwi3d_unetr_ce2_bands_20261019',weight_ce=2,objective='dice',
        training_loss='Foreground squared Dice+2CE; optional calibrated binary BANDS',
        sdf_policy={'dice':'not_applicable','dice_bands':'not_applicable'},
        calibration_policy='19 training volumes only, after 50 shared base-loss updates; preserve RNG; calibrate against Dice+2CE',
        constraint_geometry='Binary salt BANDS inner/outer, 2-voxel six-connected morphology',
        cap_includes_calibration=True,profile_sha256=digest(stage/'rescue_profile.json'),
        budget_policy='Original cap per arm; BANDS calibration reserved within the same cap',
        constraints='Conditional matched BANDS only',requested_arm_order=['dice','dice_bands'])
    write(stage/'templates/unetr_vit_b16/protocol.json',p)
    write(stage/'templates/unetr_vit_b16/profile_baseline.json',profile)
    plan=read(PARENT/'next_experiment_plan.json')
    plan.update(status='authorized',dataset='GeoFWI3D',models=['unetr_vit_b16'],folds=[0],augmentation='xy_d4_v1',
        threshold=.5,budget_flops=base.CAP,maximum_optimization_flops_total=base.CAP*2,
        schedules=schedules,bands_stopping_rule='Same 46% mean at 40%/50% of total capped arithmetic',
        bands_prefix_updates=50,bands_ramp_updates=50)
    write(stage/'plan.json',plan)
    provenance=read(stage/'source_provenance.json')
    provenance.update(ce2_parent_manifest_sha256=digest(source/'bundle_manifest.json'),
        changes=['CE weight 2','conditional binary BANDS calibrated against Dice+2CE','calibration inside cap','explicit augmentation archive check'])
    write(stage/'source_provenance.json',provenance)
    files={str(f.relative_to(stage)):digest(f) for f in stage.rglob('*') if f.is_file()
        and not f.is_relative_to(stage/'data') and '__pycache__' not in f.parts
        and not f.is_relative_to(stage/'derived_protocols') and f.name!='bundle_manifest.json'}
    for name in files:
        if name.endswith('.py'):ast.parse((stage/name).read_text())
    write(stage/'bundle_manifest.json',files)
    code=OUT/'geofwi3d_unetr_ce2_bands_code_v1.zip'
    with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*files,'bundle_manifest.json']):z.write(stage/name,name)
    uploads={k:v for k,v in read(PARENT/'package.json')['uploads'].items() if k.startswith('geofwi3d_data_part')}
    uploads[code.name]=digest(code)
    nb=read(PARENT/'GeoFWI3D_UNETR_XY_augmentation_f0s0.ipynb')
    for cell in nb['cells']:
        s=cell['source'];s=''.join(s) if isinstance(s,list) else s
        for old,new in [('results_unetr_xy_aug_f0s0_v1','results_unetr_ce2_bands_f0s0_v1'),
            ('hippo_geofwi_unetr_xy_aug_v1','hippo_geofwi_unetr_ce2_bands_v1'),
            ('augmentation_runtime.py','ce2_runtime.py'),('geofwi3d_unetr_xy_aug_code_v1.zip',code.name)]:s=s.replace(old,new)
        cell['source']=s
    s=nb['cells'][3]['source'];s=s[s.index('def sha(path):'):]
    # Reuse the previous verified dataset on the same VM without copying 319 volumes again.
    reuse="""existing=Path('/content/hippo_geofwi_unetr_xy_aug_v1/data')
if existing.is_dir() and not (BUNDLE/'data').exists():
    (BUNDLE/'data').symlink_to(existing,target_is_directory=True)
"""
    nb['cells'][3]['source']=f'EXPECTED_UPLOADS={uploads!r}\nEXPECTED_MANIFEST={digest(stage/"bundle_manifest.json")!r}\n'+reuse+s
    nb['cells'][0]['source']=f"""# GeoFWI3D — UNETR Dice + 2CE, then conditional BANDS
One fresh fold-0 / seed-0 baseline. Change CE weight from 1 to **2**; keep ViT-B/16,
XY D4 augmentation, dropout 0.1, LR 1e-4, AdamW decay 1e-5, fixed 477-update LR horizon,
BF16, batch 1, native 96³ and identical initialization. 19 train / 96 validation volumes.

**Cap per arm: {base.CAP/1e9:,.8f} GFLOPs. Baseline: {schedules['dice']['max_steps']} updates.**
Stop at 207 unless the mean original validation Dice at updates 165 and 207 is **>=46%**.
Success requires **final-budget validation Dice strictly >50%**. Best Dice never triggers BANDS.

If and only if the verified archived baseline passes, automatically run **one matched BANDS arm**
from the same fresh seed-0 initialization. Same Dice+2CE base recipe and training sequence;
50 shared base-loss updates, then training-only gradient calibration (target 0.10, p95 cap 0.50)
and a 50-update BANDS ramp. Two-voxel six-connected inner/outer salt boundaries.
BANDS fits **{schedules['dice_bands']['max_steps']} updates including its measured calibration within the same cap**.
The same 46% halfway gate applies. A pruned BANDS arm is explicitly marked as an incomplete full-budget comparison.
At most two caps in total; no automatic further seeds or recipes.

Three GPU preflight updates per launched arm are diagnostic and separately reported, as in the established convention.
Validation/setup are outside optimization accounting. Archive verification precedes checkpoint retirement.
Interruptions preserve recovery; reruns never repeat completed stages. This is a development experiment, not the 3×3 baseline.
Account **focacciafilippo@gmail.com**, **A100-SXM4-80GB + High RAM**. Run one copy.
"""
    nb['cells'][6]['source']='## Required GPU checks\nNative augmented forward/backward, finite gradients and exact model/optimizer/RNG replay. Conditional BANDS is checked only after baseline success.\n'
    nb['cells'][8]['source']='## Run or resume\nApply the 46% halfway gate; finish the baseline only if it passes. Automatically launch matched BANDS only for verified final baseline Dice >50%.\n'
    nb['cells'][9]['source']="SESSION_HOURS=2.0\n# Graceful stop: create RESULTS/'STOP'. Rerun to resume a timed pause.\ncampaign('run','--hours',str(SESSION_HOURS))\n"
    nb['cells'][10]['source']='## Results\n`ce2_bands_status.json` separates baseline qualification, conditional BANDS and technical pauses. `EXPERIMENT_FINISHED.json` records a completed or pruned job.\n'
    nb['cells'][11]['source']="campaign('report')\nstatus=json.loads((RESULTS/'ce2_bands_status.json').read_text())\nprint(json.dumps(status,indent=2))\n"
    name='GeoFWI3D_UNETR_CE2_conditional_BANDS_f0s0.ipynb';nb['metadata']['colab']['name']=name
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source']);cell['outputs']=[];cell['execution_count']=None
    write(OUT/name,nb)
    write(OUT/'package.json',dict(notebook=name,uploads=uploads,manifest_sha256=digest(stage/'bundle_manifest.json'),schedules=schedules))
    print(OUT/name)

if __name__=='__main__':prepare()
