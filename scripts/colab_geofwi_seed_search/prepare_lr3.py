"""Freeze the authorized higher-LR pilot and two matched BANDS comparisons."""
import ast
import shutil
import zipfile
from scripts.colab_geofwi_seed_search import prepare as base

ROOT=base.ROOT
PARENT=ROOT/'reports/geofwi3d_unetr_ce2_bands_20261019'
OUT=ROOT/'reports/geofwi3d_unetr_lr3_bands_20261019'
XY=ROOT/'reports/geofwi3d_unetr_xy_aug_20261019/completed_evidence'
read,write,digest=base.read,base.write,base.digest


def prepare():
    stage=OUT/'bundle';stage.mkdir(parents=True,exist_ok=True);source=PARENT/'bundle'
    for name,sha in read(source/'bundle_manifest.json').items():
        assert digest(source/name)==sha,name
        if name=='ce2_runtime.py':continue
        dst=stage/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dst)
    if not (stage/'data').exists():(stage/'data').symlink_to(source/'data',target_is_directory=True)
    for src,dst in [('ce1_bands_objective.py','rescue_objective.py'),('lr3_runtime.py','lr3_runtime.py')]:
        shutil.copyfile(ROOT/'scripts/colab_geofwi_seed_search'/src,stage/dst)
    shutil.copyfile(OUT/'profile.json',stage/'rescue_profile.json')
    profile=read(stage/'rescue_profile.json');assert profile['recipe_sha256']==digest(stage/'rescue_objective.py')
    worker=stage/'worker.py';s=worker.read_text().replace('Dice + 2CE','Dice + CE').replace('Dice+2CE','Dice+CE')
    worker.write_text(s)
    # Preserve the previously archived control bytes and validate their original manifest.
    manifest=read(XY/'archive_manifest.json')
    for name in ['config.json','completion.json','runtime_recipe.json','validation_latest.json','validation_best.json','metrics.csv','training_final.json']:
        assert digest(XY/name)==manifest['metadata'][name]['sha256'],name
        dst=stage/'reference_xy'/name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(XY/name,dst)
    write(stage/'reference_xy/archive_manifest.json',manifest)
    p=read(stage/'templates/unetr_vit_b16/protocol.json')
    schedules=p['models']['unet']['objectives']
    for arm,schedule in schedules.items():
        assert schedule['flops_per_update']==profile['costs'][arm]['update_flops']+(2 if arm=='dice_bands' else 0)
        assert schedule['calibration_flops']==profile['costs'][arm]['calibration_flops']
    p.update(name='geofwi3d_unetr_lr3_bands_20261019',weight_ce=1,
        training_loss='Foreground squared Dice+CE; optional calibrated binary BANDS',
        calibration_policy='19 training volumes only, after 50 shared base-loss updates; preserve RNG; calibrate against Dice+CE',
        profile_sha256=digest(stage/'rescue_profile.json'),
        constraints='Authorized archived-baseline BANDS and conditional higher-LR BANDS')
    write(stage/'templates/unetr_vit_b16/protocol.json',p)
    write(stage/'templates/unetr_vit_b16/profile_baseline.json',profile)
    plan=dict(status='authorized',dataset='GeoFWI3D',models=['unetr_vit_b16'],folds=[0],seeds=[0],
        augmentation='xy_d4_v1',threshold=.5,budget_flops=base.CAP,
        maximum_optimization_flops_total=base.CAP*3,schedules=schedules,
        order=['lr3','xy_bands','lr3_bands'],learning_rates={'lr3':3e-4,'xy_bands':1e-4,'lr3_bands':3e-4},
        hypothesis='Higher LR may improve learning within the short capped schedule. CE2 did not improve the half-budget gate.',
        loss_weights={'dice':1,'ce':1},fixed='ViT-B/16, same seed-0 init, XY D4/order, dropout .1, AdamW wd 1e-5, 477-update LR horizon, BF16, batch1, native96cubed',
        gate=dict(threshold=.46,rule='Mean validation Dice at 40% and 50% of total capped arithmetic >= .46',
            applies_to=['lr3','xy_bands','lr3_bands'],limitation='May reject late improvers; pruned runs are not final-budget results'),
        bands_prefix_updates=50,bands_ramp_updates=50,
        calibration='Training-only base-loss logit-gradient RMS target .10, p95 cap .50; calibrated arithmetic reserved inside cap',
        xy_bands=dict(authorized=True,reference_final=.4482058253085583,initialization='Fresh seed-0, not finetuning; archived control is not retrained',
            trigger='Unconditional user authorization, even if lr3 pruned'),
        lr3_bands=dict(authorized=True,trigger='Verified full-budget lr3 final validation Dice strictly > .50',initialization='Fresh seed-0'),
        maximum_new_runs=3,diagnostics='Three separately reported preflight steps per launched arm; established convention',
        scope='Development pilots only, not completion of 3 seeds / 3 folds')
    write(stage/'plan.json',plan);write(OUT/'experiment_plan.json',plan)
    provenance=read(stage/'source_provenance.json')
    provenance.update(parent_manifest_sha256=digest(source/'bundle_manifest.json'),
        changes=['Restore CE weight 1','LR 3e-4 pilot','Independent archived-baseline BANDS at LR 1e-4',
            'Conditional LR 3e-4 BANDS','Same .46 halfway gate for all new arms'])
    write(stage/'source_provenance.json',provenance)
    files={str(f.relative_to(stage)):digest(f) for f in stage.rglob('*') if f.is_file()
        and not f.is_relative_to(stage/'data') and '__pycache__' not in f.parts
        and not f.is_relative_to(stage/'derived_protocols') and f.name!='bundle_manifest.json'}
    for name in files:
        if name.endswith('.py'):ast.parse((stage/name).read_text())
    write(stage/'bundle_manifest.json',files)
    code=OUT/'geofwi3d_unetr_lr3_bands_code_v1.zip'
    with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*files,'bundle_manifest.json']):z.write(stage/name,name)
    uploads={k:v for k,v in read(PARENT/'package.json')['uploads'].items() if k.startswith('geofwi3d_data_part')}
    uploads[code.name]=digest(code)
    nb=read(PARENT/'GeoFWI3D_UNETR_CE2_conditional_BANDS_f0s0.ipynb')
    for cell in nb['cells']:
        s=cell['source'];s=''.join(s) if isinstance(s,list) else s
        cell['source']=s.replace('ce2_bands','lr3_bands').replace('ce2_runtime.py','lr3_runtime.py')
    s=nb['cells'][3]['source'];s=s[s.index('existing=Path'):]
    nb['cells'][3]['source']=f'EXPECTED_UPLOADS={uploads!r}\nEXPECTED_MANIFEST={digest(stage/"bundle_manifest.json")!r}\n'+s
    nb['cells'][0]['source']=f'''# GeoFWI3D — UNETR higher-LR pilot and matched BANDS
Account **focacciafilippo@gmail.com**. **A100-SXM4-80GB + High RAM**. Run one copy.

Sequential queue, fold 0 / seed 0:
1. **lr3**: Dice+CE (1:1), peak LR **3e-4**, fresh original initialization.
2. **xy_bands**: matched BANDS at LR **1e-4** against the verified archived **44.8206% final Dice** baseline. Runs independently of lr3's outcome. The archived baseline is not retrained.
3. **lr3_bands**: matched BANDS at LR **3e-4**, only if lr3's verified **final-budget Dice >50%**. Best Dice cannot trigger it.

Same ViT-B/16, XY D4 augmentation/order, dropout .1, AdamW decay 1e-5, fixed 477-update LR horizon,
BF16, batch 1, native 96³, 19 training / 96 validation volumes. Each arm starts from the same seed-0 initialization; BANDS is not checkpoint finetuning.

**Cap: {base.CAP/1e9:,.8f} GFLOPs per arm.** Baseline 414 updates; BANDS 407 including calibration.
**All new arms use the unchanged 46% halfway rule:** mean unaugmented validation Dice at 40% and 50% of capped arithmetic must be >=46%.
Baseline checkpoints are updates 165 / 207; BANDS checkpoints account for its calibration reservation.
A pruned BANDS run is not a complete final-budget comparison. This gate may reject a late improver.

BANDS: 50 shared base-loss updates, train-only gradient calibration against Dice+CE (target .10, p95 cap .50),
then a 50-update ramp; two-voxel six-connected salt boundaries. Calibration is included inside its cap.
At most three new caps total; no further seeds or recipes. Three preflight updates per launched arm are diagnostic
and separately reported under the established convention; validation/setup are outside optimization accounting.
Verified archives precede checkpoint cleanup; interruptions preserve recovery. Reruns do not repeat completed/pruned stages.
This is development work, not completion of the 3-fold/3-seed baseline.
'''
    nb['cells'][6]['source']='## GPU checks\nNative augmented forward/backward and exact model/optimizer/RNG replay. Each BANDS arm is checked immediately before its authorized launch.\n'
    nb['cells'][8]['source']='## Run or resume\nRun lr3, independent xy_bands, then conditional lr3_bands. Keep each cap and the same 46% halfway gate.\n'
    nb['cells'][10]['source']='## Results\n`lr3_bands_status.json` separates each arm and pruning. `EXPERIMENT_FINISHED.json` means all authorized arms are complete or pruned, not that Dice exceeded 50%.\n'
    name='GeoFWI3D_UNETR_LR3_and_matched_BANDS_f0s0.ipynb';nb['metadata']['colab']['name']=name
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source']);cell['outputs']=[];cell['execution_count']=None
    write(OUT/name,nb)
    write(OUT/'package.json',dict(notebook=name,uploads=uploads,manifest_sha256=digest(stage/'bundle_manifest.json'),schedules=schedules))
    print(OUT/name)


if __name__=='__main__':prepare()
