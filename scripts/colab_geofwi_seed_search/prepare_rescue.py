"""Create the bounded rescue bundle and replace the existing notebook's workflow."""
import ast
import json
from pathlib import Path
import shutil
import zipfile
from scripts.colab_geofwi_seed_search import prepare as base

ROOT=base.ROOT
PARENT=ROOT/'reports/geofwi3d_unetr_ce_pilot_20261019'
OUT=ROOT/'reports/geofwi3d_unetr_rescue_20261019'
read,write,digest=base.read,base.write,base.digest


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Frozen patch anchor changed: '+old[:100])
    return text.replace(old,new)


def prepare():
    stage=OUT/'bundle';stage.mkdir(parents=True,exist_ok=True)
    source=PARENT/'bundle'
    for name,sha in read(source/'bundle_manifest.json').items():
        if name=='pilot.py':continue
        assert digest(source/name)==sha,name
        dst=stage/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dst)
    if not (stage/'data').exists():(stage/'data').symlink_to(source/'data',target_is_directory=True)
    for name in ['rescue_objective.py','rescue_runtime.py']:
        shutil.copyfile(ROOT/'scripts/colab_geofwi_seed_search'/name,stage/name)
    shutil.copyfile(OUT/'profile.json',stage/'rescue_profile.json')
    profile=read(OUT/'profile.json');assert profile['recipe_sha256']==digest(stage/'rescue_objective.py')
    cost=profile['costs']['dice']['update_flops'];n=base.CAP//cost;half=(base.CAP//2)//cost
    p=read(stage/'templates/unetr_vit_b16/protocol.json')
    schedule=p['models']['unet']['objectives']['dice']
    for key in ['baseline_flops_per_update','early_flops_per_update','flops_per_update']:schedule[key]=cost
    for key in ['consumed_training_flops','consumed_flops','consumed_total_flops']:schedule[key]=n*cost
    schedule.update(max_steps=n,unused_flops=base.CAP-n*cost,patient_pass_epochs=n/19)
    steps=sorted({(base.CAP*i//10)//cost for i in range(1,11)})
    p.update(name='geofwi3d_unetr_rescue_20261019',dropout_rate=.1,
        training_loss='Foreground-only squared Dice + CE, weights 1:1; dropout 0.1',
        rescue_validation_steps=steps,validation_milestones=10,
        validation_target_flops=[base.CAP*i//10 for i in range(1,11)],
        profile_sha256=digest(stage/'rescue_profile.json'))
    write(stage/'templates/unetr_vit_b16/protocol.json',p)
    write(stage/'templates/unetr_vit_b16/profile_baseline.json',profile)
    worker=stage/'worker.py';s=worker.read_text()
    s=replace_once(s,"    exp.configure(p,'dice')\n", "    exp.configure(p,'dice')\n    import rescue_objective as rescue\n    exp.base.build_model=lambda alias,architecture:rescue.build_model(p['model_identity'],architecture)\n")
    start=s.index("    if p['weight_ce']")
    end=s.index('    builder=exp.base.build_model',start)
    s=s[:start]+"    exp.trainer.training_loss=rescue.training_loss\n"+s[end:]
    s=replace_once(s,"        model=builder(name,architecture)\n", "        model=builder(name,architecture)\n        rescue.dropout_audit(model)\n")
    s=replace_once(s,"dice_include_background=p['model_identity']!='nnunet',", "dice_include_background=False, dropout_rate=.1, squared_pred=True,\n        rescue_objective_sha256=io.digest(Path(__file__).parent/'rescue_objective.py'),")
    s=replace_once(s,'coherent=exc.code in (None,0,75);raise','coherent=exc.code in (None,0,75,76);raise')
    worker.write_text(s)
    trainer=stage/'source/scripts/cross_dataset_campaign/capped_core.py'
    s=trainer.read_text();before=digest(trainer)
    s=replace_once(s,'import math\n','import math\nimport os\n')
    s=replace_once(s,'    milestones = base.validation_steps(max_steps, protocol["validation_milestones"])',
        '    milestones = set(protocol["rescue_validation_steps"])\n    stage_limit = int(os.environ.get("HIPPO_STAGE_LIMIT", max_steps))\n    if not 0 < stage_limit <= max_steps:\n        raise ValueError("Invalid stage limit")')
    s=replace_once(s,'    while ledger["charged_attempts"] < max_steps:\n        if stopping():', '''    while ledger["charged_attempts"] < max_steps:
        if ledger["charged_attempts"] >= stage_limit:
            if step != ledger["charged_attempts"] or stage_limit not in milestones:
                raise ValueError("Incoherent screening boundary")
            # This extra diagnostic must not change the continuation's RNG trajectory.
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if amp else []
            mode = model.training
            try:
                training_stage = base.evaluate(model, train, device, amp)
            finally:
                model.train(mode)
                torch.set_rng_state(cpu_rng)
                if amp: torch.cuda.set_rng_state_all(cuda_rng)
            base.atomic_json(args.output / "training_stage_final.json", training_stage)
            commit()
            base.atomic_json(args.output / "stage_complete.json", {
                "status": "screen_complete", "updates": step,
                "charged_attempts": ledger["charged_attempts"],
                "training_flops": ledger["training_flops"],
                "checkpoint_sha256": base.digest(latest),
                "best_checkpoint_sha256": base.digest(args.output / "checkpoint_best.pt"),
                "full_budget_complete": False})
            raise SystemExit(76)
        if stopping():''')
    trainer.write_text(s)
    plan=read(PARENT/'rescue_experiment_plan.json')
    plan.update(dataset='GeoFWI3D',models=['unetr_vit_b16'],folds=[0],augmentation='none',threshold=.5,
        budget_flops=base.CAP,maximum_optimization_flops_total=base.CAP*3//2,
        profiled_flops_per_update=cost,half_steps=half,full_steps=n,
        maximum_planned_training_flops=(half+n)*cost)
    write(stage/'plan.json',plan)
    provenance=read(stage/'source_provenance.json')
    provenance.update(rescue_parent_manifest_sha256=digest(source/'bundle_manifest.json'),
        rescue_profile_sha256=digest(stage/'rescue_profile.json'),
        rescue_patch=dict(path='scripts/cross_dataset_campaign/capped_core.py',
            before_sha256=before,after_sha256=digest(trainer),
            reason='Pause at declared half-budget update without changing full config/schedule; preserve RNG around additional training evaluation'))
    write(stage/'source_provenance.json',provenance)
    files={str(f.relative_to(stage)):digest(f) for f in stage.rglob('*')
        if f.is_file() and not f.is_relative_to(stage/'data') and '__pycache__' not in f.parts
        and not f.is_relative_to(stage/'derived_protocols') and f.name!='bundle_manifest.json'}
    write(stage/'bundle_manifest.json',files)
    code=OUT/'geofwi3d_unetr_rescue_code_v1.zip'
    with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*files,'bundle_manifest.json']):z.write(stage/name,name)
    uploads={code.name:digest(code)}
    uploads.update({k:v for k,v in read(PARENT/'package.json')['uploads'].items() if k.startswith('geofwi3d_data_part')})
    nb=base.notebook(uploads,digest(stage/'bundle_manifest.json'))
    replacements={
        'filippofocaccia2@gmail.com':'focacciafilippo@gmail.com',
        'hippo_geofwi3d_baseline_seed_search':'hippo_geofwi3d_unetr_ce_pilot_20261019',
        'results_mixed_noaug_cap182178_v1':'results_unetr_rescue_f0s0_v1',
        'hippo_geofwi_seed_bundle_r2':'hippo_geofwi_unetr_rescue_v1',
        'hippo_geofwi_seed_venv':'hippo_geofwi_unetr_ce_pilot_venv',
        'geofwi3d_seed_search_code.zip':code.name,
        "BUNDLE/'runtime.py'":"BUNDLE/'rescue_runtime.py'",
        "20*1024**3":"24*1024**3",
        '20 GiB runtime':'24 GiB runtime',
    }
    for cell in nb['cells']:
        for old,new in replacements.items():cell['source']=cell['source'].replace(old,new)
    nb['cells'][0]['source']=f'''# GeoFWI3D — bounded UNETR rescue experiment

Two recipes, seed 0 / fold 0. Both use **dropout 0.1 + foreground-only squared Dice + CE (1:1)**.
Arm A peak LR **1e-4**, arm B **3e-5**. Same 92,802,274 parameters, fresh identical initialization,
19 training / 96 validation cases, native 96³, BF16, batch 1 and **no augmentation**.

Each arm first runs **{half} updates**, at most half the original cap. Select by mean validation Dice
at 30%, 40%, and 50% of budget (tie -> A). Resume that arm to **{n} total updates** with its
optimizer/RNG and full learning-rate schedule intact. The other arm is recorded as pruned.

The CPU full-model profile measured **{cost/1e9:.9f} GFLOPs/update**, including the new loss and dropout.
Maximum planned optimization: **{(half+n)*cost/1e9:,.3f} GFLOPs**, below the authorized
**{base.CAP*1.5/1e9:,.3f} GFLOP** total. GPU preflight uses six diagnostic updates in total;
validation/setup costs are outside the optimization cap and reported separately.

Success requires **final-budget validation Dice >50%**, not simply winning this race.
This is validation-guided recipe development, not completion of the three-seed/three-fold baseline.
Half-budget screening can miss a slower learner. No automatic extra seeds or follow-on experiments.

Account: **focacciafilippo@gmail.com**. Use **A100-SXM4-80GB + High RAM**, matching the control.
Run only one copy. Old pilot results remain in their separate results folder.
'''
    nb['cells'][6]['source']='''## Required GPU checks
For each learning rate: native 96³ forward/backward, finite gradients and exact model/optimizer/RNG
checkpoint replay with dropout active. Six diagnostic optimizer updates in total, recorded separately.
Initialization must match the archived seed-0 control. No research updates occur before these checks pass.
'''
    nb['cells'][8]['source']='''## Run or resume the bounded experiment
Runs A to half budget, B to half budget, records the selection, then resumes only the winner to full budget.
The timer pauses safely; rerunning reuses completed stages. Interrupt once and wait for recovery upload.
A hard VM loss with uncheckpointed charged work stops for an audit rather than silently replaying it.
All scientific completion and pruning records are separate from session pauses and technical failures.
'''
    nb['cells'][9]['source']="SESSION_HOURS=2.0\n# Create RESULTS/'STOP' to request a graceful stop.\ncampaign('run','--hours',str(SESSION_HOURS))\n"
    nb['cells'][10]['source']='''## Results
`rescue_status.json` contains both screening scores, the selected arm, charged compute and the winner's
final result. `RESCUE_COMPLETE.json` means this development experiment finished; `threshold_passed`
states whether final validation Dice is strictly above 50%. It never declares the 3×3 baseline complete.
Curves, confusion counts, source, config and logs are verified in Drive before obsolete checkpoints retire.
'''
    nb['cells'][11]['source']="""campaign('report')
import pandas as pd
status=json.loads((RESULTS/'rescue_status.json').read_text())
print('Status:',status['status'])
display(pd.DataFrame(status['screening']))
print('Selection:',status['selection'])
print('Training GFLOPs:',status['total_training_flops']/1e9)
if status['final']:
    r=status['final']
    print('Final validation Dice:',100*r['final_dice'],'%')
    print('Best validation Dice:',100*r['best_dice'],'%')
    print('Final training Dice:',100*r['training_dice'],'%')
    print('Final validation Dice >50%:',status['threshold_passed'])
else:print('Not complete; inspect status/logs before resuming.')
"""
    name='GeoFWI3D_UNETR_rescue_f0s0.ipynb'
    nb['metadata']['colab']['name']=name
    nb['metadata']['accelerator']='GPU';nb['metadata']['colab']['machine_shape']='hm'
    nb['metadata']['gpuClass']='premium'
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    write(OUT/name,nb)
    write(OUT/'package.json',dict(uploads=uploads,notebook=name,bundle_manifest_sha256=digest(stage/'bundle_manifest.json'),
        full_steps=n,half_steps=half,cost=cost,maximum_optimization_flops=(n+half)*cost))
    print(json.dumps(dict(notebook=name,full_steps=n,half_steps=half,cost=cost,
        optimization_gflops=(n+half)*cost/1e9),indent=2))


if __name__=='__main__':prepare()
