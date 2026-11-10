"""Build the isolated, manually launched UNETR loss pilot for focacciafilippo."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile

from scripts.colab_geofwi_seed_search import prepare as parent
from scripts.colab_geofwi_seed_search.pilot import INITIALIZATION

ROOT = parent.ROOT
OUT = ROOT/'reports/geofwi3d_unetr_ce_pilot_20261019'
OLD = parent.OUT/'bundle'
PROFILE = parent.CE/'protocols/GeoFWI3D_unetr_vit_b16'
AUDIT = ROOT/'reports/geofwi3d_unetr_cpu_audit_20261019/evidence/GeoFWI3D_unetr_vit_b16_baseline_f0s0'
read, write, digest = parent.read, parent.write, parent.digest


def prepare():
    stage = OUT/'bundle'
    stage.mkdir(parents=True, exist_ok=True)
    for name, sha in read(OLD/'bundle_manifest.json').items():
        if name.startswith('templates/') and not name.startswith('templates/unetr_vit_b16/'):
            continue
        src = OLD/name
        assert digest(src) == sha, name
        dst = stage/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    data = stage/'data'
    if not data.exists():
        data.symlink_to(OLD/'data', target_is_directory=True)
    for name, sha in read(stage/'data_manifest.json').items():
        assert digest(stage/name) == sha, name

    ce = read(PROFILE/'protocol.json')
    profile = read(PROFILE/'profile.json')
    p = read(stage/'templates/unetr_vit_b16/protocol.json')
    assert ce['budget_flops'] == p['budget_flops'] == parent.CAP
    assert ce['models']['unet']['parameters'] == p['models']['unet']['parameters'] == 92802274
    for key in ['initial_lrs', 'lr_horizon_updates', 'batch_size', 'spatial_size',
                'adamw_weight_decay', 'augmentation', 'precision']:
        assert ce.get(key) == p.get(key), key
    assert digest(PROFILE/'nnunet_architecture.json') == digest(stage/'templates/unetr_vit_b16/nnunet_architecture.json')
    assert digest(stage/'source/scripts/cross_dataset_campaign/msd_ce_constraints.py') == profile['shared_objective_sha256']
    assert digest(stage/'source/scripts/cross_dataset_campaign/binary_ce_constraints.py') == profile['objective_sha256']
    schedule = deepcopy(ce['models']['unet']['objectives']['dice'])
    cost = profile['costs']['dice']['update_flops']
    assert profile['status'] == 'profiled' and profile['loss_recipe'] == 'dice_ce'
    checks = [c for c in profile['full_model_checks'] if c['arm'] == 'dice' and not c['calibration']]
    assert {c['geometry'] for c in checks} == {'ellipsoid', 'box'}
    assert all(c['flops'] == cost for c in checks)
    assert schedule['max_steps'] == parent.CAP//cost == 414
    assert schedule['consumed_training_flops'] == 414*cost
    p['models']['unet']['objectives'] = {'dice': schedule}
    p.update(name='geofwi3d_unetr_ce_pilot_20261019', weight_ce=1,
             training_loss='Original unsquared, background-inclusive Dice + voxel cross-entropy; coefficients 1:1',
             profile_sha256=digest(PROFILE/'profile.json'),
             pilot_initialization_sha256=INITIALIZATION, folds=[0], seeds=[0],
             profile_parent_protocol_sha256=digest(PROFILE/'protocol.json'))
    write(stage/'templates/unetr_vit_b16/protocol.json', p)
    shutil.copyfile(PROFILE/'profile.json', stage/'templates/unetr_vit_b16/profile_ce_full.json')
    write(stage/'templates/unetr_vit_b16/profile_baseline.json', dict(
        parent_profile_sha256=digest(PROFILE/'profile.json'), parameters=92802274,
        baseline_schedule=schedule, cost=profile['costs']['dice'], accounting=profile['accounting']))

    # Keep the frozen trainer unchanged; fail before an update if the paired init differs.
    worker = stage/'worker.py'
    old = "        record=dict(model_state_sha256=io.model_digest(model))\n"
    new = old + "        if record['model_state_sha256'] != p['pilot_initialization_sha256']:\n            raise ValueError('Initialization differs from the archived seed-0 Dice control')\n"
    content = worker.read_text()
    assert content.count(old) == 1
    worker.write_text(content.replace(old, new))
    shutil.copyfile(ROOT/'scripts/colab_geofwi_seed_search/pilot.py', stage/'pilot.py')
    done = read(AUDIT/'completion.json')
    control = dict(model='unetr_vit_b16', fold=0, seed=0, loss='Dice',
                   final_dice=read(AUDIT/'validation_latest.json')['summary']['macro_hard_dice'],
                   best_dice=done['best_validation_hard_dice'],
                   training_dice=done['final_training_hard_dice'], updates=done['updates'],
                   initialization_sha256=INITIALIZATION,
                   source='Archived completed Dice-only seed 0 fold 0; no control retraining')
    write(stage/'dice_control.json', control)
    # Read from the archived search environment_binding.json on 19 October 2026.
    # Drive file 18z3RQm6ss2N8_FsM-3J8RnvDSoSP6knw, second-account search results.
    write(stage/'control_environment.json', dict(packages={
        'torch':'2.12.1', 'monai':'1.6.0', 'dynamic-network-architectures':'0.4.4',
        'numpy':'2.4.6', 'pandas':'3.0.3', 'scipy':'1.17.1',
        'scikit-image':'0.26.0', 'nibabel':'5.4.2', 'einops':'0.8.2'},
        gpu='NVIDIA A100-SXM4-80GB', cuda='13.0', cudnn=92000))
    for name in ['completion.json', 'config.json', 'protocol.json', 'validation_latest.json', 'metrics.csv']:
        target = stage/'control_evidence'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(AUDIT/name, target)
    plan = dict(dataset='GeoFWI3D', models=['unetr_vit_b16'], folds=[0], seeds=[0],
                augmentation='none', threshold=.5, budget_flops=parent.CAP,
                updates={'unetr_vit_b16':414}, scope='One paired pilot, no automatic seed search',
                primary_metric='Mean per-volume foreground hard Dice at final budget',
                intervention='Add CE with coefficient 1 to unchanged Dice; all other training settings fixed',
                initialization_sha256=INITIALIZATION,
                interpretation='Recipe development on validation; not an independent test or three-seed baseline')
    write(stage/'plan.json', plan)
    provenance = read(stage/'source_provenance.json')
    provenance.update(pilot_parent_bundle_sha256=digest(OLD/'bundle_manifest.json'),
                      ce_profile=str(PROFILE.relative_to(ROOT)),
                      ce_profile_sha256=digest(PROFILE/'profile.json'),
                      pilot_wrapper='scripts/colab_geofwi_seed_search/pilot.py',
                      worker_change='Fail before training if initialization differs from archived Dice control')
    write(stage/'source_provenance.json', provenance)
    files = {str(f.relative_to(stage)):digest(f) for f in stage.rglob('*')
             if f.is_file() and not f.is_relative_to(stage/'data') and '__pycache__' not in f.parts
             and f.name != 'bundle_manifest.json' and not f.is_relative_to(stage/'derived_protocols')}
    write(stage/'bundle_manifest.json', files)
    code = OUT/'geofwi3d_unetr_ce_pilot_code.zip'
    with zipfile.ZipFile(code, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*files, 'bundle_manifest.json']):
            z.write(stage/name, name)
    uploads = {code.name:digest(code)}
    uploads.update({k:v for k,v in read(parent.OUT/'package.json')['uploads'].items() if k.startswith('geofwi3d_data_part')})
    nb = parent.notebook(uploads, digest(stage/'bundle_manifest.json'))
    replacements = {
        'filippofocaccia2@gmail.com':'focacciafilippo@gmail.com',
        'hippo_geofwi3d_baseline_seed_search':'hippo_geofwi3d_unetr_ce_pilot_20261019',
        'results_mixed_noaug_cap182178_v1':'results_unetr_dice_ce_f0s0_v1',
        'hippo_geofwi_seed_bundle_r2':'hippo_geofwi_unetr_ce_pilot_v1',
        'hippo_geofwi_seed_venv':'hippo_geofwi_unetr_ce_pilot_venv',
        'geofwi3d_seed_search_code.zip':code.name,
        "BUNDLE/'runtime.py'":"BUNDLE/'pilot.py'",
    }
    for cell in nb['cells']:
        for old, new in replacements.items():
            cell['source'] = cell['source'].replace(old, new)
    nb['cells'][0]['source'] = '''# GeoFWI3D — UNETR Dice+CE paired pilot

**One run: seed 0, fold 0.** Test whether adding voxel cross-entropy improves UNETR at the existing cap.
The archived Dice-only control reached **29.12% final validation Dice** (33.04% best).
This is a proposed remedy, not a confirmed fix. No automatic seed sweep follows this pilot.

- Only loss changes: original unsquared Dice including background **+ CE**, weights 1:1.
- Same initialization hash, 19 training / 96 validation volumes, native 96³, batch 1, no augmentation.
- Same AdamW, learning-rate trajectory and BF16 training. No pretrained weights or constraints.
- **182,178.12654316 GFLOPs cap; 414 updates**, with CE included in the accounting.
- Final-budget validation Dice is primary; best validation is secondary. A final score >50% passes
  this single cell only. Three passing seeds across three folds still require separate validation.
- Preflight uses 3 diagnostic updates (~1,318.42 GFLOPs), outside the optimization cap.
  Validation/setup also incur compute. No control rerun is needed.

Sign into **focacciafilippo@gmail.com**, select **A100 GPU + High RAM**, then **Run all**.
The paired comparison requires the control's A100-SXM4-80GB and pinned software; preflight stops if they differ.
Inputs are in `MyDrive/hippo_geofwi3d_unetr_ce_pilot_20261019/input`.
Run one copy only. The notebook verifies initialization before any training update and stops if it differs.
All three folds' existing inputs are included for later reuse; this notebook trains only fold 0.
'''
    nb['cells'][6]['source'] = '''## Required UNETR GPU preflight
Checks one model only: finite loss/gradients, actual 96³ BF16 forward/backward, and exact checkpoint
replay of model, optimizer and RNG. Its 3 diagnostic updates are recorded separately.
The scientific run starts fresh from the matching archived seed-0 initialization.
'''
    nb['cells'][8]['source'] = '''## Run or resume the single pilot
The run stops after 414 updates and final evaluation, regardless of its score. It cannot start another
seed/model/fold. Rerunning reuses a verified completed archive or resumes coherent recovery.
Interrupt once and wait for recovery to finish uploading. A hard VM loss can leave charged updates
newer than the checkpoint: the notebook then stops for an audit instead of silently exceeding the cap.
'''
    nb['cells'][9]['source'] = "SESSION_HOURS=2.0\n# RESULTS/'STOP' requests a graceful stop; remove only when intentionally resuming.\ncampaign('run','--hours',str(SESSION_HOURS))\n"
    nb['cells'][10]['source'] = '''## Compare with the archived Dice-only control
`pilot_status.json` reports the paired difference in percentage points and whether final Dice exceeds
50%. `PILOT_COMPLETE.json` means this one experiment finished, including when the score is below 50%.
It never declares the three-seed/three-fold baseline complete. Curves, confusion counts, config, source,
budget ledger and logs are archived and verified before completed checkpoints are retired.
'''
    nb['cells'][11]['source'] = """campaign('report')
import pandas as pd
status=json.loads((RESULTS/'pilot_status.json').read_text())
print('Pilot status:',status['status'])
rows=[dict(recipe='Archived Dice control',**status['dice_control'])]
rows += [dict(recipe='Dice+CE pilot',**r) for r in status['runs']]
table=pd.DataFrame(rows)
display(table[['recipe','fold','seed','updates','final_dice','best_dice','training_dice']])
if status['runs']:
    print('Final Dice change (percentage points):',status['final_dice_delta_percentage_points'])
    print('Final Dice strictly >50%:',status['threshold_passed'])
    print('Stop here and review before spending more compute.')
else:print('Pilot has not completed. Resume if paused; inspect logs if it failed.')
"""
    name = 'GeoFWI3D_UNETR_DiceCE_pilot_f0s0.ipynb'
    nb['metadata']['colab']['name'] = name
    write(OUT/name, nb)
    write(OUT/'package.json', dict(uploads=uploads, notebook=name,
        bundle_manifest_sha256=digest(stage/'bundle_manifest.json'), plan=plan,
        target_account='focacciafilippo@gmail.com'))
    print(json.dumps(dict(notebook=str(OUT/name), code_bytes=code.stat().st_size,
                         data_parts=len(uploads)-1, updates=414), indent=2))


if __name__ == '__main__':
    prepare()
