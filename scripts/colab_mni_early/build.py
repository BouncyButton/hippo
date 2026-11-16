"""Build the isolated MNI seed0/fold0 timing/strength pilot from frozen evidence."""
import os
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = 'mni_nnunet_early_f0s0_20261113'
OUT = ROOT / 'reports' / NAME
FROZEN = ROOT / 'reports/mni_winning_recipes_matrix2030_53t_0519_20261109/frozen'
CAP = 53305109592450


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def consumed(n, s):
    a, b = s['prefix_updates'], s['sdf_prefix_updates']
    return min(n, a)*s['baseline_flops_per_update'] + max(0, min(n, b)-a)*s['early_flops_per_update'] + max(0, n-b)*s['flops_per_update']


def main():
    bundle = OUT / 'bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FROZEN/'source', bundle/'source', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(FROZEN/'data', bundle/'data', dirs_exist_ok=True)
    tasks = []
    variants = [('reference',20,83,.6), ('earlier_non_sdf',10,83,.6),
                ('sync20_t60',20,20,.6), ('sync10_t60',10,10,.6),
                ('sync20_t10',20,20,.1), ('sync10_t10',10,10,.1),
                ('sync20_t05',20,20,.05), ('sync10_t05',10,10,.05)]
    for arm in ('dice', 'dice_sdf_depth', 'dice_bands_edge_sdf_depth'):
        origin = FROZEN/'protocols'/('nnunet_'+arm)
        for name, prefix, sdf, target in ([('native',0,0,.6)] if arm=='dice' else variants):
            identity = arm+'__'+name
            folder = bundle/'protocols'/identity
            shutil.copytree(origin, folder, dirs_exist_ok=True)
            p = json.loads((folder/'protocol.json').read_text())
            s = p['models']['nnunet']['objectives'][arm]
            s.update(prefix_updates=prefix, sdf_prefix_updates=sdf)
            maximum = 0
            while consumed(maximum+1,s)<=CAP:
                maximum += 1
            used = consumed(maximum,s)
            s.update(max_steps=maximum, consumed_training_flops=used, consumed_flops=used,
                     consumed_total_flops=used+s['calibration_flops'],unused_flops=CAP-used,
                     patient_pass_epochs=maximum/p['train_count'])
            p.update(name=NAME, seeds=[0], folds=[0], calibration_target_ratio=target,
                     data='/content/'+NAME+'/data/MNI.pkl',
                     parent_protocol_sha256=digest(origin/'protocol.json'),
                     timing_policy=f'Non-SDF prefix {prefix}; SDF prefix {sdf}; original 20-update ramps.',
                     calibration_strength_policy='Training-only original logit-gradient calibration; target varied, p95 guard unchanged.',
                     pilot_variant=name)
            assert p['initial_lrs']=={'nnunet':.02} and p['lr_horizon_updates']==200
            assert p['budget_flops']==CAP and p['augmentation']=='none'
            assert p['calibration_p95_cap']==.75
            assert used<=CAP<consumed(maximum+1,s)
            assert p['pkl_sha256']==digest(bundle/'data/MNI.pkl')
            for file,key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
                assert digest(folder/file)==p[key]
            write(folder/'protocol.json',p)
            tasks.append(dict(id=identity,arm=arm,variant=name,prefix=prefix,sdf_prefix=sdf,target=target,
                              fold=0,seed=0,model='nnunet',kind='conventional',dataset='MNI'))
    # Start with baseline and the heaviest constrained reference to measure peak usage.
    tasks.sort(key=lambda t: (0 if t['arm']=='dice' else 1 if t['id']=='dice_bands_edge_sdf_depth__reference' else 2,t['id']))
    write(bundle/'plan.json',dict(name=NAME,cap=CAP,tasks=tasks,seed=0,fold=0,
        selection='Final-budget validation macro Dice; exploratory single-pair evidence only.',
        source='Frozen completed MNI sweep; original losses, definitions, data and LR preserved.'))
    for filename in ('runner.py','setup_cell.py'):
        shutil.copy2(HERE/filename,bundle/filename)
    storage=(ROOT/'scripts/colab_hfh/medsam_constraints_drive.py').read_text()
    storage=storage.replace("os.environ.get('HIPPO_HFH_DRIVE_ROOT', '')", "os.environ['MNI_DRIVE_ROOT']")
    storage=storage.replace('hfh_storage:', 'mni_storage:')
    (bundle/'storage_sdk.py').write_text(storage)
    shutil.copy2(ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015/bundle/requirements-colab.txt',bundle/'requirements-colab.txt')
    for path in bundle.rglob('*.py'): ast.parse(path.read_text())
    write(bundle/'manifest.json',{str(p.relative_to(bundle)):digest(p) for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    archive=OUT/(NAME+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): z.write(path,str(path.relative_to(bundle)))
    print(json.dumps(dict(package=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,runs=len(tasks))))


if __name__=='__main__': main()
