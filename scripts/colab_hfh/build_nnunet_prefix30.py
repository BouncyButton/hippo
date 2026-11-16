"""Freeze a timing-only update30 ablation over the established nine pairs."""
from copy import deepcopy
from pathlib import Path
import shutil
import zipfile
from scripts.colab_hfh.runtime import read,write,digest
from scripts.colab_hfh.nnunet_search13_shared_policy import CAP,recost
from scripts.colab_hfh.constraint_schedule import training_flops

NAME='nnunet_prefix30_shared_a100_v1'
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/hfh_colab_exploration_20261111'
SELECTION=REPORT/'nnunet_search13_shared_a100_f0s0_v1'
PREVIOUS=REPORT/'nnunet_confirm13_shared_a100_v1'
WORK=REPORT/NAME


def change_timing(original):
    p=deepcopy(original);arm=p['active_objective'];s=p['models']['nnunet']['objectives'][arm]
    p['prefix_anchor_updates']=30;p['search_recipe']['prefix']=30
    s['prefix_updates']=s['sdf_prefix_updates']=30
    recost(s)
    assert s['max_steps']-30-p['constraint_warmup_updates']+1>=15
    return p


def build():
    package=WORK/'package';package.mkdir(parents=True,exist_ok=True)
    shutil.copytree(PREVIOUS/'package/source',package/'source',dirs_exist_ok=True)
    for name in ('nnunet_prefix30_campaign.py','build_nnunet_prefix30.py','test_nnunet_prefix30.py'):
        shutil.copyfile(ROOT/'scripts/colab_hfh'/name,package/'source/scripts/colab_hfh'/name)
    original=read(PREVIOUS/'package/selected_recipes.json')
    selected=deepcopy(original)
    for recipe in selected.values():recipe['prefix']=max(30,recipe['prefix'])
    changed=[arm for arm,r in original.items() if r['prefix']<30]
    assert len(changed)==7
    runs=[];reused=[]
    for fold in range(3):
        for seed in range(3):
            for arm,variant in [('dice','native')]+[(a,r['id']) for a,r in original.items()]:
                name=f'{variant}__f{fold}_s{seed}'
                source=SELECTION/'package/protocols'/variant
                old=read(source/'protocol.json')
                spec=dict(name=name,variant=variant,arm=arm,fold=fold,seed=seed,native_name=f'native__f{fold}_s{seed}')
                if arm not in changed:
                    spec.update(source_campaign='selection' if (fold,seed)==(0,0) else 'confirmation',source_name=variant if (fold,seed)==(0,0) else name)
                    reused.append(spec);continue
                target=package/'protocols'/name
                shutil.copytree(source,target,dirs_exist_ok=True)
                p=change_timing(old)
                p.update(name=NAME+'__'+name,folds=[fold],seeds=[seed],hardware_campaign=NAME,
                    a100_native_anchor=str(Path('/content/hippo_'+NAME)/'runs'/spec['native_name']/'archive/native_update30.json'),
                    confirmation=dict(retuning=False,ablation='Only activation prefix changed; selected ramp and targets frozen',original_prefix=original[arm]['prefix']),
                    endpoint='Final shared-cap endpoint; fixed update30 timing ablation; no pruning or reselection')
                s=p['models']['nnunet']['objectives'][arm]
                assert training_flops(s['max_steps'],s)<=CAP<training_flops(s['max_steps']+1,s)
                write(target/'protocol.json',p)
                runs.append(dict(**spec,updates=s['max_steps'],optimization_flops=training_flops(s['max_steps'],s)))
    design=dict(name=NAME,cap=CAP,pairs=[(f,s) for f in range(3) for s in range(3)],runs=runs,reused=reused,
        total_runs=63,reused_runs=63,full_matrix_runs=126,selected_recipes=selected,original_recipes=original,
        max_workers=12,archive_workers=2,max_pending_archives=12,threads_per_worker=4,
        worker_module='scripts.colab_hfh.nnunet_search13_shared_worker',
        activation='Previously update20 recipes move to 30 completed native updates; first auxiliary loss on update31. Existing30 and45 recipes unchanged',
        comparison='Paired timing ablation; same native, targets, ramp, activity policy and LR schedule; calibration recomputed on training data at update30',
        interpretation='Exploratory follow-up motivated by observed confirmation variability; not an independent untouched test set',
        accounting='31.976T optimization cap per run; full prefix and activity probes charged; calibration/diagnostics separately reported',
        no_automatic_retries=True)
    write(package/'confirmation_design.json',design);write(package/'selected_recipes.json',selected)
    manifest={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(package/'manifest.json',manifest)
    archive=WORK/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'manifest.json']:z.write(package/name,name)
    write(WORK/'build_receipt.json',dict(status='prepared',package_sha256=digest(archive),bytes=archive.stat().st_size,new_runs=63,reused_runs=63,cap=CAP))
    print(read(WORK/'build_receipt.json'))


if __name__=='__main__':build()
