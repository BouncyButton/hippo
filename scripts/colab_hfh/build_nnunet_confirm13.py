"""Freeze search winners for the other eight established fold/seed pairs."""
import io
from pathlib import Path
import shutil
import zipfile
from scripts.colab_hfh.runtime import read, write, digest
from scripts.colab_hfh.constraint_schedule import training_flops

NAME = 'nnunet_confirm13_shared_a100_v1'
CAP = 31976450667090
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'
PREVIOUS = REPORT/'nnunet_search13_shared_a100_f0s0_v1'
WORK = REPORT/NAME
PAIRS = [(f,s) for f in range(3) for s in range(3) if (f,s)!=(0,0)]


def build():
    verification = read(PREVIOUS/'verification.json')
    assert verification['status']=='verified_complete'
    assert digest(PREVIOUS/'all_results.zip')==verification['all_results_sha256']
    leaders = read(PREVIOUS/'leaders.json')
    package = WORK/'package'
    package.mkdir(parents=True,exist_ok=True)
    # Copy frozen scientific source, not the possibly changed current checkout.
    shutil.copytree(PREVIOUS/'package/source',package/'source',dirs_exist_ok=True)
    for name in ('nnunet_confirm13_campaign.py','test_nnunet_confirm13.py','build_nnunet_confirm13.py'):
        shutil.copyfile(ROOT/'scripts/colab_hfh'/name,package/'source/scripts/colab_hfh'/name)
    selected = {entry['arm']:entry['best']['recipe'] for entry in leaders}
    write(package/'selected_recipes.json',selected)
    write(package/'selection_results.json',leaders)
    rows=[]
    with zipfile.ZipFile(PREVIOUS/'all_results.zip') as bundle:
        for variant in ['native']+[r['id'] for r in selected.values()]:
            # Preserve the selection-pair evidence, separately from confirmation.
            target=package/'selection_archives'/(variant+'.zip')
            target.parent.mkdir(exist_ok=True);target.write_bytes(bundle.read(variant+'.zip'))
    for fold,seed in PAIRS:
        for variant,arm in [('native','dice')]+[(r['id'],a) for a,r in selected.items()]:
            name=f'{variant}__f{fold}_s{seed}'
            source=PREVIOUS/'package/protocols'/variant
            target=package/'protocols'/name
            shutil.copytree(source,target,dirs_exist_ok=True)
            p=read(target/'protocol.json')
            p.update(name=NAME+'__'+name,folds=[fold],seeds=[seed],hardware_campaign=NAME,
                confirmation=dict(selection_fold=0,selection_seed=0,frozen_variant=variant,
                    selection_bundle_sha256=verification['all_results_sha256'],retuning=False),
                endpoint='Final shared-cap endpoint; frozen recipe confirmation; no pruning or reselection')
            if arm!='dice':
                p['a100_native_anchor']=str(Path('/content/hippo_'+NAME)/'runs'/f'native__f{fold}_s{seed}'/'archive'/f"native_update{p['prefix_anchor_updates']}.json")
            write(target/'protocol.json',p)
            schedule=p['models']['nnunet']['objectives'][arm]
            assert training_flops(schedule['max_steps'],schedule)<=CAP<training_flops(schedule['max_steps']+1,schedule)
            rows.append(dict(name=name,variant=variant,arm=arm,fold=fold,seed=seed,
                native_name=f'native__f{fold}_s{seed}',updates=schedule['max_steps'],
                optimization_flops=training_flops(schedule['max_steps'],schedule)))
    rows.sort(key=lambda r:(r['arm']!='dice',r['fold'],r['seed'],r['name']))
    design=dict(name=NAME,cap=CAP,pairs=PAIRS,runs=rows,total_runs=len(rows),
        selected_recipes=selected,selection_bundle_sha256=verification['all_results_sha256'],
        selection_leaders_sha256=digest(PREVIOUS/'leaders.json'),max_workers=12,archive_workers=2,
        max_pending_archives=12,threads_per_worker=4,
        worker_module='scripts.colab_hfh.nnunet_search13_shared_worker',
        primary_summary='Six pairs on folds1/2; fold0 seeds1/2 reported separately; selection pair excluded from confirmation mean',
        calibration='Same frozen training-only calibration rule separately at each matched native prefix; no validation retuning',
        accounting='Original shared optimization-only cap; full prefixes and activity probes charged; calibration/diagnostics separate',
        no_automatic_retries=True)
    write(package/'confirmation_design.json',design)
    manifest={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(package/'manifest.json',manifest)
    archive=WORK/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'manifest.json']:z.write(package/name,name)
    write(WORK/'build_receipt.json',dict(status='prepared',package_sha256=digest(archive),bytes=archive.stat().st_size,
        files=len(manifest),runs=len(rows),cap=CAP,pairs=PAIRS))
    print(read(WORK/'build_receipt.json'))


if __name__=='__main__':build()
