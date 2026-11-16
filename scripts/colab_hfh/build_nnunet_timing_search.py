"""Prepare immutable native replay and all feasible per-seed timing protocols."""
import io
from pathlib import Path
import shutil
from statistics import mean
import zipfile
from scripts.colab_hfh.runtime import read,write,digest
from scripts.colab_hfh.nnunet_timing_policy import NAME,CAP,set_prefix,candidate_prefixes
from scripts.colab_hfh.constraint_schedule import training_flops

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/hfh_colab_exploration_20261111'
SELECTION=REPORT/'nnunet_search13_shared_a100_f0s0_v1'
CONFIRM=REPORT/'nnunet_confirm13_shared_a100_v1'
WORK=REPORT/NAME


def build():
    assert read(CONFIRM/'independent_verification.json')['status']=='independently_verified'
    assert digest(CONFIRM/'all_results.zip')==read(CONFIRM/'independent_verification.json')['bundle_sha256']
    package=WORK/'package';package.mkdir(parents=True,exist_ok=True)
    shutil.copytree(CONFIRM/'package/source',package/'source',dirs_exist_ok=True)
    for name in ('nnunet_timing_policy.py','nnunet_timing_worker.py','nnunet_timing_campaign.py','build_nnunet_timing_search.py','test_nnunet_timing.py'):
        shutil.copyfile(ROOT/'scripts/colab_hfh'/name,package/'source/scripts/colab_hfh'/name)
    selected=read(CONFIRM/'package/selected_recipes.json');oldrows=[]
    receipts={}
    for home,origin in [(SELECTION,'selection'),(CONFIRM,'confirmation')]:
        with zipfile.ZipFile(home/'all_results.zip') as z:
            receipt=__import__('json').loads(z.read('verification.json'));state=__import__('json').loads(z.read('status.json'))
            receipts[origin]=dict(bundle_sha256=digest(home/'all_results.zip'))
            for row in state['results']:
                if row['arm']!='dice' and row['variant']!=selected[row['arm']]['id']:continue
                name=f"{row['variant']}__f{row['fold']}_s{row['seed']}"
                source_name=row.get('name',row['variant']);data=z.read(source_name+'.zip')
                assert __import__('hashlib').sha256(data).hexdigest()==receipt['archives_sha256'][source_name]
                dest=package/'reference_archives'/(name+'.zip');dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
                oldrows.append(dict(row,name=name,reference_archive=name+'.zip',source_campaign=origin))
    assert len(oldrows)==126
    write(package/'reference_results.json',oldrows)
    groups={};specs={};native=[]
    for fold in range(3):
        for seed in range(3):
            name=f'native__f{fold}_s{seed}'
            target=package/'protocols'/name;shutil.copytree(SELECTION/'package/protocols/native',target,dirs_exist_ok=True)
            p=read(target/'protocol.json');p.update(name=NAME+'__'+name,folds=[fold],seeds=[seed],hardware_campaign=NAME)
            write(target/'protocol.json',p)
            spec=dict(name=name,variant='native',arm='dice',fold=fold,seed=seed,native_name=name,updates=90)
            specs[name]=spec;native.append(spec)
    for arm,recipe in selected.items():
        source=SELECTION/'package/protocols'/recipe['id'];original=read(source/'protocol.json')
        for seed in range(3):
            key=arm+'__s'+str(seed);rows=sorted([r for r in oldrows if r['arm']==arm and r['seed']==seed],key=lambda r:r['fold'])
            gain=mean(r['native_gain_pp'] for r in rows);positive=gain>0
            prefixes=[] if positive else candidate_prefixes(original)
            groups[key]=dict(arm=arm,seed=seed,status='already_positive' if positive else 'searching',
                remaining_prefixes=prefixes,attempts=[dict(prefix=recipe['prefix'],results=rows,mean_gain_pp=gain,chosen=positive,origin='prior_completed_campaign')])
            for prefix in prefixes:
                for fold in range(3):
                    name=f'{arm}__p{prefix:02d}__f{fold}_s{seed}';target=package/'protocols'/name
                    shutil.copytree(source,target,dirs_exist_ok=True)
                    p=set_prefix(original,prefix)
                    p.update(name=NAME+'__'+name,folds=[fold],seeds=[seed],hardware_campaign=NAME,
                        a100_native_anchor=str(Path('/content/hippo_'+NAME)/'runs'/f'native__f{fold}_s{seed}'/'archive'/f'native_update{prefix}.json'),
                        endpoint='Final shared-cap endpoint; seed-specific activation tuning using all three folds; exploratory selection')
                    write(target/'protocol.json',p);s=p['models']['nnunet']['objectives'][arm]
                    assert training_flops(s['max_steps'],s)<=CAP<training_flops(s['max_steps']+1,s)
                    specs[name]=dict(name=name,variant=recipe['id'],arm=arm,fold=fold,seed=seed,native_name=f'native__f{fold}_s{seed}',updates=s['max_steps'],prefix=prefix,group=key)
    design=dict(name=NAME,cap=CAP,groups=groups,specs=specs,native_runs=native,original_recipes=selected,
        prior_receipts=receipts,reference_runs=126,initial_positive_groups=sum(g['status']=='already_positive' for g in groups.values()),
        failing_groups=sum(g['status']=='searching' for g in groups.values()),maximum_new_runs=len(specs),max_workers=12,
        archive_workers=2,max_pending_archives=12,threads_per_worker=4,minimum_full_strength_updates=15,
        stop='First strictly positive mean gain across all three folds for each combination-seed; otherwise exhaust finite feasible integer prefixes',
        no_individual_fold_pruning=True,no_seed_exclusion=True,no_automatic_retries=True,
        worker_module='scripts.colab_hfh.nnunet_timing_worker',
        interpretation='Exploratory validation-guided per-seed tuning; not independent confirmation; no guarantee all individual folds improve')
    write(package/'timing_design.json',design)
    manifest={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(package/'manifest.json',manifest)
    archive=WORK/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'manifest.json']:z.write(package/name,name)
    receipt=dict(status='prepared',sha256=digest(archive),bytes=archive.stat().st_size,groups=39,initial_positive=design['initial_positive_groups'],failing_groups=design['failing_groups'],maximum_new_runs=len(specs),cap=CAP)
    write(WORK/'build_receipt.json',receipt);print(receipt)


if __name__=='__main__':build()
