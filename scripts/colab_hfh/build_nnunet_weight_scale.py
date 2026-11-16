"""Freeze 351 strength-only trials around the 39 successful timing recipes."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
from scripts.colab_hfh.runtime import read,write,digest
from scripts.colab_hfh.nnunet_weight_scale_policy import NAME,CAP,MULTIPLIERS,scale_protocol
from scripts.colab_hfh.constraint_schedule import training_flops

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/hfh_colab_exploration_20261111'
PRIOR=REPORT/'nnunet_timing_seed_search_a100_v1'
WORK=REPORT/NAME


def build():
    audit=read(PRIOR/'independent_verification.json')
    assert audit['all_39_positive'] and digest(PRIOR/'all_results.zip')==audit['bundle_sha256']
    selected=read(PRIOR/'summary.json');assert len(selected)==39
    package=WORK/'package';package.mkdir(parents=True,exist_ok=True)
    shutil.copytree(PRIOR/'package/source',package/'source',dirs_exist_ok=True)
    for name in ('nnunet_weight_scale_policy.py','nnunet_weight_scale_campaign.py','build_nnunet_weight_scale.py','test_nnunet_weight_scale.py'):
        shutil.copyfile(ROOT/'scripts/colab_hfh'/name,package/'source/scripts/colab_hfh'/name)
    groups={};runs=[];refs=[];manifest_refs={}
    with zipfile.ZipFile(PRIOR/'all_results.zip') as bundle:
        receipt=json.loads(bundle.read('verification.json'))
        with zipfile.ZipFile(io.BytesIO(bundle.read('reference_archives.zip'))) as old:
            for g in selected:
                assert g['mean_gain_pp']>0
                key=g['key'];groups[key]=dict(arm=g['arm'],seed=g['seed'],prefix=g['selected_prefix'],reference_results=[])
                for row in g['fold_results']:
                    reference_name=row['name'];fold,seed=row['fold'],row['seed']
                    origin='timing' if reference_name in receipt['archives_sha256'] else 'reference'
                    data=bundle.read(reference_name+'.zip') if origin=='timing' else old.read(row['reference_archive'])
                    expected=hashlib.sha256(data).hexdigest()
                    if origin=='timing':assert expected==receipt['archives_sha256'][reference_name]
                    ref=dict(row,group=key,weight_multiplier=1.,reference_archive=reference_name+'.zip',reference_origin=origin,reference_archive_sha256=expected)
                    refs.append(ref);groups[key]['reference_results'].append(ref);manifest_refs[reference_name+'.zip']=expected
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        original=json.loads(z.read('protocol/protocol.json'))
                        assert original['prefix_anchor_updates']==g['selected_prefix']
                        refdir=package/'references'/reference_name;refdir.mkdir(parents=True,exist_ok=True)
                        for name in ('protocol.json',):write(refdir/name,original)
                        (refdir/'calibration.json').write_bytes(z.read('results/calibration.json'))
                        for multiplier in MULTIPLIERS:
                            name=f"{g['arm']}__w{int(multiplier*100)}__f{fold}_s{seed}"
                            target=package/'protocols'/name;target.mkdir(parents=True,exist_ok=True)
                            for item in z.namelist():
                                if item.startswith('protocol/'):
                                    suffix=Path(item).relative_to('protocol');assert '..' not in suffix.parts
                                    dest=target/suffix;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(item))
                            p=scale_protocol(original,multiplier)
                            p.update(name=NAME+'__'+name,folds=[fold],seeds=[seed],hardware_campaign=NAME,
                                weight_reference_name=reference_name,
                                a100_native_anchor=str(Path('/content/hippo_'+NAME)/'runs'/f'native__f{fold}_s{seed}'/'archive'/f"native_update{g['selected_prefix']}.json"),
                                endpoint='Final shared-cap checkpoint; weights-only grid; fixed successful activation/ramp; three-fold exploratory selection')
                            assert p['models']==original['models'] and p['activity_policy']==original['activity_policy']
                            schedule=p['models']['nnunet']['objectives'][g['arm']]
                            assert training_flops(schedule['max_steps'],schedule)<=CAP<training_flops(schedule['max_steps']+1,schedule)
                            write(target/'protocol.json',p)
                            runs.append(dict(name=name,variant=name,arm=g['arm'],fold=fold,seed=seed,group=key,
                                reference_name=reference_name,native_name=f'native__f{fold}_s{seed}',updates=schedule['max_steps'],
                                prefix=g['selected_prefix'],weight_multiplier=multiplier,optimization_flops=training_flops(schedule['max_steps'],schedule)))
            natives=[]
            state=json.loads(bundle.read('status.json'))
            for row in [r for r in state['results'] if r['arm']=='dice']:
                name=row['name'];data=bundle.read(name+'.zip');assert hashlib.sha256(data).hexdigest()==receipt['archives_sha256'][name]
                natives.append(row);manifest_refs[name+'.zip']=receipt['archives_sha256'][name]
    runs.sort(key=lambda r:(r['weight_multiplier'],-next(g['mean_gain_pp'] for g in selected if g['key']==r['group']),r['group'],r['fold']))
    design=dict(name=NAME,cap=CAP,runs=runs,total_runs=len(runs),groups=groups,multipliers=list(MULTIPLIERS),
        native_results=natives,reference_results=refs,reference_archive_sha256=manifest_refs,
        prior_bundle_sha256=audit['bundle_sha256'],prior_execution_source_sha256=read(PRIOR/'launch_provenance.json')['execution_source_zip_sha256'],
        pairs=[(f,s) for f in range(3) for s in range(3)],max_workers=12,archive_workers=2,max_pending_archives=12,threads_per_worker=4,
        worker_module='scripts.colab_hfh.nnunet_search13_shared_worker',
        strength='Multiply all calibrated anchor coefficients uniformly; preserve relative component weights; unchanged per-update activity envelope may limit effective increase',
        selection='Best mean Dice across all three folds per combination-seed, including existing1x; all351 trials completed, no outcome pruning',
        accounting='Unchanged31.976T optimization cap and schedule per run; calibration and final diagnostics separately accounted',
        interpretation='Exploratory validation-guided strength selection; no independent test claim; original1x remains eligible',no_automatic_retries=True)
    assert len(runs)==351 and len(refs)==117 and len(natives)==9
    write(package/'weight_design.json',design)
    manifest={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(package/'manifest.json',manifest)
    archive=WORK/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'manifest.json']:z.write(package/name,name)
    receipt=dict(status='prepared',package_sha256=digest(archive),bytes=archive.stat().st_size,new_runs=351,reused_controls=9,reused_1x_runs=117,multipliers=list(MULTIPLIERS),cap=CAP)
    write(WORK/'build_receipt.json',receipt);print(receipt)


if __name__=='__main__':build()
