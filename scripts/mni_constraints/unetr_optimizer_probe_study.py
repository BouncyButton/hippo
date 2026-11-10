"""Replay one helped and one harmed UNETR pair with observational Adam probes."""
from copy import deepcopy
import difflib
import shutil
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign,headroom_bands
from scripts.mni_constraints.unetr_delayed_confirm import ROOT,CAP,NAME as SOURCE
from scripts.mni_constraints.unetr_optimizer_diagnostics import instrument_source
from scripts.unetr_subfields import cap_extension as batch

NAME='mni_unetr_adam_probe53t_20261108'
MODULE='scripts.mni_constraints.unetr_optimizer_probe_study'
PARENT='mni_unetr_readiness53t_20261108'
CORE='source/scripts/cross_dataset_campaign/capped_core.py'


def tasks():
    return [dict(id=f'MNI_unetr_probe_bands_f{fold}s0',kind='conventional',dataset='MNI',
                 model='unetr_vit_b16',arm='dice_bands',protocol='delayed_t05',seed=0,fold=fold,
                 augmentation='xy_d4_v1',reference=f'f{fold}s0') for fold in (0,1)]


def protocol(old):
    p=deepcopy(old)
    p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),seeds=[0],folds=[0,1],
        decision_rule='Observational replay of original delayed5% bands on seed0 folds0 and1, previously helped/harmed. Require identical reference model hashes, calibration, and complete metric trajectories before interpreting probes. No hyperparameter change or new performance claim.',
        diagnostic_compute_accounting='Extra gradient/virtual-optimizer diagnostics reported separately; original409 optimization updates and cap unchanged. This is not a new cap-matched performance candidate.')
    return p


def prepare(_target=None):
    prior=read(ROOT/'reports'/PARENT/'readiness_results.json')
    assert prior['status']=='complete' and prior['outcome']=='exhausted','Finish readiness pilots first'
    stage=ROOT/'experiments'/NAME;source=ROOT/'experiments'/SOURCE
    if (stage/'manifest.json').exists():return verify(stage)
    manifest=read(source/'manifest.json')
    for name,sha in manifest.items():assert digest(source/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(source/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('unetr_optimizer_probe_study.py','unetr_optimizer_diagnostics.py','test_unetr_optimizer_diagnostics.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,stage/'source/scripts/mni_constraints'/name)
    shutil.copy2(source/'manifest.json',stage/'parent_manifest.json')
    reference=stage/'reference_source/capped_core.py';reference.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(stage/CORE,reference);before=reference.read_text();after=instrument_source(before)
    (stage/CORE).write_text(after)
    (stage/'instrumentation.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),after.splitlines(True),fromfile='reference/capped_core.py',tofile=CORE)))
    folder=stage/'protocols/delayed_t05'
    shutil.copytree(source/'protocols/delayed_t05',folder)
    shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
    write(folder/'protocol.json',protocol(read(folder/'parent_protocol.json')))
    references=[]
    for task in tasks():
        origin=ROOT/'reports'/SOURCE/'runs'/f"MNI_unetr_delayed_t05_f{task['fold']}s0"
        verify_archive(origin,read(origin/'archive_manifest.json'));assert read(origin/'cleanup.json')['status']=='complete'
        shutil.copytree(origin,stage/'reference_runs'/task['reference'])
        references.append(dict(task=task['id'],folder=task['reference'],archive=str(origin),manifest_sha256=digest(origin/'archive_manifest.json')))
    write(stage/'references.json',references);write(stage/'previous_results.json',prior)
    plan=dict(name=NAME,tasks=tasks(),num_runs=2,models=['unetr_vit_b16'],seeds=[0],folds=[0,1],
        budget_flops=CAP,gpu_workers=2,augmentation='xy_d4_v1',
        conventional_python=read(source/'plan.json')['conventional_python'],excluded_nodes=['gnode04'],
        graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='User explicitly authorizes continuing approaches until constraints consistently improve UNETR. Readiness did not solve consistency; observational AdamW diagnosis before more intervention. Other models paused; two GPU workers.',
        compute_accounting='Unchanged optimization cap/409 updates; extra observational diagnostic computation explicitly profiled and timed separately. Not included as new performance improvement evidence.',
        criterion='Exact replay of each reference constrained trajectory and final model; characterize logit versus parameter gradients and same-state Adam update perturbation. No optimizer/loss/data/seed changes.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(source/'manifest.json'),
        original_core_sha256=digest(reference),instrumented_core_sha256=digest(stage/CORE),
        changes='Read-only loss/gradient probes and actual parameter-delta snapshots around the unchanged optimizer step.'))
    (stage/'run.sbatch').write_text((source/'run.sbatch').read_text().replace('scripts.mni_constraints.unetr_delayed_confirm',MODULE).replace('hippo-unetr-confirm','hippo-unetr-adam'))
    donors={str(f.relative_to(stage)):str(REMOTE/SOURCE/f.relative_to(stage)) for f in stage.rglob('*')
            if f.is_file() and manifest.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None);donors['parent_manifest.json']=str(REMOTE/SOURCE/'manifest.json')
    for item in references:
        origin=REMOTE/SOURCE/'runs'/f"MNI_unetr_delayed_t05_f{next(t['fold'] for t in tasks() if t['id']==item['task'])}s0"
        for f in (stage/'reference_runs'/item['folder']).iterdir():
            if f.is_file() and f.name!='archive_acknowledgment.json':donors[str(f.relative_to(stage))]=str(origin/f.name)
    write(stage/'deployment_known_donors.json',donors);campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    manifest=read(stage/'manifest.json')
    for name,sha in manifest.items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');assert plan['tasks']==tasks() and plan['gpu_workers']==plan['num_runs']==2
    assert plan['budget_flops']==CAP and read(stage/'previous_results.json')['outcome']=='exhausted'
    parent=read(stage/'parent_manifest.json');reference=stage/'reference_source/capped_core.py'
    assert digest(reference)==parent[CORE] and (stage/CORE).read_text()==instrument_source(reference.read_text())
    for name,sha in parent.items():
        if name.startswith('source/') and name not in (CORE,'source/source_manifest.json'):assert manifest[name]==sha,name
    folder=stage/'protocols/delayed_t05';p=read(folder/'protocol.json')
    assert digest(folder/'parent_protocol.json')==parent['protocols/delayed_t05/protocol.json']
    assert p==protocol(read(folder/'parent_protocol.json'))
    assert p['pkl_sha256']==manifest['data/MNI.pkl']
    sched=p['models']['unet']['objectives']['dice_bands']
    assert sched['max_steps']==409 and sched['prefix_updates']==102 and sched['constraint_warmup_updates']==33
    assert campaign.training_flops(409,sched)<=CAP<campaign.training_flops(410,sched)
    for item in read(stage/'references.json'):
        folder=stage/'reference_runs'/item['folder'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==item['manifest_sha256']
        cfg=read(folder/'config.json');task=next(t for t in tasks() if t['id']==item['task'])
        assert cfg['seed']==task['seed'] and cfg['fold']==task['fold'] and cfg['schedule']==sched
        assert cfg['script_sha256']==digest(reference)
    return plan


def preflight(stage):
    verify(stage)
    import torch,monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints.test_unetr_optimizer_diagnostics import run
    assert monai.__version__=='1.6.0';test=run();torch.set_num_threads(4)
    folder=stage/'protocols/delayed_t05'
    batch.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective='dice_bands',fold=0))
    model=core.base.build_model('unet',read(folder/'nnunet_architecture.json')).to('meta').train()
    assert list(model(torch.zeros((1,1,64,64,64),device='meta')).shape)==[1,4,64,64,64]
    assert sum(p.numel() for p in model.parameters())==92685572
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    for fold in (0,1):
        split=read(folder/'splits_10patients.json')[fold]
        assert len(core.PreparedSubfields(frame,split['train'],DATASETS['MNI']))==20
        assert len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI']))==10
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),production_optimizer_updates=0,
                synthetic_equivalence=test,torch=torch.__version__,monai=monai.__version__,two_reference_replays=True)
    write(stage/'preflight.json',result);return result


def main():
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=headroom_bands.train
    campaign.main()


if __name__=='__main__':main()
