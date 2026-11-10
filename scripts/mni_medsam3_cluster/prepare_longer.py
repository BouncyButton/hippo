"""Single-seed follow-up, admitted only after the MNI sweep."""
import argparse
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import digest, read, write
from scripts.mni_medsam3_cluster.prepare import ROOT

NAME = 'mni_medsam3_n5_f0_s0_1000_20261110'
EXPORT = ROOT/'reports'/NAME
STAGE = EXPORT/'stage'
REMOTE = Path('/mnt/beegfsstudents/home/3160552')/NAME
PARENT = ROOT/'reports/mni_medsam3_n5_f0_s012_200_20261109/archive/frozen'
SWEEP = dict(stage='/mnt/beegfsstudents/home/3160552/mni_winning_recipes_matrix2030_53t_0519_20261109',
    plan_sha256='1161d098684b148bfe315208519417e8d755c88bab7016e613e44dcd42796b7c', expected_runs=407)


def build(updates=1000):
    if updates not in (400, 1000):
        raise ValueError('Unsupported update budget')
    NAME = f'mni_medsam3_n5_f0_s0_{updates}_20261110'
    EXPORT = ROOT/'reports'/NAME
    STAGE = EXPORT/'stage'
    REMOTE = Path('/mnt/beegfsstudents/home/3160552')/NAME
    if STAGE.exists():
        raise ValueError('Preserve existing frozen campaign')
    for name, expected in read(PARENT/'manifest.json').items():
        if digest(PARENT/name) != expected:
            raise ValueError('Parent changed: '+name)
    STAGE.mkdir(parents=True)
    for name in ('source', 'data'):
        shutil.copytree(PARENT/name, STAGE/name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('saved_folds.json', 'upstream_manifest.json', 'data_audit.json'):
        shutil.copy2(PARENT/name, STAGE/name)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, STAGE/'source/scripts/mni_medsam3_cluster'/source.name)
    old_plan = read(PARENT/'plan.json')
    old = read(PARENT/'protocols'/old_plan['tasks'][0]['protocol']/'protocol.json')
    identity = f'MNI_MedSAM3_n5_f0s0_native{updates}'
    protocol = dict(old, name=identity, updates=updates, data=str(REMOTE/'data'),
        sampling=f'{updates} updates, two uniformly sampled adjacent-slice pairs per update; original seed0 sampling method.',
        decision_rule='Single-seed longer-training diagnostic; inspect subfield Dice before any seed/fold expansion.')
    write(STAGE/'protocols'/identity/'protocol.json', protocol)
    plan = dict(old_plan, name=NAME, seeds=[0], updates_per_run=updates,
        tasks=[dict(old_plan['tasks'][0], id=identity, protocol=identity)],
        parallelism=dict(old_plan['parallelism'], workers=1), wait_for_campaigns=[SWEEP],
        storage=dict(minimum_free_gib=8, estimated_peak_beegfs_gib=2,
            note='One adapter/Adam recovery file and atomic copy; base and compiler caches on node scratch.'),
        initialization_policy='Fresh medical-pretrained LoRA initialization; completed200-update checkpoint was retired under storage policy.',
        schedule_policy=f'Same LR endpoints5e-5→5e-6 and20 warmup updates; decay spans{updates} updates.',
        estimated_training_hours=3.5*updates/1000)
    write(STAGE/'plan.json', plan)
    provenance = read(PARENT/'source_provenance.json')
    provenance.update(parent_manifest_sha256=digest(PARENT/'manifest.json'),
        parent_protocol_sha256=digest(PARENT/'protocols'/old_plan['tasks'][0]['protocol']/'protocol.json'),
        scientific_engine_changed=False,
        changes=f'200→{updates} optimizer updates; seed0 only; oneGPU; admission gated on completion of entire407-run sweep.',
        unchanged=['Five subjects and all ten L/R crops', 'Fold0 validation cohort', 'Medical adapter/base initialization',
            'Native loss, accumulation2, warmup20, LR endpoints, noaugmentation', 'Audited inference and recovery optimizations'])
    write(STAGE/'source_provenance.json', provenance)
    script = (PARENT/'run.sbatch').read_text().replace('hippo-mni-medsam3-n5',f'hippo-medsam3-n5-{updates}')
    if updates == 400:
        script = script.replace('--time=06:00:00', '--time=03:00:00')
        plan['graceful_stop_seconds'] = 9900
        write(STAGE/'plan.json', plan)
    script = script.replace('--ntasks=2','--ntasks=1').replace('--gres=gpu:4g.40gb:2','--gres=gpu:4g.40gb:1').replace('--mem=80G','--mem=40G')
    (STAGE/'run.sbatch').write_text(script)
    write(STAGE/'manifest.json', {str(p.relative_to(STAGE)):digest(p) for p in sorted(STAGE.rglob('*')) if p.is_file()})
    write(STAGE/'control.json', dict(pause=False))
    write(EXPORT/'preparation.json', dict(stage=str(REMOTE), local_stage=str(STAGE),
        manifest_sha256=digest(STAGE/'manifest.json'), subjects=protocol['support_subjects'], seed=0, fold=0,
        updates=updates, gpu_workers=1, wait_for_campaigns=[SWEEP], status='prepared_not_submitted'))
    print(EXPORT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, choices=(400, 1000), required=True)
    build(parser.parse_args().updates)
