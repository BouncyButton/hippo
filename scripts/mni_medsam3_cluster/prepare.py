"""Build a reviewable frozen campaign without changing the Colab notebook."""
from pathlib import Path
import shutil

from scripts.colab_mni_medsam3.data import select_support
from scripts.cross_dataset_campaign.common import digest, read, write

ROOT = Path(__file__).resolve().parents[2]
NAME = 'mni_medsam3_n5_f0_s012_200_20261109'
EXPORT = ROOT/'reports'/NAME
STAGE = EXPORT/'stage'
REMOTE = Path('/mnt/beegfsstudents/home/3160552')/NAME
PARENT = ROOT/'reports/medsam3_mni_colab_two_subjects_20261108/bundle'
RUNTIME = Path('/mnt/beegfsstudents/home/3160552/medsam3_runtime_20260928')


def build():
    if STAGE.exists():
        raise ValueError('Preserve existing frozen campaign')
    for name, expected in read(PARENT/'manifest.json').items():
        if digest(PARENT/name) != expected:
            raise ValueError('Parent changed: '+name)
    STAGE.mkdir(parents=True)
    shutil.copytree(PARENT/'source', STAGE/'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(PARENT/'data', STAGE/'data')
    shutil.copytree(Path(__file__).parent, STAGE/'source/scripts/mni_medsam3_cluster', ignore=shutil.ignore_patterns('__pycache__'))
    # Retain the scientific engine byte-for-byte; update only orchestration.
    framework = ROOT/'scripts/cross_dataset_campaign'
    for name in ('controller.py', 'collect.py', 'storage.py', 'adni_foundation_full_pool.py', 'foundation_gpu_worker.py'):
        shutil.copy2(framework/name, STAGE/'source/scripts/cross_dataset_campaign'/name)
    shutil.copy2(PARENT/'saved_folds.json', STAGE/'saved_folds.json')
    selection = select_support(read(PARENT/'saved_folds.json'), 5)
    tasks = []
    for seed in (0, 1, 2):
        identity = f'MNI_MedSAM3_n5_f0s{seed}_native200'
        protocol = read(PARENT/'protocol.json')
        protocol.update(selection)
        protocol.update(name=identity, seed=seed, train_subjects=5, train_crops=10,
            data=str(REMOTE/'data'), upstream=str(RUNTIME/'MedSAM3'),
            lora_weights=str(RUNTIME/'weights/best_lora_weights.pt'),
            support_selection='First five subjects of the fixed RandomState(0) fold0 ordering, both L/R crops; identical across optimizer seeds.',
            decision_rule='Inspect all three seed results and individual subfields before setting a target or expanding folds.')
        write(STAGE/'protocols'/identity/'protocol.json', protocol)
        tasks.append(dict(id=identity, protocol=identity, kind='foundation', dataset='MNI',
            subjects=5, fold=0, seed=seed, arm='dice'))
    upstream = read(ROOT/'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/frozen/source_version.json')['upstream_provenance']
    write(STAGE/'upstream_manifest.json', upstream['python_source_sha256'])
    write(STAGE/'source_provenance.json', dict(parent_manifest_sha256=digest(PARENT/'manifest.json'),
        parent_protocol_sha256=digest(PARENT/'protocol.json'), upstream=upstream,
        scientific_engine_changed=False, changes='Support 2→5 subjects; optimizer seeds0/1/2; two isolated Slurm GPU workers; local checkpoints with background verified copies',
        training_source_sha256={name:digest(STAGE/'source'/name) for name in (
            'scripts/cross_dataset_campaign/foundation.py', 'scripts/cross_dataset_campaign/foundation_msd.py',
            'scripts/colab_mni_medsam3/train.py', 'scripts/colab_mni_medsam3/optimized.py',
            'scripts/colab_medsam3/fast_eval.py')}))
    audit = read(PARENT/'data_audit.json'); audit['selection'] = selection
    write(STAGE/'data_audit.json', audit)
    plan = dict(name=NAME, tasks=tasks, augmentation='none', base_sha256=protocol['base_sha256'],
        foundation_python=str(RUNTIME/'venv/bin/python'),
        conventional_python='/mnt/beegfsstudents/home/3160552/.conda/envs/hippocampus/bin/python',
        base_staging='/tmp/hippo-mni-medsam3-3160552/sam3.pt',
        parallelism=dict(workers=2, gpus_per_worker=1, cpus_per_worker=4, gpu_type='4g.40gb', memory_gib_per_worker=40),
        storage=dict(minimum_free_gib=8, estimated_peak_beegfs_gib=3,
            note='Base3.45GB on login/node scratch. Two adapter+Adam recovery files ~0.5GB total, atomic temporary copies; conservative3GiB reserve.'),
        graceful_stop_seconds=20700, archive_wait_timeout_seconds=300,
        archive_resume_module='scripts.mni_medsam3_cluster.campaign',
        final_evaluation='All ten training crops and ten held-out crops, per-class Dice and full TP/FP/FN confusion; A/P swaps not applicable to MNI subfields')
    write(STAGE/'plan.json', plan)
    template = (ROOT/'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/frozen/run.sbatch').read_text()
    template = template.replace('hippo-adni-medsam3-full', 'hippo-mni-medsam3-n5').replace('--time=24:00:00', '--time=06:00:00')
    template = template.replace('scripts.cross_dataset_campaign.adni_foundation_full_job', 'scripts.mni_medsam3_cluster.campaign')
    template = template.replace('cd "$campaign/source"',
        'export TORCHINDUCTOR_CACHE_DIR="/tmp/hippo-mni-compiler-${SLURM_JOB_ID}"\n'
        'export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton"\n'
        'cd "$campaign/source"')
    # gnode04 is excluded by the account's current MNI cluster runs.
    template = template.replace('#SBATCH --nodes=1', '#SBATCH --nodes=1\n#SBATCH --exclude=gnode04')
    (STAGE/'run.sbatch').write_text(template)
    write(STAGE/'manifest.json', {str(p.relative_to(STAGE)):digest(p) for p in sorted(STAGE.rglob('*')) if p.is_file()})
    write(STAGE/'control.json', dict(pause=False))
    write(EXPORT/'preparation.json', dict(stage=str(REMOTE), local_stage=str(STAGE),
        manifest_sha256=digest(STAGE/'manifest.json'), subjects=selection['support_subjects'], seeds=[0,1,2], fold=0,
        updates_per_run=200, concurrent_gpu_workers=2, status='prepared_not_submitted'))
    print(EXPORT)


if __name__ == '__main__':
    build()
