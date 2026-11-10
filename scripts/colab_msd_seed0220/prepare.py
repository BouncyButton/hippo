"""Package the explicit MSD seed replacement without migrating original results."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from scripts.colab_msd_seed0220 import derivation
from scripts.colab_adni_seed014.prepare import notebook as notebook_template
from scripts.cross_dataset_campaign.storage import verify_archive

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent
OUT = ROOT/'reports/msd_colab_seeds0220_noaug_20261018'
PARENT = ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015/bundle'
CONTROL = ROOT/'reports/msd_colab_max_workers_3fold_3seed_noaug_20261015/control'
SEARCH = ROOT/'reports/msd_seed_search_10_41_20261018'
ZIP_NAME = 'msd_seeds0220_resume_noaug_code.zip'
BOOK_NAME = 'MSD_SEEDS_0_2_20_RESUME_NO_AUGMENTATION.ipynb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError('Unexpected frozen source: '+old)
    return source.replace(old, new)


def notebook(package_sha, summary, seed_sha):
    book = notebook_template(package_sha, summary['code_zip_sha256'], summary['data_zip_sha256'], seed_sha)
    substitutions = [
        ('ADNI', 'MSD'), ('adni', 'msd'),
        ('SEEDS_0_1_4', 'SEEDS_0_2_20'), ('seeds014', 'seeds0220'),
        ('seed4', 'seed20'), ('Seed4', 'Seed20'), ('seed-4', 'seed-20'),
        ('seed 4', 'seed 20'), ('Seed 4', 'Seed 20'),
        ('0/1/4', '0/2/20'), ('0/1/2 report', '0/1/2 report'), ('0/1 results', '0/2 results'),
        ('seeds 0/1', 'seeds 0/2'), ('seeds0/1', 'seeds0/2'), ('Seeds 0/1', 'Seeds 0/2'),
        ('seed 0/1', 'seed 0/2'), ('seed 2', 'seed 1'), ('Seed 2', 'Seed 1'),
        ('seed2', 'seed1'),
        ('e0b87c00231bb355aa66531f6684c361a632b3f8e393f6afbfb4939a37d0caeb', derivation.PARENT_SHA),
    ]
    # Apply seed1 exclusion before translating seed4 to seed20 (avoid prefix matches).
    substitutions = [x for x in substitutions if x[0] not in ('seed 2','Seed 2','seed2')]
    for cell in book['cells']:
        s = cell['source'].replace('seed 2', 'seed 1').replace('Seed 2', 'Seed 1').replace('seed2', 'seed1').replace('Seed2 ', 'Seed1 ')
        for old, new in substitutions:
            s = s.replace(old, new)
        s = s.replace('5% training data', '10 frozen training cases and 52 validation cases per fold')
        s = s.replace('All saved subject splits', 'All saved case splits')
        s = s.replace('original binary loss/calibration/optimizer/evaluation', 'original loss/calibration/optimizer/evaluation')
        s = s.replace('The acceleration implementation is unchanged from the A100 validation; the gate verifies that\nonly the seed selection and CLI allowance changed. No evidence is relabeled as a seed20 benchmark.',
                      'The seed20 GPU preflight compares the original and accelerated paths on your current GPU.\nIts receipt is separate from the retained original preflight.')
        s = s.replace('## 4. Resume missing seeds0/2, then run missing seed20', '## 4. Resume missing seeds 0/2, then run seed 20')
        s = s.replace('Use the original validated A100-SXM4-80GB and software versions.',
                      'Use the same GPU model and software versions recorded in your original results binding.')
        s = s.replace('every30seconds', 'every 30 seconds').replace(';22hours', '; 22 hours').replace('default8hours', 'default 8 hours')
        if cell['cell_type']=='markdown' and '## 4.' in s:
            s += ('\nThe existing speedups remain: GPU data/geometry caches, deferred evaluation readback, '
                  'unchanged batch-1 evaluation, fewer redundant best-checkpoint writes, local NVMe for new '
                  'checkpoints, fresh single-read archive verification and maximum eligible worker occupancy '
                  'without solo profiling or throughput backoff.\n')
        cell['source'] = s
    book['metadata']['colab']['name'] = BOOK_NAME
    for cell in book['cells']:
        if cell['cell_type']=='code':
            ast.parse(cell['source'])
    return book


def prepare():
    OUT.mkdir(exist_ok=True)
    package = OUT/'package'; package.mkdir(exist_ok=True)
    stage = package/'seed20_bundle'; stage.mkdir(exist_ok=True)
    original = read(PARENT/'bundle_manifest.json')
    assert sha(PARENT/'bundle_manifest.json') == derivation.PARENT_SHA
    for name, expected in original.items():
        assert sha(PARENT/name) == expected, name
        if not name.startswith('data/'):
            dest = stage/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT/name, dest)
    canonical = read(OUT/'seed20_initialization_reference.json')
    reference = read(PARENT/'initialization_reference.json')
    plan = read(PARENT/'plan.json')
    cluster_hashes, evidence = {}, []
    for model in plan['models']:
        digest = canonical['models'][model]['20']['model_sha256']
        for fold in plan['folds']:
            archive = SEARCH/'runs'/f'MSD_{model}_dice_f{fold}s20_full'
            verify_archive(archive, read(archive/'archive_manifest.json'))
            ack = read(archive/'archive_acknowledgment.json')
            assert ack['manifest_sha256'] == sha(archive/'archive_manifest.json')
            assert ack['source_verified'] and ack['source_manifest_sha256'] == sha(SEARCH/'frozen/manifest.json')
            assert read(archive/'initialization.json')['model_state_sha256'] == digest
            done = read(archive/'completion.json')
            assert done['status']=='complete' and done['budget_flops']==31976450667090
            evidence.append(dict(model=model, fold=fold, seed=20,
                archive_manifest_sha256=ack['manifest_sha256'], initialization_sha256=digest,
                final_dice=read(archive/'validation_latest.json')['summary']['macro_hard_dice']))
        cluster_hashes[model] = digest
        reference['models'][model]['20'] = canonical['models'][model]['20']
        old = PARENT/'protocols'/f'{model}_s0'
        folder = stage/'protocols'/f'{model}_s20'
        shutil.copytree(old, folder, dirs_exist_ok=True)
        protocol = read(old/'protocol.json')
        protocol['seeds'] = [20]
        protocol['baseline_reference'].update(seed=20, initialization_sha256=digest)
        write(folder/'protocol.json', protocol)
    plan['tasks'] = [dict(t, seed=20, id=t['id'][:-1]+'20', protocol=t['protocol'][:-1]+'20')
                     for t in plan['tasks'] if t['seed']==0]
    plan.update(name='msd_seed20_extension_20261018', seeds=[20], num_runs=168,
                policy='Retain Colab seeds0/2; add seed20; preserve seed1 separately. Post-selection exploratory comparison.')
    write(stage/'plan.json', plan)
    write(stage/'initialization_reference.json', reference)
    write(stage/'seed20_cluster_hashes.json', cluster_hashes)
    write(package/'seed_selection_evidence.json', dict(verified_baselines=evidence, imported_as_results=False,
        selection='Seed20 chosen on existing validation results; post-selection exploratory analysis.'))
    shutil.copyfile(OUT/'seed20_initialization_reference.json', stage/'seed20_initialization_reference.json')
    shutil.copyfile(SOURCE/'derivation.py', stage/'derivation.py')
    (stage/derivation.CLI_FILE).write_text(replace((PARENT/derivation.CLI_FILE).read_text(), derivation.CLI_OLD, derivation.CLI_NEW))
    (stage/'runtime.py').write_text(derivation.expected_runtime((PARENT/'runtime.py').read_text()))
    benchmark = 'source/scripts/cross_dataset_campaign/benchmark_msd_runtime.py'
    (stage/benchmark).write_text(replace((PARENT/benchmark).read_text(), 'base.seed_everything(0);model=', "base.seed_everything(p['baseline_reference']['seed']);model="))
    manifest = {str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*'))
                if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts and not p.is_symlink()}
    manifest['data/msd_hippocampus_full.pkl'] = original['data/msd_hippocampus_full.pkl']
    write(stage/'bundle_manifest.json', manifest)
    (stage/'data').mkdir(exist_ok=True)
    data = stage/'data/msd_hippocampus_full.pkl'
    if not data.exists(): data.symlink_to(PARENT/'data/msd_hippocampus_full.pkl')
    proof = derivation.verify(stage, PARENT)
    write(OUT/'derivation_verification.json', proof)
    for name, root in [('control02', PARENT), ('control20', stage)]:
        control = package/name
        shutil.copytree(CONTROL, control, dirs_exist_ok=True)
        scheduler = (control/'scheduler.py').read_text()
        scheduler = replace(scheduler, 'plan = self.r.verify_bundle(self.root)', 'plan = self.r.verify_bundle(self.root)\n        self.total_tasks = len(plan["tasks"])')
        scheduler = replace(scheduler, '{len(ids)}/504 verified runs', '{len(ids)}/{self.total_tasks} verified runs')
        scheduler = replace(scheduler, '{len(self.rows)}/504;', '{len(self.rows)}/{self.total_tasks};')
        (control/'scheduler.py').write_text(scheduler)
        m = read(control/'control_manifest.json')
        m['scientific_bundle_sha256'] = sha(root/'bundle_manifest.json')
        m['files']['scheduler.py'] = sha(control/'scheduler.py')
        write(control/'control_manifest.json', m)
    for name in ('campaign.py', 'phase.py', 'preflight.py'):
        shutil.copyfile(SOURCE/name, package/name)
    write(package/'selection.json', dict(seeds=[0,2,20], folds=[0,1,2], total=504,
        reused_seeds=[0,2], seed20_tasks=168, augmentation='none',
        original_results='results_noaug_v1', new_results='results_seeds0220_noaug_v1',
        reason='User requested replacement of seed1 after observing validation results; exploratory, post-selection.'))
    files = {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
             if p.is_file() and not p.is_symlink() and p.name!='package_manifest.json' and '__pycache__' not in p.parts}
    write(package/'package_manifest.json', files)
    archive = OUT/ZIP_NAME
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in [*files, 'package_manifest.json']: z.write(package/name, name)
    summary = read(PARENT.parent/'export_summary.json')
    write(OUT/BOOK_NAME, notebook(sha(archive), summary, sha(stage/'bundle_manifest.json')))
    write(OUT/'export_summary.json', dict(status='prepared_pending_Colab_seed20_GPU_preflight',
        notebook=BOOK_NAME, code_zip=ZIP_NAME, code_zip_sha256=sha(archive),
        seed20_bundle_sha256=sha(stage/'bundle_manifest.json'), parent_bundle_sha256=derivation.PARENT_SHA,
        total_target_runs=504, retained_seeds=[0,2], new_seed=20, seed20_tasks=168,
        augmentation='none', gpu_tested=False, source_derivation=proof,
        original_results_untouched=True, dataset_upload_required=False, cluster_jobs_submitted=0))
    print(OUT)


if __name__=='__main__': prepare()
