"""Read-only audit of downloaded completed Colab records against cluster archives.

Runs locally on CPU; never imports the trainer, contacts Colab, or modifies a run.
Only findings.json in this audit directory is written.
"""
import ast
import functools
import hashlib
import json
import math
import random
import statistics
import tarfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO / 'reports/medsam3_msd_colab_3fold_3seed_noaug_20261015'
BUNDLE = REPO / 'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle'
SOURCE = ORIGINAL / 'original/source'


def read(path):
    return json.loads(path.read_text())


@functools.lru_cache(maxsize=None)
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def original_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1
    # Execute just the archived pure sampling/LR function, not module startup.
    exec(compile(ast.Module(body=matches, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def classify_exact_differences(report):
    """Keep every exact mismatch, separately identify inspected packaging/roundoff.

    No check is discarded and no training tolerance or file is changed.
    """
    explained = {}
    for relative in ('thesis/new_constraints/__init__.py', 'thesis/new_constraints/bands/__init__.py'):
        published = BUNDLE / 'source' / relative
        if published.read_bytes() == b'':
            explained['original_source:' + relative] = {
                'category': 'package_initialization_difference',
                'original_sha256': sha(SOURCE / relative), 'colab_sha256': sha(published),
                'description': 'Original imports/re-exports replaced by empty package initializer; actual selected loss modules retained.',
            }
    lr_function = original_function(SOURCE / 'scripts/train_medsam3_few_shot.py',
                                    'lr_for_update', {'math': math})
    for run in report['runs']:
        history = read(HERE / 'drive' / run['id'] / 'training_history.json')
        for row in history:
            expected = lr_function(row['update'] - 1, 200, 20, 5e-5, 5e-6)
            observed = row['lr']
            if expected != observed and abs(expected - observed) <= 2 * math.ulp(observed):
                assert np.float32(expected) == np.float32(observed)
                explained[run['id'] + f":lr:{row['update']}"] = {
                    'category': 'local_recomputation_roundoff',
                    'colab': observed, 'original_function_on_local_mac': expected,
                    'absolute_difference': abs(expected - observed),
                    'ulps': abs(expected - observed) / math.ulp(observed),
                    'same_float32_value': True,
                }
    report['exact_mismatches'] = report.pop('failures', report.get('exact_mismatches', []))
    report['explained_exact_differences'] = {
        check['check']: explained[check['check']]
        for check in report['exact_mismatches'] if check['check'] in explained
    }
    report['unexpected_failures'] = [c for c in report['exact_mismatches'] if c['check'] not in explained]
    different_source = {name.removeprefix('original_source:') for name in explained if name.startswith('original_source:')}
    report['byte_identical_archived_project_files'] = [
        name for name in report['byte_identical_archived_project_files'] if name not in different_source]
    report['status'] = ('unexpected_artifact_mismatch' if report['unexpected_failures'] else
                        'scientific_inputs_match_with_documented_implementation_differences')
    return report


def main():
    checks = []

    def check(label, condition):
        checks.append({'check': label, 'passed': bool(condition)})

    manifest = read(BUNDLE / 'bundle_manifest.json')
    bundle_sha = sha(BUNDLE / 'bundle_manifest.json')
    for relative, expected in manifest.items():
        check('bundle_file:' + relative, sha(BUNDLE / relative) == expected)
    print('Verified frozen bundle files.', flush=True)
    for filename in ('environment_binding.json', 'preflight.json'):
        root = read(HERE / 'drive/root' / filename)
        check('drive_binding:' + filename, root['bundle_sha256'] == bundle_sha)

    source_manifest = read(SOURCE / 'SOURCE_MANIFEST.json')
    source_matches = []
    for relative, expected in source_manifest.items():
        published = BUNDLE / 'source' / relative
        if published.exists():
            check('original_source:' + relative, sha(SOURCE / relative) == expected == sha(published))
            source_matches.append(relative)
    upstream_matches = []
    with tarfile.open(ORIGINAL / 'assets/medsam3_upstream.tar.gz') as archive:
        for member in archive.getmembers():
            published = BUNDLE / 'upstream' / member.name
            if member.isfile() and published.is_file():
                check('upstream:' + member.name,
                      hashlib.sha256(archive.extractfile(member).read()).hexdigest() == sha(published))
                upstream_matches.append(member.name)

    historical = read(BUNDLE / 'existing_results.json')
    check('historical_report_sha', sha(REPO / historical['provenance']['source']) ==
          historical['provenance']['source_sha256'])
    historic_runs = {r['id']: r for r in historical['runs']}
    old_records = historical['provenance']['records']
    data_manifest = read(BUNDLE / 'data_manifest.json')
    for relative, expected in data_manifest.items():
        check('data_file:' + relative, sha(BUNDLE / 'data' / relative) == expected)
    print('Verified source snapshot and local data hashes.', flush=True)
    base_sha = sha(BUNDLE / 'weights/sam3.pt')
    adapter_sha = sha(BUNDLE / 'weights/best_lora_weights.pt')
    check('base_checkpoint_matches_archive', base_sha ==
          sha(ORIGINAL / 'assets/sam3.pt'))
    check('medical_adapter_matches_archive', adapter_sha ==
          sha(ORIGINAL / 'assets/best_lora_weights.pt'))
    print('Verified base and medical adapter checkpoint files.', flush=True)
    schedule_function = original_function(SOURCE / 'scripts/train_medsam3_constraint_few_shot.py',
                                          'schedule', {'random': random})
    lr_function = original_function(SOURCE / 'scripts/train_medsam3_few_shot.py',
                                    'lr_for_update', {'math': math})
    runs = []
    verified_archive_files = 0
    label_hashes = {}
    for folder in sorted((HERE / 'drive').glob('MSD_MedSAM3_baseline_*')):
        name = folder.name
        archived = read(folder / 'archive_manifest.json')
        check(name + ':bundle_binding', archived['bundle_sha256'] == bundle_sha)
        for filename, expected in archived['files'].items():
            local = folder / filename
            if local.exists():
                check(name + ':archive_hash:' + filename, sha(local) == expected)
                verified_archive_files += 1
        task = archived['task']
        fold, seed = task['fold'], task['seed']
        original_protocol = read(ORIGINAL / f'original/results/fold{fold}_seed{seed}/protocol.json')
        cloud_protocol = read(folder / 'protocol.json')
        expected_protocol = dict(original_protocol, augmentation='none',
                                 budget_policy='original_200_updates_not_flop_capped', arm='baseline')
        check(name + ':entire_protocol', cloud_protocol == expected_protocol)
        old_training = old_records[f'results/fold{fold}_seed{seed}/results.json']['summary']['training']['baseline']
        summary = read(folder / 'training_summary.json')
        for field in ('frozen_before', 'frozen_after', 'adapter_before', 'updates'):
            check(name + ':initialization:' + field, summary[field] == old_training[field])
        check(name + ':frozen_unchanged', summary['frozen_before'] == summary['frozen_after'])
        check(name + ':baseline_weights_empty', summary['weights'] == {})
        for relative, expected in original_protocol['data_hashes'].items():
            check(name + ':original_data_hash:' + relative, data_manifest[relative] == expected)
        check(name + ':split_hash', original_protocol['split_sha256'] ==
              sha(SOURCE / 'datasets/Dataset101_MSD/splits_final.json'))
        check(name + ':base_hash', original_protocol['base_sha256'] == base_sha)
        indices = {case: range(nib.as_closest_canonical(nib.load(
            BUNDLE / 'data/imagesTr' / f'{case}_0000.nii.gz')).shape[2])
            for case in original_protocol['train_cases']}
        expected_schedule = schedule_function(SimpleNamespace(indices=indices), original_protocol)
        schedule = read(folder / 'slab_schedule.json')
        check(name + ':all_400_slab_pairs', schedule == [list(pair) for pair in expected_schedule])
        check(name + ':schedule_summary_hash', sha(folder / 'slab_schedule.json') == summary['schedule_sha256'])
        history = read(folder / 'training_history.json')
        check(name + ':200_contiguous_updates', [r['update'] for r in history] == list(range(1, 201)))
        for row in history:
            expected_lr = lr_function(row['update'] - 1, original_protocol['updates'],
                                      original_protocol['warmup_updates'], original_protocol['learning_rate'],
                                      original_protocol['final_learning_rate'])
            check(name + f":lr:{row['update']}", row['lr'] == expected_lr)
            check(name + f":baseline_loss:{row['update']}", row['total'] == row['native'])
        recipe = read(folder / 'runtime_recipe.json')
        for key, value in {'augmentation': 'none', 'optimizer': 'AdamW', 'weight_decay': .01,
                           'gradient_clip_norm': 1., 'updates': 200, 'accumulation': 2,
                           'learning_rate': 5e-5, 'final_learning_rate': 5e-6,
                           'warmup_updates': 20, 'constraint_warmup_updates': 20,
                           'trainable_parameters': 18497664, 'learned_feature_cache': False,
                           'optimization_flop_cap': None,
                           'native_loss_source_sha256': source_manifest['scripts/train_medsam3_few_shot.py']}.items():
            check(name + ':recipe:' + key, recipe[key] == value)
        evaluation = read(folder / 'evaluation.json')['records']
        completion = read(folder / 'completion.json')
        for split, key, metric in [('training_support', 'train_cases', 'training_dice'),
                                   ('validation', 'evaluation_cases', 'validation_dice')]:
            rows = [r for r in evaluation if r['split'] == split]
            cases = [r['case'] for r in rows]
            check(name + ':evaluated_cases:' + split, cases == original_protocol[key])
            check(name + ':dice_aggregation:' + split,
                  abs(statistics.mean(r['ap_mean_dice'] for r in rows) - completion[metric]) < 1e-14)
        for row in evaluation:
            case = row['case']
            if case not in label_hashes:
                truth = nib.as_closest_canonical(nib.load(BUNDLE / 'data/labelsTr' / f'{case}.nii.gz'))
                label_hashes[case] = hashlib.sha256(truth.get_fdata().astype(np.uint8).tobytes()).hexdigest()
            check(name + ':evaluation_truth_hash:' + case, row['truth_sha256'] == label_hashes[case])
        check(name + ':checkpoint_receipt', completion['checkpoint_sha256'] == archived['checkpoint_sha256'])
        old = historic_runs[name]
        runs.append(dict(id=name, fold=fold, seed=seed, training_cases=len(original_protocol['train_cases']),
                         validation_cases=len(original_protocol['evaluation_cases']),
                         colab_dice=completion['validation_dice'], cluster_dice=old['validation_dice'],
                         delta_pp=100 * (completion['validation_dice'] - old['validation_dice']),
                         cluster_adapter_after=old_training['adapter_after'],
                         colab_adapter_after=summary['adapter_after']))

    failures = [c for c in checks if not c['passed']]
    report = dict(status='artifact_checks_passed_with_code_equivalence_qualification' if not failures else 'audit_failed',
                  scope='Nine completed baseline archives only; local CPU analysis, no live runtime access or mutation',
                  drive_account='filippofocaccia2@gmail.com (Second account)',
                  bundle_sha256=bundle_sha, checks=len(checks), failures=failures,
                  verified_downloaded_archive_files=verified_archive_files,
                  bundle_files=len(manifest), data_files=len(data_manifest),
                  byte_identical_archived_project_files=source_matches,
                  byte_identical_upstream_files=upstream_matches,
                  reconstructed_slab_pairs=400 * len(runs), checked_learning_rates=200 * len(runs),
                  original_raw_slab_schedules_available=False,
                  schedules_comparison='Reconstructed with original archived function, original protocols and verified image dimensions',
                  starting_base_file_sha256=base_sha,
                  starting_medical_adapter_file_sha256=adapter_sha,
                  checks_detail=checks, runs=runs,
                  cluster_mean_dice=statistics.mean(r['cluster_dice'] for r in runs),
                  colab_mean_dice=statistics.mean(r['colab_dice'] for r in runs),
                  mean_absolute_delta_pp=statistics.mean(abs(r['delta_pp']) for r in runs),
                  code_qualification={
                      'shared_source': 'Frozen model loader, preprocessing, native loss, sampling and LR functions match byte-for-byte where bundled.',
                      'wrapper': 'Colab uses a separate training loop with checkpoint/resume and extra SDF/depth diagnostics.',
                      'baseline_objective': 'Original: native + ramp*(0*bands + 0*edge). Extension: native + ramp*sum(empty weights). Same mathematical objective; different backward graph.',
                      'causality': 'The impact of the zero-weight backward branches has not been tested on this model/GPU. No root-cause attribution is made.',
                      'preflight_coverage': 'Prior strict audit compares shared per-term losses/gradients and optimizer resume, not the original full baseline train_arm loop against the new baseline loop.',
                      'calibration': 'Fresh Colab calibration is intentional for constrained runs; baseline weights are empty, so calibration coefficients do not explain baseline differences.',
                      'runtime': 'Exact historical numerical environment has not been fully reconstructed. Original requested a 3g.40gb MIG allocation; Colab records A100-SXM4-80GB.'},
                  preservation='No running job, notebook, package, checkpoint, calibration or result was changed. No GPU experiments launched.')
    report = classify_exact_differences(report)
    (HERE / 'findings.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'checks', 'unexpected_failures', 'verified_downloaded_archive_files',
                                            'bundle_files', 'data_files', 'reconstructed_slab_pairs',
                                            'checked_learning_rates', 'cluster_mean_dice', 'colab_mean_dice')}, indent=2))
    if report['unexpected_failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
