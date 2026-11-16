"""Bind a complete/pending HFH matrix snapshot to independently checked archives."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import zipfile

from scripts.colab_hfh.constraint_report import matrix_rows
from scripts.colab_hfh.summarize_constraints import verify as verify_constraint
from scripts.colab_hfh.summarize_strict_bands import verify as verify_band


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_equal(actual, expected, label):
    if isinstance(expected, float):
        valid = math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        valid = actual == ("" if expected is None else str(expected))
    if not valid:
        raise ValueError(f"Matrix differs from source at {label}: {actual!r} != {expected!r}")


def verify_native(path, reference):
    if digest(path) != reference['archive_sha256']:
        raise ValueError('Native archive identity differs')
    with zipfile.ZipFile(path) as archive:
        read = lambda name: json.loads(archive.read('results/' + name))
        for name, expected in read('archive_manifest.json').items():
            if Path(name).name != name or hashlib.sha256(archive.read('results/' + name)).hexdigest() != expected:
                raise ValueError('Native archive member differs: ' + name)
        for key, name in (('result', 'result.json'), ('configuration', 'config.json'),
                          ('initialization', 'initialization.json'),
                          ('validation', 'validation_latest.json'), ('training', 'training_final.json')):
            if read(name) != reference[key]:
                raise ValueError('Native reference differs from archived ' + name)
        protocol = archive.read('protocol/protocol.json')
        if hashlib.sha256(protocol).hexdigest() != reference['baseline_protocol_sha256']:
            raise ValueError('Native protocol binding differs')
        done, ledger = read('completion.json'), read('budget_ledger.json')
        if done['status'] != 'complete' or ledger['training_flops'] != reference['result']['optimization_flops']:
            raise ValueError('Native completion/accounting differs')


def audit(root, matrix_path, state_path):
    read = lambda path: json.loads(path.read_text())
    plan = read(root/'constraint_matrix_plan.json')
    base = read(root/'bands_v1_payload.json')
    state = read(state_path)
    expected = matrix_rows(plan, base, state)
    observed = list(csv.DictReader(matrix_path.open()))
    identity = lambda row: (row['model'], row['objective'], int(row['fold']), int(row['seed']))
    lookup = {identity(row): row for row in observed}
    if len(observed) != 504 or len(lookup) != 504 or set(lookup) != {identity(row) for row in expected}:
        raise ValueError('Matrix does not have the exact 504 unique declared identities')
    for row in expected:
        for field, value in row.items():
            check_equal(lookup[identity(row)][field], value, f'{identity(row)}/{field}')

    native_paths = list((root/'recipe_analysis').glob('*.zip')) + list((root/'baseline_v2').rglob('*.zip'))
    native_index = {digest(path): path for path in native_paths}
    evidence, protocols = [], {}
    for row in expected:
        if row['status'] != 'verified':
            continue
        model, objective, fold, seed = identity(row)
        key = f'{model}_f{fold}_s{seed}'
        reference = base['references'][key]
        if objective == 'dice':
            path = native_index[reference['archive_sha256']]
            verify_native(path, reference)
            result = reference['result']
            score, delta = result['validation_dice'], 0.
        elif row['recipe'] in ('basic_unet16_r09', 'bands_v1'):
            directory = (root/'bands_analysis/selected_recipes/basic_unet16_r09'
                         if row['recipe'] == 'basic_unet16_r09' else root/'bands_analysis/current')
            path = directory/(key+'.zip')
            result = verify_band(path, base['references'])
            score, delta = result['band_dice'], result['delta_dice_pp']
        else:
            path = root/'constraints_analysis'/row['recipe']/(key+'.zip')
            result = verify_constraint(path, base['references'])
            score, delta = result['constraint_dice'], result['delta_dice_pp']
        for field, value in (('final_dice', score), ('delta_pp', delta),
                             ('optimization_flops', result['optimization_flops']),
                             ('updates', result['updates'])):
            check_equal(lookup[identity(row)][field], value, f'{identity(row)}/archive/{field}')
        if result['optimization_flops'] > plan['budget_flops_per_run']:
            raise ValueError('Optimization cap exceeded')
        with zipfile.ZipFile(path) as archive:
            protocol_sha = hashlib.sha256(archive.read('protocol/protocol.json')).hexdigest()
        protocols.setdefault((model, objective, row['recipe']), set()).add(protocol_sha)
        evidence.append(dict(model=model, objective=objective, fold=fold, seed=seed,
                             recipe=row['recipe'], archive=str(path.relative_to(root)),
                             archive_sha256=digest(path), protocol_sha256=protocol_sha,
                             final_dice=score, delta_pp=delta))
    if any(len(values) != 1 for values in protocols.values()):
        raise ValueError('One selected recipe mixes scientific protocols across its pairs')
    return dict(snapshot_matrix_sha256=digest(matrix_path), snapshot_state_sha256=digest(state_path),
                declared_conditions=504, independently_verified_cells=len(evidence),
                pending_cells=504-len(evidence), full_matrix_complete=len(evidence)==504,
                verified_by_model=dict(Counter(row['model'] for row in evidence)),
                scope='Exact matrix identities and reported values checked against frozen references, '
                      'archive manifests, protocols, paired metrics and optimization accounting. '
                      'Calibration/diagnostic allowances remain separate from the optimization cap. '
                      'An independently verified result may be negative; verification is not qualification.',
                evidence=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('reports/hfh_colab_exploration_20261111'))
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.matrix or args.root/'full_constraint_matrix_current.csv',
                   args.state or args.root/'latest_full_search_state.json')
    target = args.root/'independent_full_matrix_audit.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'evidence'}, indent=2))
