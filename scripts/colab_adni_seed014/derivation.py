"""Verify the narrow seed-4 extension against the retained scientific bundle."""
import hashlib
import json
from pathlib import Path

PARENT_SHA = 'e0b87c00231bb355aa66531f6684c361a632b3f8e393f6afbfb4939a37d0caeb'
CLI_FILE = 'source/scripts/cross_dataset_campaign/capped_core.py'
CLI_OLD = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)'
CLI_NEW = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2, 4), required=True)'
CHANGED = {'runtime.py', 'gpu_preflight.py', 'initialization_reference.json', 'plan.json', CLI_FILE}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, parent):
    parent_manifest = parent/'bundle_manifest.json'
    if sha(parent_manifest) != PARENT_SHA:
        raise ValueError('Wrong parent scientific bundle; preserve it for inspection')
    original = read(parent_manifest)
    for name, expected in original.items():
        if sha(parent/name) != expected:
            raise ValueError('Parent frozen file changed: '+name)
        if name not in CHANGED and sha(root/name) != expected:
            raise ValueError('Scientific source or original protocol changed: '+name)
    source = (root/CLI_FILE).read_text()
    if source.count(CLI_NEW) != 1 or hashlib.sha256(source.replace(CLI_NEW, CLI_OLD).encode()).hexdigest() != original[CLI_FILE]:
        raise ValueError('Only the seed-4 CLI allowance may change in the trainer')
    old_ref = read(parent/'initialization_reference.json')
    new_ref = read(root/'initialization_reference.json')
    canonical = read(root/'seed4_initialization_reference.json')
    cluster = read(root/'seed4_cluster_hashes.json')
    plan = read(root/'plan.json')
    old_plan = read(parent/'plan.json')
    if plan['models'] != old_plan['models'] or plan['arms'] != old_plan['arms'] or plan['seeds'] != [4] or plan['folds'] != [0, 1, 2]:
        raise ValueError('Unexpected seed-4 experiment scope')
    expected_tasks = []
    for task in old_plan['tasks']:
        if task['seed'] != 0:
            continue
        expected_tasks.append(dict(task, seed=4, id=task['id'][:-1]+'4', protocol=task['protocol'][:-1]+'4'))
    if plan['tasks'] != expected_tasks or len(expected_tasks) != 168:
        raise ValueError('Expected all and only 168 seed-4 tasks')
    for model in plan['models']:
        for seed in ('0', '1', '2'):
            if new_ref['models'][model][seed] != old_ref['models'][model][seed]:
                raise ValueError('Original initialization reference changed')
        ref = canonical['models'][model]['4']
        if new_ref['models'][model]['4'] != ref or ref['model_sha256'] != cluster[model]:
            raise ValueError('Seed-4 canonical initialization differs from completed cluster controls')
        for fold in (0, 1, 2):
            before = parent/'protocols'/f'{model}_f{fold}_s0'
            after = root/'protocols'/f'{model}_f{fold}_s4'
            p = read(before/'protocol.json')
            p['seeds'] = [4]
            p['baseline_run_id'] = f'ADNI_{model}_dice_f{fold}s4'
            p['baseline_reference'] = dict(p['baseline_reference'], seed=4,
                initialization_sha256=cluster[model])
            if read(after/'protocol.json') != p:
                raise ValueError('Seed-4 scientific recipe changed: '+model)
            for file in before.iterdir():
                if file.is_file() and file.name != 'protocol.json' and sha(after/file.name) != sha(file):
                    raise ValueError('Seed-4 split/profile/architecture changed')
    return dict(status='verified_seed_only_derivation', parent_bundle_sha256=PARENT_SHA,
                seed4_bundle_sha256=sha(root/'bundle_manifest.json'), new_tasks=168,
                training_source_change='CLI accepts seed 4; numerical code unchanged')
