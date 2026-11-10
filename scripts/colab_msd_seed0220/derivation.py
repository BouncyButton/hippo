"""Verify the narrow seed-20 extension against the retained scientific bundle."""
import hashlib
import json
from pathlib import Path

PARENT_SHA = '3b9d0c30f71873b6f834ae3b1fd989bf33e37bef7682fdda094b656daff3620b'
CLI_FILE = 'source/scripts/cross_dataset_campaign/capped_core.py'
CLI_OLD = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)'
CLI_NEW = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2, 20), required=True)'
CHANGED = {'runtime.py', 'source/scripts/cross_dataset_campaign/benchmark_msd_runtime.py', 'initialization_reference.json', 'plan.json', CLI_FILE}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected parent source: '+old)
    return source.replace(old, new)


def expected_runtime(source):
    source = source.replace('504', '168')
    source = checked_replace(source, 'for seed in (0, 1, 2):', 'for seed in (20,):')
    source = checked_replace(source, "f'{name}_s0'", "f'{name}_s20'")
    source = checked_replace(source, "if len([x for x in r if x['fold']==f]) == 3", "if len([x for x in r if x['fold']==f]) == 1")
    source = checked_replace(source, 'completed=len(r), expected=9', 'completed=len(r), expected=3')
    source = checked_replace(source, 'each averaging three seeds', 'each containing seed20')
    return source



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
        raise ValueError('Only the seed-20 CLI allowance may change in the trainer')
    if (root/'runtime.py').read_text() != expected_runtime((parent/'runtime.py').read_text()):
        raise ValueError('Unexpected runtime change beyond seed20 scope')
    benchmark = 'source/scripts/cross_dataset_campaign/benchmark_msd_runtime.py'
    expected_benchmark = checked_replace((parent/benchmark).read_text(),
        'base.seed_everything(0);model=', "base.seed_everything(p['baseline_reference']['seed']);model=")
    if (root/benchmark).read_text() != expected_benchmark:
        raise ValueError('Unexpected GPU diagnostic change')
    old_ref = read(parent/'initialization_reference.json')
    new_ref = read(root/'initialization_reference.json')
    canonical = read(root/'seed20_initialization_reference.json')
    cluster = read(root/'seed20_cluster_hashes.json')
    plan = read(root/'plan.json')
    old_plan = read(parent/'plan.json')
    if plan['models'] != old_plan['models'] or plan['arms'] != old_plan['arms'] or plan['seeds'] != [20] or plan['folds'] != [0, 1, 2]:
        raise ValueError('Unexpected seed-20 experiment scope')
    expected_tasks = []
    for task in old_plan['tasks']:
        if task['seed'] != 0:
            continue
        expected_tasks.append(dict(task, seed=20, id=task['id'][:-1]+'20', protocol=task['protocol'][:-1]+'20'))
    if plan['tasks'] != expected_tasks or len(expected_tasks) != 168:
        raise ValueError('Expected all and only 168 seed-20 tasks')
    for model in plan['models']:
        for seed in ('0', '1', '2'):
            if new_ref['models'][model][seed] != old_ref['models'][model][seed]:
                raise ValueError('Original initialization reference changed')
        ref = canonical['models'][model]['20']
        if new_ref['models'][model]['20'] != ref or ref['model_sha256'] != cluster[model]:
            raise ValueError('Seed-20 canonical initialization differs from completed cluster controls')
        before = parent/'protocols'/f'{model}_s0'
        after = root/'protocols'/f'{model}_s20'
        p = read(before/'protocol.json')
        p['seeds'] = [20]
        p['baseline_reference'] = dict(p['baseline_reference'], seed=20,
            initialization_sha256=cluster[model])
        if read(after/'protocol.json') != p:
            raise ValueError('Seed-20 scientific recipe changed: '+model)
        for file in before.iterdir():
            if file.is_file() and file.name != 'protocol.json' and sha(after/file.name) != sha(file):
                raise ValueError('Seed-20 split/profile/architecture changed')
    return dict(status='verified_seed_only_derivation', parent_bundle_sha256=PARENT_SHA,
                seed20_bundle_sha256=sha(root/'bundle_manifest.json'), new_tasks=168,
                training_source_change='CLI accepts seed 20; numerical code unchanged')
