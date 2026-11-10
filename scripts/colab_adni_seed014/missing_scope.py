"""Pinned continuation scope; historical archives are never visited by this runner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_package(package):
    for name, expected in read(package/'package_manifest.json').items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(package/relative) != expected:
            raise ValueError('Continuation package changed: ' + name)


def missing_plan(plan, package, seeds):
    """Filter only after the caller verifies the entire original scientific bundle."""
    scope = read(package/'continuation.json')
    selected = scope['seed01_tasks'] if list(seeds) == [0, 1] else scope['seed4_tasks']
    if list(seeds) not in ([0, 1], [4]):
        raise ValueError('Unexpected continuation phase')
    expected = {t['id']: t for t in plan['tasks'] if t['seed'] in seeds}
    if len({t['id'] for t in selected}) != len(selected):
        raise ValueError('Duplicate continuation task')
    if any(expected.get(t['id']) != t for t in selected):
        raise ValueError('Continuation identity differs from frozen science')
    old = read(package/'historical_snapshot.json')['runs']
    old_ids = {r['id'] for r in old}
    if len(old_ids) != len(old) or old_ids.intersection(t['id'] for t in selected):
        raise ValueError('Historical and pending work overlap')
    complement = old_ids if list(seeds) == [0, 1] else set()
    if set(expected) != {t['id'] for t in selected} | complement:
        raise ValueError('Continuation does not cover exactly the missing work')
    for row in old if list(seeds) == [0, 1] else []:
        if any(row.get(k) != v for k, v in expected[row['id']].items()):
            raise ValueError('Saved historical task identity differs')
    result = deepcopy(plan)
    result.update(tasks=selected, seeds=list(seeds), num_runs=len(selected))
    return result


def require_recovery(sweep, scope):
    """Never restart an imported interrupted run if any charged state was lost."""
    done = {r['id'] for r in sweep.rows}
    for identity, receipt in scope['recovery'].items():
        if identity in done:
            continue  # This continuation's full completed-archive verifier just passed.
        folder = sweep.durable/'recovery'/identity
        if not (folder/'charged_ledger.json').is_file() or not (folder/'latest.json').is_file():
            raise RuntimeError('Required interrupted recovery is missing; never restart: ' + identity)
        ledger, pointer = read(folder/'charged_ledger.json'), read(folder/'latest.json')
        if ledger['config_sha256'] != receipt['ledger']['config_sha256']:
            raise ValueError('Recovery configuration changed: ' + identity)
        for key in ('charged_attempts', 'training_flops', 'calibration_attempts', 'calibration_flops'):
            if ledger.get(key, 0) < receipt['ledger'].get(key, 0):
                raise ValueError('Recovery accounting regressed: ' + identity)
        if pointer['charged_attempts'] != ledger['charged_attempts']:
            raise RuntimeError('Recovery checkpoint is behind charges; preserve for audit: ' + identity)
        # The original restore path still validates the complete TAR hash, every
        # member, checkpoint update count, calibration and full ledger equality.


def install(sweep, package, seeds):
    scope = read(package/'continuation.json')
    allowed = {t['id'] for t in (scope['seed01_tasks'] if list(seeds) == [0, 1] else scope['seed4_tasks'])}
    historical = {r['id'] for r in read(package/'historical_snapshot.json')['runs']}
    original_load, original_baseline, original_launch = sweep.load, sweep.restore_baseline, sweep.launch

    def restore_baseline(task):
        identity = f"ADNI_{task['model']}_dice_f{task['fold']}s{task['seed']}"
        if task['arm'] == 'dice' or identity not in scope['baseline_initializations']:
            return original_baseline(task)
        if identity not in historical or task['id'] not in allowed:
            raise ValueError('Unapproved historical baseline dependency')
        source = package/'baselines'/identity/'initialization.json'
        if digest(source) != scope['baseline_initializations'][identity]:
            raise ValueError('Pinned baseline initialization file changed')
        protocol = sweep.r.read(sweep.root/'protocols'/task['protocol']/'protocol.json')
        if read(source) != {'model_state_sha256': protocol['baseline_reference']['initialization_sha256']}:
            raise ValueError('Historical baseline initialization differs from frozen protocol')
        # Historical completion is trusted from the frozen, previously produced
        # inventory at the user's request, not represented as freshly verified.
        # Only this exact initialization dependency is staged; no fake archive.
        target = sweep.root/'work'/identity/'initialization.json'
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Existing baseline initialization differs; preserve it')
        sweep.r.copy_verified(source, target)

    def load():
        marker = sweep.durable.parent/'missing_only_bootstrap.json'
        if not marker.is_file() or read(marker).get('continuation_sha256') != digest(package/'continuation.json'):
            raise RuntimeError('Continuation import did not finish; run setup before training')
        original_load()
        if any(t['id'] not in allowed for t in sweep.pending):
            raise ValueError('Historical/out-of-scope task reached pending queue')
        if list(seeds) == [0, 1]:
            require_recovery(sweep, scope)
            full = {t['id']: t for t in read(sweep.root/'plan.json')['tasks']}
            for identity in scope['baseline_initializations']:
                if identity not in historical:
                    raise ValueError('Missing saved baseline completion evidence')
                sweep.done_tasks.append(full[identity])
        print('SCOPE: old completed archives excluded; only continuation results are checked. '
              f'{len(sweep.pending)} pending in this phase.', flush=True)

    def launch(task):
        if task['id'] not in allowed or task['id'] in historical:
            raise ValueError('Refusing a historical or out-of-scope launch')
        return original_launch(task)

    sweep.load, sweep.restore_baseline, sweep.launch = load, restore_baseline, launch
