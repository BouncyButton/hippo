"""Copy a stopped SAM3 campaign to its owner's other account, with no replay."""
import copy
import hashlib
import json
from pathlib import Path
import zipfile

EVENT = '20261021_to_filippofocaccia2'
SOURCE_STATE = '1L-s3J9w5wuP4vzLLvnKUR5lm44EZqjwd'
SOURCE_RESULTS = '1EDy575RclOhGBxhuylaHszLA67wT80qU'
DESTINATION_RESULTS = '1ABU0S3cjFNozoYH4tD1DR_5aVp0z-kBz'
ACCOUNT = 'filippofocaccia2@gmail.com'
BUNDLE_SHA = 'bd128e499273277477f204a8d182906b669b377eae008c6d169d8ad4fe0c69cd'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_snapshot(path, key, state):
    """Check the recovery ledger independently, before progress can overwrite it."""
    from pathlib import PurePosixPath
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read('_snapshot_manifest.json'))
        if len(names) != len(set(names)) or set(names) != set(manifest) | {'_snapshot_manifest.json'}:
            raise ValueError('Unexpected or duplicate snapshot members: ' + key)
        for name, expected in manifest.items():
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or str(p) != name:
                raise ValueError('Unsafe snapshot member')
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError('Snapshot member checksum differs: ' + name)
        kind, _, task = key.partition('/')
        if kind in ('recovery', 'progress') and task in state['charges']:
            ledger = json.loads(archive.read(f'work/{task}/checkpoint_ledger.json'))
            if ledger != state['charges'][task]:
                raise ValueError('Clean-stop checkpoint does not match charged updates: ' + task)
            if kind == 'recovery' and f'work/{task}/checkpoint_latest.pt' not in manifest:
                raise ValueError('Recovery checkpoint missing: ' + task)


def migrate(cloud, directory, expected_sha):
    """Commit destination state only after every referenced snapshot is verified."""
    if cloud.folder != DESTINATION_RESULTS:
        raise ValueError('Unexpected destination folder')
    cloud.check_account()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / 'source_campaign_state.json'
    cloud.download(SOURCE_STATE, source_path, expected_sha)
    source = json.loads(source_path.read_text())
    if source.get('bundle_sha256') != BUNDLE_SHA:
        raise ValueError('Source scientific bundle changed')
    objects = source['objects']
    for task in source['completed']:
        if 'completed/' + task not in objects:
            raise ValueError('Completed result archive missing: ' + task)
    for task in source['charges']:
        if task not in source['completed'] and 'recovery/' + task not in objects:
            raise ValueError('Charged run has no recovery: ' + task)
    found = [m for m in cloud.children() if m['name'] == 'campaign_state.json']
    if len(found) > 1:
        raise ValueError('Duplicate destination campaign state')
    if found:
        current = directory / 'destination_campaign_state.json'
        cloud.download(found[0]['id'], current)
        state = json.loads(current.read_text())
        audit = state.get('account_migrations', {}).get(EVENT, {})
        if (audit.get('source_sha256') != expected_sha or
                state.get('bundle_sha256') != BUNDLE_SHA or
                not set(source['completed']).issubset(state['completed'])):
            raise ValueError('Destination belongs to another campaign or lost completed runs')
        for task, result in source['completed'].items():
            if state['completed'][task] != result:
                raise ValueError('Previously completed result changed: ' + task)
        print('Existing migrated campaign retained:', len(state['completed']), '/126 complete', flush=True)
        return state

    state = copy.deepcopy(source)
    state['objects'] = {}
    mapping = {}
    for key, receipt in objects.items():
        path = directory / (receipt['local_sha256'] + '.zip')
        cloud.download(receipt['id'], path, receipt['local_sha256'])
        verify_snapshot(path, key, source)
        new = cloud.put_immutable(path, 'sam3', task=key.replace('/', '_'))
        if new['local_sha256'] != receipt['local_sha256']:
            raise ValueError('Copied snapshot changed')
        state['objects'][key] = new
        mapping[key] = {'source_id': receipt['id'], 'destination_id': new['id'],
                        'sha256': receipt['local_sha256']}
        path.unlink()
        print('Verified migration:', key, flush=True)

    # A resumed source invalidates this transfer; never commit a stale campaign.
    cloud.download(SOURCE_STATE, directory / 'source_rechecked.json', expected_sha)
    original = cloud.put_immutable(source_path, 'migration_source', task=EVENT)
    audit = dict(source_account='focacciafilippo@gmail.com', destination_account=ACCOUNT,
                 source_state_id=SOURCE_STATE, source_results=SOURCE_RESULTS,
                 destination_results=DESTINATION_RESULTS, source_sha256=expected_sha,
                 original_state=original, objects=mapping,
                 completed_at_transfer=len(source['completed']),
                 original_retired=source.get('retired', []),
                 policy='Transport/account migration only; no charged updates replayed; scientific bundle unchanged.')
    state.setdefault('account_migrations', {})[EVENT] = audit
    # These are historical source-owned files. The original state archive retains
    # their receipts; the destination must never delete them.
    state['retired'] = []
    target = directory / 'destination_campaign_state.json'
    target.write_text(json.dumps(state, indent=2) + '\n')
    receipt = cloud.upload(target, 'campaign_state.json')
    cloud.download(receipt['id'], directory / 'destination_readback.json', digest(target))
    print('Migration verified:', len(state['completed']), '/126 complete;',
          126-len(state['completed']), 'remaining.', flush=True)
    return state
