"""Verify completed guides and controls; reuse frozen anatomy without new fits."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    p = json.loads((root/'protocol.json').read_text())
    prior = Path(p['prior_root'])
    assert root.name == p['name'] and os.getuid() == 1395
    if (root/'submission.json').exists() or (root/'submission_attempt.json').exists():
        raise RuntimeError('Cannot restage after a submission attempt')
    assert json.loads((prior/'status.json').read_text())['state'] == 'complete'
    old_protocol = prior/'source/docs/experiments/unwrapped_swin_20261006/protocol.json'
    assert sha(old_protocol) == p['prior_protocol_sha256']
    files = {str(old_protocol): sha(old_protocol)}
    destination, target = root/'atlas', (prior/'atlas').resolve()
    assert target.is_dir()
    if destination.is_symlink():
        assert destination.resolve() == target
    elif destination.exists():
        raise ValueError('Expected a shared atlas link')
    else:
        destination.symlink_to(target, target_is_directory=True)
    for fraction in ('5pct', 'full'):
        path = prior/'atlas'/fraction/'manifest.json'
        manifest = json.loads(path.read_text())
        assert manifest['complete'] and manifest['protocol_sha256'] == p['atlas_protocol_sha256']
        assert set(manifest['cases']) == set(p['splits'][fraction]['train']+p['splits'][fraction]['val'])
        files[str(path)] = sha(path)
        for name, expected in manifest['artifacts'].items():
            artifact = path.parent/name
            assert sha(artifact) == expected
            files[str(artifact)] = expected
        run = prior/'runs'/fraction/'dice'
        record = json.loads((run/'completion.json').read_text())
        assert record['identity']['arm'] == 'dice' and record['identity']['guide_fold'] is None
        for subset in ('train', 'val'):
            assert record['identity'][subset] == p['splits'][fraction][subset]
        files[str(run/'completion.json')] = sha(run/'completion.json')
        for name, expected in record['artifacts'].items():
            if name == 'best.pt':
                continue
            artifact = run/name
            assert sha(artifact) == expected
            files[str(artifact)] = expected
    record = dict(prior_root=str(prior), files=files, geometry_reused=True, completed_training_repeated=False)
    (root/'reference_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(verified_reference_files=len(files), shared_inputs=['atlas'])))


if __name__ == '__main__':
    main()
