"""Verify and share frozen 5% guide inputs before submission; no guide fitting."""
import json
import os
from pathlib import Path
from .prepare import sha, NAME


def main():
    root = Path(__file__).resolve().parents[3]
    p = json.loads((root/'protocol.json').read_text())
    if root.name != NAME or p['name'] != NAME or os.getuid() != 1395:
        raise RuntimeError('Incorrect staging location')
    if (root/'submission_attempt.json').exists() or (root/'submission.json').exists():
        raise RuntimeError('Cannot restage after submission')
    for name, expected in json.loads((root/'payload_manifest.json').read_text()).items():
        if sha(root/name) != expected:
            raise ValueError('Payload changed: '+name)
    prior = Path(p['prior_root'])
    target = (prior/'atlas/5pct').resolve()
    manifest_path = target/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['protocol_sha256'] == p['atlas_protocol_sha256']
    assert set(manifest['cases']) == set(p['splits']['5pct']['train']+p['splits']['5pct']['val'])
    files = {str(manifest_path): sha(manifest_path)}
    for name, expected in manifest['artifacts'].items():
        path = target/name
        if sha(path) != expected:
            raise ValueError('Frozen atlas changed: '+str(path))
        files[str(path)] = expected
    destination = root/'atlas/5pct'
    destination.parent.mkdir(exist_ok=True)
    if destination.is_symlink():
        assert destination.resolve() == target
    elif destination.exists():
        raise RuntimeError('Expected a shared atlas link')
    else:
        destination.symlink_to(target, target_is_directory=True)
    assert sha(p['pkl']) == p['pkl_sha256']
    result = dict(files=files, shared_atlas=str(target), guide_fits=0, training_cases=10, validation_cases=52)
    (root/'reference_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(verified_files=len(files), shared_atlas=str(target))))


if __name__ == '__main__':
    main()
