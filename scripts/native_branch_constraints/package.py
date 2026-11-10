"""Package the tested source and protocol without data or model weights."""
import json
import shutil
import tarfile
from .prepare import ROOT, DOC, build, sha


def main():
    p = build()
    out = ROOT/'evaluation_output/native_branch_constraints_20261008'
    checks = [json.loads((out/name).read_text()) for name in ('unit_checks.json', 'model_smoke.json')]
    assert all(c['passed'] for c in checks)
    assert checks[1]['native_mri_only'] and checks[1]['all_three_groups_updated']
    for result in checks:
        for name, expected in result['tested_sources'].items():
            assert sha(ROOT/name) == expected, 'Source changed after checks: '+name
    stage = out/'payload'
    if stage.exists():
        raise RuntimeError('Use a fresh payload directory; never merge stale files into a deployment')
    stage.mkdir(parents=True)
    for name, expected in p['source_sha256'].items():
        assert sha(ROOT/name) == expected
        destination = stage/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    for name in ('protocol.json', 'README.md'):
        shutil.copy2(DOC/name, stage/name)
    value = dict(passed=True, protocol_sha256=sha(DOC/'protocol.json'), checks=checks,
                 source_sha256=p['source_sha256'], note='CPU verification passed. Allocated MONAI 1.6 bf16 CUDA preflight gates every scientific run.')
    (stage/'local_checks.json').write_text(json.dumps(value, indent=2)+'\n')
    shutil.copy2(stage/'local_checks.json', DOC/'local_checks.json')
    manifest = {str(f.relative_to(stage)): sha(f) for f in stage.rglob('*') if f.is_file()}
    (stage/'payload_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = out/'payload.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for file in sorted(stage.rglob('*')):
            if file.is_file():
                tar.add(file, arcname=str(file.relative_to(stage)))
    result = dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size, protocol_identity=p['identity_sha'])
    (out/'payload_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
