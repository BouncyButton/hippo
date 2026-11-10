"""Build a content-bound source payload only after local checks pass."""
import json
from pathlib import Path
import shutil
import tarfile
from .prepare import ROOT,DOC,build,sha


def main():
    p=build()
    out=ROOT/'evaluation_output/unwrapped_swin_campaign/followup'
    out.mkdir(parents=True,exist_ok=True)
    tests=ROOT/'evaluation_output/unwrapped_swin_campaign/followup_unit_checks.json'
    smoke=ROOT/'evaluation_output/unwrapped_swin_campaign/followup_model_smoke.json'
    unit,model=[json.loads(path.read_text()) for path in (tests,smoke)]
    assert unit['passed'] and model['passed'] and model['unit_gate_exactly_matches_prior']
    stage=out/'payload';stage.mkdir(exist_ok=True)
    for name,expected in p['source_sha256'].items():
        assert sha(ROOT/name)==expected
        dest=stage/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(DOC/'protocol.json',stage/'protocol.json')
    shutil.copy2(DOC/'README.md',stage/'README.md')
    shutil.copy2(tests,stage/'local_unit_checks.json');shutil.copy2(smoke,stage/'local_model_smoke.json')
    checks=dict(passed=True,protocol_sha256=sha(DOC/'protocol.json'),unit=unit,model=model,
                source_sha256=p['source_sha256'],note='Local CPU checks; production MONAI 1.6 CUDA checks run before any scientific training.')
    (stage/'local_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    shutil.copy2(stage/'local_checks.json',DOC/'local_checks.json')
    manifest={str(f.relative_to(stage)):sha(f) for f in stage.rglob('*') if f.is_file() and f.name!='payload_manifest.json'}
    (stage/'payload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out/'payload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for f in sorted(stage.rglob('*')):
            if f.is_file():tar.add(f,arcname=str(f.relative_to(stage)))
    result=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,protocol_identity=p['identity_sha'])
    (out/'payload_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
