"""Content-bound deployment after local loss/restart/native-model checks pass."""
import json
import shutil
import tarfile
from .prepare import ROOT, DOC, build, sha


def main():
    p = build()
    out = ROOT/'evaluation_output/tissue_profiles_20261007'
    out.mkdir(parents=True, exist_ok=True)
    unit = json.loads((out/'unit_checks.json').read_text())
    model = json.loads((out/'model_smoke.json').read_text())
    multiscale = json.loads((out/'multiscale_model_smoke.json').read_text())
    branches = json.loads((out/'branch_model_checks.json').read_text())
    informed = [json.loads((out/f'informed_{v}_smoke.json').read_text()) for v in ('none', 'pooled', 'multiscale')]
    mri_views = [json.loads((out/f'{arm}_{variant}_smoke.json').read_text()) for arm in
                 ('plane_mri_context', 'unfold_mri_context', 'plane_mri_attention', 'unfold_mri_attention',
                  'native_mri_context', 'native_mri_attention') for variant in ('none', 'pooled', 'multiscale')]
    assert unit['passed'] and model['passed'] and model['full_native_network_trained']
    assert multiscale['passed'] and multiscale['variant'] == 'multiscale' and multiscale['full_native_network_trained']
    assert unit['cancellation_example']['pooled_loss'] == 0 and unit['cancellation_example']['multiscale_loss'] > 0
    assert branches['passed'] and len(branches['cases']) == 8
    assert {(r['arm'], r['variant']) for r in branches['cases']} == {
        (arm, variant) for arm in ['plane_position', 'unfold_position', 'plane_attention', 'unfold_attention']
        for variant in ['pooled', 'multiscale']}
    assert all(r['passed'] and r['branch_and_fusion_gradients_checked'] for r in branches['cases'])
    assert all(r['passed'] and r['full_native_network_trained'] and r['input_channel_gradients'] for r in informed)
    assert all(r['passed'] and r['mri_only_auxiliary'] and r['branch_and_fusion_gradients_checked'] for r in mri_views)
    for result in [unit, model, multiscale, *informed, *mri_views]:
        for name, expected in result['tested_sources'].items():
            assert sha(ROOT/name) == expected, 'Source changed since verification: '+name
    # The original branch architecture and scientific losses are unchanged.
    prior = out/'before_informed_amendment/tissue_profiles'
    for name in ('views.py', 'profiles.py', 'multiscale.py'):
        assert sha(prior/name) == sha(ROOT/'scripts/tissue_profiles'/name)
    stage = out/'payload'
    stage.mkdir(exist_ok=True)
    for name, expected in p['source_sha256'].items():
        assert sha(ROOT/name) == expected
        destination = stage/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    shutil.copy2(DOC/'protocol.json', stage/'protocol.json')
    shutil.copy2(DOC/'README.md', stage/'README.md')
    checks = dict(passed=True, protocol_sha256=sha(DOC/'protocol.json'), unit=unit, model=model, multiscale=multiscale, branches=branches,
                  informed=informed, mri_views=mri_views,
                  preserved_branch_checks='Earlier eight CPU branch/loss checks reused because branch models and both losses are byte-identical; unit training orchestration and native smokes rerun. Each production run gets fresh CUDA preflight.',
                  source_sha256=p['source_sha256'],
                  note='Local CPU checks passed; production MONAI 1.6 bf16 CUDA and real-case coverage checks run inside the dependent allocation before scientific training.')
    (stage/'local_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    shutil.copy2(stage/'local_checks.json', DOC/'local_checks.json')
    manifest = {str(f.relative_to(stage)): sha(f) for f in stage.rglob('*') if f.is_file() and f.name != 'payload_manifest.json'}
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
