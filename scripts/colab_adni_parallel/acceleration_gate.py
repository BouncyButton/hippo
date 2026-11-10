"""Read-only acceptance of the returned A100 equivalence evidence.

This supplements the original scientific preflight. It never creates or changes
an environment/preflight receipt, runs a model, or relaxes a numerical check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(control, root, binding, preflight):
    manifest = read(control/'control_manifest.json')
    spec = manifest['acceleration']
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(control/name) != expected:
            raise ValueError('Control package changed: '+name)
    if (binding['bundle_sha256'] != manifest['scientific_bundle_sha256']
            or sha(root/'bundle_manifest.json') != binding['bundle_sha256']):
        raise ValueError('Acceleration evidence belongs to another scientific bundle')
    if preflight != dict(status='passed', **binding):
        raise RuntimeError('The existing original GPU preflight must have passed in this exact environment. Nothing was reset.')
    evidence = control/spec['evidence_file']
    if (Path(spec['evidence_file']).name != spec['evidence_file']
            or sha(evidence) != spec['evidence_sha256']
            or sha(control/'acceleration_candidate.py') != spec['candidate_sha256']):
        raise ValueError('Acceleration source/evidence hash mismatch')
    with zipfile.ZipFile(evidence) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Unreadable or ambiguous acceleration evidence')
        report = json.loads(archive.read('summary.json'))
        if report['binding'] != binding:
            raise RuntimeError('The acceleration was validated on a different GPU/software environment. '
                               'Restore the validated A100 environment; no results or checks were changed.')
        if (report['status'] != 'passed_equivalence' or report['models'] != spec['models']
                or report['checked'] != spec['models'] or report['extra_scientific_runs'] != 0
                or report['campaign_results_modified'] is not False
                or report['comparison'] != 'exact; no tolerance changes'
                or report['candidate_sha256'] != spec['candidate_sha256']
                or report['validator_sha256'] != spec['validator_sha256']):
            raise ValueError('Incomplete acceleration validation')
        for name, expected in (('acceleration_candidate.py', spec['candidate_sha256']),
                               ('validate_acceleration.py', spec['validator_sha256'])):
            if hashlib.sha256(archive.read('validator_source/'+name)).hexdigest() != expected:
                raise ValueError('Evidence source does not match: '+name)
        for model in spec['models']:
            original, candidate = (json.loads(archive.read(model+'_'+path+'.json'))
                                   for path in ('original', 'candidate'))
            if (original['status'] != 'passed_path_checks' or candidate['status'] != 'passed_path_checks'
                    or original['scientific'] != candidate['scientific']):
                raise ValueError('Scientific outputs differ: '+model)
            science = candidate['scientific']
            protocol = root/'protocols'/f'{model}_f0_s0'/'protocol.json'
            if (science['protocol_sha256'] != sha(protocol) or science['augmentation'] != 'none'
                    or science['scientific_updates'] != 0 or science['diagnostic_updates'] != 3
                    or len(science['data_hashes']) != 64):
                raise ValueError('Diagnostic coverage/recipe differs: '+model)
            if not all(candidate['acceleration'][name] > 0 for name in
                       ('sdf_hits', 'depth_hits', 'skipped_best_checkpoint_writes')):
                raise ValueError('Acceleration hooks were not exercised: '+model)
    return dict(status='verified_external_acceleration', binding=binding, original_preflight=preflight,
                control_sha256=sha(control/'control_manifest.json'), **spec,
                scope='Exact fold0/seed0 all-model diagnostic equivalence; unchanged frozen 504-task study. '
                      'No full-sweep throughput claim and no new experiments.')


def existing(runtime, control, root, durable):
    """Never call binding(): even an empty campaign may not silently rebind."""
    binding = dict(bundle_sha256=sha(root/'bundle_manifest.json'), environment=runtime.environment())
    if read(durable/'environment_binding.json') != binding:
        raise RuntimeError('Existing environment binding differs. Restore the original environment; nothing was rewritten.')
    receipt = verify(control, root, binding, read(durable/'preflight.json'))
    return binding, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    sys.path.insert(0, str(root))
    import runtime
    runtime.activate(root)
    print('PHASE: verify unchanged scientific files on local SSD', flush=True)
    runtime.verify_bundle(root)
    print('PHASE: check existing preflight/environment and returned A100 evidence; no GPU trials', flush=True)
    _, receipt = existing(runtime, Path(__file__).resolve().parent, root, args.durable)
    print('PASSED: original preflight retained; exact acceleration evidence verified for '+', '.join(receipt['models']), flush=True)


if __name__ == '__main__':
    main()
