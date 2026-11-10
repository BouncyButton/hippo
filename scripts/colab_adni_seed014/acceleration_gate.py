"""Accept unchanged A100 backend evidence only after verifying seed-only derivation."""
import importlib.util
import json
from pathlib import Path
import sys
from hashlib import sha256


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(control, root, binding, preflight):
    manifest = read(control/'control_manifest.json')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(control/name) != expected:
            raise ValueError('Seed-4 control package changed: '+name)
    if (sha(root/'bundle_manifest.json') != manifest['scientific_bundle_sha256']
            or binding['bundle_sha256'] != manifest['scientific_bundle_sha256']
            or preflight != dict(status='passed', **binding)):
        raise ValueError('Seed-4 bundle or its required GPU preflight differs')
    parent = root.parent/'hippo_adni_bundle'
    sys.path.insert(0, str(root))
    import derivation
    derived = derivation.verify(root, parent)
    parent_control = control.parent/'control01'
    spec = importlib.util.spec_from_file_location('original_adni_acceleration_gate', parent_control/'acceleration_gate.py')
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    parent_binding = dict(bundle_sha256=derivation.PARENT_SHA, environment=binding['environment'])
    # Original evidence remains bound to its original bundle/environment. It is
    # never relabeled as seed-4 evidence. The separate seed-4 preflight is required.
    receipt = original.verify(parent_control, parent, parent_binding, dict(status='passed', **parent_binding))
    if manifest['acceleration'] != {key: receipt[key] for key in manifest['acceleration']}:
        raise ValueError('Backend evidence/candidate identity changed')
    return dict(status='verified_seed4_backend_derivation', binding=binding,
                original_preflight=preflight, control_sha256=sha(control/'control_manifest.json'),
                **manifest['acceleration'], derivation=derived,
                parent_evidence_receipt=receipt,
                scope='Unchanged backend tested on original fold0/seed0 A100; seed4 separately preflighted. No seed4 throughput claim.')


def existing(runtime, control, root, durable):
    binding = dict(bundle_sha256=sha(root/'bundle_manifest.json'), environment=runtime.environment())
    if read(durable/'environment_binding.json') != binding:
        raise ValueError('Seed-4 environment changed; nothing was reset')
    return binding, verify(control, root, binding, read(durable/'preflight.json'))
