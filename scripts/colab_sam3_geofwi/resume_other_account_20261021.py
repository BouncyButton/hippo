"""Account-routing entry point; frozen training and source verification stay intact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def admission_seconds(charge):
    """Budget conservatively for full runs, allowing short evaluation resumes."""
    return 15 * 60 if charge and charge.get('step') == 25 else 45 * 60


def apply_assignment(replication, config):
    if 'colab_tasks' not in config:
        return
    allowed = set(config['colab_tasks'])
    original_tasks = replication.tasks
    original_run = replication.run_one
    if not allowed.issubset({t['id'] for t in original_tasks()}):
        raise ValueError('Unknown task in Colab assignment')
    replication.tasks = lambda: [t for t in original_tasks() if t['id'] in allowed]
    def run_before_deadline(root, task, transport):
        needed = admission_seconds(transport.state['charges'].get(task['id']))
        if time.time() + needed > config['stop_compute_epoch']:
            print('DEFER TO CLUSTER: insufficient time before 15:00:', task['id'], flush=True)
            raise SystemExit(75)
        return original_run(root, task, transport)
    replication.run_one = run_before_deadline

ACCOUNT = 'filippofocaccia2@gmail.com'
RESULTS = '1ABU0S3cjFNozoYH4tD1DR_5aVp0z-kBz'
SOURCE_STATE = '1L-s3J9w5wuP4vzLLvnKUR5lm44EZqjwd'
BUNDLE_SHA = 'bd128e499273277477f204a8d182906b669b377eae008c6d169d8ad4fe0c69cd'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bundle', type=Path, required=True)
    args, _ = parser.parse_known_args()
    root = args.bundle.resolve()
    if str(root) != '/content/hippo_sam3_126_v1':
        raise ValueError('Keep the original bundle path for checkpoint/protocol compatibility')
    manifest_bytes = (root / 'bundle_manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != BUNDLE_SHA:
        raise ValueError('Frozen scientific bundle changed')
    manifest = json.loads(manifest_bytes)
    for name in ('replication.py', 'replication_store.py', 'store.py', 'request_gate.py'):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError('Frozen entry point changed: ' + name)
    config = json.loads(Path(__file__).with_suffix('.json').read_text())
    sys.path.insert(0, str(root))
    import replication_store
    replication_store.ACCOUNT = ACCOUNT
    replication_store.RESULTS = RESULTS
    cloud = replication_store.Cloud(RESULTS)
    cloud.check_account()
    # This read runs in the parent AND every child. Resuming the old notebook
    # changes the original mutable state and stops the new worker before updates.
    meta = cloud.metadata(SOURCE_STATE)
    if meta.get('md5Checksum') != config['source_state_md5']:
        raise RuntimeError('Original campaign state changed after handoff; stop both sessions and reconcile.')
    import replication
    apply_assignment(replication, config)
    # The original launcher dispatches children through its module __file__.
    # Route those children through this same account check without changing one
    # byte in the source bundle or weakening its verification.
    replication.__file__ = str(Path(__file__).resolve())
    replication.main()


if __name__ == '__main__':
    main()
