"""Execute the eight requested runs sequentially and preserve separate results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .utils import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root/'source'
    protocol_path = source/'docs/experiments/depth_swinunetr_20261006/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    pkl = Path('/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    # Unit checks and CUDA checks execute on the compute node, never login node.
    with (root/'preflight.log').open('a') as log:
        subprocess.run([sys.executable, '-m', 'scripts.depth_supervision.test_depth'], cwd=source,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, '-m', 'scripts.depth_supervision.preflight', '--protocol', str(protocol_path),
                        '--pkl', str(pkl), '--root', str(root)], cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = []
    for fraction in ('5pct', 'full'):
        for arm in protocol['arms']:
            output = root/'runs'/fraction/arm
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(root/'status.json', dict(state='running', fraction=fraction, arm=arm,
                                                completed=len(records), started_at_unix=time.time()))
            with (output/'training.log').open('a') as log:
                result = subprocess.run([sys.executable, '-m', 'scripts.depth_supervision.train',
                    '--protocol', str(protocol_path), '--pkl', str(pkl), '--fraction', fraction,
                    '--arm', arm, '--output', str(output), '--cache', str(root/'geometry_cache')],
                    cwd=source, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                failure = dict(fraction=fraction, arm=arm, returncode=result.returncode, failed_at_unix=time.time())
                atomic_json(output/'failure.json', failure)
                records.append(failure)
            else:
                records.append(json.loads((output/'completion.json').read_text()))
            atomic_json(root/'summary.json', dict(protocol_sha256=digest(protocol_path), runs=records))
    failed = [r for r in records if 'returncode' in r]
    atomic_json(root/'status.json', dict(state='failed' if failed else 'complete', runs=len(records), failures=len(failed)))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
