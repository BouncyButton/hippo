"""Import unchanged native training state into the isolated repair campaign."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from scripts.depth_supervision.utils import atomic_json, digest


def restore(root):
    if (root / 'submission_attempt.json').exists():
        raise RuntimeError('Do not change state after submission')
    p = json.loads((root / 'source/docs/experiments/unwrapped_swin_20261006/protocol.json').read_text())
    source = Path(p['resume_from_root'])
    previous = json.loads((source / 'source/docs/experiments/unwrapped_swin_20261006/protocol.json').read_text())
    if previous != json.loads((root / 'prior_protocol.json').read_text()):
        raise ValueError('Previous protocol differs from frozen evidence')
    active = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i|%T'], text=True).splitlines()
    if any(line.split('|')[0] == '686405' for line in active):
        raise RuntimeError('Original job must remain stopped before importing checkpoints')
    if previous['identity_sha'] != p['native_training_identity_sha']:
        raise ValueError('Original training identity changed')
    for key, value in previous.items():
        if key not in ('name', 'identity_sha', 'source_sha256') and p[key] != value:
            raise ValueError('Native experiment changed: ' + key)
    artifacts, completed_main, completed_guides, interrupted = {}, [], [], []

    def link_file(original):
        rel = original.relative_to(source)
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(original)
        if not target.exists():
            # Checkpoint writers use atomic replacement, so later continuation
            # cannot alter the preserved original checkpoint through this link.
            os.link(original, target)
        if digest(target) != expected:
            raise ValueError('Imported state changed: ' + str(rel))
        artifacts[str(rel)] = expected

    paths = [source / 'runs' / f / a for f in ('5pct', 'full') for a in ('dice', 'depth')]
    paths += sorted((source / 'guides').glob('*/*'))
    for directory in paths:
        if not directory.exists():
            continue
        done = directory / 'completion.json'
        if done.exists():
            record = json.loads(done.read_text())
            if record['identity']['protocol_sha256'] != previous['identity_sha']:
                raise ValueError('Unexpected completed training identity')
            for name, expected in record['artifacts'].items():
                if digest(directory / name) != expected:
                    raise ValueError('Saved result changed: ' + str(directory / name))
                link_file(directory / name)
            link_file(done)
            (completed_guides if 'guides' in directory.relative_to(source).parts else completed_main).append(str(directory.relative_to(source)))
        elif (directory / 'latest.pt').exists():
            for name in ('latest.pt', 'best.pt', 'config.json', 'metrics.csv', 'validation_best.json'):
                if (directory / name).exists():
                    link_file(directory / name)
            interrupted.append(str(directory.relative_to(source)))
    if len(completed_main) != 3 or len(completed_guides) != 2 or interrupted != ['runs/full/depth']:
        raise ValueError('Saved state differs from the audited continuation plan')
    for path in (source / 'loss_geometry').glob('*.npz'):
        link_file(path)
    manifest = dict(previous_root=str(source), previous_protocol_sha256=previous['identity_sha'],
                    completed_main_runs=completed_main, completed_guides=completed_guides,
                    interrupted_runs=interrupted, artifacts=artifacts)
    atomic_json(root / 'resume_manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    restore(parser.parse_args().root)
