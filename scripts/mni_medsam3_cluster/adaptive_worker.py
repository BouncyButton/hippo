"""Resume a verified milestone, keeping optimizer/RNG/history; run the native trainer."""
import argparse
from pathlib import Path
import sys

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_medsam3_cluster import worker


def migrate(parent, output, protocol_path):
    import torch
    target = {**read(protocol_path), 'arm': 'dice'}
    old = read(parent/'protocol.json')
    allowed = {'updates', 'name', 'sampling', 'decision_rule'}
    if set(old) != set(target) or any(old[k] != target[k] for k in old if k not in allowed):
        raise ValueError('Continuation changes scientific inputs beyond the update horizon')
    if target['updates'] != old['updates'] + 100:
        raise ValueError('Only a100-update continuation is authorized')
    done = read(parent/'completion.json')
    checkpoint = parent/'checkpoint_latest.pt'
    source_sha = digest(checkpoint)
    if source_sha != done['checkpoint_sha256'] or done['updates'] != old['updates']:
        raise ValueError('Parent checkpoint is not the evaluated milestone')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['step'] != old['updates'] or len(state['history']) != state['step'] or state['protocol_sha256'] != digest(parent/'protocol.json'):
        raise ValueError('Invalid parent checkpoint history/protocol')
    output.mkdir(parents=True, exist_ok=True)
    write(output/'protocol.json', target)
    state['protocol_sha256'] = digest(output/'protocol.json')
    worker.copy_verified(parent/'calibration.json', output/'calibration.json')
    temporary = output/'checkpoint_migrating.tmp'
    torch.save(state, temporary)
    check = torch.load(temporary, map_location='cpu', weights_only=False)
    if check['step'] != state['step'] or check['protocol_sha256'] != state['protocol_sha256']:
        raise ValueError('Continuation checkpoint verification failed')
    for name, value in state['adapter'].items():
        if not torch.equal(value, check['adapter'][name]):
            raise ValueError('Adapter changed during continuation transfer')
    if digest(checkpoint) != source_sha:
        raise ValueError('Parent changed during continuation')
    temporary.replace(output/'checkpoint_latest.pt')
    write(output/'training_history.json', state['history'])
    write(output/'resume_provenance.json', dict(status='verified', parent_path=str(parent),
        parent_checkpoint_sha256=source_sha, resumed_at_update=state['step'],
        checkpoint_sha256=digest(output/'checkpoint_latest.pt'), protocol_sha256=state['protocol_sha256'],
        optimizer_and_rng_retained=True, history_replayed=False,
        learning_rate_policy='Keep recorded historical learning rates; use original cosine function with expanded horizon for the next100 updates.'))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--resume-parent', type=Path)
    args, remaining = parser.parse_known_args()
    inputs = argparse.ArgumentParser(add_help=False)
    inputs.add_argument('--output', type=Path, required=True)
    inputs.add_argument('--protocol', type=Path, required=True)
    known, _ = inputs.parse_known_args(remaining)
    if args.resume_parent and not (known.output/'checkpoint_latest.pt').exists():
        migrate(args.resume_parent, known.output, known.protocol)
    sys.argv = [sys.argv[0], *remaining]
    try:
        worker.main()
    except SystemExit as exc:
        if exc.code == 75:
            write(known.output/'requested_stop.json', dict(status='checkpoint_saved',
                checkpoint_sha256=digest(known.output/'checkpoint_latest.pt'),
                updates=len(read(known.output/'training_history.json'))))
        raise


if __name__ == '__main__':
    main()
