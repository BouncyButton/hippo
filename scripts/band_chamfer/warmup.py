"""Copy the verified common prefix without recomputing training or consuming RNG."""
import json
from pathlib import Path
import torch
from scripts.depth_supervision.utils import atomic_json, digest, save_checkpoint
from .train import identity


def migrate(state, p, entry):
    state['identity'] = identity(p, entry)
    if 'history' in state:
        if len(state['history']) != p['calibration']['prefix_epochs'] or state['calibration'] is not None:
            raise ValueError('Warmup must end before all constraints')
        for row in state['history']:
            if any(row[k] != 0 for k in ('constraint_scale', 'train_bands', 'train_edge', 'coefficient_bands', 'coefficient_edge')):
                raise ValueError('Warmup contains constraint training')
            row.pop('train_edge'); row.pop('coefficient_edge')
            row.update(train_pair=0., train_surface=0., coefficient_pair=0., coefficient_surface=0.)
            row.update(train_ap=0., train_ap_ce=0., coefficient_ap=0., coefficient_ap_ce=0.)
    return state


def prepare_prefix(p, root):
    prefix = p['shared_warmup']
    source = Path(prefix['folder'])
    output = root/'shared_warmup'
    marker = output/'provenance.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['source'] != prefix or record['protocol_identity'] != p['identity_sha']:
            raise ValueError('Warmup provenance changed')
        for name, expected in record['artifacts'].items():
            if digest(output/name) != expected:
                raise ValueError('Warmup changed')
        return record
    output.mkdir(exist_ok=True)
    for name, expected in prefix['artifacts'].items():
        if digest(source/name) != expected:
            raise ValueError('Original verified warmup changed: '+name)
    for name in ('latest.pt', 'best.pt'):
        state = torch.load(source/name, map_location='cpu', weights_only=False)
        if state['identity']['protocol_sha256'] != prefix['protocol_identity']:
            raise ValueError('Original warmup identity mismatch')
        save_checkpoint(output/name, migrate(state, p, p['runs'][0]))
    record = dict(source=prefix, protocol_identity=p['identity_sha'],
                  artifacts={n: digest(output/n) for n in ('latest.pt', 'best.pt')})
    atomic_json(marker, record)
    return record


def seed_run(p, root, entry):
    source = root/'shared_warmup'
    record = json.loads((source/'provenance.json').read_text())
    output = root/'runs'/entry['id']
    marker = output/'shared_warmup.json'
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise ValueError('Run prefix changed')
        return
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Refuse to replace a partially seeded run')
    for name, expected in record['artifacts'].items():
        if digest(source/name) != expected:
            raise ValueError('Warmup changed')
        state = torch.load(source/name, map_location='cpu', weights_only=False)
        state['identity'] = identity(p, entry)
        save_checkpoint(output/name, state)
    atomic_json(marker, record)
