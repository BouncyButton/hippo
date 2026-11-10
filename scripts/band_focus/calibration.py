"""Check and freeze every arm on the shared training-only warmup before runs."""
import argparse
import json
import math
from pathlib import Path
import torch
from scripts.depth_supervision.utils import atomic_json, digest, restore_rng, seed_all
from . import train
from .geometry import Targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(4)
    seed_all(0)
    p = json.loads((root/'protocol.json').read_text())
    state = torch.load(root/'shared_warmup/latest.pt', map_location='cpu', weights_only=False)
    warmup_sha = digest(root/'shared_warmup/latest.pt')
    device = torch.device('cuda')
    model = train.model_factory(device)
    model.load_state_dict(state['model'])
    patients = train.Patients(train._load_pkl_dataframe(p['pkl']), p['splits']['5pct']['train'], p['native_geometry'])
    summary = []
    for entry in p['runs']:
        restore_rng(state['rng'])
        cache = Targets(root, p, entry)
        result = train.calibrate(model, patients, cache, device, p['calibration'], entry)
        if entry['band']['kind'] == 'canonical':
            expected = p['shared_warmup']['calibration']['coefficients']['bands']
            if not math.isclose(result['coefficients']['bands'], expected, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('Original canonical calibration mismatch: '+entry['id'])
        reference = p.get('reference_controls', {}).get(entry['id'])
        if reference:
            previous = json.loads((Path(p['predecessor_root'])/'runs'/reference/'config.json').read_text())
            expected = previous['calibration']['coefficients']
            if set(result['coefficients']) != set(expected) or any(
                    not math.isclose(result['coefficients'][k],v,rel_tol=1e-6,abs_tol=1e-8)
                    for k,v in expected.items()):
                raise ValueError('Prior-control calibration mismatch: '+entry['id'])
        folder = root/'calibration'/entry['id']
        folder.mkdir(parents=True, exist_ok=True)
        record = dict(identity=train.identity(p, entry), warmup_sha256=warmup_sha,
                      calibration=result, geometry=cache.statistics())
        path = folder/'record.json'
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError('Do not replace prior calibration silently: '+entry['id'])
        atomic_json(path, record)
        row = dict(id=entry['id'], coefficients=result['coefficients'], statistics=result['statistics'])
        summary.append(row)
        print(json.dumps(row), flush=True)
    atomic_json(root/'calibration_preflight.json', dict(passed=True, warmup_sha256=warmup_sha,
        objectives_checked=len(summary), training_cases=len(patients), arms=summary,
        artifacts={e['id']: digest(root/'calibration'/e['id']/'record.json') for e in p['runs']}))


if __name__ == '__main__':
    main()
