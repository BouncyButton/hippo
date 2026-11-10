"""Run real 64-cube CUDA forward/backward checks inside the Slurm allocation."""
import argparse
import json
from pathlib import Path
import time
import monai
import torch

from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from .losses import ARMS, combined_loss, components
from .train import GeometryCache, Patients, autocast, model_factory, verify_protocol
from .utils import atomic_json, digest, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    verify_protocol(p, Path(__file__).resolve().parents[2])
    assert monai.__version__ == p['required_monai']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert digest(args.pkl) == p['pkl_sha256']
    torch.set_num_threads(4)
    seed_all(0)
    patients = Patients(_load_pkl_dataframe(args.pkl), p['splits']['5pct']['train'][:2], p['native_geometry'])
    cache = GeometryCache(args.root/'geometry_cache', p['geometry'])
    device = torch.device('cuda')
    model = model_factory(device)
    assert sum(v.numel() for v in model.parameters()) == p['model_parameters']
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
    records = []
    for arm in ARMS:
        item = patients[0]
        geometry = cache.get(item)
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        with autocast(device):
            logits = model(item['image'][None].to(device))
        values = components(logits, item['label'][None].to(device), geometry, arm)
        loss = combined_loss(values, {'depth': .1}, 1.)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.stack([torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None]).all()
        optimizer.step()
        torch.cuda.synchronize()
        records.append(dict(arm=arm, components={k: float(v.detach()) for k,v in values.items()},
                            seconds=time.monotonic()-start))
    # This discarded smoke model never supplies production weights.
    for item in patients.items:
        g = cache.get(item)
        assert abs(float(g['weights'].sum())-1) < 1e-6
    atomic_json(args.root/'preflight.json', dict(passed=True, torch=torch.__version__, monai=monai.__version__,
                gpu=torch.cuda.get_device_name(0), records=records, training_cases=[x['case_name'] for x in patients.items],
                validation_cases_used=0, protocol_sha256=digest(args.protocol)))
    print('CUDA preflight passed', flush=True)


if __name__ == '__main__':
    main()
