"""Check real-data weight conservation and both SwinUNETR backward passes."""
import contextlib
import io
import json
from pathlib import Path
import time
import unittest

import numpy as np
import torch

from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from . import test_variants
from .geometry import GeometryConfig, build_geometry
from .losses import ARMS, partial_class_weights, components, combined_loss
from .train import Patients, model_factory, verify_protocol
from .utils import atomic_json, digest, seed_all

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(4)
    out = ROOT/'evaluation_output/depth_weight_variants_20261006'
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT/'docs/experiments/depth_weight_variants_20261006/protocol.json'
    p = json.loads(path.read_text())
    verify_protocol(p, ROOT)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_variants))
    (out/'unit_checks.log').write_text(stream.getvalue())
    if not tests.wasSuccessful():
        raise RuntimeError('Unit checks failed; see unit_checks.log')
    data_path = ROOT/'datasets/Dataset101_MSD/msd_hippocampus_full.pkl'
    assert digest(data_path) == p['local_pkl_sha256']
    data = _load_pkl_dataframe(data_path)
    names = p['splits']['full']['train'] + p['splits']['full']['val']
    patients = Patients(data, names, p['native_geometry'])
    records = []
    start = time.monotonic()
    first = None
    for i, item in enumerate(patients.items):
        y = item['label'][0].numpy()
        g = build_geometry(y, item['affine'], GeometryConfig(**p['geometry']), fields=False)
        w = partial_class_weights(y, g)
        assert np.isfinite(w).all() and np.isclose(w.sum(), 1.)
        assert (w[g['weights'] == 0] == 0).all()
        roi = g['weights'] > 0
        max_ratio = float((w[roi] / g['weights'][roi]).max())
        assert max_ratio <= p['weight_variants']['class_boost_cap'] + 1e-5
        max_mass_error = max(abs(float(w[g['bin_ids'] == b].sum())-float(g['weights'][g['bin_ids'] == b].sum())) for b in g['bin_values'])
        assert max_mass_error < 1e-6
        records.append(dict(case_name=item['case_name'], max_weight_ratio=max_ratio,
            max_band_mass_error=max_mass_error, foreground_weight_before=float(g['weights'][y>0].sum()),
            foreground_weight_after=float(w[y>0].sum())))
        if i == 0:
            first = dict(g, class_weights=w)
        if (i+1) % 52 == 0:
            print(json.dumps(dict(checked=i+1, total=len(names), seconds=time.monotonic()-start)), flush=True)
    atomic_json(out/'weight_audit.json', dict(passed=True, cases=records))
    seed_all(0)
    model = model_factory(torch.device('cpu'))
    assert sum(v.numel() for v in model.parameters()) == p['model_parameters']
    item = patients[0]
    smoke = []
    for arm in ARMS:
        model.zero_grad(set_to_none=True)
        values = components(model(item['image'][None]), item['label'][None], first, arm)
        loss = combined_loss(values, {'depth':.1})
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        assert sum(float(v.grad.square().sum()) for v in model.parameters() if v.grad is not None) > 0
        smoke.append(dict(arm=arm, loss=float(loss.detach()), components=list(values)))
        print(json.dumps(smoke[-1]), flush=True)
    result = dict(passed=True, protocol_sha256=digest(path), unit_tests=tests.testsRun,
        geometry_cases=len(records), real_swin_backward=smoke, model_parameters=p['model_parameters'],
        model_check_training_case=item['case_name'], validation_used_for_training_or_calibration=False,
        weight_audit_sha256=digest(out/'weight_audit.json'), unit_log_sha256=digest(out/'unit_checks.log'))
    atomic_json(out/'local_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
