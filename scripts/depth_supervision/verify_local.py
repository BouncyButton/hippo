"""Verify geometry on all MSD cases and one real SwinUNETR backward step."""
import contextlib
import io
import json
from pathlib import Path
import time
import unittest

import numpy as np
import torch

from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from . import test_depth
from .geometry import GeometryConfig, build_geometry
from .losses import combined_loss, components
from .train import Patients, model_factory, verify_protocol
from .utils import atomic_json, digest, seed_all

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(4)
    output = ROOT/'evaluation_output/depth_swinunetr_20261006'
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT/'docs/experiments/depth_swinunetr_20261006/protocol.json'
    protocol = json.loads(path.read_text())
    verify_protocol(protocol, ROOT)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_depth))
    (output/'unit_checks.log').write_text(log.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError('Unit checks failed; see unit_checks.log')
    data = _load_pkl_dataframe(ROOT/'datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
    names = protocol['splits']['full']['train'] + protocol['splits']['full']['val']
    patients = Patients(data, names, protocol['native_geometry'])
    records = []
    first_geometry = None
    start = time.monotonic()
    for i, item in enumerate(patients.items):
        labels = item['label'][0].numpy()
        geometry = build_geometry(labels, item['affine'], GeometryConfig(**protocol['geometry']))
        if i == 0:
            first_geometry = geometry
        for key in ('sdf_3d', 'sdf_layer'):
            for c in (1, 2):
                assert (geometry[key][c-1][labels == c] < 0).all()
                assert (geometry[key][c-1][labels != c] > 0).all()
        records.append(dict(case_name=item['case_name'], bins=len(geometry['bin_values']),
            roi_voxels=int(geometry['roi_voxels']), layer_grid_shape=geometry['layer_grid_shape'].tolist(),
            axis_gap=float(geometry['principal_axis_gap']),
            raster_sign_disagreement=geometry['layer_raster_sign_disagreement'].tolist()))
        if (i+1) % 32 == 0:
            print(json.dumps(dict(geometry_checked=i+1, total=len(patients), seconds=time.monotonic()-start)), flush=True)
    atomic_json(output/'geometry_audit.json', dict(cases=records, all_cases_passed=True, seconds=time.monotonic()-start))
    seed_all(0)
    model = model_factory(torch.device('cpu'))
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == protocol['model_parameters']
    item = patients[0]
    values = components(model(item['image'][None]), item['label'][None], first_geometry, 'depth_sdf_layer')
    loss = combined_loss(values, {'depth': .1, 'sdf': .1})
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None) > 0
    checks = dict(passed=True, protocol_sha256=digest(path), unit_tests=result.testsRun, geometry_cases=len(records),
                  real_swin_backward=True, model_parameters=parameters,
                  model_check_training_case=item['case_name'],
                  validation_used_for_training_or_calibration=False,
                  geometry_audit_sha256=digest(output/'geometry_audit.json'),
                  unit_log_sha256=digest(output/'unit_checks.log'))
    atomic_json(output/'local_checks.json', checks)
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
