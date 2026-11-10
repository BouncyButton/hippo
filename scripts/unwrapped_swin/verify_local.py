"""Validate cut-shell geometry against real predicted masks before submission."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients
from scripts.depth_supervision.utils import atomic_json, digest
from .cache import field_of_view
from .geometry import build_atlas
from .test_pipeline import GeometryTests, real_model_smoke


def main():
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / 'docs/experiments/unwrapped_swin_20261006/protocol.json'
    p = json.loads(protocol_path.read_text())
    out = root / 'evaluation_output/unwrapped_swin_campaign'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests))
    assert tests.wasSuccessful()
    paths = {
        'full': root / 'evaluation_output/unwrapped_input_prototype/source/validation_predictions_full_depth.npz',
    }
    patients = Patients(_load_pkl_dataframe(str(root / 'datasets/Dataset101_MSD/msd_hippocampus_full.pkl')),
                        p['splits']['full']['val'], p['native_geometry'])
    reports, failures, example = [], [], None
    for source, path in paths.items():
        with np.load(path) as predictions:
            for item in patients.items:
                name = item['case_name']
                try:
                    image = item['image'][0].numpy()
                    fov = field_of_view(image.shape, p['native_geometry'][name]['native_shape'])
                    atlas, audit = build_atlas(image, predictions[name], item['affine'], fov)
                    assert all(c['return_support_fraction'] >= .999 for c in audit['charts_audit'])
                    assert all(c['flipped_triangles'] == 0 for c in audit['surface'])
                    if name == 'hippocampus_164':
                        example = atlas
                        np.savez_compressed(out / 'atlas164.npz', **atlas)
                        atomic_json(out / 'atlas164.json', audit)
                    reports.append(dict(case=name, source=source, **audit))
                    print(json.dumps(dict(case=name, charts=audit['charts'], checked=len(reports))), flush=True)
                except Exception as error:
                    failures.append(dict(case=name, source=source, error=repr(error)))
                    print(json.dumps(failures[-1]), flush=True)
    atomic_json(out / 'geometry_checks.json', dict(reports=reports, failures=failures))
    if failures:
        raise RuntimeError('Real-mask geometry failures; do not submit')
    model_check = real_model_smoke(example)
    atomic_json(out / 'model_smoke.json', model_check)
    training_check = json.loads((out / 'training_smoke.json').read_text())
    assert training_check['passed'] and training_check['code_sha256'] == digest(root / 'scripts/unwrapped_swin/train.py')
    record = dict(passed=True, protocol_sha256=digest(protocol_path), unit_tests=tests.testsRun,
                  source_sha256=p['source_sha256'], real_masks=len(reports), geometry_sha256=digest(out / 'geometry_checks.json'),
                  model=model_check, training=training_check,
                  note='Local MONAI 1.4 CPU; MONAI 1.6 CUDA preflight additionally required inside submitted job.')
    atomic_json(out / 'local_checks.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
