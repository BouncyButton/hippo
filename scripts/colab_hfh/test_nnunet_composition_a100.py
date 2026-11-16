from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.colab_hfh import nnunet_composition_calibration as calibration
from scripts.colab_hfh.nnunet_composition_a100_campaign import prepare_protocols
from scripts.colab_hfh.nnunet_composition_a100_worker import native_anchor, source

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'reports/hfh_colab_exploration_20261111/nnunet_composition_v1/package'


class A100Tests(unittest.TestCase):
    def test_control_protocols_keep_native_scientific_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_protocols(root,PACKAGE,True)
            for name in ('native','band_edge','band_edge_depth'):
                before = json.loads((PACKAGE/'controls'/name/'protocol/protocol.json').read_text())
                after = json.loads((root/'protocols'/name/'protocol.json').read_text())
                self.assertEqual({k:v for k,v in after.items() if k not in ('a100_native_anchor','hardware_campaign')},before)

    def test_fresh_anchor_is_required_and_uses_same_coefficient_rule(self):
        parent = json.loads((PACKAGE/'controls/band_edge/results/calibration.json').read_text())
        triple = json.loads((PACKAGE/'controls/band_edge_depth/results/calibration.json').read_text())
        original = deepcopy(calibration.ANCHOR)
        new = dict(after_update=60,model_state_sha256='a'*64)
        try:
            calibration.ANCHOR = new
            with self.assertRaises(ValueError):calibration.derive_specification(parent,triple,'attenuated_band_edge')
            parent['anchor_model_sha256']=triple['anchor_model_sha256']=new['model_state_sha256']
            a = calibration.derive_specification(parent,triple,'attenuated_band_edge')
            b = calibration.derive_specification(parent,triple,'group_band_edge_depth')
            self.assertEqual(a['weights'],{k:triple['weights'][k] for k in ('bands','edge')})
            self.assertEqual(b['weights']['edge'],parent['weights']['edge'])
            self.assertAlmostEqual(b['expected_envelope']['median'],.1)
            self.assertEqual(b['anchor_model_sha256'],new['model_state_sha256'])
        finally:
            calibration.ANCHOR = original

    def test_anchor_requires_exact_update_and_generated_workers_compile(self):
        for is_native in (False,True):
            value=source(is_native)
            compile(value,'a100_worker','exec')
            self.assertIn('nnunet_composition_a100_worker',value)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'anchor.json'
            path.write_text(json.dumps(dict(after_update=59,model_state_sha256='a'*64)))
            with self.assertRaises(ValueError):native_anchor(dict(a100_native_anchor=str(path)))
            path.write_text(json.dumps(dict(after_update=60,model_state_sha256='a'*64)))
            self.assertEqual(native_anchor(dict(a100_native_anchor=str(path)))['after_update'],60)


if __name__=='__main__':unittest.main()
