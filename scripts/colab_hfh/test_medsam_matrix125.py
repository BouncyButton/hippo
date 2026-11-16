"""Full factorial coverage, recipe preservation and resume checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.colab_hfh.medsam_matrix125_campaign import ALL_ARMS,TERMS,RECIPE,Matrix125,matrix_phase_source
from scripts.colab_hfh.medsam_matrix125_worker import worker_source


class MatrixTests(unittest.TestCase):
    def test_full_factorial(self):
        self.assertEqual(len(ALL_ARMS),16)
        self.assertEqual(len(set(ALL_ARMS)),16)
        self.assertEqual({frozenset(a.split('_')[1:]) for a in ALL_ARMS},
                         {frozenset(t for i,t in enumerate(TERMS) if mask&(1<<i)) for mask in range(16)})

    def test_original_worker_and_no_negative_gate(self):
        source=worker_source();compile(source,'worker','exec')
        self.assertNotIn('install(original,',source)
        self.assertIn("'dice_edge', 'dice_edge_sdf'",source)
        self.assertIn('install_spatial_evaluation(matched)',source)
        phase=matrix_phase_source();compile(phase,'scheduler','exec')
        self.assertIn("fail_fast=RECIPE,'full16','results',False",phase)
        self.assertIn("self.prepare(RECIPE,arm,0,0)",phase)

    def test_resume_and_frozen_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'medsam_constraints_strict_v1/protocols';base.mkdir(parents=True)
            reference=dict(updates=200,seed=0,fold=0,frozen_learning_rates=[.001]*200,
                           frozen_sample_schedule=[['case',0]]*400,gradient_ratio_target=.1)
            (base/'strict_native__dice__f0_s0.json').write_text(json.dumps(reference))
            control=dict(archive_sha256='verified-native',protocol=dict(reference,updates=125,frozen_sample_schedule=[['case',0]]*250))
            obj=SimpleNamespace(root=root,work=root/'matrix',durable=root/'durable',controls={(0,0):control},state={'geometry':{'tensor_spacing_mm':[2,1,1]}})
            for arm in ALL_ARMS[1:]:
                first=Matrix125.prepare(obj,RECIPE,arm,0,0)
                second=Matrix125.prepare(obj,RECIPE,arm,0,0)
                self.assertEqual(first['protocol'],second['protocol'])
                p=first['protocol'];self.assertEqual(p['updates'],125);self.assertEqual(p['learning_rate_horizon_updates'],200)
                self.assertEqual(p['frozen_learning_rates'],control['protocol']['frozen_learning_rates'])
                self.assertEqual(p['frozen_sample_schedule'],control['protocol']['frozen_sample_schedule'])
                for term in arm.split('_')[1:]:self.assertEqual(p['gradient_ratio_targets'][term],.01 if term=='bands' else .1)
            with self.assertRaises(ValueError):Matrix125.prepare(obj,RECIPE,'dice_sdf',1,0)


if __name__=='__main__':unittest.main()
