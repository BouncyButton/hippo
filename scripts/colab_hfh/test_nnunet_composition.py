import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.colab_hfh.nnunet_composition_calibration import (
    ARMS, CAP, derive_specification, make_protocol, select_fixed_weights, check_anchor)
from scripts.colab_hfh.nnunet_composition_campaign import check_campaign_completion, check_prior_attempt, retire
from scripts.colab_hfh.nnunet_composition_worker import install, source
from scripts.colab_hfh.constraint_schedule import training_flops

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'
PARENT = REPORT/'nnunet_long_matrix_v1/dice_bands_edge'
TRIPLE = REPORT/'nnunet_long_matrix_v1/dice_bands_edge_depth'


def read(path):
    return json.loads(path.read_text())


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.parent = read(PARENT/'results/calibration.json')
        self.triple = read(TRIPLE/'results/calibration.json')

    def test_frozen_hypotheses_and_strength(self):
        a = derive_specification(self.parent,self.triple,'attenuated_band_edge')
        b = derive_specification(self.parent,self.triple,'group_band_edge_depth')
        self.assertEqual(a['weights'],{n:self.triple['weights'][n] for n in ('bands','edge')})
        self.assertAlmostEqual(a['expected_envelope']['median'],.0648953022823677)
        self.assertEqual(b['weights']['edge'],self.parent['weights']['edge'])
        self.assertAlmostEqual(b['weights']['bands']/b['weights']['depth'],
                               self.triple['weights']['bands']/self.triple['weights']['depth'])
        self.assertAlmostEqual(b['expected_envelope']['median'],.1)
        self.assertLess(b['expected_envelope']['p95'],.5)
        expected = read(REPORT/'nnunet_research_decision_20261113/combination_evidence.json')['arms']
        for spec,key in ((a,'proposed_attenuated_band_edge'),(b,'proposed_group_calibrated_band_edge_depth')):
            for term,weight in spec['weights'].items():
                self.assertAlmostEqual(weight,expected[key]['weights'][term],places=14)

    def test_protocol_retains_matched_native_and_cap(self):
        for variant in ARMS:
            folder = PARENT if variant == 'attenuated_band_edge' else TRIPLE
            ref = read(folder/'protocol/protocol.json')
            saved = deepcopy(ref)
            spec = derive_specification(self.parent,self.triple,variant)
            p = make_protocol(ref,spec)
            self.assertEqual(ref,saved)
            changed = {k for k in p if p[k] != ref.get(k)}
            self.assertEqual(changed,{'name','composition','constraint_calibration','constraint_definition','endpoint','models'})
            s = p['models']['nnunet']['objectives'][ARMS[variant]]
            self.assertEqual(s['max_steps'],179)
            self.assertLessEqual(training_flops(179,s),CAP)
            self.assertGreater(training_flops(180,s),CAP)
            self.assertEqual(s['calibration_flops'],saved['models']['nnunet']['objectives'][ARMS[variant]]['calibration_flops']+10000000)

    def test_worker_installs_actual_fixed_selector_and_checks_anchor(self):
        file = ROOT/'scripts/cross_dataset_campaign/capped_core.py'
        fn = next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        for variant in ARMS:
            spec = derive_specification(self.parent,self.triple,variant)
            folder = PARENT if variant == 'attenuated_band_edge' else TRIPLE
            p = make_protocol(read(folder/'protocol/protocol.json'),spec)
            ns = {};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(file),'exec'),ns)
            trainer = SimpleNamespace(run=ns['run'],base=SimpleNamespace(),__file__=str(file))
            def original(*args, **kwargs):
                weights,stats = trainer.select_weights(spec['anchor_rows'],spec['objective'])
                return dict(cases=spec['anchor_rows'],weights=weights,statistics=stats,
                            anchor_model_sha256=spec['anchor_model_sha256'])
            trainer.calibrate = original
            install(trainer,p,spec['objective'])
            result = trainer.calibrate()
            self.assertTrue(result['composition_anchor_verified'])
            self.assertEqual(result['weights'],spec['weights'])
            self.assertIn('matched_readiness_anchor',trainer.run.__code__.co_consts)
            self.assertEqual(trainer.base.learning_rate('nnunet',100,179,.1),.07)
            # Later diagnostics must measure the frozen coefficients, not retune.
            later = deepcopy(spec['anchor_rows'])
            for row in later:
                g = np.array(row['gradient_gram']);g[1:,:]*=2;g[:,1:]*=2
                row['gradient_gram']=g.tolist()
            weights,_ = select_fixed_weights(later,spec['objective'],spec)
            self.assertEqual(weights,spec['weights'])
            bad = deepcopy(result);bad['anchor_model_sha256']='0'*64
            with self.assertRaises(ValueError):check_anchor(bad,spec)
        compile(source(),'composition_worker','exec')

    def test_completion_gate_rejects_partial_or_active_campaign(self):
        unetr = dict(status='complete',unetr_target_pairs=9,three_model_verified_pairs=378)
        medsam = dict(status='complete_4_component_constraints',workers=[],results=[
            dict(component=c,arm=a,fold=0,seed=0,updates=125)
            for c in ('noise','gain') for a in ('dice_bands','dice_edge_depth')])
        check_campaign_completion(unetr,medsam)
        for field,value in [('status','running_component_constraints'),('workers',[{'pid':1}]),('results',medsam['results'][:3])]:
            bad = deepcopy(medsam);bad[field]=value
            with self.assertRaises(ValueError):check_campaign_completion(unetr,bad)
        with self.assertRaises(ValueError):check_campaign_completion({**unetr,'unetr_target_pairs':6},medsam)

    def test_retirement_checks_all_files_before_deleting(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            output,durable = Path(temp)/'run',Path(temp)/'archive'
            output.mkdir();durable.mkdir()
            (output/'checkpoint_latest.pt').write_bytes(b'latest')
            (output/'checkpoint_best.pt').write_bytes(b'best')
            done = dict(checkpoint_latest_sha256=hashlib.sha256(b'latest').hexdigest(),checkpoint_best_sha256='0'*64)
            (output/'completion.json').write_text(json.dumps(done))
            with self.assertRaises(ValueError):retire(output,durable)
            self.assertTrue((output/'checkpoint_latest.pt').exists())
            self.assertTrue((output/'checkpoint_best.pt').exists())

    def test_remote_attempt_cannot_be_reset_by_new_vm(self):
        remote = dict(status='running',run_id='first')
        with self.assertRaises(RuntimeError):check_prior_attempt(None,remote)
        with self.assertRaises(RuntimeError):check_prior_attempt(dict(run_id='replacement'),remote)
        check_prior_attempt(dict(run_id='first'),remote)
        check_prior_attempt(None,None)

    def test_launcher_requires_deliberate_start_before_runtime_access(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)/'must_not_exist'
            result = subprocess.run([sys.executable,str(ROOT/'scripts/colab_hfh/launch_nnunet_composition.py'),
                '--package','missing.zip','--sha256','0'*64,'--root',str(destination)],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('No launch without --start',result.stderr)
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
