import ast
import json
from pathlib import Path
import tempfile
import zipfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.colab_hfh.constraint_diagnostics import audit, fixed_weight_statistics


class GradientDiagnosticTests(unittest.TestCase):
    def row(self,native=1.):
        return dict(gradient_names=['native','bands'],gradient_gram=[[native,0.],[0.,1.]],case_name='training_case')

    def test_fixed_coefficients_detect_drift(self):
        anchor=fixed_weight_statistics([self.row()],'dice_bands',{'bands':.1})
        final=fixed_weight_statistics([self.row(.01)],'dice_bands',{'bands':.1})
        self.assertAlmostEqual(anchor['joint']['median'],.1)
        self.assertAlmostEqual(final['joint']['median'],1.)

    def test_joint_gradient_accounts_for_cancellation(self):
        row=dict(gradient_names=['native','bands','edge'],
                 gradient_gram=[[1,0,0],[0,1,-1],[0,-1,1]])
        result=fixed_weight_statistics([row],'dice_bands_edge',{'bands':1.,'edge':1.})
        self.assertEqual(result['joint']['median'],0.)
        self.assertEqual(result['bands']['median'],1.)

    def test_probe_keeps_original_coefficients_and_separate_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            initial=dict(weights={'bands':.1},cases=[self.row()],anchor_model_sha256='before')
            (output/'calibration.json').write_text(json.dumps(initial))
            def calibrate(*args):
                return dict(weights={'bands':.01},cases=[self.row(.01)],anchor_model_sha256='after')
            audit(calibrate,None,[],'dice_bands',None,False,None,{},output,{'calibration_flops':100})
            result=json.loads((output/'gradient_strength_final.json').read_text())
            self.assertEqual(result['weights'],initial['weights'])
            self.assertEqual(result['optimization_updates'],0)
            self.assertEqual(result['diagnostic_flops_allowance'],200)
            self.assertAlmostEqual(result['joint_median_change_factor'],10.)

    def test_final_probe_integration_compiles(self):
        from scripts.colab_hfh.constraint_worker import install
        path=Path(__file__).resolve().parents[1]/'cross_dataset_campaign/capped_core.py'
        source=path.read_text();tree=ast.parse(source)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        run_source=ast.get_source_segment(source,function)
        trainer=SimpleNamespace(run=lambda:None,calibrate=lambda *a,**k:{},__file__=str(path))
        protocol=dict(constraint_component_targets={'bands':.1},calibration_p95_cap=.5,
                      constraint_joint_target=.1,constraint_joint_cap=.5)
        with patch('scripts.colab_hfh.constraint_geometry.install'),\
             patch('scripts.colab_hfh.constraint_worker.inspect.getsource',return_value=run_source):
            install(trainer,protocol,'dice_bands')
        self.assertIn('audit',trainer.run.__code__.co_names)

    def test_archive_keeps_diagnostic_cost_separate_and_checks_training_cases(self):
        from scripts.colab_hfh.constraint_archive import paired_result
        export=Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'
        references=json.loads((export/'bands_v1_payload.json').read_text())['references']
        path=export/'constraints_analysis/nnunet__dice_depth__r00/nnunet_f0_s1.zip'
        with tempfile.TemporaryDirectory() as directory,zipfile.ZipFile(path) as zipped:
            output=Path(directory)
            for name in zipped.namelist():
                if name.startswith('results/') and name.endswith('.json'):
                    (output/Path(name).name).write_bytes(zipped.read(name))
            protocol=json.loads(zipped.read('protocol/protocol.json'))
            initial=json.loads((output/'calibration.json').read_text())
            original=paired_result(output,references['nnunet_f0_s1'],protocol)
            arm=protocol['active_objective'];schedule=protocol['models'][protocol['trainer_alias']]['objectives'][arm]
            stats=fixed_weight_statistics(initial['cases'],arm,initial['weights'])
            report=dict(status='complete',optimization_updates=0,diagnostic_flops_allowance=2*schedule['calibration_flops'],
                weights=initial['weights'],anchor_model_sha256=initial['anchor_model_sha256'],
                cases=initial['cases'],anchor=stats,final=stats)
            diagnostic=output/'gradient_strength_final.json';diagnostic.write_text(json.dumps(report))
            result=paired_result(output,references['nnunet_f0_s1'],protocol)
            self.assertEqual(result['optimization_flops'],original['optimization_flops'])
            self.assertGreater(result['post_training_diagnostic_flops_allowance'],0)
            report['cases'][0]['case_name']='not_a_training_case';diagnostic.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                paired_result(output,references['nnunet_f0_s1'],protocol)


if __name__=='__main__':unittest.main()
