import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.colab_hfh.prepare_bands import build,EXPORT,RATES
from scripts.colab_hfh.bands_worker import worker_source
from scripts.colab_hfh.bands_archive import paired_result


class BandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload=build()

    def test_complete_matched_matrix_and_budget(self):
        payload=self.payload
        self.assertEqual(set(payload['references']),{f'{m}_f{f}_s{s}' for m in RATES for f in range(3) for s in range(3)})
        for model,files in payload['protocols'].items():
            p=files['protocol.json'];alias=p['trainer_alias'];control=p['models'][alias]['objectives']['dice'];band=p['models'][alias]['objectives']['dice_bands']
            self.assertEqual(control['max_steps'],band['max_steps'])
            self.assertEqual(p['initial_lrs'][alias],RATES[model])
            self.assertLessEqual(band['consumed_training_flops'],p['budget_flops'])
            self.assertEqual(band['consumed_total_flops'],band['consumed_training_flops']+band['calibration_flops'])
            for fold in range(3):
                split=json.loads(files['splits_10patients.json'])[fold]
                self.assertEqual(len(split['train']),10);self.assertFalse(set(split['train'])&set(split['val']))
                for seed in range(3):
                    c=payload['references'][f'{model}_f{fold}_s{seed}']['configuration']
                    self.assertEqual(c['train_cases'],split['train']);self.assertEqual(c['validation_cases'],split['val'])
                    self.assertEqual(c['lr_horizon_updates'],p['lr_horizon_updates'])

    def test_worker_preserves_native_optimizer_and_checks_initialization(self):
        source=worker_source();ast.parse(source)
        self.assertIn('exp.base.build_loss_and_optimizer = exp.ORIGINAL_LOSS_OPTIMIZER',source)
        self.assertIn("exp.configure(p, 'dice_bands')",source)
        self.assertIn('BAND initialization differs from matched baseline',source)

    def fixture(self,folder):
        reference=deepcopy(self.payload['references']['nnunet_f0_s0'])
        config=reference['configuration'];config['objective']='dice_bands'
        p=self.payload['protocols']['nnunet']['protocol.json']
        files={'config.json':config,'initialization.json':reference['initialization'],
            'completion.json':dict(objective='dice_bands',discarded_attempts=0,updates=90,consumed_training_flops=1,calibration_flops=1),
            'calibration.json':dict(status='complete',weights={'bands':1.},cases=[{'case_name':c} for c in config['train_cases']],target_ratio=p['calibration_target_ratio'],p95_cap=p['calibration_p95_cap']),
            'validation_latest.json':deepcopy(reference['validation']),'training_final.json':reference['training']}
        files['validation_latest.json']['summary']['macro_hard_dice']=.4
        for name,value in files.items():(folder/name).write_text(json.dumps(value))
        return reference,p

    def test_regressions_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);reference,p=self.fixture(folder)
            result=paired_result(folder,reference,p)
            self.assertFalse(result['improved']);self.assertLess(result['delta_dice_pp'],0)

    def test_validation_cases_cannot_enter_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);reference,p=self.fixture(folder)
            path=folder/'calibration.json';value=json.loads(path.read_text());value['cases'][0]['case_name']=reference['configuration']['validation_cases'][0];path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'frozen training cases'):paired_result(folder,reference,p)

    def test_mismatched_initialization_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);reference,p=self.fixture(folder)
            (folder/'initialization.json').write_text('{"model_state_sha256":"wrong"}')
            with self.assertRaisesRegex(ValueError,'Initialization'):paired_result(folder,reference,p)


if __name__=='__main__':unittest.main()
