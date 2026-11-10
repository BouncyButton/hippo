"""Regression: consecutive conditions and resume must use their own paths."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

sys.dont_write_bytecode = True
WORKER_PATH = Path(__file__).with_name('capped_replication_worker.py')


class ReplicationPathTest(unittest.TestCase):
    def test_two_consecutive_conditions_and_resume(self):
        spec = importlib.util.spec_from_file_location('path_test_worker', WORKER_PATH)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        entries = [dict(model=m, objective=a, fold=0, seed=1, trainer_alias='unet', sdf_policy='not_applicable')
                   for m,a in [('nnunet','dice'),('nnunet','dice_bands'),('basic_unet16','dice')]]
        entries[-1].update(fold=2,seed=2)
        plan = dict(runs=entries, reused_runs=[], runtime_versions={'torch':'fixture','monai':'fixture'})
        trained = []
        job_ids = []
        def fake_train(command, **kwargs):
            flags = dict(zip(command[3::2], command[4::2]))
            output = Path(flags['--output']); output.mkdir(parents=True)
            trained.append(flags['--objective'])
            validation = dict(cases=[dict(anterior_hard_dice=.7,posterior_hard_dice=.6) for _ in range(52)],
                              summary=dict(macro_hard_dice=.65))
            for name in ('validation_best.json','validation_latest.json'):
                (output/name).write_text(json.dumps(validation))
            (output/'config.json').write_text('{}')
            for name in ('checkpoint_best.pt','checkpoint_latest.pt'):
                (output/name).write_bytes(b'fixture checkpoint')
        def fake_verify(scratch, model, arm, directory, fold, seed):
            if len(trained) == 2:
                worker.STOP = True
            return dict(model=model,objective=arm,fold=fold,seed=seed,updates=90,epochs=9.,
                        final_dice=.65,best_dice=.65,anterior_dice=.7,posterior_dice=.6,
                        optimization_gflops=31976.45,calibration_gflops=0.)
        def fake_checkpoint_proofs(output,row):
            return {name:dict(file_sha256=worker.digest(output/name),configuration_verified=True,finite_tensors=True)
                    for name in ('checkpoint_best.pt','checkpoint_latest.pt')}
        modules = {'torch':types.SimpleNamespace(__version__='fixture'),
                   'monai':types.SimpleNamespace(__version__='fixture'),
                   'scripts.capped_replication_verify':types.SimpleNamespace(verify_run=fake_verify)}
        with tempfile.TemporaryDirectory(prefix='hippo-path-regression-') as tmp:
            stage=Path(tmp)/'stage'; source=stage/'source'; root=Path(tmp)/'results'
            directory=source/'experiments'/worker.CAMPAIGN; directory.mkdir(parents=True)
            (directory/'plan.json').write_text(json.dumps(plan));(stage/'source_manifest.json').write_text('{}')
            try:
                for run_number in (1,2):
                    job_id='regression-'+uuid.uuid4().hex;job_ids.append(job_id)
                    worker.STOP=False
                    args=['worker','--stage',str(stage),'--root',str(root),'--pkl',str(Path(tmp)/'fixture.pkl')]
                    with patch.object(worker,'check_frozen',return_value=plan), \
                         patch.object(worker.os,'getuid',return_value=1395), \
                         patch.object(worker.subprocess,'run',side_effect=fake_train), \
                         patch.object(worker.subprocess,'check_output',return_value='user pool_home 1.0GiB/93.13GiB'), \
                         patch.object(worker.shutil,'disk_usage',return_value=types.SimpleNamespace(free=20*1024**3)), \
                         patch.object(worker,'verify_checkpoints',side_effect=fake_checkpoint_proofs), \
                         patch.dict(sys.modules,modules), patch.dict(worker.os.environ,{'SLURM_JOB_ID':job_id}), \
                         patch.object(sys,'argv',args):
                        worker.main()
                    if run_number == 1:
                        for entry in entries[:2]:
                            destination=root/entry['model']/entry['objective']/'fold0/seed1/run'
                            row=worker.verify_stored(destination)
                            self.assertEqual((row['model'],row['objective']), (entry['model'],entry['objective']))
                        self.assertFalse((root/'basic_unet16/dice/fold0/seed1/run').exists())
                self.assertEqual(trained,['dice','dice_bands','dice'])
                self.assertEqual(len(json.loads((root/'summary.json').read_text())['runs']),3)
            finally:
                for job_id in job_ids:
                    path=Path('/tmp')/f'hippo-replication-1395-{job_id}'
                    if path.exists():shutil.rmtree(path)


if __name__=='__main__':
    if len(sys.argv) > 1:
        WORKER_PATH = Path(sys.argv[1])
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReplicationPathTest)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
