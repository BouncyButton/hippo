"""Promotion, offline execution, and protected-recovery regression checks."""
import json
import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import winning_matrix2030 as study,winning_matrix_runner as runner


class WinningMatrixTests(unittest.TestCase):
    def _shutdown_case(self, exit_code):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'source').mkdir();write(p/'control.json',dict(pause=False))
            plan=dict(tasks=[dict(id='pending')],num_runs=1,reused_runs=503,
                graceful_stop_seconds=600,storage=dict(minimum_free_gib=16))
            state={'stopping':False}
            class Child:
                pid=123
                code=None
                def poll(self):return self.code
                def send_signal(self,sig):self.code=exit_code
                def wait(self):return self.code
            with patch.dict(os.environ,{'SLURM_JOB_ID':'test','CUDA_VISIBLE_DEVICES':'0,1'}), \
                 patch.object(study,'verify',return_value=plan),patch.object(runner,'install_signals'), \
                 patch.object(runner,'stopping',side_effect=lambda:state['stopping']), \
                 patch.object(runner,'quota',return_value=dict(free_gib=40)), \
                 patch.object(runner,'start_worker',return_value=Child()), \
                 patch.object(runner.campaign,'command_for',return_value=[]), \
                 patch.object(runner,'retire_recovery'),patch.object(runner.controller,'submit'), \
                 patch.object(runner.controller,'cancel_successor') as cancel, \
                 patch.object(runner.time,'sleep',side_effect=lambda _:state.update(stopping=True)):
                if exit_code==-signal.SIGUSR1:
                    runner.run(p);cancel.assert_not_called()
                    self.assertEqual(read(p/'status.json')['status'],'continuing_at_time_limit')
                    self.assertFalse((p/'failure.json').exists())
                else:
                    with self.assertRaises(RuntimeError):runner.run(p)
                    cancel.assert_called_once()
                    self.assertEqual(read(p/'status.json')['status'],'failed')

    def test_requested_shutdown_keeps_continuation(self):
        self._shutdown_case(-signal.SIGUSR1)

    def test_unrelated_crash_still_fails_and_updates_status(self):
        self._shutdown_case(-signal.SIGSEGV)

    def test_stop_signal_during_worker_import_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'slow_worker.py').write_text(
                "from pathlib import Path\nimport time,sys\n"
                "Path('importing').touch()\ntime.sleep(.5)\n"
                "from scripts.cross_dataset_campaign.common import install_signals,stopping\n"
                "install_signals()\nsys.exit(75 if stopping() else 0)\n")
            env={**os.environ,'PYTHONPATH':str(p)+os.pathsep+os.environ.get('PYTHONPATH','')}
            launch=getattr(runner,'start_worker',subprocess.Popen)
            child=launch([sys.executable,'-u','-m','slow_worker'],cwd=p,env=env)
            try:
                limit=time.monotonic()+10
                while not (p/'importing').exists() and time.monotonic()<limit:time.sleep(.01)
                self.assertTrue((p/'importing').exists())
                child.send_signal(signal.SIGUSR1)
                self.assertEqual(child.wait(timeout=10),75)
            finally:
                if child.poll() is None:child.kill();child.wait()

    def test_stop_signal_immediately_after_spawn_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'quick_worker.py').write_text(
                "import sys\nfrom scripts.cross_dataset_campaign.common import stopping\n"
                "sys.exit(75 if stopping() else 0)\n")
            env={**os.environ,'PYTHONPATH':str(p)+os.pathsep+os.environ.get('PYTHONPATH','')}
            launch=getattr(runner,'start_worker',subprocess.Popen)
            for _ in range(5):
                child=launch([sys.executable,'-u','-m','quick_worker'],cwd=p,env=env)
                child.send_signal(signal.SIGUSR1)
                self.assertEqual(child.wait(timeout=10),75)

    def test_promotion_requires_nine_pairs_and_swin_reference_gain(self):
        rows=[dict(seed=s,fold=f,gain_pp=.2,vs_original_pp=.1,ca4_vs_original_pp=0.,ca4_gain_pp=.1) for s,f in study.PAIRS]
        self.assertEqual(study.promotion_score('nnunet',rows),[.2,.2])
        self.assertEqual(study.promotion_score('swinunetr',rows),[1,0.,.1,.2])
        self.assertIsNone(study.promotion_score('nnunet',rows[:-1]))
        self.assertIsNone(study.promotion_score('nnunet',rows[:-1]+[rows[0]]))
        rows[0]['gain_pp']=0.;self.assertIsNone(study.promotion_score('nnunet',rows));rows[0]['gain_pp']=.2
        rows[0]['vs_original_pp']=-2.;self.assertIsNone(study.promotion_score('swinunetr',rows));rows[0]['vs_original_pp']=.1
        rows[0]['ca4_vs_original_pp']=-.01;self.assertIsNone(study.promotion_score('swinunetr',rows))

    def test_class_consistency_tradeoff_is_explicit_and_requires_all_nine(self):
        rows=[dict(seed=s,fold=f,gain_pp=.5,ca4_gain_pp=.1,vs_original_pp=-.2,ca4_vs_original_pp=.3) for s,f in study.PAIRS]
        self.assertEqual(study.promotion_score('swinunetr',rows),[1,.3,-.2,.5])
        rows[0]['ca4_gain_pp']=0.;self.assertIsNone(study.promotion_score('swinunetr',rows))
        rows[0]['ca4_gain_pp']=.1;rows[0]['gain_pp']=-.01;self.assertIsNone(study.promotion_score('swinunetr',rows))

    def test_confirmation_recomputes_gains_and_includes_reused_pilots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'mni_nnunet_swin_refine2030_round_test';tasks=[];controls=[]
            for i,(seed,fold) in enumerate(sorted(study.PAIRS)):
                identity=f'candidate_f{fold}s{seed}'
                tasks.append(dict(id=identity,model='swinunetr',candidate='early05',protocol='swinunetr_early05',seed=seed,fold=fold))
                d=stage/('reused_pilot_runs' if i<2 else 'runs')/identity
                write(d/'completion.json',dict(status='complete',discarded_attempts=0));write(d/'config.json',dict(budget_flops=study.CAP))
                write(d/'archive_manifest.json',{})
                write(d/'validation_latest.json',dict(summary=dict(macro_hard_dice=.705,CA4_DG_hard_dice=.6)))
                for arm,macro,ca4 in (('dice',.7,.59),('dice_bands',.707,.595)):
                    name=f'{arm}_f{fold}s{seed}';controls.append(dict(id=name,model='swinunetr',seed=seed,fold=fold,arm=arm))
                    c=stage/'controls'/name;write(c/'archive_manifest.json',{});write(c/'validation_latest.json',dict(summary=dict(macro_hard_dice=macro,CA4_DG_hard_dice=ca4)))
            write(stage/'manifest.json',{});write(stage/'controls.json',controls)
            write(stage/'plan.json',dict(tasks=tasks,budget_flops=study.CAP,deadline=study.START))
            with patch.object(study,'REMOTE',root),patch.object(study,'verify_archive'):
                result=study.confirmed_candidates([stage]);winner=result['swinunetr']
                self.assertEqual(len(winner['rows']),9)
                self.assertEqual(sum('reused_pilot_runs' in r['source'] for r in winner['rows']),2)
                write(stage/'runs'/tasks[-1]['id']/'validation_latest.json',dict(summary=dict(macro_hard_dice=.69,CA4_DG_hard_dice=.6)))
                self.assertIsNone(study.confirmed_candidates([stage])['swinunetr'])

    def test_offline_workers_complete_without_fake_archive_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'source').mkdir();write(p/'control.json',dict(pause=False))
            tasks=[dict(id='run'+str(i)) for i in range(3)]
            plan=dict(tasks=tasks,num_runs=3,reused_runs=501,graceful_stop_seconds=600,storage=dict(minimum_free_gib=16))
            def command(stage,plan,task):
                code="import pathlib,json,time; p=pathlib.Path("+repr(str(stage/'runs'/task['id']))+"); time.sleep(.05); (p/'checkpoint_latest.pt').write_bytes(b'retain'); (p/'completion.json').write_text(json.dumps(dict(status='complete',discarded_attempts=0)))"
                return [sys.executable,'-c',code]
            with patch.dict(os.environ,{'SLURM_JOB_ID':'test','CUDA_VISIBLE_DEVICES':'0,1'}), \
                 patch.object(study,'verify',return_value=plan),patch.object(runner,'install_signals'), \
                 patch.object(runner,'stopping',return_value=False),patch.object(runner,'quota',return_value=dict(free_gib=40)), \
                 patch.object(runner.campaign,'command_for',side_effect=command),patch.object(runner,'start_worker',side_effect=subprocess.Popen),patch.object(runner,'prepare'), \
                 patch.object(runner,'retire_recovery'),patch.object(runner.controller,'submit') as submit, \
                 patch.object(runner.controller,'cancel_successor') as cancel,patch.object(runner,'retire') as retire:
                runner.run(p)
                submit.assert_called_once();cancel.assert_called_once();retire.assert_not_called()
            status=read(p/'status.json');self.assertEqual(status['status'],'complete_pending_archive')
            self.assertEqual(status['training_completed'],3)
            for t in tasks:
                d=p/'runs'/t['id'];self.assertTrue((d/'checkpoint_latest.pt').exists())
                self.assertFalse((d/'archive_acknowledgment.json').exists());self.assertFalse((d/'cleanup.json').exists())

    def test_recovery_cleanup_requires_ack_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'new';origin=root/study.PARENT/'runs/cell';origin.mkdir(parents=True)
            write(origin.parent.parent/'control.json',dict(pause=True))
            checkpoint=origin/'checkpoint_latest.pt';checkpoint.write_bytes(b'protected')
            ledger=origin/'budget_ledger.json';write(ledger,dict(charged_attempts=144))
            write(stage/'recovery.json',dict(id='cell',files={f.name:digest(f) for f in (checkpoint,ledger)}))
            with patch.object(runner,'REMOTE',root):
                runner.retire_recovery(stage);self.assertTrue(checkpoint.exists())
                d=stage/'runs/cell';write(d/'cleanup.json',dict(status='complete'))
                with self.assertRaises(FileNotFoundError):runner.retire_recovery(stage)
                write(d/'archive_acknowledgment.json',dict(status='verified'))
                checkpoint.write_bytes(b'changed')
                with self.assertRaises(AssertionError):runner.retire_recovery(stage)
                self.assertTrue(checkpoint.exists());checkpoint.write_bytes(b'protected')
                runner.retire_recovery(stage)
                self.assertFalse(checkpoint.exists());self.assertTrue(ledger.exists())
                self.assertTrue((stage/'recovery_resolved.json').exists())


if __name__=='__main__':unittest.main()
