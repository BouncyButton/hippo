"""Preservation, adaptive scheduling and real-process stop/resume checks."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_medsam3 import parallel as p, runtime as rt
from scripts.colab_medsam3.prepare_parallel import notebook
from scripts.colab_medsam3.stage_parallel import stage_parallel


def gpu(**extra):
    return dict(total_gib=80., used_gib=15., utilization=35., processes={},
                host_free_gib=150., cpu_workers=6, **extra)


def worker(i, step=1, phase='training'):
    return p.Worker(dict(id=str(i), arm='baseline'), SimpleNamespace(pid=i), None, -1000,
                    dict(step=0, volumes=0), peak_gib=12., state=dict(step=step, volumes=0, phase=phase))


class TuningTests(unittest.TestCase):
    def test_uses_more_than_two_when_throughput_improves(self):
        tuner = p.AdaptiveLimit(window=180)
        workers = [worker(1)]
        self.assertIsNone(tuner.observe(0, gpu(), workers))
        workers[0].state['step'] = 21
        self.assertIn('trying 2', tuner.observe(180, gpu(), workers))
        workers.append(worker(2))
        tuner.observe(181, gpu(), workers)
        for w in workers: w.state['step'] += 20
        self.assertIn('trying 3', tuner.observe(361, gpu(), workers))
        workers.append(worker(3))
        tuner.observe(362, gpu(), workers)
        for w in workers: w.state['step'] += 20
        self.assertIn('trying 4', tuner.observe(542, gpu(), workers))
        self.assertEqual(tuner.target, 4)

    def test_small_gain_reduces_target_without_stopping_workers(self):
        tuner = p.AdaptiveLimit(window=180)
        workers = [worker(1)]
        tuner.observe(0, gpu(), workers)
        workers[0].state['step'] += 30
        tuner.observe(180, gpu(), workers)
        workers.append(worker(2))
        tuner.observe(181, gpu(), workers)
        for w in workers: w.state['step'] += 15
        self.assertIn('returning to 1', tuner.observe(361, gpu(), workers))
        self.assertEqual(len(workers), 2)
        self.assertEqual(tuner.target, 1)
        # A different phase can still explore additional workers.
        workers = [worker(1, step=200, phase='evaluating')]
        tuner.observe(362, gpu(), workers)
        workers[0].state['volumes'] = 12
        self.assertIn('trying 2', tuner.observe(542, gpu(), workers))

    def test_memory_accounts_for_unseen_peaks_and_staggering(self):
        tuner = p.AdaptiveLimit()
        tuner.target = 5
        workers = [worker(1), worker(2)]
        telemetry = gpu()
        telemetry.update(used_gib=35., processes={1: 4., 2: 4.})
        self.assertFalse(tuner.can_launch(telemetry, workers))
        telemetry['used_gib'] = 15.
        self.assertTrue(tuner.can_launch(telemetry, workers))
        workers[1].state['step'] = 0
        self.assertFalse(tuner.can_launch(telemetry, workers))
        telemetry['cpu_workers'] = 2
        self.assertEqual(tuner.resource_limit(telemetry), 2)

    def test_mixed_phases_do_not_compare_update_rates(self):
        tuner = p.AdaptiveLimit(window=180)
        tuner.target = 2
        workers = [worker(1), worker(2, 200, 'evaluating')]
        tuner.observe(0, gpu(), workers)
        workers[0].state['step'] += 20
        workers[1].state['volumes'] += 10
        self.assertIn('Mixed phases', tuner.observe(180, gpu(), workers))
        self.assertEqual(tuner.rates, {})
        self.assertEqual(tuner.target, 3)


class StagingTests(unittest.TestCase):
    def fixture(self, folder):
        folder = Path(folder)
        package, data, base = folder/'code.zip', folder/'data.zip', folder/'base.pt'
        bundle, controller, results = folder/'bundle', folder/'controller', folder/'results'
        base.write_bytes(b'original base')
        image = b'original data'
        data_manifest = json.dumps({'image.nii.gz': hashlib.sha256(image).hexdigest()}).encode()
        engine = b'original engine'
        scientific_manifest = json.dumps({'engine.py': hashlib.sha256(engine).hexdigest()}).encode()
        members = {'bundle/engine.py': engine, 'bundle/data_manifest.json': data_manifest,
                   'bundle/bundle_manifest.json': scientific_manifest, 'controller/parallel.py': b'controller'}
        manifest = {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}
        with zipfile.ZipFile(package, 'w') as archive:
            for k, v in members.items(): archive.writestr(k, v)
            archive.writestr('package_manifest.json', json.dumps(manifest))
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('data/image.nii.gz', image)
        return (package, data, base, bundle, controller, results, rt.digest(package),
                hashlib.sha256(scientific_manifest).hexdigest(), rt.digest(data), rt.digest(base))

    def test_idempotent_staging_preserves_checkpoint_archive_and_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(folder)
            stage_parallel(*args)
            bundle, _, results = args[3:6]
            (bundle/'work/task').mkdir()
            rt.write(bundle/'work/task/checkpoint_ledger.json', {'step': 110})
            (bundle/'work/task/checkpoint_latest.pt').write_bytes(b'checkpoint at 110')
            (results/'runs/done').mkdir(parents=True)
            (results/'runs/done/archive_manifest.json').write_bytes(b'archive')
            (results/'STOP').write_text('manual pause')
            before = {str(x): rt.digest(x) for x in [bundle/'work/task/checkpoint_latest.pt',
                bundle/'work/task/checkpoint_ledger.json', results/'runs/done/archive_manifest.json', results/'STOP']}
            stage_parallel(*args)
            self.assertEqual(before, {x: rt.digest(x) for x in before})

    def test_existing_scheduler_blocks_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(folder)
            with p.SweepLock(args[3]):
                with self.assertRaisesRegex(RuntimeError, 'sweep is active'):
                    stage_parallel(*args)
            self.assertFalse((args[3]/'engine.py').exists())

    def test_changed_science_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(folder)
            stage_parallel(*args)
            (args[3]/'engine.py').write_bytes(b'modified externally')
            with self.assertRaisesRegex(ValueError, 'Existing file differs'):
                stage_parallel(*args)
            self.assertEqual((args[3]/'engine.py').read_bytes(), b'modified externally')

    def test_complete_notebook_has_valid_python_and_original_identity(self):
        nb = notebook({'medsam3_msd_data.zip': 'data', 'sam3.pt': 'base'}, 'code')
        for cell in nb['cells']:
            if cell['cell_type'] == 'code': ast.parse(cell['source'])
        source = '\n'.join(c['source'] for c in nb['cells'])
        self.assertIn('results_matched200_noaug_108_v1', source)
        self.assertIn(p.FROZEN_MANIFEST, source)
        self.assertIn('MAX_WORKERS = 0', source)
        self.assertNotIn("torch.set_num_threads", source)


FAKE_ENGINE = r'''
import argparse, os, signal, time
from pathlib import Path
import runtime as r
parser=argparse.ArgumentParser()
parser.add_argument('--bundle',type=Path);parser.add_argument('--durable',type=Path);parser.add_argument('--task');parser.add_argument('--deadline')
a=parser.parse_args()
out=a.bundle/'work'/a.task;rec=a.durable/'recovery'/a.task
stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGUSR1,stop)
with (out/'launches.txt').open('a') as f:f.write(str(os.getpid())+'\n')
r.write(out/'protocol.json',dict(updates=200,augmentation='none'))
r.write(out/'calibration.json',dict(weight=.1))
sha=r.digest(out/'protocol.json')
history=r.read(out/'training_history.json') if (out/'training_history.json').exists() else []
if not history and (out/'checkpoint_latest.pt').exists():
    history=r.read(out/'checkpoint_latest.pt')['history']
writer=r.RecoveryWriter(out,rec,interval=9999)
def commit():
    r.write(out/'checkpoint_latest.pt',dict(history=history))
    r.write(out/'training_history.json',history)
    r.write(out/'checkpoint_ledger.json',dict(step=len(history),protocol_sha256=sha))
if not (out/'checkpoint_latest.pt').exists():commit()
try:
    for i in range(len(history),200):
        if stopping:raise SystemExit(75)
        r.write(rec/'charged_ledger.json',dict(step=i+1,protocol_sha256=sha))
        history.append(dict(update=i+1,value=(i+1)**2))
        commit()
        time.sleep(.003)
    for name in ('training_summary.json','slab_schedule.json','runtime_recipe.json'):
        r.write(out/name,dict(updates=200))
    r.write(out/'evaluation.json',dict(records=[dict(case='case0')]))
    (out/'curves.png').write_bytes(b'curve')
    r.write(out/'completion.json',dict(updates=200,status='complete',checkpoint_sha256=r.digest(out/'checkpoint_latest.pt')))
finally:writer.close(force=True)
'''


class ProcessTests(unittest.TestCase):
    def fixture(self, directory, count=4):
        root, durable = Path(directory)/'bundle', Path(directory)/'results'
        root.mkdir(); durable.mkdir()
        (root/'runtime.py').write_bytes(Path(rt.__file__).read_bytes())
        (root/'fake_engine.py').write_text(FAKE_ENGINE)
        rt.write(root/'bundle_manifest.json', {})
        tasks = [dict(id=f'task{i}', fold=0, seed=i, arm='baseline' if i < 3 else 'sdf') for i in range(count)]
        rt.write(root/'plan.json', dict(tasks=tasks))
        binding = {'bundle_sha256': rt.digest(root/'bundle_manifest.json')}
        rt.write(durable/'preflight.json', dict(status='passed', **binding))
        def quota(path): rt.write(path/'storage_quota.json',dict(limit=100*1024**3,usage=0))
        def report(root,durable): return dict(new_completed=sum(rt.completed(root,durable,t) for t in tasks))
        def child(root,durable,*args,deadline=None):
            task=args[-1]
            return [sys.executable,str(root/'fake_engine.py'),'--bundle',str(root),'--durable',str(durable),'--task',task],os.environ.copy()
        runtime = SimpleNamespace(**{k:getattr(rt,k) for k in ('read','write','digest','copy_verified','relative','restore','check_ledger','archive_run','completed')},
            verify_bundle=lambda _:dict(tasks=tasks),binding=lambda *_:binding,FOLDS=(0,),fold_calibration=lambda *_:None,
            quota=quota,child=child,report=report)
        return root, durable, runtime, tasks

    def test_three_processes_clean_stop_resume_no_replay_or_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root,durable,runtime,tasks = self.fixture(directory)
            supervisor = p.Supervisor(runtime,root,durable,poll=.005,hours=1)
            def telemetry():
                if len(supervisor.active)==3:
                    states=[p.progress(rt,root/'work'/t) for t in supervisor.active]
                    if all(s['step']>=10 for s in states):supervisor.request_stop()
                return gpu()
            supervisor.telemetry=telemetry
            with patch.object(p.AdaptiveLimit,'can_launch',lambda _,g,w:len(w)<3), \
                 patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                supervisor.run()
            prefixes = {}
            for task in tasks[:3]:
                output=root/'work'/task['id']; recovery=durable/'recovery'/task['id']
                self.assertGreaterEqual(rt.read(output/'checkpoint_ledger.json')['step'],10)
                p.verify_recovery(rt,output,recovery)
                prefixes[task['id']]=rt.read(output/'training_history.json')
            # Simulate a new VM: restore each checkpoint and its preserved log from Drive.
            shutil.rmtree(root/'work')
            resumed = p.Supervisor(runtime,root,durable,poll=.005,hours=1,telemetry=gpu)
            with patch.object(p.AdaptiveLimit,'can_launch',lambda _,g,w:len(w)<3), \
                 patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                resumed.run()
            self.assertEqual(resumed.finished,{t['id'] for t in tasks})
            for task in tasks:
                archive=durable/'runs'/task['id']
                self.assertTrue(rt.completed(root,durable,task))
                rows=rt.read(archive/'training_history.json')
                self.assertEqual([r['update'] for r in rows],list(range(1,201)))
                if task['id'] in prefixes:self.assertEqual(rows[:len(prefixes[task['id']])],prefixes[task['id']])
                self.assertFalse((root/'work'/task['id']/'checkpoint_latest.pt').exists())
                self.assertFalse((durable/'recovery'/task['id']/'latest.json').exists())
            again = p.Supervisor(runtime,root,durable,poll=.005,telemetry=gpu)
            again.run()
            self.assertEqual(again.launched,0)

    def test_shared_lock_rejects_second_supervisor(self):
        with tempfile.TemporaryDirectory() as folder:
            root,durable,runtime,_=self.fixture(folder)
            with p.SweepLock(root):
                with self.assertRaisesRegex(RuntimeError,'previous sweep is still active'):
                    p.Supervisor(runtime,root,durable).run()

    def test_lost_charged_update_blocks_without_modifying_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root,durable,_,tasks=self.fixture(folder)
            output=root/'work'/tasks[0]['id']; output.mkdir(parents=True)
            rec=durable/'recovery'/tasks[0]['id']
            rt.write(output/'checkpoint_ledger.json',dict(step=110))
            (output/'checkpoint_latest.pt').write_bytes(b'keep checkpoint')
            rt.write(rec/'charged_ledger.json',dict(step=111))
            before=rt.digest(output/'checkpoint_latest.pt')
            with self.assertRaisesRegex(RuntimeError,'lost charged updates'):
                p.prepare_task(rt,root,durable,tasks[0])
            self.assertEqual(before,rt.digest(output/'checkpoint_latest.pt'))

    def test_manual_stop_is_preserved_and_launches_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            root,durable,runtime,_=self.fixture(folder)
            (durable/'STOP').write_text('user pause')
            supervisor=p.Supervisor(runtime,root,durable,poll=.005,telemetry=gpu)
            supervisor.run()
            self.assertEqual(supervisor.launched,0)
            self.assertEqual((durable/'STOP').read_text(),'user pause')

    def test_partial_evaluation_recovery_is_bound_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root,durable,_,tasks=self.fixture(folder)
            task=tasks[0];output=root/'work'/task['id'];output.mkdir(parents=True)
            (output/'checkpoint_latest.pt').write_bytes(b'checkpoint')
            rt.write(output/'checkpoint_ledger.json',dict(step=200))
            rt.write(output/'evaluation.json',dict(records=[dict(case='already evaluated')]))
            (output/'train.log').write_text('update 200')
            p.save_supplement(rt,root,durable,task)
            (output/'evaluation.json').unlink()
            p.restore_supplement(rt,root,durable,task)
            self.assertEqual(len(rt.read(output/'evaluation.json')['records']),1)
            (output/'checkpoint_latest.pt').write_bytes(b'different checkpoint')
            with self.assertRaisesRegex(ValueError,'different checkpoint'):
                p.restore_supplement(rt,root,durable,task)

    def test_failure_drains_other_processes_and_keeps_all_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root,durable,runtime,tasks=self.fixture(directory,count=3)
            failing=FAKE_ENGINE.replace("history.append(dict(update=i+1,value=(i+1)**2))",
                "if a.task=='task1' and i==20: raise RuntimeError('synthetic failure after charge')\n        history.append(dict(update=i+1,value=(i+1)**2))")
            (root/'fake_engine.py').write_text(failing)
            supervisor=p.Supervisor(runtime,root,durable,poll=.005,hours=1,telemetry=gpu)
            with patch.object(p.AdaptiveLimit,'can_launch',lambda _,g,w:len(w)<3), \
                 patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                    supervisor.run()
            self.assertFalse(supervisor.active)
            for task in tasks:
                output=root/'work'/task['id'];recovery=durable/'recovery'/task['id']
                self.assertTrue((output/'checkpoint_latest.pt').exists())
                self.assertTrue((recovery/'latest.json').exists())
                if task['id']=='task1':
                    with self.assertRaisesRegex(RuntimeError,'lost charged updates'):
                        p.prepare_task(rt,root,durable,task)
                else:
                    p.verify_recovery(rt,output,recovery)

    def test_stop_during_python_import_does_not_kill_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root,durable,runtime,tasks=self.fixture(directory,count=1)
            # The launcher is ready before the engine imports. The first stop is
            # ignored until the engine registers its own handler, then retried.
            (root/'fake_engine.py').write_text('import time\ntime.sleep(.25)\n'+FAKE_ENGINE)
            supervisor=p.Supervisor(runtime,root,durable,poll=.005,hours=1)
            def telemetry():
                if supervisor.active:supervisor.request_stop()
                return gpu()
            supervisor.telemetry=telemetry
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                supervisor.run()
            self.assertIsNone(supervisor.failure)
            self.assertFalse(supervisor.active)
            output=root/'work'/tasks[0]['id'];recovery=durable/'recovery'/tasks[0]['id']
            if not rt.completed(root,durable,tasks[0]):p.verify_recovery(rt,output,recovery)


if __name__ == '__main__':
    unittest.main()
