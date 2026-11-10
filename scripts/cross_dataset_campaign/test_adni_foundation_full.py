"""Exercise the full queue, archive gates, recovery, and actual ADNI geometry."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign import adni_foundation_full_job as job
from scripts.cross_dataset_campaign import adni_foundation_full_pool as pool
from scripts.cross_dataset_campaign.common import read, write, digest


class PoolTests(unittest.TestCase):
    def exercise(self, mode='normal', resume=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stage=Path(directory);write(stage/'control.json',dict(pause=False))
            plan=dict(tasks=job.tasks(),foundation_python='python',parallelism=dict(workers=2),
                      storage=dict(minimum_free_gib=12),archive_wait_timeout_seconds=3)
            now=[0];launched=[];cleaned=[];processes=[];published=[]
            if resume:
                for task in plan['tasks'][:10]:
                    write(stage/'runs'/task['id']/'cleanup.json',dict(status='complete'));cleaned.append(task['id'])
                write(stage/'runs'/plan['tasks'][10]['id']/'completion.json',dict(status='complete'))
            stack.enter_context(patch.object(pool.time,'monotonic',side_effect=lambda:now[0]))
            stack.enter_context(patch.object(pool.time,'sleep',side_effect=lambda _:now.__setitem__(0,now[0]+1)))
            stack.enter_context(patch.object(pool,'quota',return_value=dict(free_gib=60)))
            def stopping():
                if mode=='pause' and now[0]>=1:
                    write(stage/'control.json',dict(pause=True));return True
                return False
            stack.enter_context(patch.object(pool,'stopping',side_effect=stopping))
            def prepare(output,task):
                self.assertTrue((output/'completion.json').exists())
                if mode!='offline':write(output/'archive_acknowledgment.json',dict(status='verified'))
            def retire(output):
                self.assertTrue((output/'archive_acknowledgment.json').exists())
                write(output/'cleanup.json',dict(status='complete'));cleaned.append(output.name)
            stack.enter_context(patch.object(pool,'prepare',side_effect=prepare))
            stack.enter_context(patch.object(pool,'retire',side_effect=retire))
            class Process:
                def __init__(inner,args,**kwargs):
                    inner.output=Path(args[args.index('--output')+1]);inner.start=now[0];inner.code=None;inner.signals=[]
                    task=next(t for t in plan['tasks'] if t['id']==inner.output.name)
                    if task['arm']!='dice':
                        self.assertTrue(all(t['id'] in cleaned for t in plan['tasks'] if t['arm']=='dice'))
                    self.assertNotIn(task['id'],cleaned)
                    self.assertIn('--gres=gpu:4g.40gb:1',args);self.assertIn('--mem=40G',args)
                    self.assertIn('scripts.cross_dataset_campaign.adni_foundation_full_worker',args)
                    launched.append(task['id']);processes.append(inner)
                    self.assertLessEqual(sum(p.code is None for p in processes),2)
                def poll(inner):
                    if inner.code is None and now[0]-inner.start>=2:
                        inner.code=1 if mode=='failed' and inner.output.name==plan['tasks'][0]['id'] else 0
                        if inner.code==0:write(inner.output/'completion.json',dict(status='complete'))
                    return inner.code
                def send_signal(inner,sig):inner.signals.append(sig);inner.code=75
                def wait(inner,timeout=None):return inner.poll()
                def terminate(inner):inner.code=-15
            stack.enter_context(patch.object(pool.subprocess,'Popen',side_effect=Process))
            publish=lambda state,workers:published.append((state,len(workers)))
            if mode=='failed':
                with self.assertRaisesRegex(RuntimeError,'trainer exited 1'):pool.dispatch(stage,plan,Path('/tmp/base'),publish)
                result='failed'
            else:result=pool.dispatch(stage,plan,Path('/tmp/base'),publish)
            return result,launched,cleaned,published,processes

    def test_full_sweep_has_nine_baselines_before_constraints(self):
        result,launched,cleaned,published,_=self.exercise()
        self.assertEqual(result,'complete');self.assertEqual(len(launched),126)
        self.assertEqual(len(set(launched)),126);self.assertEqual(len(cleaned),126)
        self.assertTrue(any(n==2 for _,n in published))
        self.assertEqual({(t['fold'],t['seed']) for t in job.tasks()[:9]},
                         {(f,s) for f in (0,1,3) for s in (0,1,2)})

    def test_resume_skips_cleaned_and_completed_training(self):
        result,launched,cleaned,_,_=self.exercise(resume=True)
        self.assertEqual(result,'complete');self.assertEqual(len(launched),115);self.assertEqual(len(set(cleaned)),126)
        self.assertNotIn(job.tasks()[10]['id'],launched)

    def test_offline_preserves_results_without_launching_more(self):
        result,launched,cleaned,_,_=self.exercise('offline')
        self.assertEqual(result,'waiting_for_archive_offline');self.assertEqual(len(launched),2);self.assertEqual(cleaned,[])

    def test_pause_drains_both_workers(self):
        result,launched,cleaned,_,processes=self.exercise('pause')
        self.assertEqual(result,'paused');self.assertEqual(len(launched),2);self.assertEqual(cleaned,[])
        self.assertTrue(all(p.signals for p in processes))

    def test_failure_stops_new_launches(self):
        result,launched,_,_,_=self.exercise('failed')
        self.assertEqual(result,'failed');self.assertEqual(len(launched),2)

    def test_incomplete_baseline_cleanup_blocks_auxiliaries(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);tasks=job.tasks()
            for task in tasks[:9]:write(stage/'runs'/task['id']/'cleanup.json',dict(status='complete'))
            write(stage/'runs'/tasks[8]['id']/'cleanup.json',dict(status='pending'))
            self.assertIsNone(pool.next_ready_task(stage,tasks,{tasks[8]['id']}))

    def test_submit_reuses_owned_active_job(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);write(stage/'submissions/123.json',dict(job_id='123'))
            with patch.object(job,'verify'),patch.object(job.controller,'queue',return_value=[['123','RUNNING','own']]),patch.object(job.controller,'submit') as submit:
                self.assertEqual(job.submit(stage),dict(job_id='123'));submit.assert_not_called()

    def test_duplicate_gpu_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory)
            for i in range(2):write(stage/f'gpu_probe/123/{i}.json',dict(status='passed',visible_devices=1,slurm_job_id='123',device_uuid='same'))
            with patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch.object(pool.subprocess,'run'):
                with self.assertRaisesRegex(RuntimeError,'distinct'):pool.verify_gpu_probe(stage,dict(foundation_python='python'))


def support_check(stage):
    import numpy as np
    import torch
    from scripts.cross_dataset_campaign.foundation import Volumes,SupportSlabs
    from scripts.adni_baselines.core import case_name
    torch.set_num_threads(2);job.verify(stage);results=[];geometry=[];cache={}
    saved=read(stage/'saved_folds.json');provenance=read(stage/'split_provenance.json')
    p=read(stage/'protocols/ADNI_f0_s0/protocol.json');volumes=Volumes(p)
    by_case={case_name(r):str(r.subject_id) for r in volumes.dataframe.itertuples()}
    if by_case!=provenance['case_to_subject']:raise ValueError('Actual subject mapping differs')
    for selection in read(stage/'split_audit.json')['selections']:
        if selection!=job.select_support(saved,by_case,selection['fold']):raise ValueError('Saved support is not reproducible')
        for seed in job.SEEDS:
            p=read(stage/'protocols'/f"ADNI_f{selection['fold']}_s{seed}"/'protocol.json')
            if len(p['evaluation_cases'])!=54 or len({by_case[c] for c in p['evaluation_cases']})!=27 or set(p['support_subjects']) & {by_case[c] for c in p['evaluation_cases']}:
                raise ValueError('Validation size or subject leakage differs')
            schedules=[]
            for arm in job.ARMS:
                active=tuple(k for k in ('sdf','depth') if k in arm);key=(tuple(p['train_cases']),active)
                protocol={**p,'arm':arm}
                if key not in cache:
                    slabs=SupportSlabs(volumes,protocol,None);cache[key]=slabs
                    for case in p['train_cases']:
                        _,labels=volumes.load(case);truth=torch.from_numpy(labels.transpose(2,1,0).copy()).long()
                        sdf=slabs.sdf[case][0];depth=slabs.depth[case]
                        if not torch.isfinite(sdf).all() or not torch.isfinite(depth).all():raise ValueError('Nonfinite geometry')
                        if 'sdf' in active:
                            if sdf.abs().max()>1 or not (sdf[truth==1]<0).all() or not (sdf[truth==0]>0).all():raise ValueError('Wrong SDF sign or bounds')
                        elif torch.count_nonzero(sdf):raise ValueError('Inactive SDF')
                        if 'depth' in active:
                            if depth.min()<0 or not torch.isclose(depth.sum(),torch.tensor(1.),atol=1e-6):raise ValueError('Depth not normalized')
                            logits=torch.randn((2,*truth.shape),requires_grad=True);logp=logits.log_softmax(0)
                            gathered=logp.gather(0,truth[None])[0]
                            whole=-(gathered*depth).sum()
                            sampled=torch.stack([-(gathered[z]*depth[z]).sum()*len(depth) for z in range(len(depth))]).mean()
                            if not torch.allclose(whole,sampled,atol=1e-6):raise ValueError('Biased depth estimator')
                            g=torch.autograd.grad(sampled,logits)[0]
                            if not torch.isfinite(g).all() or g.abs().sum()==0:raise ValueError('Invalid depth gradient')
                        elif torch.count_nonzero(depth):raise ValueError('Inactive depth')
                        geometry.append(dict(case=case,active=list(active),shape=list(truth.shape),sdf_range=[float(sdf.min()),float(sdf.max())],depth_sum=float(depth.sum())))
                slabs=cache[key];schedule=slabs.schedule(protocol);calibration=slabs.calibration_schedule(4)
                if len(slabs.inventory)!=2 or len(schedule)!=400 or len(calibration)!=8 or len(set(calibration))!=8:raise ValueError('Wrong support or schedule')
                if schedule!=slabs.schedule(protocol) or {c for c,_ in schedule}!=set(p['train_cases']):raise ValueError('Sampling differs')
                schedules.append(schedule);results.append(dict(fold=p['fold'],seed=seed,arm=arm,training_pairs=400,calibration_pairs=8))
            if any(s!=schedules[0] for s in schedules):raise ValueError('Unmatched arm sampling')
    write(stage/'support_preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),runs=results,geometry=geometry,
        source_coordinate_policy=read(stage/'plan.json')['geometry_policy']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path);args=parser.parse_args()
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PoolTests)).wasSuccessful():raise SystemExit(1)
    if args.stage:support_check(args.stage)


if __name__=='__main__':main()
