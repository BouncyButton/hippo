"""Check held-out subjects, matched schedules and two independent fold workers."""
import argparse
from contextlib import ExitStack
import importlib
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign import adni_foundation_folds_job as job
from scripts.cross_dataset_campaign.common import read, write


def load_parallel():
    """Test the staged dispatcher, or generate the same source in a checkout."""
    name = 'scripts.cross_dataset_campaign.foundation_fold_parallel'
    if importlib.util.find_spec(name) is not None:
        return importlib.import_module(name)
    from scripts.cross_dataset_campaign import foundation_parallel
    source_path = Path(foundation_parallel.__file__)
    module = types.ModuleType(name)
    module.__file__ = str(source_path.with_name('foundation_fold_parallel.py'))
    source = job.parallel_source(source_path.read_text())
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


class FoldTests(unittest.TestCase):
    def test_folds_overlap_but_arms_wait_for_archive_cleanup(self):
        parallel = load_parallel()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stage=Path(directory);write(stage/'control.json',dict(pause=False))
            plan=dict(tasks=job.tasks(),foundation_python='python',parallelism=dict(datasets=['ADNI_f1','ADNI_f3']),
                      storage=dict(minimum_free_gib=10),archive_wait_timeout_seconds=3)
            now=[0];launched=[];cleaned=[];processes=[];concurrency=[]
            stack.enter_context(patch.object(parallel.time,'monotonic',side_effect=lambda:now[0]))
            stack.enter_context(patch.object(parallel.time,'sleep',side_effect=lambda _:now.__setitem__(0,now[0]+1)))
            stack.enter_context(patch.object(parallel,'stopping',return_value=False))
            stack.enter_context(patch.object(parallel,'quota',return_value=dict(free_gib=60)))
            stack.enter_context(patch.object(parallel,'prepare',side_effect=lambda output,task:write(output/'archive_acknowledgment.json',dict(status='verified'))))
            def retire(output):
                self.assertTrue((output/'archive_acknowledgment.json').exists())
                write(output/'cleanup.json',dict(status='complete'));cleaned.append(output.name)
            stack.enter_context(patch.object(parallel,'retire',side_effect=retire))
            class Process:
                def __init__(inner,args,**kwargs):
                    inner.output=Path(args[args.index('--output')+1]);inner.start=now[0];inner.code=None
                    task=next(t for t in plan['tasks'] if t['id']==inner.output.name)
                    group=[t for t in plan['tasks'] if t['fold']==task['fold']]
                    index=group.index(task)
                    if index:self.assertIn(group[index-1]['id'],cleaned)
                    self.assertIn('--gres=gpu:4g.40gb:1',args);self.assertIn('--mem=40G',args)
                    launched.append(task);processes.append(inner)
                    self.assertLessEqual(sum(p.code is None for p in processes),2)
                def poll(inner):
                    if inner.code is None and now[0]-inner.start>=2:
                        inner.code=0;write(inner.output/'completion.json',dict(status='complete'))
                    return inner.code
                def send_signal(inner,sig):inner.code=75
            stack.enter_context(patch.object(parallel.subprocess,'Popen',side_effect=Process))
            result=parallel.dispatch(stage,plan,Path('/tmp/base'),lambda status,workers:concurrency.append(len(workers)))
            self.assertEqual(result,'complete');self.assertEqual(len(cleaned),6);self.assertIn(2,concurrency)
            self.assertEqual({t['fold'] for t in launched[:2]},{1,3})
            for fold in job.FOLDS:
                self.assertEqual([t['arm'] for t in launched if t['fold']==fold],list(job.ARMS))

    def test_submit_reuses_owned_active_job(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);write(stage/'submissions/123.json',dict(job_id='123'))
            with patch.object(job,'verify'),patch.object(job.controller,'queue',return_value=[['123','RUNNING','own']]),patch.object(job.controller,'submit') as submit:
                self.assertEqual(job.submit(stage),dict(job_id='123'));submit.assert_not_called()


def support_check(stage):
    import torch
    from scripts.cross_dataset_campaign.foundation import Volumes, SupportSlabs
    from scripts.adni_baselines.core import case_name
    torch.set_num_threads(2)
    job.verify(stage);results=[]
    saved=read(stage/'saved_folds.json');provenance=read(stage/'split_provenance.json')
    for selection in read(stage/'split_audit.json')['selections']:
        if selection!=job.select_support(saved,provenance['case_to_subject'],selection['fold']):
            raise ValueError('Saved support selection is not reproducible')
    for fold in job.FOLDS:
        p=read(stage/'protocols'/f'ADNI_f{fold}'/'protocol.json');volumes=Volumes(p)
        by_case={case_name(r):str(r.subject_id) for r in volumes.dataframe.itertuples()}
        if by_case!=provenance['case_to_subject']:
            raise ValueError('Actual dataset subject mapping differs')
        if (len(p['evaluation_cases'])!=54 or len({by_case[c] for c in p['evaluation_cases']})!=27
                or set(p['support_subjects']) & {by_case[c] for c in p['evaluation_cases']}):
            raise ValueError('Validation size or subject leakage differs')
        schedules=[]
        for arm in job.ARMS:
            protocol={**p,'arm':arm};slabs=SupportSlabs(volumes,protocol,None)
            schedule=slabs.schedule(protocol);calibration=slabs.calibration_schedule(4)
            if len(slabs.inventory)!=2 or len(schedule)!=400 or len(calibration)!=8:
                raise ValueError('Wrong support/update/calibration inventory')
            if schedule!=slabs.schedule(protocol) or {c for c,_ in schedule}!=set(p['train_cases']):
                raise ValueError('Sampling is not reproducible')
            schedules.append(schedule)
            results.append(dict(fold=fold,arm=arm,inventory=slabs.inventory,training_pairs=len(schedule),calibration_pairs=len(calibration)))
        if any(s!=schedules[0] for s in schedules):raise ValueError('Unmatched arms')
    write(stage/'support_preflight.json',dict(status='passed',runs=results))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path);args=parser.parse_args()
    from scripts.cross_dataset_campaign import test_foundation_parallel as inherited
    parallel = load_parallel()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(FoldTests),
                            unittest.defaultTestLoader.loadTestsFromTestCase(inherited.ParallelTests)])
    with patch.object(inherited,'parallel',parallel):
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    if args.stage:support_check(args.stage)


if __name__=='__main__':main()
