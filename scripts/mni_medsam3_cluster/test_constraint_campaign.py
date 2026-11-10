from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np

from scripts.mni_medsam3_cluster import constraint_campaign as campaign
from scripts.mni_medsam3_cluster.constraint_support import depth_geometry
from scripts.depth_supervision.geometry import build_geometry, GeometryConfig
from scripts.cross_dataset_campaign.common import read, write, digest, ARMS


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_gate_requires_every_pair_not_a_good_average(self):
        rows=[dict(fold=f,seed=s,updates=400,validation_dice=.7) for f in range(3) for s in (0,1)]
        self.assertEqual(len(campaign.cohort_rows(rows)),6)
        rows[-1]['validation_dice']=.59
        self.assertIsNone(campaign.cohort_rows(rows))
        rows.append(dict(fold=2,seed=1,updates=500,validation_dice=.6))
        self.assertEqual(campaign.cohort_rows(rows)[-1]['updates'],500)
        rows.append(dict(fold=2,seed=1,updates=600,validation_dice=.9))
        self.assertEqual(campaign.cohort_rows(rows)[-1]['updates'],500)

    def test_baseline_watcher_does_not_write_verification_receipts(self):
        task=dict(id='baseline',fold=0,seed=0,updates=400)
        out=self.root/'runs/baseline'
        write(out/'protocol.json',dict(updates=400))
        write(out/'evaluation.json',dict(mean=.7))
        write(out/'milestone_observed.json',dict(time=1))
        write(out/'completion.json',dict(status='complete',updates=400,validation_dice=.7,training_dice=.8,
            evaluation_sha256=digest(out/'evaluation.json'),protocol_sha256=digest(out/'protocol.json')))
        write(out/'archive_manifest.json',dict(completion_sha256=digest(out/'completion.json'),
            metadata={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.glob('*.json')}))
        before={str(p):digest(p) for p in self.root.rglob('*') if p.is_file()}
        rows=campaign.baseline_records(self.root,dict(tasks=[task]))
        self.assertEqual(rows[0]['validation_dice'],.7)
        self.assertEqual(before,{str(p):digest(p) for p in self.root.rglob('*') if p.is_file()})

    def test_all_pilot_pairs_finish_before_full_matrix(self):
        tasks=[dict(id=f'{a}_{f}_{s}',arm=a,fold=f,seed=s,phase='pilot' if a in campaign.PILOT else 'matrix')
               for a in ARMS if a!='dice' for f in range(3) for s in (0,1)]
        self.assertEqual(len(tasks),78)
        pilot=[t for t in tasks if t['phase']=='pilot']
        self.assertEqual(len(pilot),12)
        self.assertEqual(campaign.next_tasks(tasks,pilot[:-1],{pilot[-1]['id']}),[])
        ready=campaign.next_tasks(tasks,pilot,set())
        self.assertEqual(len(ready),66)
        self.assertTrue(all(t['phase']=='matrix' for t in ready))
        self.assertEqual(campaign.next_tasks(tasks,tasks,set()),[])

    def test_paired_protocol_retains_historical_schedule_and_lr(self):
        for count in (400,500):
            out=self.root/'runs'/f'MNI_MedSAM3_n5_f0s0_native{count}'
            write(out/'protocol.json',dict(updates=count,accumulation=2,arm='dice',seed=0,fold=0))
            write(out/'slab_schedule.json',[['a',i%5] for i in range(count*2)])
            write(out/'training_history.json',[dict(update=i+1,lr=.01 if i<400 else .003) for i in range(count)])
            write(out/'training_summary.json',dict(adapter_before='adapter',frozen_before='base'))
            write(out/'completion.json',dict(updates=count))
        row=dict(id='MNI_MedSAM3_n5_f0s0_native500',fold=0,seed=0,updates=500,validation_dice=.61)
        p=campaign.paired_protocol(self.root,row)
        self.assertEqual(p['frozen_learning_rates'],[.01]*400+[.003]*100)
        self.assertEqual(len(p['frozen_sample_schedule']),1000)
        self.assertEqual(p['requested_arms'],list(ARMS))
        self.assertNotIn('arm',p)

    def test_depth_weights_preserve_reference_and_ignore_subfield_ids(self):
        labels=np.zeros((15,23,17),dtype=np.uint8)
        labels[4:11,3:20,5:12]=1;labels[5:10,8:14,6:11]=2
        affine=np.eye(4);config=GeometryConfig()
        expected=build_geometry(labels,affine,config,fields=False)['weights']
        actual=depth_geometry(build_geometry,labels,affine,config,fields=False)['weights']
        np.testing.assert_array_equal(actual,expected)
        labels[6:9,10:12,7:10]=3;original=labels.copy()
        actual=depth_geometry(build_geometry,labels,affine,config,fields=False)['weights']
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(labels,original)
        self.assertAlmostEqual(float(actual.sum()),1.,places=6)

    def test_depth_adapter_rejects_invalid_labels_and_classwise_fields(self):
        with self.assertRaises(ValueError):depth_geometry(build_geometry,np.full((3,3,3),4),np.eye(4),GeometryConfig())
        with self.assertRaises(ValueError):depth_geometry(build_geometry,np.zeros((3,3,3)),np.eye(4),GeometryConfig(),fields=True)

    def test_completed_gpu_gate_bound_to_protocol_and_no_updates(self):
        t=dict(protocol='f0s0',output=self.root/'preflight')
        p=self.root/'protocols/f0s0/protocol.json';write(p,dict(seed=0))
        write(t['output']/'complete.json',dict(status='passed',optimizer_updates=0,adapter_before='a',adapter_after='a',protocol_sha256=digest(p)))
        self.assertTrue(campaign.gate_passed(self.root,t))
        write(p,dict(seed=1))
        with self.assertRaises(ValueError):campaign.gate_passed(self.root,t)

    def test_calibration_stop_resumes_without_optimizer_checkpoint(self):
        write(self.root/'protocol.json',dict(seed=0))
        write(self.root/'calibration_partial.json',dict(rows=[1]))
        write(self.root/'requested_stop.json',dict(files={n:digest(self.root/n) for n in ('protocol.json','calibration_partial.json')}))
        self.assertTrue(campaign.saved_stop(self.root))
        write(self.root/'calibration_partial.json',dict(rows=[2]))
        self.assertFalse(campaign.saved_stop(self.root))

    def test_launcher_error_requires_successful_exact_step(self):
        write(self.root/'gpu_worker.json',dict(slurm_job_id='12',slurm_step_id='3'))
        for text,expected in [('12.3|COMPLETED|0:0|\n',True),('12.3|FAILED|73:0|\n',False),('12.4|COMPLETED|0:0|\n',False)]:
            with patch.object(campaign.subprocess,'check_output',return_value=text):
                self.assertEqual(campaign.step_succeeded(self.root),expected)

    def test_offline_completion_never_authorizes_deletion(self):
        with patch.object(campaign,'retire') as retire:
            campaign.cleanup(self.root,[dict(id='run')]);retire.assert_not_called()
            write(self.root/'runs/run/archive_acknowledgment.json',dict(status='verified'))
            campaign.cleanup(self.root,[dict(id='run')]);retire.assert_called_once()

    def test_dispatch_runs_two_lanes_and_advances_without_laptop_ack(self):
        tasks=[dict(id=str(i),phase='pilot' if i<3 else 'matrix') for i in range(5)]
        write(self.root/'cohort.json',dict(tasks=tasks))
        done=[];active=set();launched=[];peak=[0];statuses=[]
        class Process:
            def __init__(self,command,**kwargs):
                identity=command[0]
                if int(identity)>=3:
                    assert {r['id'] for r in done}>={'0','1','2'}
                active.add(identity);peak[0]=max(peak[0],len(active));launched.append(identity)
                self.identity=identity
            def poll(self):active.discard(self.identity);return 0
        with ExitStack() as stack:
            for name,value in (
                ('completed',lambda stage,all_tasks:list(done)),
                ('finish',lambda stage,t:done.append(t)),('cleanup',lambda *args:None),
                ('report',lambda *args:None),('stopping',lambda:False),
                ('quota',lambda:dict(free_gib=40)),
                ('command',lambda stage,plan,t,base,lane:[t['id']])):
                stack.enter_context(patch.object(campaign,name,value))
            stack.enter_context(patch.object(campaign.subprocess,'Popen',Process))
            stack.enter_context(patch.object(campaign.time,'sleep',lambda seconds:None))
            stack.enter_context(patch.object(campaign.pool,'signal_workers',lambda workers:None))
            result=campaign.dispatch(self.root,dict(storage=dict(minimum_free_gib=8)),Path('/base'),
                lambda status,workers:statuses.append(status))
        self.assertEqual(result,'complete');self.assertEqual(peak[0],2)
        self.assertEqual(launched,['0','1','2','3','4'])
        self.assertEqual(list(self.root.glob('runs/*/archive_acknowledgment.json')),[])


if __name__=='__main__':unittest.main()
