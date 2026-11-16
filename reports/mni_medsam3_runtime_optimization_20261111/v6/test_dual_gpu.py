import unittest
from dual_gpu import allocation_flags, dual_lane, free_full_nodes, pending_decision
from mni_runtime import preferred_profile


def node(name, used=0):
    return f'NodeName={name} State=MIXED CPUTot=24 CPUAlloc=8 RealMemory=500000 AllocMem=81920 Gres=gpu:7g.80gb:1,gpu:4g.40gb:3 GresUsed=gpu:7g.80gb:{used},gpu:4g.40gb:1'


class DualGpuTests(unittest.TestCase):
    def test_requires_two_distinct_available_nodes(self):
        self.assertEqual(free_full_nodes(node('a')+'\n'+node('a')), ['a'])
        self.assertEqual(preferred_profile(node('a')+'\n'+node('b')), 'dual_full')
        self.assertEqual(preferred_profile(node('a')+'\n'+node('b',1)), 'mixed')
        self.assertEqual(preferred_profile(node('a')+'\n'+node('b'),allow_dual=False), 'mixed')

    def test_dual_allocation_is_two_gpus_and_eighty_gib_host_total(self):
        flags=allocation_flags('dual_full')
        for value in ['--nodes=2','--ntasks-per-node=1','--gres=gpu:7g.80gb:1','--mem=40G']:
            self.assertIn(value,flags)

    def test_lanes_are_pinned_to_distinct_nodes(self):
        command=['srun','--nodes=1','--gres=gpu:4g.40gb:1','--gpus-per-task=4g.40gb:1','python']
        a=dual_lane(command,0,['a','b']);b=dual_lane(command,1,['a','b'])
        self.assertIn('--nodelist=a',a);self.assertIn('--nodelist=b',b)
        self.assertIn('--gres=gpu:7g.80gb:1',a)
        self.assertIn('--gpus-per-task=7g.80gb:1',b)
        with self.assertRaises(ValueError):dual_lane(command,0,['a','a'])

    def test_dependency_wait_does_not_consume_fallback_timer(self):
        original=dict(job_id='2',dependency='1',eligible_seen_at=0)
        record,cancel=pending_decision(original,[('1','RUNNING'),('2','PENDING')],100000)
        self.assertFalse(cancel);self.assertNotIn('eligible_seen_at',record)
        self.assertEqual(original['eligible_seen_at'],0)

    def test_pending_successor_falls_back_after_fifteen_minutes(self):
        record,cancel=pending_decision(dict(job_id='2',dependency='1'),[('2','PENDING')],100)
        self.assertFalse(cancel)
        self.assertFalse(pending_decision(record,[('2','PENDING')],999)[1])
        self.assertTrue(pending_decision(record,[('2','PENDING')],1000)[1])

    def test_running_missing_and_already_cancelled_jobs_are_protected(self):
        record=dict(job_id='2',dependency='1',eligible_seen_at=0)
        self.assertFalse(pending_decision(record,[('2','RUNNING')],10000)[1])
        self.assertFalse(pending_decision(record,[],10000)[1])
        self.assertFalse(pending_decision(dict(record,status='fallback_cancelled'),[('2','PENDING')],10000)[1])


if __name__=='__main__':unittest.main()
