"""CPU checks of offline continuation and strict archival retirement gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.mni_medsam3_cluster.offline_ops import offline_ops as ops


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stage = Path(self.tmp.name)
        ops.VALIDATED.clear()
        self.plan = dict(seeds=[0, 1, 2], threshold=.60, maximum_updates=1000,
            tasks=[dict(id=f'f{f}s{s}u{u}', fold=f, seed=s, updates=u)
                   for f in range(3) for s in range(3) for u in range(400,1001,100)])

    def fixture(self, seed=0, score=.5):
        task = next(t for t in self.plan['tasks'] if (t['fold'],t['seed'],t['updates'])==(0,seed,400))
        out = self.stage/'runs'/task['id']
        ops.write(out/'protocol.json', dict(updates=400))
        ops.write(out/'evaluation.json', dict(validation_dice=score))
        ops.write(out/'milestone_observed.json', dict(time=seed+1))
        ops.write(out/'completion.json', dict(status='complete', updates=400,
            validation_dice=score, training_dice=.8,
            protocol_sha256=ops.digest(out/'protocol.json'), evaluation_sha256=ops.digest(out/'evaluation.json')))
        manifest = dict(completion_sha256=ops.digest(out/'completion.json'),
            metadata={p.name:dict(bytes=p.stat().st_size,sha256=ops.digest(p)) for p in out.glob('*.json')})
        ops.write(out/'archive_manifest.json', manifest)
        return task, out

    def test_unarchived_verified_failure_unlocks_increment(self):
        task,out = self.fixture()
        rows = ops.records(self.stage,self.plan)
        ready = ops.next_trials(self.plan,rows,[],{'f0s1u400','f0s2u400'})
        self.assertEqual(ready[0]['updates'],500)
        self.assertFalse((out/'archive_acknowledgment.json').exists())
        self.assertFalse(ops.read(self.stage/'offline_state/verified'/f"{task['id']}.json")['off_beegfs_archive'])

    def test_corrupt_evaluation_blocks_continuation(self):
        task,out = self.fixture()
        ops.write(out/'evaluation.json',dict(corrupt=True))
        with self.assertRaises(ValueError):ops.validate_milestone(self.stage,task)

    def test_no_archive_never_retires(self):
        task,out = self.fixture()
        with patch.object(ops.study,'retire_ready') as retire:
            ops.retire_archived(self.stage,self.plan,[task],[],set())
            self.assertEqual(retire.call_args.args[2],[])
        self.assertFalse((out/'archive_acknowledgment.json').exists())

    def test_actual_archive_uses_existing_retirement_verification(self):
        task,out = self.fixture()
        ops.write(out/'archive_acknowledgment.json',dict(status='verified'))
        with patch.object(ops.study,'retire_ready') as retire:
            ops.retire_archived(self.stage,self.plan,[task],[],set())
            self.assertEqual(retire.call_args.args[2],[task])

    def test_two_qualified_seeds_unlock_folds_and_third_initial_trial(self):
        self.fixture(0,.61);self.fixture(1,.62)
        rows=ops.records(self.stage,self.plan)
        selected=ops.selected_seeds(self.stage,rows,self.plan)
        self.assertEqual(selected,[0,1])
        ready=ops.next_trials(self.plan,rows,selected,set())
        self.assertEqual({(t['fold'],t['seed'],t['updates']) for t in ready},
            {(0,2,400),(1,0,400),(1,1,400),(2,0,400),(2,1,400)})

    def test_base_delivery_is_cluster_local_and_scope_checked(self):
        job='123';target=f'/tmp/hippo-sam3-{ops.os.getuid()}-{job}.pt'
        ops.write(self.stage/'submissions'/f'{job}.json',{})
        ops.write(self.stage/'base_request.json',dict(job_id=job,destination=target,expected_sha256='abc'))
        with patch.object(ops.subprocess,'run') as run:
            ops.deliver_base(self.stage,dict(base_sha256='abc'),[(job,'RUNNING')],Path('/persistent/sam3.pt'))
            self.assertEqual(run.call_args.args[0],['sbcast','--jobid=123','--force','/persistent/sam3.pt',target])
        self.assertEqual(ops.read(self.stage/f'base_ready_{job}.json')['delivered_by'],'cluster_side_service')
        ops.write(self.stage/'base_request.json',dict(job_id=job,destination='/wrong',expected_sha256='abc'))
        with self.assertRaises(ValueError):ops.deliver_base(self.stage,dict(base_sha256='abc'),[(job,'RUNNING')],Path('/persistent/sam3.pt'))

    def test_occupied_queue_prevents_submission(self):
        ops.write(self.stage/'control.json',dict(controller_owner='offline_v1',pause=False))
        with patch.object(ops.controller,'queue',return_value=[('9','RUNNING'),('10','PENDING')]), patch.object(ops.subprocess,'check_output') as submit:
            self.assertIsNone(ops.submit(self.stage,self.plan))
            submit.assert_not_called()

    def test_pause_prevents_submission(self):
        ops.write(self.stage/'control.json',dict(controller_owner='offline_v1',pause=True))
        with self.assertRaises(RuntimeError):ops.submit(self.stage,self.plan)


if __name__=='__main__':unittest.main()
