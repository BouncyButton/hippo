from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign.common import read, write
from scripts.mni_medsam3_cluster import seed_search as search, constraint_campaign as matrix


class SeedSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan=dict(threshold=.6,maximum_updates=1000,seeds=[3],candidate_seed=3,
            tasks=[dict(id=f'f{f}s3u{u}',fold=f,seed=3,updates=u) for f in range(3) for u in range(400,1001,100)])

    def test_failed_fold0_continues_and_qualified_folds_are_not_retrained(self):
        rows=[dict(fold=0,seed=3,updates=400,validation_dice=.59),dict(fold=1,seed=3,updates=400,validation_dice=.7)]
        tasks=search.next_baselines(self.plan,rows,[3],set())
        self.assertEqual([(t['fold'],t['updates']) for t in tasks],[(2,400),(0,500)])
        rows[0]['updates']=1000
        self.assertEqual([t['fold'] for t in search.next_baselines(self.plan,rows,[3],set())],[2])

    def test_no_candidate_created_before_two_seed_matrix_finishes(self):
        with patch.object(search,'prior_ready',return_value=(False,77)),patch.object(search,'make_baseline') as create:
            result=search.tick(self.root,dict(previous_matrix_stage='/previous'),dict(candidate_seed=2,history=[]),[])
            create.assert_not_called();self.assertEqual(result['phase'],'waiting_for_two_seed_matrix')

    def test_exhausted_baseline_advances_seed_without_launching_constraints(self):
        state=dict(candidate_seed=3,history=[])
        rows=[dict(fold=f,seed=3,updates=1000,validation_dice=.59) for f in range(3)]
        with ExitStack() as stack:
            for name,value in [('prior_ready',lambda *a:(True,78)),('clean_registered',lambda *a:None),
                ('make_baseline',lambda *a:self.root/'baseline'),('register',lambda *a:None),
                ('baseline_verify',lambda *a:self.plan),('own_jobs',lambda *a:[])]:
                stack.enter_context(patch.object(search,name,value))
            stack.enter_context(patch.object(matrix,'baseline_records',return_value=rows))
            stack.enter_context(patch.object(search.ops,'deliver_base'))
            create=stack.enter_context(patch.object(search,'make_matrix'))
            result=search.tick(self.root,dict(previous_matrix_stage='/prior',base_source='/base'),state,[])
            create.assert_not_called()
        self.assertEqual(result['phase'],'advancing_candidate');self.assertEqual(state['candidate_seed'],4)
        self.assertEqual(read(self.root/'state.json')['history'][0]['outcome'],'baseline_below60_at1000')

    def test_failed_pilot_advances_seed_without_full_matrix(self):
        state=dict(candidate_seed=3,history=[])
        rows=[dict(fold=f,seed=3,updates=400,validation_dice=.7) for f in range(3)]
        cohort=dict(baselines=rows,tasks=[])
        with ExitStack() as stack:
            for name,value in [('prior_ready',lambda *a:(True,78)),('clean_registered',lambda *a:None),
                ('make_baseline',lambda *a:self.root/'baseline'),('make_matrix',lambda *a:self.root/'matrix'),
                ('register',lambda *a:None),('baseline_verify',lambda *a:self.plan),('own_jobs',lambda *a:[])]:
                stack.enter_context(patch.object(search,name,value))
            for name,value in [('baseline_records',lambda *a:rows),('verify',lambda *a:{}),
                ('resolve',lambda *a:cohort),('completed',lambda *a:[]),('cleanup',lambda *a:None),
                ('report',lambda *a:None),('pilot_decision',lambda *a:dict(status='not_improved'))]:
                stack.enter_context(patch.object(matrix,name,value))
            stack.enter_context(patch.object(search.ops,'deliver_base'))
            submit=stack.enter_context(patch.object(search.ops,'submit'))
            result=search.tick(self.root,dict(previous_matrix_stage='/prior',base_source='/base'),state,[])
            submit.assert_not_called()
        self.assertEqual(result['phase'],'advancing_candidate');self.assertEqual(state['candidate_seed'],4)

    def test_negative_pilot_stops_gpu_dispatch_before_any_matrix_fit(self):
        baselines=[dict(fold=f,seed=3,updates=400,validation_dice=.7) for f in range(3)]
        tasks=matrix.inventory(baselines)
        rows=[dict(**t,validation_dice=.69) for t in tasks if t['phase']=='pilot']
        write(self.root/'cohort.json',dict(baselines=baselines,tasks=tasks))
        with ExitStack() as stack:
            for name,value in [('completed',lambda *a:rows),('cleanup',lambda *a:None),
                ('report',lambda *a:None),('stopping',lambda:False)]:
                stack.enter_context(patch.object(matrix,name,value))
            stack.enter_context(patch.object(matrix.time,'sleep'))
            stack.enter_context(patch.object(matrix.pool,'signal_workers'))
            launch=stack.enter_context(patch.object(matrix.subprocess,'Popen'))
            result=matrix.dispatch(self.root,dict(conditional_matrix_seed=3),Path('/base'),lambda *a:None)
            launch.assert_not_called()
        self.assertEqual(result,'complete_pilot_no_improvement')
        self.assertEqual(read(self.root/'pilot_decision.json')['status'],'not_improved')

    def test_generated_candidate_freezes_21_protocols_and_resumes_same_stage(self):
        reference=Path(__file__).resolve().parents[2]/'reports/mni_medsam3_n5_adaptive60_2gpu_20261110/stage'
        if not reference.exists():self.skipTest('Local frozen baseline fixture not installed')
        from scripts.cross_dataset_campaign.common import digest
        write(self.root/'plan.json',dict(baseline_stage=str(reference),baseline_manifest_sha256=digest(reference/'manifest.json')))
        write(self.root/'manifest.json',{})
        write(self.root/'source/sentinel.json',dict(immutable=True))
        (self.root/'run.sbatch').write_text('python -m scripts.mni_medsam3_cluster.constraint_campaign run --stage "$1"')
        stage=search.make_baseline(self.root,3)
        plan=search.baseline_verify(stage)
        self.assertEqual(len(plan['tasks']),21)
        self.assertIn('source/sentinel.json',read(stage/'manifest.json'))
        self.assertIn('seed_search run-baseline',(stage/'run.sbatch').read_text())
        self.assertEqual(search.make_baseline(self.root,3),stage)
        self.assertEqual(read(stage/'selected_seeds.json')['seeds'],[3])


if __name__=='__main__':unittest.main()
