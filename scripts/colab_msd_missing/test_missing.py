"""Selection, launch eligibility and upload-volume tests; no GPU or Drive writes."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'reports/msd_missing_324_second_account_20261018/package'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'scripts/colab_msd_cloud'))
from scope import historical_baselines, merged_rows, restrict
from traffic import DEFAULT_LIMIT, WINDOW_SECONDS, reserved_window
from store import DriveBlocked


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class MissingScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope=json.loads((PACKAGE/'missing_scope.json').read_text())
        cls.reference=json.loads((PACKAGE/'historical_reference.json').read_text())
        cls.parent=json.loads((PACKAGE/'original_bundle/plan.json').read_text())
        cls.seed20=json.loads((PACKAGE/'seed20_bundle/plan.json').read_text())
    def test_exact_missing_set_is_disjoint_and_complete(self):
        selected=restrict(self.parent,[0,2],self.scope)['tasks']+restrict(self.seed20,[20],self.scope)['tasks']
        missing={t['id'] for t in selected};excluded={t['id'] for t in self.scope['excluded_completed_tasks']}
        self.assertEqual(len(selected),324);self.assertEqual(len(missing),324)
        self.assertEqual(len(excluded),180);self.assertFalse(missing&excluded)
        self.assertEqual(missing|excluded,{t['id'] for t in self.scope['all_tasks']})
        self.assertEqual({t['seed'] for t in selected},{0,2,20})
        self.assertEqual({t['fold'] for t in selected},{0,1,2})
    def test_changed_scientific_task_rejected(self):
        changed=json.loads(json.dumps(self.parent))
        target=restrict(self.parent,[0,2],self.scope)['tasks'][0]['id']
        next(t for t in changed['tasks'] if t['id']==target)['seed']=99
        with self.assertRaises(ValueError):restrict(changed,[0,2],self.scope)
    def test_three_missing_result_archives_are_scheduled(self):
        audit=json.loads((PACKAGE/'recovery_plan.json').read_text())
        self.assertEqual(len(audit['rerun_completed']),3)
        self.assertTrue({r['id'] for r in audit['rerun_completed']}<={t['id'] for t in self.scope['missing_tasks']})
        cp=[r for r in audit['interrupted'] if r.get('available_checkpoint')]
        self.assertEqual(len(cp),1)
        self.assertEqual(cp[0]['available_checkpoint']['id'],'1wQMNczpf3vSG5NJVWn_koQs8q1s8s1Gf')
    def test_old_baselines_enable_constraints_without_archives_or_reruns(self):
        from scripts.colab_msd_parallel.scheduler import candidates
        tasks=restrict(self.parent,[0,2],self.scope)['tasks']
        baseline=historical_baselines(self.scope,[0,2])
        for task in tasks:
            if task['arm']!='dice' and any((b['model'],b['fold'],b['seed'])==(task['model'],task['fold'],task['seed']) for b in baseline):
                eligible=candidates([task],set(),baseline)
                self.assertEqual(eligible,[task])
        self.assertFalse({t['id'] for t in baseline}&{t['id'] for t in tasks})
    def test_old_results_never_become_new_completions(self):
        rows=merged_rows([],self.reference,self.scope)
        self.assertEqual(len(rows),180)
        with self.assertRaises(ValueError):merged_rows([self.reference[0]],self.reference,self.scope)
    def test_notebook_dependency_roots_are_only_new_account(self):
        source=(PACKAGE/'cloud/store.py').read_text()
        self.assertIn("ACCOUNT = 'filippofocaccia2@gmail.com'",source)
        self.assertIn("CAMPAIGN = '1fErhcTI519LV2wpPtbzLjiHd5dzxRUQu'",source)
        new_prepare=(PACKAGE/'cloud/missing.py').read_text()
        self.assertNotIn('preserved_results.zip',new_prepare)
        self.assertNotIn('imported_results',new_prepare)
        self.assertNotIn('recovery_evidence.zip',new_prepare)


class TrafficTests(unittest.TestCase):
    def test_budget_blocks_before_an_excess_upload(self):
        value=reserved_window({},DEFAULT_LIMIT,'full',100000.)
        with self.assertRaises(DriveBlocked):reserved_window(value,1,'extra',100001.)
    def test_window_expires_without_erasing_recent_reservations(self):
        value=reserved_window({},100,'old',100000.,1000)
        value=reserved_window(value,200,'new',100001.,1000)
        result=reserved_window(value,1,'latest',100000.+WINDOW_SECONDS,1000)
        self.assertEqual(result['reserved_bytes'],201)
    def test_failed_upload_reservation_is_not_refunded_silently(self):
        first=reserved_window({},600,'failed',100000.,1000)
        with self.assertRaises(DriveBlocked):reserved_window(first,600,'retry',100001.,1000)


if __name__=='__main__':unittest.main()
