import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.colab_hfh.constraint_search import run, run_all, read, write, qualifies, near_threshold_followup, prefer_mean


class AutomaticPhaseTests(unittest.TestCase):
    def test_mean_ranking_keeps_the_all_nine_gate(self):
        incumbent=[dict(fold=f,seed=s,delta_dice_pp=2.,constraint_dice=.75)
                   for f in range(3) for s in range(3)]
        better=[{**r,'delta_dice_pp':3.} for r in incumbent]
        self.assertTrue(prefer_mean(better,incumbent))
        self.assertFalse(prefer_mean(incumbent,better))
        self.assertFalse(prefer_mean(better[:-1],incumbent))
        better[0]['delta_dice_pp']=.99
        self.assertFalse(prefer_mean(better,incumbent))
        self.assertTrue(prefer_mean(incumbent,better))

    def test_model_scope_does_not_launch_other_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write(root/'constraint_matrix_plan.json',dict(calibration_implementation={'source_sha256':{}},
                verified_band_recipes={},cost_references={'swinunetr':{'canonical_costs':{}}}))
            write(root/'bands_v1_payload.json',{})
            launches=[]
            class FakeDrive:
                folder='remote:'
                def __init__(self,*a):pass
                def call(self,*a):return '[]'
                def upload(self,*a):pass
            def payload(base,model,*a,**k):
                launches.append(model);return {}
            def launch(command,**kwargs):
                work=Path(command[command.index('--work')+1])
                rows=[dict(fold=f,seed=s,delta_dice_pp=2.,constraint_dice=.75)
                      for f in range(3) for s in range(3)]
                write(work/'campaign_status.json',dict(status='complete_requested_matrix',results=rows))
                (root/'constraint_search/STOP').touch()
            specs=[dict(prefix=0,ramp=50,target=t,cap=.5) for t in (.1,.03,.3)]
            report=({'verified':63,'conditions':504},root/'matrix.csv',root/'summary.json')
            with patch('scripts.colab_hfh.constraint_search.Drive',FakeDrive),\
                 patch('scripts.colab_hfh.constraint_search.recipes',return_value=specs),\
                 patch('scripts.colab_hfh.constraint_search.make_payload',side_effect=payload),\
                 patch('scripts.colab_hfh.constraint_search.subprocess.run',side_effect=launch),\
                 patch('scripts.colab_hfh.constraint_report.write_report',return_value=report):
                run(argparse.Namespace(root=root,hours=8,pilot=False,phase='screen',models=['swinunetr'],optimize_mean=False))
            self.assertEqual(launches,['swinunetr'])
            self.assertEqual(read(root/'constraint_search/search_status.json')['model_scope'],['swinunetr'])

    def test_near_followup_does_not_lower_qualification(self):
        first=dict(fold=0,seed=1,delta_dice_pp=.9962152752,constraint_dice=.7588)
        self.assertTrue(near_threshold_followup([first]))
        rows=[dict(fold=f,seed=s,delta_dice_pp=1.5,constraint_dice=.77)
              for f in range(3) for s in range(3)]
        rows[1]=first
        self.assertFalse(qualifies(rows))
        self.assertFalse(near_threshold_followup(rows))
        rows[1]={**first,'delta_dice_pp':1.}
        self.assertTrue(qualifies(rows))

    def test_followup_margin_and_dice_floor(self):
        row=dict(delta_dice_pp=.9,constraint_dice=.5)
        self.assertTrue(near_threshold_followup([row]))
        self.assertFalse(near_threshold_followup([{**row,'delta_dice_pp':.899}]))
        self.assertFalse(near_threshold_followup([{**row,'constraint_dice':.499}]))
        self.assertFalse(near_threshold_followup([{**row,'delta_dice_pp':1.}]))

    def test_near_candidate_runs_full_matrix_and_remains_unqualified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            models=('nnunet','swinunetr','basic_unet16','unetr_vit_b16')
            write(root/'constraint_matrix_plan.json',dict(
                calibration_implementation={'source_sha256':{}},
                verified_band_recipes={'basic_unet16':{},'unetr_vit_b16':{}},
                cost_references={m:{'canonical_costs':{}} for m in models}))
            write(root/'bands_v1_payload.json',{})
            spec=dict(prefix=22,ramp=9,target=.1,cap=.5)
            specs=[spec,{**spec,'target':.03},{**spec,'target':.3}]
            rows=[dict(fold=f,seed=s,delta_dice_pp=1.5,constraint_dice=.77)
                  for f,s in [(0,1),(0,0),(0,2),(2,1),(2,0),(2,2),(1,1),(1,0),(1,2)]]
            rows[0]['delta_dice_pp']=.95
            write(root/'constraint_search/search_status.json',dict(status='prepared',selected={},
                attempts={f'nnunet__dice_depth__r{i:02d}':dict(model='nnunet',objective='dice_depth',
                    candidate=i,specification=specs[i],status='rejected',
                    results=[{**rows[0],'delta_dice_pp':.1}]) for i in (1,2)}))
            launches=[]
            class FakeDrive:
                folder='remote:'
                def __init__(self,*a):pass
                def call(self,*a):return '[]'
                def upload(self,*a):pass
            def payload(*a,**k):return {'screening':k['screening']}
            def launch(command,**kwargs):
                work=Path(command[command.index('--work')+1])
                selected=read(Path(command[command.index('--payload')+1]))['screening']
                launches.append(selected)
                write(work/'campaign_status.json',dict(
                    status='rejected_pair' if selected else 'complete_requested_matrix',
                    results=rows[:1] if selected else rows))
                if not selected:(root/'constraint_search/STOP').touch()
            report=({'verified':63,'conditions':504},root/'matrix.csv',root/'summary.json')
            with patch('scripts.colab_hfh.constraint_search.Drive',FakeDrive),\
                 patch('scripts.colab_hfh.constraint_search.recipes',return_value=specs),\
                 patch('scripts.colab_hfh.constraint_search.make_payload',side_effect=payload),\
                 patch('scripts.colab_hfh.constraint_search.subprocess.run',side_effect=launch),\
                 patch('scripts.colab_hfh.constraint_report.write_report',return_value=report):
                run(argparse.Namespace(root=root,hours=8,pilot=False,phase='screen'))
            state=read(root/'constraint_search/search_status.json')
            record=state['attempts']['nnunet__dice_depth__r00']
            self.assertEqual(launches,[True,False])
            self.assertEqual(state['status'],'paused')
            self.assertEqual(record['status'],'rejected')
            self.assertTrue(record['full_matrix_complete'])
            self.assertFalse(record['final_matrix_frozen'])
            self.assertEqual(state['selected']['nnunet__dice_depth']['qualification'],
                             'not_qualified_consistent_near_threshold_all_nine')

    def exercise(self, first_status='screen_complete', stop=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=argparse.Namespace(root=root,hours=8,pilot=False,phase='all')
            calls=[]
            def fake_run(current):
                calls.append(current)
                state={'status':first_status if current.phase=='screen' else 'matrix_complete'}
                write(root/'constraint_search/search_status.json',state)
                if stop:(root/'constraint_search/STOP').touch()
            with patch('scripts.colab_hfh.constraint_search.run',side_effect=fake_run),\
                 patch('scripts.colab_hfh.constraint_search.time.time',side_effect=[100,100,200]):
                run_all(args)
            return calls

    def test_freezes_before_matrix_and_shares_deadline(self):
        calls=self.exercise()
        self.assertEqual([c.phase for c in calls],['screen','matrix'])
        self.assertEqual(calls[0].hours,8)
        self.assertAlmostEqual(calls[1].hours,8-100/3600)

    def test_paused_screen_never_starts_matrix(self):
        self.assertEqual([c.phase for c in self.exercise('paused')],['screen'])

    def test_stop_at_boundary_is_respected(self):
        self.assertEqual([c.phase for c in self.exercise(stop=True)],['screen'])


if __name__=='__main__':unittest.main()
