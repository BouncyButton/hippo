import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.colab_hfh.constraint_mean_search import (
    POLICY, complete, qualifies, freeze_qualified, candidate_order, resume_queue,
    prioritize_band_followup, run_mean)
from scripts.colab_hfh.bands_campaign import read, write


def rows(delta=1.5):
    return [dict(fold=f, seed=s, delta_dice_pp=delta, constraint_dice=.75)
            for f in range(3) for s in range(3)]


class MeanPolicyTests(unittest.TestCase):
    def test_explicit_band_scope_never_launches_other_objectives(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);spec=dict(prefix=45,ramp=9,target=.3,cap=.75,inner_weight=.75)
            write(root/'bands_v1_payload.json',{})
            write(root/'constraint_matrix_plan.json',dict(calibration_implementation={'source_sha256':{}},
                cost_references={'nnunet':{'canonical_costs':{}}}))
            old={'retained':True}
            write(root/'constraint_search/search_status.json',dict(criterion='old',
                selected={'swinunetr__dice_bands':old},attempts={}))
            launches=[]
            class FakeDrive:
                def __init__(self,*a):pass
                def upload(self,*a):pass
            def launch(command,**kw):
                trial=Path(command[command.index('--work')+1]);launches.append(trial.name)
                write(trial/'campaign_status.json',dict(status='complete_requested_matrix',results=rows(1.2)))
            report=({'verified':72,'conditions':504},root/'matrix.csv',root/'summary.json')
            with patch('scripts.colab_hfh.constraint_mean_search.Drive',FakeDrive),\
                 patch('scripts.colab_hfh.constraint_search.recipes',return_value=[spec]),\
                 patch('scripts.colab_hfh.constraint_search.diagnosis_recipes',return_value=[]),\
                 patch('scripts.colab_hfh.constraint_mean_search.make_payload',return_value={}),\
                 patch('scripts.colab_hfh.constraint_mean_search.subprocess.run',side_effect=launch),\
                 patch('scripts.colab_hfh.constraint_report.write_report',return_value=report):
                run_mean(argparse.Namespace(root=root,hours=8,models=['nnunet'],
                    objectives=['dice_bands','dice_bands_depth','dice_sdf']))
            self.assertEqual(launches,['nnunet__dice_bands__r00','nnunet__dice_bands_depth__r00'])
            final=read(root/'constraint_search/search_status.json')
            self.assertEqual(final['objective_scope'],['dice_bands','dice_bands_depth'])
            self.assertEqual(final['selected']['swinunetr__dice_bands'],old)
            self.assertEqual(final['status'],'all_target_arms_mean_qualified')

    def test_band_priority_waits_for_depth_preserves_active_and_does_not_repeat(self):
        queue=[('nnunet','dice_sdf_depth'),('swinunetr','dice_depth'),('nnunet','dice_bands')]
        specs=[dict(prefix=22,ramp=9,target=.3,cap=.75,inner_weight=.9)]
        state={'attempts':{}}
        self.assertEqual(prioritize_band_followup(state,queue,specs),queue)
        state['attempts']['nnunet__dice_depth__r04']={'results':rows(.1)}
        self.assertEqual(prioritize_band_followup(state,queue,specs),[queue[2],*queue[:2]])
        state['current']='other'
        state['attempts']['other']={'status':'running','results':rows()[:2]}
        self.assertEqual(prioritize_band_followup(state,queue,specs),queue)
        state.pop('current')
        state['attempts']['nnunet__dice_bands__r00']={'results':rows(.2)}
        self.assertEqual(prioritize_band_followup(state,queue,specs),queue)

    def test_diagnosis_priority_preserves_active_work_and_excludes_completed(self):
        specs=[{'target':.1},{'target':.3}]
        state={'attempts':{'nnunet__dice_bands__r00':{'status':'running','results':rows()[:2]}}}
        self.assertEqual(candidate_order(state,'nnunet','dice_bands',specs,[specs[1]]),[0,1])
        state['attempts']['nnunet__dice_bands__r00']['status']='rejected'
        self.assertEqual(candidate_order(state,'nnunet','dice_bands',specs,[specs[1]]),[1,0])
        state['attempts']['nnunet__dice_bands__r01']={'results':rows(.2)}
        self.assertEqual(candidate_order(state,'nnunet','dice_bands',specs,[specs[1]]),[0])

    def test_moderate_band_follows_completed_combination_without_repeating(self):
        queue=[('nnunet','dice_sdf_depth'),('nnunet','dice_bands'),('swinunetr','dice_depth')]
        specs=[dict(prefix=22,ramp=9,target=.3,cap=.75,inner_weight=.9),
               dict(prefix=45,ramp=9,target=.3,cap=.75,inner_weight=.75)]
        state={'attempts':{'nnunet__dice_depth__r04':{'results':rows(.1)},
                           'nnunet__dice_bands__r00':{'results':rows(.2)},
                           'nnunet__dice_bands_depth__r11':{'results':rows(.3)}}}
        self.assertEqual(prioritize_band_followup(state,queue,specs),[queue[1],queue[0],queue[2]])
        state['current']='active_combination'
        state['attempts']['active_combination']={'status':'running','results':rows()[:2]}
        self.assertEqual(prioritize_band_followup(state,queue,specs),queue)
        state.pop('current')
        state['attempts']['nnunet__dice_bands__r01']={'results':rows(.4)}
        self.assertEqual(prioritize_band_followup(state,queue,specs),queue)

    def test_report_failure_resume_advances_without_repeating_the_arm(self):
        queue=[('nnunet','dice_bands_sdf'),('nnunet','dice_bands'),('swinunetr','dice_depth')]
        state=dict(current='trial',attempts={'trial':dict(model='nnunet',objective='dice_bands_sdf',results=rows(.1))})
        self.assertEqual(resume_queue(state,queue),queue[1:]+queue[:1])
        state['attempts']['trial']['results']=rows()[:3]
        self.assertEqual(resume_queue(state,queue),queue)
        state.pop('current');state['next_arm_key']='nnunet__dice_bands'
        self.assertEqual(resume_queue(state,queue),queue[1:]+queue[:1])

    def test_mean_accepts_regressions_but_requires_all_nine(self):
        r = rows(); r[0]['delta_dice_pp'] = -.2
        self.assertTrue(qualifies(r))
        self.assertFalse(qualifies(r[:-1]))
        r[0]['constraint_dice'] = .499
        self.assertFalse(qualifies(r))

    def test_threshold_identity_and_nan(self):
        self.assertTrue(qualifies(rows(1.)))
        self.assertFalse(qualifies(rows(.999)))
        r=rows(); r[0]=r[1].copy()
        self.assertFalse(complete(r))
        r=rows(); r[0]['delta_dice_pp']=float('nan')
        self.assertFalse(qualifies(r))

    def test_freeze_is_per_arm_and_leaves_other_models_untouched(self):
        state=dict(selected={'unetr_vit_b16__dice_depth':{'retained':True}},attempts={})
        for index,arm in enumerate(('dice_bands','dice_depth')):
            state['attempts'][f'swinunetr__{arm}__r00']=dict(model='swinunetr',objective=arm,
                candidate=0,specification={},results=rows(2. if index==0 else .2))
        freeze_qualified(state,['swinunetr'])
        self.assertEqual(set(state['selected']),{'unetr_vit_b16__dice_depth','swinunetr__dice_bands'})
        frozen=deepcopy(state['selected']['swinunetr__dice_bands'])
        state['attempts']['swinunetr__dice_bands__r01']=dict(model='swinunetr',objective='dice_bands',
            candidate=1,specification={},results=rows(3.))
        freeze_qualified(state,['swinunetr'])
        self.assertEqual(state['selected']['swinunetr__dice_bands'],frozen)

    def test_common_pilot_priority_not_unequal_subset_average(self):
        state={'attempts':{}}
        for i,pilot in enumerate((.6,.8)):
            r=[dict(fold=0,seed=1,delta_dice_pp=pilot)]
            if i==0:r.append(dict(fold=2,seed=1,delta_dice_pp=10.))
            state['attempts'][f'nnunet__dice_depth__r{i:02d}']={'results':r,'status':'rejected'}
        self.assertEqual(candidate_order(state,'nnunet','dice_depth',[{},{}]),[1,0])

    def test_resumes_rejected_pair_without_gate_and_never_launches_frozen_arms(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); spec=dict(prefix=22,ramp=9,target=.3,cap=1.)
            write(root/'bands_v1_payload.json',{})
            write(root/'constraint_matrix_plan.json',dict(calibration_implementation={'source_sha256':{}},
                cost_references={'nnunet':{'canonical_costs':{}}}))
            state=dict(criterion='old each-pair rule',selected={},attempts={})
            state['attempts']['nnunet__dice_bands_sdf__r00']=dict(model='nnunet',objective='dice_bands_sdf',
                candidate=0,specification=spec,status='rejected',results=rows(.8)[:1])
            state['attempts']['swinunetr__dice_bands__r00']=dict(model='swinunetr',objective='dice_bands',
                candidate=0,specification=spec,status='rejected',results=rows(2.))
            write(root/'constraint_search/search_status.json',state)
            launches=[]
            class FakeDrive:
                def __init__(self,*a):pass
                def upload(self,*a):pass
            def launch(command,**kw):
                payload=read(Path(command[command.index('--payload')+1]))
                self.assertFalse(payload['screening'])
                self.assertEqual(payload['evaluation_policy'],POLICY)
                trial=Path(command[command.index('--work')+1]);launches.append(trial.name)
                r=rows(2.);r[0]['delta_dice_pp']=-.1
                write(trial/'campaign_status.json',dict(status='complete_requested_matrix',results=r))
                (root/'constraint_search/STOP').touch()
            report=({'verified':72,'conditions':504},root/'matrix.csv',root/'summary.json')
            with patch('scripts.colab_hfh.constraint_mean_search.Drive',FakeDrive),\
                 patch('scripts.colab_hfh.constraint_search.recipes',return_value=[spec]),\
                 patch('scripts.colab_hfh.constraint_search.diagnosis_recipes',return_value=[]),\
                 patch('scripts.colab_hfh.constraint_mean_search.make_payload',side_effect=lambda *a,**kw:{'screening':kw['screening']}),\
                 patch('scripts.colab_hfh.constraint_mean_search.subprocess.run',side_effect=launch),\
                 patch('scripts.colab_hfh.constraint_report.write_report',return_value=report):
                run_mean(argparse.Namespace(root=root,hours=8,models=['nnunet','swinunetr']))
            self.assertEqual(launches,['nnunet__dice_bands_sdf__r00'])
            final=read(root/'constraint_search/search_status.json')
            self.assertEqual(final['selected']['swinunetr__dice_bands']['frozen_under_policy'],POLICY['version'])
            self.assertEqual(final['selected']['nnunet__dice_bands_sdf']['frozen_under_policy'],POLICY['version'])
            self.assertEqual(final['attempts']['nnunet__dice_bands_sdf__r00']['previous_selection_decision']['status'],'rejected')


if __name__ == '__main__': unittest.main()
