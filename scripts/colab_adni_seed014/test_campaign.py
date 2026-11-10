"""CPU-only packaging, retention, scope, supervisor and failure checks."""
import ast
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_adni_seed014 import campaign as c, derivation, prepare as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = p.read(p.PARENT/'plan.json')
        cls.seed4 = p.read(p.OUT/'package/seed4_bundle/plan.json')
        cls.plan = dict(cls.parent, tasks=c.filtered_plan(cls.parent, (0,1))['tasks']+cls.seed4['tasks'])

    def test_exact_504_target_with_only_168_new_seed_tasks(self):
        self.assertEqual(len(self.plan['tasks']),504)
        self.assertEqual({t['seed'] for t in self.plan['tasks']},{0,1,4})
        self.assertEqual(len({t['id'] for t in self.plan['tasks']}),504)
        self.assertEqual(len(self.seed4['tasks']),168)
        original={t['id']:t for t in self.parent['tasks']}
        for task in self.plan['tasks']:
            if task['seed'] != 4:self.assertEqual(task,original[task['id']])

    def test_every_combined_gain_has_same_model_fold_seed_baseline(self):
        rows=[dict(t,dice=60+t['seed']+5*(t['arm']!='dice')) for t in self.plan['tasks']]
        report=c.comparison(rows,self.plan,(0,1,4),{})
        self.assertEqual(report['completed'],504)
        self.assertTrue(all(r['gain_pp']==(0 if r['arm']=='dice' else 5) for r in report['runs']))
        self.assertTrue(all(r['mean_dice'] is not None for r in report['aggregate']))
        partial=c.comparison(rows[:-1],self.plan,(0,1,4),{})
        self.assertIsNone(partial['aggregate'][-1]['mean_dice'])

    def test_seed2_duplicates_and_wrong_identity_rejected(self):
        task=self.plan['tasks'][0];r=dict(task,dice=70.)
        for rows in ([r,r],[dict(r,seed=2)],[dict(r,fold=2)]):
            with self.assertRaises(ValueError):c.comparison(rows,self.plan,(0,1,4),{})

    def test_nonfinite_report_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            c.comparison([dict(self.plan['tasks'][0],dice=float('nan'))],self.plan,(0,1,4),{})

    def test_missing_baseline_never_filled_from_another_seed(self):
        t=next(t for t in self.plan['tasks'] if t['arm']=='dice_bands' and t['seed']==4)
        r=c.comparison([dict(t,dice=70)],self.plan,(0,1,4),{})
        self.assertIsNone(r['runs'][0]['gain_pp'])


class PackageTests(unittest.TestCase):
    def test_complete_extension_bundle_and_exact_derivation(self):
        root=p.OUT/'package/seed4_bundle'
        runtime=c.load(root/'runtime.py','seed4_runtime_test')
        self.assertEqual(len(runtime.verify_bundle(root)['tasks']),168)
        self.assertEqual(derivation.verify(root,p.PARENT)['status'],'verified_seed_only_derivation')

    def test_original_package_remains_bound_and_seed4_cli_is_only_trainer_change(self):
        self.assertEqual(p.sha(p.PARENT/'bundle_manifest.json'),derivation.PARENT_SHA)
        for name,digest in p.read(p.PARENT/'bundle_manifest.json').items():
            self.assertEqual(p.sha(p.PARENT/name),digest,name)
        source=(p.OUT/'package/seed4_bundle'/derivation.CLI_FILE).read_text()
        self.assertIn(derivation.CLI_NEW,source)
        self.assertEqual(source.replace(derivation.CLI_NEW,derivation.CLI_OLD), (p.PARENT/derivation.CLI_FILE).read_text())

    def test_notebook_compiles_and_preserves_directory_names(self):
        book=p.read(p.OUT/p.BOOK_NAME)
        for cell in book['cells']:
            if cell['cell_type']=='code':
                ast.parse(cell['source']);self.assertEqual(cell['outputs'],[])
        text='\n'.join(cell['source'] for cell in book['cells'])
        self.assertIn("ORIGINAL_RESULTS=DRIVE_ROOT/'results_noaug_v1'",text)
        self.assertIn("RESULTS=DRIVE_ROOT/'results_seeds014_noaug_v1'",text)
        self.assertNotIn('shutil.rmtree',text)
        self.assertIn('seed4 preflight',text)

    def test_zip_crc_manifest_and_sources_match(self):
        with zipfile.ZipFile(p.OUT/p.ZIP_NAME) as z:
            self.assertIsNone(z.testzip())
            manifest=json.loads(z.read('package_manifest.json'))
            for name,digest in manifest.items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest,name)
            self.assertNotIn('seed4_bundle/data/adni_hippocampus_full.pkl',z.namelist())
            for name in ('campaign.py','phase.py','preflight.py'):
                self.assertEqual(z.read(name),(p.SOURCE/name).read_bytes())

    def test_seed4_canonical_hashes_match_completed_controls(self):
        ref=p.read(p.OUT/'seed4_initialization_reference.json')['models']
        base=p.ROOT/'reports/adni_msd_four_models_seeds456_baselines_20261015/runs'
        for model,values in ref.items():
            for fold in (0,1,2):
                original=p.read(base/f'ADNI_{model}_dice_f{fold}s4/initialization.json')
                self.assertEqual(values['4']['model_sha256'],original['model_state_sha256'])

    def test_acceleration_derivation_requires_new_preflight_and_same_environment(self):
        control=p.OUT/'package/control04'
        root=p.OUT/'package/seed4_bundle'
        gate=c.load(control/'acceleration_gate.py','extension_gate_test')
        with zipfile.ZipFile(control/'acceleration_evidence.zip') as z:
            binding=json.loads(z.read('summary.json'))['binding']
        new=dict(binding,bundle_sha256=p.sha(root/'bundle_manifest.json'))
        with self.assertRaises(ValueError):gate.verify(control,root,new,dict(status='passed',**binding))
        # Production layout uses the untouched parent beside the extension.
        with tempfile.TemporaryDirectory() as temp:
            here=Path(temp)
            (here/'hippo_adni_bundle').symlink_to(p.PARENT,target_is_directory=True)
            ext=here/'hippo_adni_seed4_bundle';ext.symlink_to(root,target_is_directory=True)
            with patch.dict(sys.modules,{'derivation':derivation}):
                receipt=gate.verify(control,ext,new,dict(status='passed',**new))
                self.assertEqual(receipt['derivation']['new_tasks'],168)
                bad=deepcopy(new);bad['environment']['gpu']='different GPU'
                with self.assertRaises(RuntimeError):gate.verify(control,ext,bad,dict(status='passed',**bad))


FAKE_PHASE = r'''
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command')
for name in ('bundle','durable','control','snapshot','other-stop'):p.add_argument('--'+name,type=Path)
p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--workers');p.add_argument('--max-runs',type=int);p.add_argument('--hours');p.add_argument('--lock-fd',type=int)
a=p.parse_args();os.fstat(a.lock_fd)
log=a.snapshot.parent/'launched_phases.jsonl'
with log.open('a') as f:f.write(json.dumps(a.seeds)+'\n')
if (a.durable/'inject_failure').exists():sys.exit(9)
rows=[dict(t,dice=60.+t['seed']) for t in json.loads((a.bundle/'plan.json').read_text())['tasks'] if t['seed'] in a.seeds]
value=dict(runs=rows,session_launched=0 if a.command=='report' else min(1,a.max_runs),session_stop_reason='test clean stop' if (a.durable/'inject_pause').exists() else None)
a.snapshot.write_text(json.dumps(value))
'''


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.package=self.root/'package';self.package.mkdir()
        shutil.copyfile(p.SOURCE/'campaign.py',self.package/'campaign.py')
        (self.package/'phase.py').write_text(FAKE_PHASE)
        self.parent=self.root/'hippo_adni_bundle';self.new=self.root/'hippo_adni_seed4_bundle'
        self.old_results=self.root/'original';self.results=self.root/'new'
        for path in (self.parent,self.new,self.old_results,self.results):path.mkdir()
        for path in (self.parent,self.new):(path/'work').mkdir()
        task=lambda s:dict(id=f'ADNI_nnunet_dice_f0s{s}',model='nnunet',arm='dice',fold=0,seed=s)
        base=dict(models=['nnunet'],arms=['dice'],folds=[0],tasks=[task(s) for s in (0,1,2)])
        write(self.parent/'plan.json',base)
        write(self.new/'plan.json',dict(base,tasks=[task(4)]))
        (self.new/'runtime.py').write_text('import json\ndef write(path,value):path.write_text(json.dumps(value))\n')
        for name in ('comparison.json','recovery/charged_ledger.json','recovery/checkpoint.pt','runs/completed/archive_manifest.json'):
            path=self.old_results/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'preserve these original bytes')
        self.before={str(x.relative_to(self.old_results)):x.read_bytes() for x in self.old_results.rglob('*') if x.is_file()}
        self.cmd=[sys.executable,str(self.package/'campaign.py'),'run','--parent-bundle',str(self.parent),'--parent-results',str(self.old_results),'--bundle',str(self.new),'--durable',str(self.results)]

    def run_command(self):return subprocess.run(self.cmd,capture_output=True,text=True)

    def check_preserved(self):
        self.assertEqual(self.before,{str(x.relative_to(self.old_results)):x.read_bytes() for x in self.old_results.rglob('*') if x.is_file()})

    def test_completed_original_archives_recovery_and_report_preserved(self):
        r=self.run_command();self.assertEqual(r.returncode,0,r.stderr)
        self.check_preserved()
        rows=p.read(self.results/'comparison.json')['runs']
        self.assertEqual({r['seed'] for r in rows},{0,1,4})
        self.assertFalse((self.results/'.adni_seeds014_controller_lock').exists())

    def test_failure_does_not_start_next_phase_or_overwrite_report(self):
        (self.old_results/'inject_failure').write_text('yes')
        before=b'previous verified combined report';(self.results/'comparison.json').write_bytes(before)
        r=self.run_command();self.assertNotEqual(r.returncode,0)
        self.assertEqual((self.results/'launched_phases.jsonl').read_text().splitlines(),['[0, 1]'])
        self.assertEqual((self.results/'comparison.json').read_bytes(),before)
        (self.old_results/'inject_failure').unlink();self.check_preserved()

    def test_pause_prevents_seed4_launch(self):
        (self.old_results/'inject_pause').write_text('yes')
        r=self.run_command();self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual((self.results/'launched_phases.jsonl').read_text().splitlines(),['[0, 1]'])

    def test_manual_stop_is_not_cleared_or_ignored(self):
        stop=self.old_results/'STOP';stop.write_text('manual')
        r=self.run_command();self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(stop.read_text(),'manual')
        self.assertFalse((self.results/'launched_phases.jsonl').exists())

    def test_duplicate_local_controller_rejected(self):
        with (self.parent/'work/sweep.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            r=self.run_command();self.assertNotEqual(r.returncode,0)
        self.assertFalse((self.results/'launched_phases.jsonl').exists());self.check_preserved()

    def test_stale_drive_controller_ownership_preserved(self):
        marker=self.results/'.adni_seeds014_controller_lock';marker.mkdir()
        (marker/'owner.json').write_text('old owner')
        r=self.run_command();self.assertNotEqual(r.returncode,0)
        self.assertEqual((marker/'owner.json').read_text(),'old owner')
        self.assertFalse((self.results/'launched_phases.jsonl').exists());self.check_preserved()


if __name__=='__main__':unittest.main()
