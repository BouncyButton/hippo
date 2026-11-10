"""Local-only continuation, import, provenance and resume tests; no GPU jobs."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts.colab_adni_seed014 import missing_scope as m, prepare_missing as p

PACKAGE = p.OUT/'package'
sys.path.insert(0, str(p.SOURCE))
from scripts.colab_adni_seed014 import bootstrap_missing as b


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.plan = p.read(PACKAGE/'original_bundle/plan.json')
        self.seed4 = p.read(PACKAGE/'seed4_bundle/plan.json')
        self.scope = p.read(PACKAGE/'continuation.json')

    def test_exact_175_no_overlap_no_seed2_no_legacy_rescheduling(self):
        first = m.missing_plan(self.plan, PACKAGE, [0, 1])['tasks']
        later = m.missing_plan(self.seed4, PACKAGE, [4])['tasks']
        old = p.read(PACKAGE/'historical_snapshot.json')['runs']
        self.assertEqual((len(first), len(later), len(old)), (7, 168, 329))
        ids = [r['id'] for r in first+later+old]
        self.assertEqual(len(set(ids)), 504)
        self.assertEqual({r['seed'] for r in first+later+old}, {0, 1, 4})

    def test_scope_rejects_changed_scientific_task(self):
        bad = deepcopy(self.plan)
        task = next(t for t in bad['tasks'] if t['id'] == self.scope['seed01_tasks'][0]['id'])
        task['fold'] = 0
        with self.assertRaises(ValueError): m.missing_plan(bad, PACKAGE, [0, 1])

    def test_scope_rejects_unauthorized_seed_selection(self):
        with self.assertRaises(ValueError): m.missing_plan(self.plan, PACKAGE, [2])

    def test_original_plan_is_not_mutated(self):
        before = deepcopy(self.plan)
        m.missing_plan(self.plan, PACKAGE, [0, 1])
        self.assertEqual(before, self.plan)

    def test_historical_snapshot_sha_and_baseline_initialization_match_protocol(self):
        self.assertEqual(m.digest(PACKAGE/'historical_snapshot.json'), self.scope['historical_snapshot_sha256'])
        for task in self.scope['seed01_tasks']:
            baseline = f"ADNI_{task['model']}_dice_f{task['fold']}s{task['seed']}"
            init = PACKAGE/'baselines'/baseline/'initialization.json'
            self.assertEqual(m.digest(init), self.scope['baseline_initializations'][baseline])
            protocol = p.read(PACKAGE/'original_bundle/protocols'/task['protocol']/'protocol.json')
            self.assertEqual(p.read(init), {'model_state_sha256':protocol['baseline_reference']['initialization_sha256']})


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.package = self.root/'package'; self.drive = self.root/'drive'
        self.package.mkdir(); self.drive.mkdir()
        self.scope = deepcopy(p.read(PACKAGE/'continuation.json'))
        # Small local stand-ins, while import/immutability logic is unchanged.
        for name in self.scope['static_receipts']:
            target = self.package/'import_state'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PACKAGE/'import_state'/name, target)
        for task, record in self.scope['recovery'].items():
            record['size'] = 5
            target = self.drive/'results_noaug_v1/recovery'/task/record['pointer']['file']
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'bytes')
        write(self.package/'continuation.json', self.scope)
        self.refresh_manifest()

    def refresh_manifest(self):
        write(self.package/'package_manifest.json', {str(f.relative_to(self.package)):m.digest(f)
            for f in self.package.rglob('*') if f.is_file() and f.name!='package_manifest.json'})

    def snapshot(self):
        return {str(f.relative_to(self.drive)):f.read_bytes() for f in self.drive.rglob('*') if f.is_file()}

    def test_first_import_preserves_exact_ledgers_and_pointers_no_old_runs(self):
        b.bootstrap(self.package, self.drive)
        for task, r in self.scope['recovery'].items():
            d = self.drive/'results_noaug_v1/recovery'/task
            self.assertEqual(p.read(d/'charged_ledger.json'),r['ledger'])
            self.assertEqual(p.read(d/'latest.json'),r['pointer'])
        self.assertFalse(list(self.drive.rglob('runs')))

    def test_second_setup_does_not_reset_newer_recovery(self):
        b.bootstrap(self.package, self.drive)
        task, record = next(iter(self.scope['recovery'].items()))
        folder = self.drive/'results_noaug_v1/recovery'/task
        ledger = dict(record['ledger'], charged_attempts=999, training_flops=99999999999999)
        write(folder/'charged_ledger.json',ledger)
        write(folder/'latest.json',dict(file='new.tar',sha256='new',charged_attempts=999))
        (folder/record['pointer']['file']).unlink()
        before = self.snapshot(); b.bootstrap(self.package, self.drive)
        self.assertEqual(before, self.snapshot())

    def test_setup_after_completed_cleanup_does_not_resurrect_recovery(self):
        b.bootstrap(self.package, self.drive)
        folder = next((self.drive/'results_noaug_v1/recovery').iterdir())
        shutil.rmtree(folder)
        before = self.snapshot(); b.bootstrap(self.package, self.drive)
        self.assertEqual(before,self.snapshot())

    def test_missing_archive_blocks_import_without_bootstrap_receipt(self):
        next(self.drive.rglob('*.tar')).unlink()
        with self.assertRaises(RuntimeError): b.bootstrap(self.package,self.drive)
        self.assertFalse((self.drive/'missing_only_bootstrap.json').exists())

    def test_conflicting_preexisting_ledger_never_overwritten(self):
        task = next(iter(self.scope['recovery']))
        path = self.drive/'results_noaug_v1/recovery'/task/'charged_ledger.json'
        write(path,dict(charged_attempts=9999))
        before = path.read_bytes()
        with self.assertRaises(ValueError): b.bootstrap(self.package,self.drive)
        self.assertEqual(before,path.read_bytes())

    def test_partial_import_rerun_idempotent(self):
        task, record = next(iter(self.scope['recovery'].items()))
        folder = self.drive/'results_noaug_v1/recovery'/task
        write(folder/'charged_ledger.json',record['ledger'])
        b.bootstrap(self.package,self.drive)
        self.assertEqual(p.read(folder/'latest.json'),record['pointer'])

    def test_import_preserves_stop(self):
        (self.drive/'results_noaug_v1/STOP').write_text('manual')
        b.bootstrap(self.package,self.drive)
        self.assertEqual((self.drive/'results_noaug_v1/STOP').read_text(),'manual')

    def test_active_controller_blocks_import_without_mutation(self):
        (self.drive/'results_noaug_v1/.adni_controller_lock').mkdir()
        before = self.snapshot()
        with self.assertRaises(RuntimeError): b.bootstrap(self.package,self.drive)
        self.assertEqual(before,self.snapshot())

    def test_changed_scientific_receipt_blocks(self):
        b.bootstrap(self.package,self.drive)
        path = self.drive/'results_noaug_v1/environment_binding.json';write(path,dict(gpu='other'))
        with self.assertRaises(ValueError):b.bootstrap(self.package,self.drive)

    def test_missing_preflight_after_import_never_silently_recreated(self):
        b.bootstrap(self.package,self.drive)
        (self.drive/'results_noaug_v1/preflight.json').unlink()
        with self.assertRaises(RuntimeError): b.bootstrap(self.package,self.drive)


class RecoveryTests(unittest.TestCase):
    setUp = ImportTests.setUp
    refresh_manifest = ImportTests.refresh_manifest
    def runner(self):
        b.bootstrap(self.package,self.drive)
        return SimpleNamespace(rows=[],durable=self.drive/'results_noaug_v1')

    def test_exact_initial_recovery_accepted(self):
        m.require_recovery(self.runner(),self.scope)

    def test_missing_pointer_cannot_restart(self):
        sweep=self.runner();next(sweep.durable.rglob('latest.json')).unlink()
        with self.assertRaises(RuntimeError):m.require_recovery(sweep,self.scope)

    def test_accounting_regression_blocks(self):
        sweep=self.runner();path=next(sweep.durable.rglob('charged_ledger.json'))
        value=p.read(path);value['training_flops']=0;write(path,value)
        with self.assertRaises(ValueError):m.require_recovery(sweep,self.scope)

    def test_charge_checkpoint_gap_blocks(self):
        sweep=self.runner();path=next(sweep.durable.rglob('charged_ledger.json'))
        value=p.read(path);value['charged_attempts']+=1;write(path,value)
        with self.assertRaises(RuntimeError):m.require_recovery(sweep,self.scope)

    def test_completed_and_verified_continuation_may_retire_recovery(self):
        sweep=self.runner();sweep.rows=[{'id':x} for x in self.scope['recovery']]
        shutil.rmtree(sweep.durable/'recovery')
        m.require_recovery(sweep,self.scope)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source';self.source.mkdir()
        self.scope=p.read(PACKAGE/'continuation.json');self.task=self.scope['seed01_tasks'][0]
        for name in ('plan.json',):shutil.copyfile(PACKAGE/'original_bundle'/name,self.source/name)
        protocol=self.task['protocol'];folder=self.source/'protocols'/protocol;folder.mkdir(parents=True)
        shutil.copyfile(PACKAGE/'original_bundle/protocols'/protocol/'protocol.json',folder/'protocol.json')
        self.sweep=SimpleNamespace(load=Mock(),restore_baseline=Mock(),launch=Mock(return_value=True),
            root=self.source,durable=self.root/'drive/results_noaug_v1',pending=[self.task],done_tasks=[],rows=[],
            r=SimpleNamespace(read=p.read,copy_verified=lambda a,b:(b.parent.mkdir(parents=True,exist_ok=True),shutil.copyfile(a,b))))
        write(self.root/'drive/missing_only_bootstrap.json',dict(continuation_sha256=m.digest(PACKAGE/'continuation.json')))
        m.install(self.sweep,PACKAGE,[0,1])

    def test_baseline_dependency_staged_locally_without_old_archive_verifier(self):
        self.sweep.restore_baseline(self.task)
        baseline=f"ADNI_{self.task['model']}_dice_f2s1"
        self.assertEqual((self.source/'work'/baseline/'initialization.json').read_bytes(),
                         (PACKAGE/'baselines'/baseline/'initialization.json').read_bytes())

    def test_old_task_launch_explicitly_refused(self):
        old=p.read(PACKAGE/'historical_snapshot.json')['runs'][0]
        with self.assertRaises(ValueError):self.sweep.launch(old)

    def test_unfinished_import_blocks_before_scheduler_load(self):
        (self.root/'drive/missing_only_bootstrap.json').unlink()
        with self.assertRaises(RuntimeError):self.sweep.load()

    def test_existing_baseline_mismatch_not_overwritten(self):
        baseline=f"ADNI_{self.task['model']}_dice_f2s1"
        path=self.source/'work'/baseline/'initialization.json';write(path,dict(model_state_sha256='bad'))
        with self.assertRaises(ValueError):self.sweep.restore_baseline(self.task)
        self.assertEqual(p.read(path)['model_state_sha256'],'bad')


class PackageTests(unittest.TestCase):
    def test_all_packaged_files_and_zip_crc(self):
        m.verify_package(PACKAGE)
        with zipfile.ZipFile(p.OUT/p.ZIP_NAME) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
            self.assertFalse(any('/work/' in name or '/runs/' in name for name in z.namelist()))

    def test_control_and_scientific_code_byte_identical_to_v3(self):
        for folder in ('control01','control04','seed4_bundle'):
            for source in (p.PRIOR/'package'/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts:
                    self.assertEqual(m.digest(source),m.digest(PACKAGE/folder/source.relative_to(p.PRIOR/'package'/folder)))
        with zipfile.ZipFile(p.ORIGINAL/'adni_replication_code.zip') as z:
            for name in z.namelist():
                if not name.endswith('/'):self.assertEqual(z.read(name),(PACKAGE/'original_bundle'/name).read_bytes())

    def test_notebook_compiles_has_175_and_correct_account_no_old_scan(self):
        book=p.read(p.OUT/p.BOOK_NAME)
        for c in book['cells']:
            if c['cell_type']=='code':ast.parse(c['source']);self.assertEqual(c['outputs'],[])
        code='\n'.join(c['source'] for c in book['cells'] if c['cell_type']=='code')
        self.assertIn('MAX_NEW_RUNS=175',code)
        self.assertIn("account=='filippofocaccia2@gmail.com'",code)
        self.assertNotIn('adni_replication_code.zip',code)
        self.assertIn(p.ZIP_NAME,code)
        self.assertIn(m.digest(p.OUT/p.ZIP_NAME),code)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.package=self.root/'package';self.package.mkdir()
        for name in ('campaign.py','missing_scope.py','historical_snapshot.json','continuation.json'):
            shutil.copyfile(PACKAGE/name,self.package/name)
        (self.package/'control01').mkdir()
        shutil.copyfile(PACKAGE/'control01/drive_io.py',self.package/'control01/drive_io.py')
        self.parent,self.new=self.root/'original',self.root/'seed4'
        self.old_results,self.results=self.root/'oldresults',self.root/'newresults'
        for f in (self.parent,self.new,self.old_results,self.results):f.mkdir()
        for f,source in ((self.parent,'original_bundle'),(self.new,'seed4_bundle')):
            (f/'work').mkdir();shutil.copyfile(PACKAGE/source/'plan.json',f/'plan.json')
        (self.new/'runtime.py').write_text('import json\ndef write(path,value):path.write_text(json.dumps(value))\n')
        self.sentinel=self.old_results/'runs/historical/do_not_scan'
        self.sentinel.parent.mkdir(parents=True);self.sentinel.write_bytes(b'unchanged historical bytes')
        (self.package/'phase.py').write_text('''
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command')
for name in ('bundle','durable','control','snapshot','other-stop'):p.add_argument('--'+name,type=Path)
p.add_argument('--seeds',nargs='+',type=int)
for name in ('lock-fd','workers','max-runs','hours'):p.add_argument('--'+name)
a=p.parse_args();package=Path(__file__).parent
with (a.snapshot.parent/'phases.txt').open('a') as f:f.write(str(a.seeds)+'\\n')
if (a.durable/'inject_failure').exists():raise RuntimeError('fixture failure')
scope=json.loads((package/'continuation.json').read_text())
tasks=scope['seed01_tasks' if a.seeds==[0,1] else 'seed4_tasks']
rows=[dict(t,dice=61.0,archive_validation='verified_in_continuation') for t in tasks]
a.snapshot.write_text(json.dumps(dict(runs=rows,session_launched=0,
 session_stop_reason='fixture pause' if (a.durable/'inject_pause').exists() else None)))
''')
        write(self.package/'package_manifest.json',{str(x.relative_to(self.package)):m.digest(x)
            for x in self.package.rglob('*') if x.is_file()})
        self.command=[sys.executable,str(self.package/'campaign.py'),'run','--parent-bundle',str(self.parent),
                      '--parent-results',str(self.old_results),'--bundle',str(self.new),'--durable',str(self.results)]

    def run_command(self):return subprocess.run(self.command,capture_output=True,text=True)

    def test_combines_only_175_continuations_with_329_labelled_historical_rows(self):
        result=self.run_command();self.assertEqual(result.returncode,0,result.stderr)
        value=p.read(self.results/'comparison.json')
        self.assertEqual((value['completed'],value['historical_saved_count'],value['verified_continuation_count']),(504,329,175))
        self.assertFalse(value['historical_archives_rescanned'])
        self.assertEqual(self.sentinel.read_bytes(),b'unchanged historical bytes')
        self.assertEqual(sum(r['archive_validation']=='historical_saved_snapshot_not_rescanned' for r in value['runs']),329)

    def test_failure_does_not_start_seed4_or_replace_previous_report(self):
        (self.old_results/'inject_failure').touch();target=self.results/'comparison.json';target.write_text('old report')
        result=self.run_command();self.assertNotEqual(result.returncode,0)
        self.assertEqual((self.results/'phases.txt').read_text().splitlines(),['[0, 1]'])
        self.assertEqual(target.read_text(),'old report')

    def test_manual_stop_preserved_without_phase_launch(self):
        marker=self.old_results/'STOP';marker.write_text('manual')
        result=self.run_command();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(marker.read_text(),'manual');self.assertFalse((self.results/'phases.txt').exists())

    def test_clean_phase_stop_prevents_next_phase(self):
        (self.old_results/'inject_pause').touch()
        result=self.run_command();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.results/'phases.txt').read_text().splitlines(),['[0, 1]'])

    def test_duplicate_controller_cannot_launch(self):
        with (self.parent/'work/sweep.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=self.run_command();self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.results/'phases.txt').exists())

    def test_stale_ownership_is_not_deleted(self):
        lease=self.results/'.adni_seeds014_controller_lock';lease.mkdir();(lease/'owner.json').write_text('previous owner')
        result=self.run_command();self.assertNotEqual(result.returncode,0)
        self.assertEqual((lease/'owner.json').read_text(),'previous owner')


class UploadPartsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.input=self.root/'input';self.input.mkdir()
        self.ssd=self.root/'ssd';self.ssd.mkdir()
        self.target=self.root/'extracted'
        source=self.root/'tiny.zip'
        with zipfile.ZipFile(source,'w') as z:z.writestr('proof.txt',b'exact payload')
        content=source.read_bytes();half=len(content)//2;self.expected=m.digest(source)
        for i,part in enumerate((content[:half],content[half:]),1):
            (self.input/(p.ZIP_NAME+f'.part{i:02d}')).write_bytes(part)
        book=p.read(p.OUT/p.BOOK_NAME)
        tree=ast.parse(book['cells'][2]['source'])
        stage=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage_zip')
        namespace=dict(Path=lambda value:self.ssd if value=='/content' else Path(value),
                       phase=lambda label:nullcontext(),INPUT=self.input,sha256=m.digest,shutil=shutil,zipfile=zipfile)
        exec(compile(ast.Module(body=[stage],type_ignores=[]),'notebook_stage','exec'),namespace)
        self.stage=namespace['stage_zip']

    def test_actual_notebook_rejoins_and_checks_before_extracting(self):
        self.stage(p.ZIP_NAME,self.expected,self.target)
        self.assertEqual((self.target/'proof.txt').read_bytes(),b'exact payload')
        self.assertEqual(m.digest(self.ssd/p.ZIP_NAME),self.expected)

    def test_corrupt_or_incomplete_parts_cannot_extract(self):
        (self.input/(p.ZIP_NAME+'.part02')).write_bytes(b'bad')
        with self.assertRaises(AssertionError):self.stage(p.ZIP_NAME,self.expected,self.target)
        self.assertFalse(self.target.exists())

    def test_cached_ssd_code_does_not_reread_drive(self):
        self.stage(p.ZIP_NAME,self.expected,self.target)
        shutil.rmtree(self.input)
        self.stage(p.ZIP_NAME,self.expected,self.target)
        self.assertEqual((self.target/'proof.txt').read_bytes(),b'exact payload')


if __name__=='__main__':unittest.main()
