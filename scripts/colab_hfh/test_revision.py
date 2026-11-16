"""Revision checks: unchanged scientific controls, safe staging and checkpointing."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.colab_hfh.revise_exploration import EXPORT, REVISION
from scripts.colab_hfh.storage import required_assets
from scripts.colab_hfh.transport import assemble_asset


class RevisionTests(unittest.TestCase):
    def test_only_declared_protocol_changes_and_original_data(self):
        allowed = {'lr_horizon_updates', 'checkpoint_interval_seconds', 'learning_rate_policy',
                   'campaign_revision', 'hfh_v1_protocol_sha256', 'hfh_v1_lr_horizon_updates', 'pkl_sha256'}
        for path in (REVISION/'bundle/protocols').glob('*/protocol.json'):
            revised = json.loads(path.read_text().replace('/content/hippo_hfh_baselines_v2/', '/content/hippo_hfh/'))
            original_path = EXPORT/'bundle/protocols'/path.parent.name/'protocol.json'
            original = json.loads(original_path.read_text())
            self.assertEqual({k:v for k,v in revised.items() if k not in allowed},
                             {k:v for k,v in original.items() if k not in allowed})
            self.assertEqual(revised['hfh_v1_protocol_sha256'], hashlib.sha256(original_path.read_bytes()).hexdigest())
            schedule = revised['models'][revised['trainer_alias']]['objectives']['dice']
            self.assertEqual(revised['lr_horizon_updates'], schedule['max_steps'])
            self.assertEqual(schedule['max_steps'], revised['budget_flops']//schedule['flops_per_update'])
            self.assertEqual((path.parent/'splits_10patients.json').read_bytes(),
                (original_path.parent/'splits_10patients.json').read_bytes())
        for path in (REVISION/'bundle/data').iterdir():
            self.assertEqual(path.read_bytes(), (EXPORT/'bundle/data'/path.name).read_bytes())

    def test_baseline_never_selects_sam_weights(self):
        transport = {'assets': {'bundle.zip': {'bytes': 1}, 'sam3.pt': {'bytes': 100}}}
        self.assertEqual(list(required_assets(transport, 'baselines')), ['bundle.zip'])
        self.assertEqual(len(required_assets(transport, 'medsam3')), 2)
        self.assertFalse((REVISION/'bundle/weights').exists())
        self.assertFalse((REVISION/'bundle/upstream').exists())

    def test_parts_verified_before_publication_and_local_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b'original bytes'
            part = root/'part'; part.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            spec = {'assets': {'input': {'bytes': len(data), 'sha256': digest,
                    'parts': [{'name':'part', 'bytes':len(data), 'sha256':digest}]}}}
            calls = []
            def fetch(p):
                calls.append(p['name']); return part
            target = root/'input'
            assemble_asset('input', root, target, spec, fetch_part=fetch)
            assemble_asset('input', root, target, spec, fetch_part=fetch)
            self.assertEqual(calls, ['part'])
            self.assertEqual(target.read_bytes(), data)
            target.unlink(); part.write_bytes(b'corrupted data')
            with self.assertRaises(ValueError):
                assemble_asset('input', root, target, spec, fetch_part=fetch)
            self.assertFalse(target.exists())

    def test_actual_checkpoint_closure_throttles_and_forces_clean_stop(self):
        source = (REVISION/'bundle/source/scripts/cross_dataset_campaign/capped_core.py').read_text()
        run = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        commit = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == 'commit')
        body = textwrap.indent(ast.unparse(commit), '    ')
        factory = '''def harness(args, base, time, torch):
    checkpoint_interval, last_checkpoint_at = 60, time.monotonic()
    checkpoint_writes, checkpoint_seconds = 0, 0.
    model = optimizer = augmentation_rng = torch
    configuration = {}
    step = best_step = epoch = cursor = 1
    best_score = .6
    ledger = {'charged_attempts':1}
    permutation, history, losses = [0], [{'step':1}], []
    best_checkpoint = {'step':1}
    best_validation = latest_validation = {'score':.6}
    latest = args.output/'checkpoint_latest.pt'
    amp = False
'''+body+'\n    return commit\n'
        namespace = {}
        exec(factory, namespace)
        now = [0.]
        saves, metadata = [], {}
        fake = SimpleNamespace(state_dict=lambda:{'weight':7}, get_rng_state=lambda:b'rng', get_state=lambda:b'aug')
        base = SimpleNamespace(save_checkpoint=lambda p,s:saves.append((p.name,copy.deepcopy(s))),
            atomic_json=lambda p,s:metadata.update({p.name:copy.deepcopy(s)}),
            write_metrics=lambda p,s:metadata.update({p.name:copy.deepcopy(s)}))
        commit = namespace['harness'](SimpleNamespace(output=Path('/tmp/example')),base,
                                     SimpleNamespace(monotonic=lambda:now[0]),fake)
        commit(force=False)
        self.assertFalse(saves)
        self.assertIn('validation_latest.json',metadata)
        now[0] = 61
        commit(force=False)
        self.assertEqual(len(saves),2)
        now[0] = 62
        commit(force=False)
        self.assertEqual(len(saves),2)
        commit()  # Forced clean stop/final save, even only one second later.
        self.assertEqual(len(saves),4)
        self.assertEqual(saves[-2][1]['torch_rng'],b'rng')
        self.assertEqual(saves[-2][1]['charged_attempts'],1)
        self.assertEqual(metadata['checkpoint_io.json']['writes'],2)

    def test_notebooks_and_bundle_are_bound(self):
        manifest = json.loads((REVISION/'bundle/manifest.json').read_text())
        for name,digest in manifest.items():
            self.assertEqual(hashlib.sha256((REVISION/'bundle'/name).read_bytes()).hexdigest(),digest)
        from scripts.colab_hfh import revise_exploration
        with patch.object(revise_exploration, 'write') as capture:
            revise_exploration.build_notebooks()
        notebooks = [nb for (path, nb), _ in capture.call_args_list if path.suffix == '.ipynb']
        self.assertEqual(len(notebooks), 2)
        for notebook in notebooks:
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    ast.parse(cell['source'])
            self.assertIn('fetch_part=fetch_part',notebook['cells'][2]['source'])
            self.assertNotIn('sha(target)',notebook['cells'][1]['source'])
        namespace = {}
        exec((REVISION/'bundle/source/scripts/colab_hfh/policy.py').read_text(),namespace)
        self.assertEqual(namespace['cells']('baselines')[0],('nnunet',0))
        complete = {cell:.5 for cell in namespace['cells']('baselines')}
        self.assertEqual(namespace['decision']('baselines',complete)[0],'accepted')
        complete[('swinunetr',2)] = .49
        self.assertEqual(namespace['decision']('baselines',complete)[0],'rejected')


if __name__ == '__main__':
    unittest.main()
