"""Scientific-scope, archive integrity and interruption regression checks."""
import ast
import base64
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts.colab_msd import runtime as r
from scripts.colab_msd.prepare import notebook

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015'


class BundleTests(unittest.TestCase):
    def test_uploads_reconstruct_a_verified_504_cell_noaug_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('msd_data_verified.zip', 'msd_replication_code.zip'):
                with zipfile.ZipFile(EXPORT/name) as archive:
                    archive.extractall(root)
            plan = r.verify_bundle(root)
            self.assertEqual(len(plan['tasks']), 504)
            references = r.read(root/'initialization_reference.json')['models']
            for model in plan['models']:
                for seed in (0, 1, 2):
                    p = r.read(root/'protocols'/f'{model}_s{seed}'/'protocol.json')
                    self.assertEqual(references[model][str(seed)]['model_sha256'],
                                     p['baseline_reference']['initialization_sha256'])
                    original = r.read(ROOT/'reports/msd_mixed_losses_nnunet60_others10_20261014/frozen/protocols'/model/'protocol.json')
                    # Only scope/portability/initialization-provenance fields may differ.
                    allowed = {'name','data','folds','seeds','parent_protocol_sha256','baseline_reference'}
                    self.assertEqual({k:v for k,v in p.items() if k not in allowed},
                                     {k:v for k,v in original.items() if k not in allowed})
            pfile = root/'protocols/nnunet_s0/protocol.json'
            p = r.read(pfile); p['augmentation'] = 'mild_v1'; r.write(pfile,p)
            # Even if someone regenerates the file inventory, augmentation is rejected.
            manifest = r.read(root/'bundle_manifest.json')
            manifest[str(pfile.relative_to(root))] = r.digest(pfile)
            r.write(root/'bundle_manifest.json',manifest)
            with self.assertRaisesRegex(ValueError, 'Scientific settings'):
                r.verify_bundle(root)

    def test_notebook_has_no_outputs_and_all_code_cells_compile(self):
        book = r.read(EXPORT/'MSD_3fold_3seed_no_augmentation.ipynb')
        self.assertEqual(book['nbformat'],4)
        for cell in book['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                ast.parse(cell['source'])

    def test_setup_failure_exposes_child_stderr_and_keeps_log(self):
        functions = [node for cell in notebook('code', 'data')['cells']
                     if cell['cell_type'] == 'code'
                     for node in ast.parse(cell['source']).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'run_setup_logged']
        self.assertEqual(len(functions), 1, 'Setup must explicitly capture subprocess output')
        scope = dict(subprocess=subprocess, Path=Path)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<setup>', 'exec'), scope)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'install.log'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, 'ERROR: dependency conflict'):
                    scope['run_setup_logged']([sys.executable, '-c',
                        "import sys; print('installer started', flush=True); "
                        "print('ERROR: dependency conflict', file=sys.stderr); sys.exit(7)"], log)
            for text in ('installer started', 'ERROR: dependency conflict'):
                self.assertIn(text, output.getvalue())
                self.assertIn(text, log.read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                scope['run_setup_logged']([sys.executable, '-c', "print('installed')"], log)
            self.assertIn('installed', log.read_text())

    def test_pip_bootstrap_only_when_pip_probe_fails(self):
        functions = [node for cell in notebook('code', 'data')['cells']
                     if cell['cell_type'] == 'code'
                     for node in ast.parse(cell['source']).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'ensure_training_pip']
        runner = Mock()
        scope = dict(subprocess=subprocess, run_setup_logged=runner)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<setup>', 'exec'), scope)
        with patch('urllib.request.urlretrieve') as download, patch.object(subprocess, 'run') as probe:
            probe.return_value.returncode = 0
            scope['ensure_training_pip']('venv/python', 'get-pip.py', 'setup.log')
            download.assert_not_called()
            runner.assert_not_called()
            probe.return_value.returncode = 1
            scope['ensure_training_pip']('venv/python', 'get-pip.py', 'setup.log')
            download.assert_called_once_with('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')
            runner.assert_called_once_with(['venv/python', 'get-pip.py'], 'setup.log')


class RecoveryTests(unittest.TestCase):
    def setup_snapshot(self, root):
        out = root/'work'/'cell'; out.mkdir(parents=True)
        durable = root/'drive'/'cell'
        (out/'checkpoint_latest.pt').write_bytes(b'original optimizer, RNG, sampling and model checkpoint')
        ledger = dict(charged_attempts=3,training_flops=300,calibration_flops=0)
        r.write(out/'budget_ledger.json',ledger)
        r.write(out/'config.json',dict(augmentation='none'))
        r.write(durable/'charged_ledger.json',ledger)
        return out, durable

    def test_coherent_snapshot_round_trip_and_async_inode_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out,durable = self.setup_snapshot(root)
            writer = r.RecoveryWriter(out,durable,interval=0)
            writer.capture()
            # The real trainer atomically replaces its checkpoint, never mutates its inode.
            next_file = out/'next.pt'; next_file.write_bytes(b'newer checkpoint')
            next_file.replace(out/'checkpoint_latest.pt')
            writer.close(force=False)
            recovered = root/'recovered'
            r.restore_recovery(recovered,durable)
            self.assertEqual((recovered/'checkpoint_latest.pt').read_bytes(),
                             b'original optimizer, RNG, sampling and model checkpoint')

    def test_newer_compute_charges_block_silent_budget_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out,durable = self.setup_snapshot(root)
            writer = r.RecoveryWriter(out,durable); writer.close()
            ledger = r.read(durable/'charged_ledger.json'); ledger['charged_attempts'] += 1
            r.write(durable/'charged_ledger.json',ledger)
            with self.assertRaisesRegex(RuntimeError,'budget audit'):
                r.restore_recovery(root/'restore',durable)

    def test_corrupt_transfer_does_not_replace_last_verified_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out,durable = self.setup_snapshot(root)
            writer = r.RecoveryWriter(out,durable); writer.close()
            previous = r.read(durable/'latest.json')
            writer = r.RecoveryWriter(out,durable)
            with patch.object(r,'copy_verified',side_effect=IOError('transfer failed')):
                with self.assertRaises(IOError):
                    writer.close()
            self.assertEqual(r.read(durable/'latest.json'),previous)
            self.assertTrue((durable/previous['file']).exists())

    def test_hash_corruption_blocks_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out,durable = self.setup_snapshot(root)
            writer = r.RecoveryWriter(out,durable); writer.close()
            previous = r.read(durable/'latest.json')
            (durable/previous['file']).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'hash differs'):
                r.restore_recovery(root/'restore',durable)


class InitializationTests(unittest.TestCase):
    def test_portability_restores_exact_values_without_consuming_extra_randomness(self):
        import torch
        from monai.networks.layers import weight_init
        original = weight_init._no_grad_trunc_normal_

        def builder(protocol, architecture):
            model = torch.nn.Linear(8, 8)
            weight_init._no_grad_trunc_normal_(model.weight, 0., .02, -2., 2.)
            return model

        calls = []
        def capture(tensor, mean, std, a, b):
            before = r.tensor_digest(torch.get_rng_state())
            value = original(tensor, mean, std, a, b)
            calls.append(dict(shape=list(value.shape), dtype=str(value.dtype), mean=mean,
                std=std, a=a, b=b, rng_before=before, rng_after=r.tensor_digest(torch.get_rng_state()),
                data=base64.b64encode(value.detach().numpy().tobytes()).decode()))
            return value

        torch.manual_seed(0)
        with patch.object(weight_init, '_no_grad_trunc_normal_', capture):
            expected = builder({}, {})
        after = torch.get_rng_state().clone()
        reference = dict(model_sha256=r.model_digest(expected), rng_after_model=r.tensor_digest(after),
            calls=calls, tensors={k:dict(sha256=r.tensor_digest(v)) for k,v in expected.state_dict().items()})
        protocol = dict(baseline_reference=dict(initialization_sha256=reference['model_sha256']))

        def roundoff(tensor, mean, std, a, b):
            value = original(tensor, mean, std, a, b)
            with torch.no_grad():
                value.add_(1e-9)
            return value

        torch.manual_seed(0)
        with patch.object(weight_init, '_no_grad_trunc_normal_', roundoff):
            actual, audit = r.canonical_initialization(builder, protocol, {}, reference)
            self.assertIs(weight_init._no_grad_trunc_normal_, roundoff)
        self.assertEqual(r.model_digest(actual), r.model_digest(expected))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assertGreater(audit['portable_calls'][0]['max_abs_difference'], 0)

        torch.manual_seed(1)
        with self.assertRaisesRegex(ValueError, 'incoming RNG'):
            r.canonical_initialization(builder, protocol, {}, reference)
        self.assertIs(weight_init._no_grad_trunc_normal_, original)

        def changed_other_weights(protocol, architecture):
            model = builder(protocol, architecture)
            with torch.no_grad():
                model.bias.add_(.1)
            return model
        torch.manual_seed(0)
        with self.assertRaisesRegex(ValueError, 'outside portable tensors'):
            r.canonical_initialization(changed_other_weights, protocol, {}, reference)

        def wrong_distribution(tensor, mean, std, a, b):
            value = original(tensor, mean, std, a, b)
            with torch.no_grad():
                value.add_(.1)
            return value
        torch.manual_seed(0)
        with patch.object(weight_init, '_no_grad_trunc_normal_', wrong_distribution):
            with self.assertRaises(AssertionError):
                r.canonical_initialization(builder, protocol, {}, reference)


if __name__ == '__main__':
    unittest.main()
