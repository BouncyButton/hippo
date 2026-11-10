"""Regression checks for exact initialization, retained recovery, and v2 packaging."""
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_adni import runtime as r
from scripts.colab_adni_seed014 import initialization_compat as fix, prepare_initfix as p


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from monai.networks.layers import weight_init
        cls.torch = torch
        cls.weight_init = weight_init
        torch.set_num_threads(2)

    def setUp(self):
        torch, layer = self.torch, self.weight_init
        self.native = layer._no_grad_trunc_normal_
        def builder(protocol, architecture):
            model = torch.nn.Linear(32, 16)
            layer._no_grad_trunc_normal_(model.weight, 0., .02, -2., 2.)
            return model
        self.builder = builder
        calls = []
        def capture(tensor, mean, std, a, b):
            before = r.tensor_digest(torch.get_rng_state())
            value = self.native(tensor, mean, std, a, b)
            calls.append(dict(shape=list(value.shape), dtype=str(value.dtype), mean=mean,
                std=std, a=a, b=b, rng_before=before, rng_after=r.tensor_digest(torch.get_rng_state()),
                data=base64.b64encode(value.detach().numpy().tobytes()).decode()))
            return value
        torch.manual_seed(1)
        with patch.object(layer, '_no_grad_trunc_normal_', capture):
            self.expected = builder({}, {})
        self.rng = torch.get_rng_state().clone()
        self.reference = dict(model_sha256=r.model_digest(self.expected),
            rng_after_model=r.tensor_digest(self.rng), calls=calls,
            tensors={k:dict(sha256=r.tensor_digest(v)) for k,v in self.expected.state_dict().items()})
        self.protocol = dict(model_identity='swinunetr',
            baseline_reference=dict(initialization_sha256=self.reference['model_sha256']))
        self.runtime = SimpleNamespace(canonical_initialization=r.canonical_initialization)
        fix.install(self.runtime)

    def run_fixed(self, builder=None, reference=None):
        self.torch.manual_seed(1)
        return self.runtime.canonical_initialization(builder or self.builder,
            self.protocol, {}, reference or self.reference)

    def test_reported_scale_of_native_error_rejected_then_corrected_exactly(self):
        def drift(tensor, *args):
            value = self.native(tensor, *args)
            with self.torch.no_grad():
                value.add_(1.45e-6)
            return value
        with patch.object(self.weight_init, '_no_grad_trunc_normal_', drift):
            self.torch.manual_seed(1)
            with self.assertRaises(AssertionError):
                r.canonical_initialization(self.builder, self.protocol, {}, self.reference)
            actual, audit = self.run_fixed()
            self.assertIs(self.weight_init._no_grad_trunc_normal_, drift)
        self.assertEqual(r.model_digest(actual), r.model_digest(self.expected))
        self.assertTrue(self.torch.equal(self.rng, self.torch.get_rng_state()))
        self.assertTrue(audit['rng_preserved'])
        self.assertTrue(audit['initialization_compat']['tolerance_unchanged'])

    def test_different_seed_still_rejected_and_native_hook_restored(self):
        self.torch.manual_seed(2)
        with self.assertRaisesRegex(ValueError, 'incoming RNG'):
            self.runtime.canonical_initialization(self.builder, self.protocol, {}, self.reference)
        self.assertIs(self.weight_init._no_grad_trunc_normal_, self.native)

    def test_nonportable_weights_still_checked(self):
        def wrong(p, a):
            model = self.builder(p, a)
            with self.torch.no_grad():model.bias.add_(.1)
            return model
        with self.assertRaisesRegex(ValueError, 'outside portable tensors'):
            self.run_fixed(wrong)

    def test_wrong_inverse_cdf_still_fails_original_tolerance(self):
        from scipy.special import erfinv
        with patch('scipy.special.erfinv', side_effect=lambda x: erfinv(x)+.001):
            with self.assertRaises(AssertionError):self.run_fixed()
        self.assertIs(self.weight_init._no_grad_trunc_normal_, self.native)

    def test_rng_consumption_and_reference_tampering_rejected(self):
        ref = deepcopy(self.reference); ref['calls'][0]['rng_after'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'RNG consumption'):self.run_fixed(reference=ref)
        ref = deepcopy(self.reference); ref['model_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'reference differs'):self.run_fixed(reference=ref)

    def test_other_models_are_delegated_without_changing_initializer(self):
        def original(*args):
            self.assertIs(self.weight_init._no_grad_trunc_normal_, self.native)
            return 'unchanged'
        runtime = SimpleNamespace(canonical_initialization=original)
        fix.install(runtime)
        for model in ('nnunet', 'unetr_vit_b16', 'basic_unet16'):
            self.assertEqual(runtime.canonical_initialization(None,dict(model_identity=model),{},{}), 'unchanged')

    def test_install_idempotent_and_unsupported_recipe_blocked(self):
        before = self.runtime.canonical_initialization;fix.install(self.runtime)
        self.assertIs(before,self.runtime.canonical_initialization)
        for tensor, std in ((self.torch.zeros(2,dtype=self.torch.float64),.02),
                            (self.torch.zeros(2),.1)):
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                fix.stable_trunc_normal(tensor,0.,std,-2.,2.)


class ReceiptTests(unittest.TestCase):
    def test_receipt_requires_correct_binding_hashes_rng_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'bundle';control=base/'control';durable=base/'drive'
            for d in (root,control,durable):d.mkdir()
            r.write(root/'plan.json',dict(seeds=[0,1,2]))
            refs={str(s):dict(model_sha256='model'+str(s),rng_after_model='rng'+str(s),calls=[{}]) for s in (0,1)}
            r.write(root/'initialization_reference.json',dict(models=dict(swinunetr=refs)))
            (control/'control_manifest.json').write_text('{}')
            (control/'initialization_compat.py').write_bytes(Path(fix.__file__).read_bytes())
            binding=dict(bundle_sha256='same bundle',environment=dict(gpu='A100'))
            receipt=dict(status='passed_cpu_initialization',binding=binding,
                control_sha256=r.digest(control/'control_manifest.json'),
                compat_sha256=r.digest(control/'initialization_compat.py'),seeds=[0,1],optimizer_updates=0,
                checks=[dict(seed=s,model_sha256=refs[str(s)]['model_sha256'],rng_after_model=refs[str(s)]['rng_after_model'],portable_calls=1,folds_checked=[0,1,2]) for s in (0,1)])
            preserved={'charged_ledger.json':b'charged work','checkpoint_latest.pt':b'checkpoint','preflight.json':b'original preflight'}
            for name, data in preserved.items():(durable/name).write_bytes(data)
            r.write(durable/fix.RECEIPT_NAME,receipt)
            self.assertEqual(fix.verify_receipt(r,control,root,durable,binding),receipt)
            for key,bad in [('binding',{}),('control_sha256','bad'),('compat_sha256','bad'),('seeds',[0]),('optimizer_updates',1),('checks',[])]:
                r.write(durable/fix.RECEIPT_NAME,dict(receipt,**{key:bad}))
                with self.assertRaises((RuntimeError,ValueError)):fix.verify_receipt(r,control,root,durable,binding)
            for name,data in preserved.items():self.assertEqual((durable/name).read_bytes(),data)


class PackageTests(unittest.TestCase):
    def test_zip_integrity_frozen_bundles_and_training_backend_unchanged(self):
        with zipfile.ZipFile(p.OUT/p.ZIP_NAME) as archive:
            self.assertIsNone(archive.testzip())
            manifest=json.loads(archive.read('package_manifest.json'))
            for name,sha in manifest.items():self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),sha,name)
            self.assertFalse(any('/data/' in n or '/work/' in n for n in archive.namelist()))
            for name in manifest:
                if name.startswith('seed4_bundle/') or name in ('campaign.py','phase.py'):
                    self.assertEqual(archive.read(name),(p.old.OUT/'package'/name).read_bytes(),name)
            for control in ('control01','control04'):
                for name in ('scheduler.py','archive_verification.py','local_storage.py','acceleration_candidate.py','acceleration_evidence.zip'):
                    path=control+'/'+name
                    self.assertEqual(archive.read(path),(p.old.OUT/'package'/path).read_bytes(),path)
                m=json.loads(archive.read(control+'/control_manifest.json'))
                for name,sha in m['files'].items():self.assertEqual(hashlib.sha256(archive.read(control+'/'+name)).hexdigest(),sha)
                self.assertIn('verify_receipt(runtime, control, root, durable, binding)',archive.read(control+'/acceleration_gate.py').decode())
                worker=archive.read(control+'/worker.py').decode()
                self.assertLess(worker.index('install(runtime)'),worker.index('entry.main()'))
        for name,sha in r.read(p.old.PARENT/'bundle_manifest.json').items():
            self.assertEqual(r.digest(p.old.PARENT/name),sha,name)

    def test_complete_notebook_uses_same_results_and_new_zip_with_no_reset(self):
        book=r.read(p.OUT/p.BOOK_NAME)
        text='\n'.join(c['source'] for c in book['cells'])
        for cell in book['cells']:
            if cell['cell_type']=='code':ast.parse(cell['source']);self.assertEqual(cell['outputs'],[])
        for value in (p.ZIP_NAME,r.digest(p.OUT/p.ZIP_NAME),"ORIGINAL_RESULTS=DRIVE_ROOT/'results_noaug_v1'",
                      "RESULTS=DRIVE_ROOT/'results_seeds014_noaug_v1'",'preflight_initfix.py',"campaign('run'","campaign('report'"):
            self.assertIn(value,text)
        self.assertNotIn('shutil.rmtree',text)
        self.assertNotIn('unlink(',text)
        self.assertNotIn("PACKAGE/'preflight.py'",text)


if __name__ == '__main__':unittest.main()
