"""Returned-evidence gate and production hook-order tests; no CUDA or campaigns."""
import copy
import hashlib
import json
from functools import partial
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.colab_adni_parallel import acceleration_gate as gate
from scripts.colab_adni_parallel.prepare import EXPORT, ORIGINAL, notebook
from scripts.colab_adni_parallel.worker import instrument_trainer
from scripts.colab_adni_parallel.test_acceleration import ModuleFixture, tensor_tree
from scripts.colab_adni_parallel.acceleration_candidate import RuntimeAcceleration
from scripts.cross_dataset_campaign.conventional import evaluate as original_evaluate


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.control = Path(self.tmp.name)/'control'
        shutil.copytree(EXPORT/'control', self.control)
        self.root = ORIGINAL/'bundle'
        with zipfile.ZipFile(self.control/'acceleration_evidence.zip') as z:
            self.binding = json.loads(z.read('summary.json'))['binding']
        self.preflight = dict(status='passed', **self.binding)

    def test_returned_evidence_exact_source_all_four_models_and_no_receipt_writes(self):
        before = {p.name:p.read_bytes() for p in self.control.iterdir()}
        receipt = gate.verify(self.control, self.root, self.binding, self.preflight)
        self.assertEqual(receipt['status'], 'verified_external_acceleration')
        self.assertEqual(len(receipt['models']), 4)
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.control.iterdir()})
        source = Path(__file__).with_name('acceleration_candidate.py')
        self.assertEqual(source.read_bytes(), (self.control/source.name).read_bytes())

    def test_failed_or_missing_original_preflight_is_never_waived(self):
        for receipt in ({}, dict(self.preflight, status='failed')):
            with self.assertRaisesRegex(RuntimeError, 'original GPU preflight'):
                gate.verify(self.control, self.root, self.binding, receipt)

    def test_gpu_and_software_change_block_without_rebinding(self):
        durable = Path(self.tmp.name)/'results'; durable.mkdir()
        (durable/'environment_binding.json').write_text(json.dumps(self.binding))
        (durable/'preflight.json').write_text(json.dumps(self.preflight))
        before = {p.name:p.read_bytes() for p in durable.iterdir()}
        changed = copy.deepcopy(self.binding['environment']); changed['gpu'] = 'NVIDIA H100'
        runtime = SimpleNamespace(environment=lambda:changed)
        with self.assertRaisesRegex(RuntimeError, 'binding differs'):
            gate.existing(runtime, self.control, self.root, durable)
        self.assertEqual(before, {p.name:p.read_bytes() for p in durable.iterdir()})
        changed_binding = dict(self.binding, environment=changed)
        with self.assertRaisesRegex(RuntimeError, 'different GPU/software'):
            gate.verify(self.control, self.root, changed_binding, dict(status='passed', **changed_binding))
        changed = copy.deepcopy(self.binding)
        changed['environment']['packages']['torch'] = 'different'
        with self.assertRaisesRegex(RuntimeError, 'different GPU/software'):
            gate.verify(self.control, self.root, changed, dict(status='passed', **changed))

    def test_source_tampering_and_scientific_report_mismatch_block(self):
        p = self.control/'acceleration_candidate.py'; saved = p.read_bytes()
        p.write_bytes(saved+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'package changed'):
            gate.verify(self.control, self.root, self.binding, self.preflight)
        p.write_bytes(saved)
        evidence = self.control/'acceleration_evidence.zip'
        with zipfile.ZipFile(evidence) as z:
            contents = {name:z.read(name) for name in z.namelist()}
        value = json.loads(contents['nnunet_candidate.json'])
        value['scientific']['losses'][0] += .0000001
        contents['nnunet_candidate.json'] = json.dumps(value).encode()
        with zipfile.ZipFile(evidence, 'w') as z:
            for name, data in contents.items():z.writestr(name, data)
        # Even a coherently rehashed package cannot turn a failed comparison into
        # a pass merely by leaving the summary status at passed_equivalence.
        manifest = gate.read(self.control/'control_manifest.json')
        manifest['files']['acceleration_evidence.zip'] = gate.sha(evidence)
        manifest['acceleration']['evidence_sha256'] = gate.sha(evidence)
        (self.control/'control_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Scientific outputs differ'):
            gate.verify(self.control, self.root, self.binding, self.preflight)

    def test_notebook_preflight_routes_only_to_read_only_evidence_gate(self):
        book = notebook('code', 'data', 'control', 'science')
        cell = book['cells'][6]['source']
        self.assertIn("if command == 'preflight':", cell)
        self.assertIn("str(CONTROL/'acceleration_gate.py')", cell)
        self.assertIn("RESULTS = DRIVE_ROOT/'results_noaug_v1'", book['cells'][2]['source'])
        self.assertIn('No augmentation', book['cells'][2]['source'])


class ProductionHookTests(unittest.TestCase):
    def test_after_binding_hooks_preserve_optimizer_rng_charges_and_latest_capture(self):
        import torch
        for optimizer_class in (torch.optim.SGD, torch.optim.AdamW):
            signatures = []
            for accelerated in (False, True):
                torch.manual_seed(73)
                data = [dict(image=torch.randn(1,3,3,3), label=torch.randint(0,2,(1,3,3,3)), case_name=str(i))
                        for i in range(2)]
                stats = dict(completed_updates=0, phase='starting')
                calls, captures, ledgers = [], [], []
                class Dataset:
                    def __init__(self, dataframe, names):self.items=[copy.deepcopy(dataframe[n]) for n in names]
                    def __getitem__(self, i):return self.items[i]
                    def __len__(self):return len(self.items)
                def save(path, value):
                    calls.append(path.name); torch.save(value, path)
                    if path.name == 'checkpoint_latest.pt':captures.append(stats['ledger'].copy())
                def atomic_json(path, value):
                    if Path(path).name == 'budget_ledger.json':ledgers.append(value.copy())
                base = ModuleFixture(PreparedPatients=Dataset, save_checkpoint=save,
                    _load_pkl_dataframe=lambda:data, build_model=lambda:torch.nn.Conv3d(1,2,1),
                    evaluate=partial(original_evaluate, classes=['background','hippocampus']),
                    atomic_json=atomic_json,
                    build_loss_and_optimizer=lambda model:(None, optimizer_class(model.parameters(), lr=.01)))
                trainer = ModuleFixture(calibrate=lambda:dict(weight=.1),
                    signed_distance_maps=lambda y:y.float()*.5,
                    geometry_weights=lambda y,a,p:y.float()*.25,
                    training_loss=lambda prediction,sdf,depth:(prediction-sdf-depth).square().mean())
                exp = SimpleNamespace(base=base, trainer=trainer)
                protocol = dict(dataset='ADNI', augmentation='none', classes=['background','hippocampus'])
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary)
                    def train():
                        dataset=base.PreparedPatients(base._load_pkl_dataframe(), [0,1])
                        model=base.build_model(); _, optimizer=base.build_loss_and_optimizer(model)
                        base.atomic_json('config.json', dict(max_steps=3)); trainer.calibrate()
                        best=dict(weight=torch.zeros(2), step=0)
                        losses, gradients = [], []
                        for step,i in enumerate((0,1,0),1):
                            image,label=dataset[i]['image'][None],dataset[i]['label'][None]
                            sdf=trainer.signed_distance_maps(label)
                            depth=trainer.geometry_weights(label,torch.eye(4).numpy(),{})
                            optimizer.zero_grad();loss=trainer.training_loss(model(image),sdf,depth)
                            loss.backward();gradients.append(tensor_tree([p.grad for p in model.parameters()]))
                            stats['ledger']=dict(charged_attempts=step,training_flops=step*100)
                            base.atomic_json('budget_ledger.json',stats['ledger']);optimizer.step()
                            losses.append(float(loss.detach()))
                            base.save_checkpoint(out/'checkpoint_latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
                            base.save_checkpoint(out/'checkpoint_best.pt',best)
                        return dict(losses=losses, gradients=gradients, state=tensor_tree(model.state_dict()),
                            optimizer=tensor_tree(optimizer.state_dict()), evaluation=base.evaluate(model,dataset,'cpu',False),
                            rng=torch.get_rng_state().tolist(), ledgers=ledgers, captures=captures)
                    trainer.main=train
                    entry=SimpleNamespace(scientific=SimpleNamespace(experiment=exp))
                    candidate=RuntimeAcceleration(exp,protocol,'cpu')
                    instrument_trainer(entry, stats, (lambda:candidate) if accelerated else None)
                    # Simulate a recipe rebind AFTER installing main instrumentation;
                    # installation must happen inside trainer.main, not prematurely.
                    trainer.signed_distance_maps=lambda y:y.float()*.5
                    rebound=trainer.signed_distance_maps
                    signatures.append(trainer.main())
                    self.assertEqual(calls.count('checkpoint_latest.pt'),3)
                    self.assertEqual(calls.count('checkpoint_best.pt'),1 if accelerated else 3)
                    self.assertEqual(stats['completed_updates'],3)
                    if accelerated:
                        self.assertGreater(stats['acceleration_counters']['sdf_hits'],0)
                        self.assertIs(trainer.signed_distance_maps,rebound)
                        self.assertIs(base.PreparedPatients,Dataset)
                        self.assertIs(base.save_checkpoint,save)
            self.assertEqual(signatures[0],signatures[1])

    def test_exception_removes_acceleration_hooks_without_touching_recovery(self):
        base=ModuleFixture(PreparedPatients=object,evaluate=lambda:None,save_checkpoint=lambda *a:None)
        trainer=ModuleFixture(signed_distance_maps=lambda x:x,geometry_weights=lambda *a:None,
            main=lambda:None)
        exp=SimpleNamespace(base=base,trainer=trainer)
        candidate=RuntimeAcceleration(exp,dict(dataset='ADNI',augmentation='none',classes=['bg','fg']),'cpu')
        original=dict(base.__dict__)
        entry=SimpleNamespace(scientific=SimpleNamespace(experiment=exp))
        instrument_trainer(entry,{},lambda:candidate)
        # Observation fails immediately due to the deliberately incomplete fake
        # trainer. Context cleanup must run even before the training loop starts.
        with self.assertRaises(AttributeError):trainer.main()
        for name,value in original.items():self.assertIs(getattr(base,name),value)


if __name__ == '__main__':unittest.main()
