"""CPU equivalence/safety tests. CUDA equivalence remains an explicit user-run gate."""
from contextlib import nullcontext
import copy
from functools import partial
from pathlib import Path
import tempfile
import sys
import fcntl
import subprocess
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'baselines/nnUNet'))
from scripts.colab_adni_parallel.acceleration_candidate import RuntimeAcceleration, evaluate
from scripts.colab_adni_parallel.validate_acceleration import compare
from scripts.cross_dataset_campaign.conventional import evaluate as original_evaluate


class ModuleFixture(SimpleNamespace):
    __hash__ = object.__hash__


def tensor_tree(value):
    if isinstance(value, torch.Tensor):return value.detach().tolist()
    if isinstance(value, dict):return {k:tensor_tree(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):return [tensor_tree(v) for v in value]
    return value


class CandidateTests(unittest.TestCase):
    def test_binary_metrics_case_order_and_rng_are_exact(self):
        for batch_count in (1,3,7):
            results=[]
            for function in (original_evaluate,evaluate):
                torch.manual_seed(17)
                model=torch.nn.Conv3d(1,2,1)
                data=[dict(image=torch.randn(1,4,4,4),label=torch.randint(0,2,(1,4,4,4)),case_name=str(i))
                      for i in range(batch_count)]
                result=function(model,data,torch.device('cpu'),False,['background','hippocampus'])
                results.append((result,torch.get_rng_state().tolist()))
            self.assertEqual(results[0],results[1])

    def test_cache_and_checkpoint_hooks_preserve_loss_gradients_optimizer_and_rng(self):
        for optimizer_class in (torch.optim.SGD,torch.optim.AdamW):
            signatures=[]
            for accelerated in (False,True):
                torch.manual_seed(42)
                data=[dict(image=torch.randn(1,4,4,4),label=torch.randint(0,2,(1,4,4,4)),case_name=str(i)) for i in range(3)]
                class Dataset:
                    def __init__(self,frame,names): self.items=[copy.deepcopy(frame[i]) for i in names]
                    def __getitem__(self,i):return self.items[i]
                    def __len__(self):return len(self.items)
                calls=[]
                def save(path,value):calls.append(path.name);torch.save(value,path)
                base=ModuleFixture(PreparedPatients=Dataset,save_checkpoint=save,
                    evaluate=partial(original_evaluate,classes=['background','hippocampus']))
                trainer=ModuleFixture(signed_distance_maps=lambda labels:labels.float()*.5,
                    geometry_weights=lambda labels,affine,settings:labels.float()*.25)
                exp=SimpleNamespace(base=base,trainer=trainer)
                candidate=RuntimeAcceleration(exp,dict(dataset='ADNI',augmentation='none',classes=['background','hippocampus']),'cpu')
                with tempfile.TemporaryDirectory() as tmp, candidate if accelerated else nullcontext():
                    dataset=base.PreparedPatients(data,[0,1,2]);model=torch.nn.Conv3d(1,2,1)
                    optimizer=optimizer_class(model.parameters(),lr=.01)
                    losses=[];grads=[]
                    for i in (0,1,0):
                        optimizer.zero_grad();item=dataset[i];label=item['label'][None]
                        sdf=trainer.signed_distance_maps(label)
                        depth=trainer.geometry_weights(label,torch.eye(4).numpy(),{})
                        loss=(model(item['image'][None])-sdf-depth).square().mean()
                        loss.backward();losses.append(loss.item())
                        grads.append([p.grad.tolist() for p in model.parameters()]);optimizer.step()
                    state=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},step=3)
                    path=Path(tmp)/'checkpoint_best.pt'
                    base.save_checkpoint(path,state);before=path.read_bytes();base.save_checkpoint(path,state)
                    self.assertEqual(path.read_bytes(),before)
                    self.assertEqual(calls.count('checkpoint_best.pt'),1 if accelerated else 2)
                    base.save_checkpoint(Path(tmp)/'checkpoint_latest.pt',state)
                    base.save_checkpoint(Path(tmp)/'checkpoint_latest.pt',state)
                    self.assertEqual(calls.count('checkpoint_latest.pt'),2)
                    signatures.append(dict(losses=losses,grads=grads,model=[p.tolist() for p in model.parameters()],
                        optimizer=tensor_tree(optimizer.state_dict()),rng=torch.get_rng_state().tolist(),
                        metrics=base.evaluate(model,dataset,torch.device('cpu'),False)))
                    if accelerated:
                        self.assertGreater(candidate.stats['sdf_hits'],0)
                        self.assertGreater(candidate.stats['depth_hits'],0)
                self.assertIs(base.PreparedPatients,Dataset)
            self.assertEqual(signatures[0],signatures[1])

    def test_mutated_labels_and_geometry_invalidate_cache(self):
        c=RuntimeAcceleration(SimpleNamespace(),dict(dataset='ADNI',augmentation='none',classes=['bg','fg']),'cpu')
        label=torch.zeros(1,1,2,2,2);calls=[]
        def operation(value,*args):calls.append(1);return value.clone()
        c.cached('sdf',operation,label);c.cached('sdf',operation,label)
        self.assertEqual(len(calls),1)
        label.add_(1);value=c.cached('sdf',operation,label)
        self.assertEqual(len(calls),2);self.assertTrue(torch.equal(value,label))

    def test_exact_comparison_fails_on_any_changed_gradient_or_calibration(self):
        value=dict(scientific=dict(gradients=['same'],calibration=dict(weights={'bands':1})))
        compare(value,copy.deepcopy(value))
        changed=copy.deepcopy(value);changed['scientific']['calibration']['weights']['bands']=1.000001
        with self.assertRaisesRegex(ValueError,'calibration'):compare(value,changed)

    def test_mutating_best_payload_cannot_suppress_a_required_write(self):
        value=dict(model={'weight':torch.zeros(3)},step=1)
        original=RuntimeAcceleration.content_version(value)
        value['model']['weight'].add_(1)
        self.assertNotEqual(original,RuntimeAcceleration.content_version(value))
        original=RuntimeAcceleration.content_version(value)
        value['step']=2
        self.assertNotEqual(original,RuntimeAcceleration.content_version(value))

    def test_candidate_rejects_augmentation_and_multiclass_recipes(self):
        for dataset,aug,classes in [('MSD','none',['bg','fg']),('ADNI','mild_v1',['bg','fg']),('ADNI','none',['bg','a','p'])]:
            with self.assertRaises(ValueError):
                RuntimeAcceleration(SimpleNamespace(),dict(dataset=dataset,augmentation=aug,classes=classes),'cpu')

    def test_validator_refuses_existing_controller_without_touching_results(self):
        script=Path(__file__).with_name('validate_acceleration.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';(root/'work').mkdir(parents=True)
            results=Path(tmp)/'results';results.mkdir()
            (results/'preserve').write_bytes(b'original archive and charge data')
            # Import-only fixture: entering any verification/model path fails.
            (root/'runtime.py').write_text('def activate(root): return root\n')
            command=[sys.executable,str(script),'--bundle',str(root),'--results',str(results),
                     '--output',str(Path(tmp)/'diagnostic')]
            lockpath=root/'work/sweep.lock'
            with lockpath.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('campaign is active',result.stderr)
            (results/'.adni_controller_lock').mkdir()
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('ownership marker is present',result.stderr)
            self.assertEqual((results/'preserve').read_bytes(),b'original archive and charge data')
            self.assertFalse((Path(tmp)/'diagnostic').exists())


if __name__=='__main__':unittest.main()
