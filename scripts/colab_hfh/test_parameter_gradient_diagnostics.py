import unittest
from copy import deepcopy

import torch

from scripts.colab_hfh.parameter_gradient_diagnostics import probe, conservative_flops_allowance, validate_report


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 1)
        self.dropout = torch.nn.Dropout(.5)
        self.register_buffer('calls', torch.tensor(0))

    def forward(self, image):
        self.calls.add_(1)
        full = self.dropout(self.conv(image))
        return [full, torch.nn.functional.avg_pool3d(full, 2), torch.nn.functional.avg_pool3d(full, 4)]


class ToyCriterion(torch.nn.Module):
    weight_factors = (.6, .4, 0.)

    def loss(self, output, target):
        return torch.nn.functional.cross_entropy(output, target[:, 0])

    def forward(self, outputs, targets):
        return sum(w*self.loss(o,t) for w,o,t in zip(self.weight_factors,outputs,targets) if w)


class GradientProbeTests(unittest.TestCase):
    def test_decomposition_and_probe_purity(self):
        torch.manual_seed(42)
        model = ToyModel().train()
        model.dropout.eval()  # Mixed modes must survive the probe.
        image = torch.randn(1,1,8,8,8)
        labels = torch.zeros(1,1,8,8,8,dtype=torch.long)
        labels[:,:,2:6,2:6,2:6] = 1
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        state = {k:v.clone() for k,v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        grads = [p.grad.clone() for p in model.parameters()]
        report = probe(model,image,labels,ToyCriterion(),
            lambda outputs,y: .3*torch.nn.functional.cross_entropy(outputs[0],y[:,0]))
        self.assertEqual(report['gradient_names'],['native_total','native_head_0','native_head_1','weighted_auxiliary'])
        gram = report['gradient_gram']
        # Total gradient is the sum of both weighted native head gradients.
        self.assertAlmostEqual(gram[0][0],gram[1][1]+2*gram[1][2]+gram[2][2],places=6)
        self.assertEqual(report['optimizer_updates'],0)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(all(torch.equal(v,state[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),grads)))

    def test_failure_restores_modes_and_buffers(self):
        model=ToyModel().train()
        def bad_loss(*args):
            raise RuntimeError('Probe failure')
        with self.assertRaisesRegex(RuntimeError,'Probe failure'):
            probe(model,torch.ones(1,1,8,8,8),torch.zeros(1,1,8,8,8,dtype=torch.long),ToyCriterion(),bad_loss)
        self.assertTrue(model.training)
        self.assertEqual(model.calls.item(),0)

    def test_separate_probe_allowance(self):
        self.assertEqual(conservative_flops_allowance(355220395319,2,3),3552203953190)

    def test_archive_audit_rejects_changed_metric_and_cost(self):
        class FourHeads(ToyModel):
            def forward(self,image):
                heads=super().forward(image)
                return [*heads,torch.nn.functional.avg_pool3d(heads[0],8)]
        class FourHeadCriterion(ToyCriterion):
            weight_factors=(4/7,2/7,1/7,0.)
        model=FourHeads()
        row=probe(model,torch.randn(1,1,8,8,8),torch.zeros(1,1,8,8,8,dtype=torch.long),
                  FourHeadCriterion(),lambda outputs,y: .1*outputs[0].square().mean())
        names=['subject_01_h1','subject_02_h1']
        config=dict(model_identity='nnunet',seed=1,fold=0,train_cases=['subject_01_h1','subject_01_h2','subject_02_h1'])
        report=dict(status='complete',phase='after_final_training_and_validation_evaluation',
            optimization_updates=0,diagnostic_flops_allowance=1000,selected_training_cases=names,
            model_sha256_before='a'*64,model_sha256_after='a'*64,weights={'bands':.1},
            cases=[dict(case_name=name,**row) for name in names])
        self.assertEqual(validate_report(report,config,{'baseline_flops_per_update':100},{'bands':.1},'a'*64),1000)
        bad=deepcopy(report);bad['cases'][0]['parameter_gradient_norms']['weighted_auxiliary']+=.01
        with self.assertRaisesRegex(ValueError,'summary differs'):
            validate_report(bad,config,{'baseline_flops_per_update':100},{'bands':.1})
        bad=deepcopy(report);bad['diagnostic_flops_allowance']=0
        with self.assertRaisesRegex(ValueError,'accounting differs'):
            validate_report(bad,config,{'baseline_flops_per_update':100},{'bands':.1})


if __name__=='__main__':
    unittest.main()
