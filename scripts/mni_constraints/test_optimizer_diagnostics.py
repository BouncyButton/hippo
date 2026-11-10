"""Instrumentation must not change optimizer state or model parameters."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from scripts.nnunet_band_objective import native_loss
from scripts.mni_constraints.optimizer_diagnostics import OptimizerDiagnostics
from scripts.cross_dataset_campaign.common import read


class InstrumentationTest(unittest.TestCase):
    def test_diagnostics_preserve_exact_training(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        reference = torch.nn.Sequential(torch.nn.Conv3d(1, 8, 3, padding=1), torch.nn.LeakyReLU(), torch.nn.Conv3d(8, 4, 1))
        observed = deepcopy(reference)
        loss = DC_and_CE_loss(dict(batch_dice=True, smooth=1e-5, do_bg=False, ddp=False), {}, dice_class=MemoryEfficientSoftDiceLoss)
        weights = np.array([1., .5, .25, 0.])
        criterion = DeepSupervisionWrapper(loss, weights / weights.sum())
        optimizers = [torch.optim.SGD(m.parameters(), lr=.01, momentum=.95, nesterov=True, weight_decay=3e-5) for m in (reference, observed)]
        images = torch.randn(3, 1, 16, 16, 16)
        labels = torch.randint(0, 4, (3, 1, 16, 16, 16))
        with tempfile.TemporaryDirectory() as temp:
            diagnostics = OptimizerDiagnostics(observed, optimizers[1], Path(temp), probe_steps=(1, 3))
            for step in range(1, 4):
                for index, (model, optimizer) in enumerate(zip((reference, observed), optimizers)):
                    optimizer.zero_grad(set_to_none=True)
                    z = model(images[step - 1:step])
                    outputs = [z] + [F.avg_pool3d(z, k) for k in (2, 4, 8)]
                    value = native_loss(outputs, labels[step - 1:step], criterion)
                    if index:
                        diagnostics.observe(outputs, labels[step - 1:step], criterion, value, step, 'synthetic', False)
                    value.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
                    if index:
                        diagnostics.before_step(norm, .01)
                    optimizer.step()
                    if index:
                        diagnostics.after_step()
                for a, b in zip(reference.parameters(), observed.parameters()):
                    self.assertTrue(torch.equal(a, b))
                    self.assertTrue(torch.equal(optimizers[0].state[a]['momentum_buffer'], optimizers[1].state[b]['momentum_buffer']))
            result = read(Path(temp) / 'optimizer_diagnostics.json')
            self.assertEqual(len(result['rows']), 3)
            self.assertEqual(result['optimizer_updates_added'], 0)
            self.assertEqual(sum(r['extra_parameter_gradient_calls'] for r in result['rows']), 8)
            self.assertGreater(result['total_diagnostic_profiled_flops'], 0)


if __name__ == '__main__':
    unittest.main()
