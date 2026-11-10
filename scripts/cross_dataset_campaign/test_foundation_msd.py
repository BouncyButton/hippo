"""Verify the fallback against the successful MSD optimizer trajectory."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from scripts.cross_dataset_campaign import foundation as f
from scripts.cross_dataset_campaign import foundation_msd as msd
from scripts.cross_dataset_campaign.common import write
from scripts.train_medsam3_constraint_few_shot import train_arm


class OriginalMSDTests(unittest.TestCase):
    def test_native_probe_restores_rng_and_does_not_update_parameters(self):
        class Slabs:
            def calibration_schedule(self, count):
                return [('case', 0)]
        parameter = torch.nn.Parameter(torch.tensor([.3, .7]))
        def forward(*args):
            random.random(); np.random.random()
            loss = (parameter * torch.rand(2)).square().mean()
            return dict(native=loss, bands=loss * .3, edge=loss * .2)
        with tempfile.TemporaryDirectory() as directory, patch.object(f, 'forward_slab', forward):
            output = Path(directory)
            write(output / 'protocol.json', {})
            before = parameter.detach().clone()
            state = torch.get_rng_state().clone()
            python_state, numpy_state = random.getstate(), np.random.get_state()
            msd.native_training_check(None, None, [parameter], Slabs(),
                dict(calibration_seed=19, calibration_slabs_per_volume=4), output)
            self.assertTrue(torch.equal(before, parameter))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(python_state, random.getstate())
            self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
            self.assertIsNone(parameter.grad)

    def test_resumable_loop_matches_successful_msd_train_arm(self):
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = torch.nn.Module()
                self.block.register_parameter('lora_A', torch.nn.Parameter(torch.tensor([.3, .7])))
                self.block.register_parameter('lora_B', torch.nn.Parameter(torch.tensor([.1, .2])))
                self.aux = torch.nn.Module()
                self.aux.register_parameter('lora_A', torch.nn.Parameter(torch.tensor([.4, .6])))
                self.register_parameter('frozen', torch.nn.Parameter(torch.tensor([2.]), requires_grad=False))
        class Slabs:
            indices = {'case': [0, 1, 2]}
            def schedule(self, p):
                return [('case', i % 3) for i in range(p['updates'] * p['accumulation'])]
        def forward(model, criterion, slabs, case, anchor):
            noise = torch.rand(2)
            native = ((model.block.lora_A * model.block.lora_B - noise - .1 * anchor)**2).mean()
            bands = (model.aux.lora_A - noise).square().mean()
            edge = (model.block.lora_A + model.aux.lora_A - noise).square().mean()
            return dict(native=native, bands=bands, edge=edge)
        def original_forward(*args):
            losses = forward(*args)
            return tuple(losses[k] for k in ('native', 'bands', 'edge'))
        protocol = dict(seed=0, updates=6, accumulation=2, learning_rate=.01,
            final_learning_rate=.001, warmup_updates=2, constraint_warmup_updates=2)
        with tempfile.TemporaryDirectory() as temporary, patch.object(f, 'forward_slab', forward), patch.object(f, 'stopping', return_value=False):
            root = Path(temporary)
            for arm, weights in [('baseline', (0., 0.)), ('bands', (.1, 0.)), ('bands_edge', (.1, .2))]:
                before = Adapter(); after = Adapter(); slabs = Slabs()
                target = root / (arm + '_resumable')
                write(target / 'protocol.json', protocol)
                f.train(after, None, [p for p in after.parameters() if p.requires_grad], slabs,
                    protocol, dict(weights=dict(zip(('bands', 'edge'), weights))), arm, target)
                train_arm(before, None, [p for p in before.parameters() if p.requires_grad], slabs,
                    protocol, dict(bands=dict(weight=.1), edge=dict(weight=.2)), arm,
                    root / (arm + '_original'), slabs.schedule(protocol), forward_fn=original_forward)
                for name, value in before.state_dict().items():
                    self.assertTrue(torch.equal(value, after.state_dict()[name]), (arm, name))

    def test_compiled_graph_supports_repeated_backward(self):
        import torch._functorch.config as config
        if not hasattr(config, 'donated_buffer'):
            self.skipTest('Older local PyTorch has no donated buffers; checked on cluster runtime')
        previous = config.donated_buffer
        try:
            msd.configure_retained_backward()
            self.assertFalse(config.donated_buffer)
            @torch.compile(backend='aot_eager')
            def objective(x):
                return x.sin().square().sum()
            value = torch.tensor([.3, .7], requires_grad=True)
            loss = objective(value)
            first = torch.autograd.grad(loss, value, retain_graph=True)[0]
            second = torch.autograd.grad(loss * 2, value)[0]
            self.assertTrue(torch.allclose(second, 2 * first))
        finally:
            config.donated_buffer = previous


if __name__ == '__main__':
    unittest.main(verbosity=2)
