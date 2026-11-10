"""Checkpoint placement and exact optimizer replay, without research training."""
import ast
from copy import deepcopy
import io
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]


def checkpoint_loader(path):
    """Exercise the actual load expression without importing GPU training dependencies."""
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == 'torch' and node.func.attr == 'load']
    assert len(calls) == 1
    return compile(ast.Expression(calls[0]), str(path), 'eval')


class ResumeTests(unittest.TestCase):
    def test_both_checkpoint_paths_keep_host_state_on_cpu(self):
        files = [ROOT/'scripts/colab_geofwi_seed_search/worker.py',
                 ROOT/'reports/geofwi3d_colab_seed_search_20261018/bundle/source/scripts/cross_dataset_campaign/capped_core.py']
        for path in files:
            with self.subTest(path=path):
                stream = io.BytesIO()
                torch.save({'step': torch.tensor(1.), 'moment': torch.ones(3)}, stream)
                stream.seek(0)
                # Meta exposes accidental whole-checkpoint accelerator remapping without a GPU.
                loaded = eval(checkpoint_loader(path), dict(torch=torch, path=stream,
                              latest=stream, device=torch.device('meta')))
                self.assertEqual(loaded['step'].device.type, 'cpu')
                self.assertEqual(loaded['moment'].device.type, 'cpu')

    def test_adamw_optimizer_places_moments_but_keeps_cpu_step(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.AdamW([parameter], foreach=False)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        target = torch.nn.Parameter(torch.empty(3, device='meta'))
        restored = torch.optim.AdamW([target], foreach=False)
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual(restored.state[target]['step'].device.type, 'cpu')
        self.assertEqual(restored.state[target]['exp_avg'].device.type, 'meta')
        self.assertEqual(restored.state[target]['exp_avg_sq'].device.type, 'meta')

    def replay(self, device):
        for kind in ('adamw', 'sgd'):
            with self.subTest(device=device, optimizer=kind):
                torch.manual_seed(17)
                model = torch.nn.Linear(3, 2).to(device)
                optimizer = (torch.optim.AdamW(model.parameters(), foreach=False) if kind == 'adamw'
                             else torch.optim.SGD(model.parameters(), lr=.01, momentum=.95))
                x = torch.ones(4, 3, device=device)
                def step():
                    optimizer.zero_grad()
                    loss = model(x).square().mean()
                    loss.backward(); optimizer.step()
                    return loss.detach().clone()
                step()
                stream = io.BytesIO()
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
                expected_loss = step()
                expected_model = deepcopy(model.state_dict())
                expected_optimizer = deepcopy(optimizer.state_dict())
                stream.seek(0)
                saved = torch.load(stream, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
                self.assertTrue(torch.equal(expected_loss, step()))
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, expected_model[key]))
                for key, state in optimizer.state_dict()['state'].items():
                    for name, value in state.items():
                        expected = expected_optimizer['state'][key][name]
                        self.assertEqual(value.device, expected.device)
                        self.assertTrue(torch.equal(value, expected))

    def test_exact_cpu_replay(self):
        self.replay('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally; notebook preflight required')
    def test_exact_cuda_replay(self):
        self.replay('cuda')


if __name__ == '__main__':
    unittest.main()
