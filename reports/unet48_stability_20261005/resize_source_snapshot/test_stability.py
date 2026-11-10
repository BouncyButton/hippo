"""Regression checks for protocol isolation and budget-independent learning rates."""
import unittest
from train_stability import fixed_learning_rate, install


class StabilityTests(unittest.TestCase):
    def test_resize_decoder_isolates_upsampling(self):
        import torch
        from monai.networks.nets import UNet
        from train_unet48_budget_pilot import KWARGS
        from train_resize_decoder import build_resize_model
        torch.manual_seed(0)
        original = UNet(**KWARGS)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(0)
        replacement = build_resize_model()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in original.parameters()),
                         sum(p.numel() for p in replacement.parameters()))
        old = original.model[2][0].conv
        new = replacement.model[2][0].conv[1]
        self.assertTrue(torch.equal(old.weight.transpose(0, 1), new.weight))
        old_state = original.state_dict()
        for key, value in replacement.state_dict().items():
            if not key.startswith('model.2.0.conv.'):
                self.assertTrue(torch.equal(value, old_state[key]), key)
        torch.set_num_threads(4)
        image = torch.randn(1, 1, 64, 64, 64)
        result = replacement(image)
        self.assertEqual(result.shape, (1, 3, 64, 64, 64))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(new.weight.grad).all())

    def test_absolute_schedule(self):
        self.assertAlmostEqual(fixed_learning_rate(0, 1e-4), 1e-6)
        self.assertEqual(fixed_learning_rate(99, 1e-4), 1e-4)
        self.assertEqual(fixed_learning_rate(3000, 1e-4), 1e-4)
        with self.assertRaises(ValueError):
            fixed_learning_rate(-1, 1e-4)

    def test_actual_optimizer_and_loss(self):
        import torch
        import compute_constraint_objectives as objectives
        recipe = {'include_background': False, 'betas': [.9, .99], 'warmup_updates': 100}
        trainer = install(recipe)
        model = torch.nn.Conv3d(1, 3, 1)
        criterion, optimizer = trainer.base.build_loss_and_optimizer(
            'unet', model, {'initial_lrs': {'unet': 1e-4}, 'adamw_weight_decay': 1e-5})
        self.assertFalse(criterion.include_background)
        self.assertEqual(optimizer.param_groups[0]['betas'], (.9, .99))
        self.assertEqual(trainer.base.learning_rate('unet', 500, 800, 1e-4),
                         trainer.base.learning_rate('unet', 500, 4000, 1e-4))
        logits = model(torch.randn(1, 1, 4, 4, 4))
        labels = torch.randint(0, 3, (1, 1, 4, 4, 4))
        actual = objectives.training_loss(logits, labels, 'dice', {}, .2)
        self.assertTrue(torch.equal(actual, criterion(logits, labels)))
        actual.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
