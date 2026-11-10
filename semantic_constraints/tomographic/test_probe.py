import unittest

import numpy as np
import torch
from monai.losses import DiceLoss

from .probe import corruptions, dice_loss, gradient_details, scores, step_record, synthetic_logits, summarize


class TestProbe(unittest.TestCase):
    def test_zero_gradient_summary_is_json_safe(self):
        import json
        z = torch.zeros(1, 3, 4, 4, 4)
        y = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        grad = torch.zeros_like(z)
        baseline = scores(z, y)
        row = dict(kind="correct", method="zero", gradient=gradient_details(z, y, grad, grad, grad),
                   steps=[step_record(z, y, grad, "rms", .1, baseline)])
        result = summarize([row])
        self.assertIsNone(result[0]["median_cosine_dice"])
        json.dumps(result, allow_nan=False)

    def test_dice_value_and_gradient_match_monai(self):
        torch.manual_seed(5)
        z = torch.randn(1, 3, 5, 5, 5, requires_grad=True)
        y = torch.randint(3, (1, 5, 5, 5))
        ours = dice_loss(z, y)
        expected = DiceLoss(to_onehot_y=True, softmax=True)(z, y[:, None])
        torch.testing.assert_close(ours, expected)
        torch.testing.assert_close(torch.autograd.grad(ours, z)[0], torch.autograd.grad(expected, z)[0])

    def test_synthetic_argmax_and_nontrivial_corruptions(self):
        truth = np.zeros((20, 20, 20), dtype=np.int64)
        truth[5:15, 4:10, 5:15] = 2
        truth[5:15, 10:16, 5:15] = 1
        for kind, candidate in corruptions(truth):
            decoded = synthetic_logits(candidate).argmax(1)[0].numpy()
            np.testing.assert_array_equal(decoded, candidate)
            self.assertEqual(bool(np.array_equal(candidate, truth)), kind == "correct")

    def test_cross_entropy_direction_and_step_accounting(self):
        torch.manual_seed(5)
        z = torch.randn(1, 3, 5, 5, 5, requires_grad=True)
        y = torch.randint(3, (1, 5, 5, 5))
        grad, = torch.autograd.grad(torch.nn.functional.cross_entropy(z, y), z)
        dg, = torch.autograd.grad(dice_loss(z, y), z)
        report = gradient_details(z.detach(), y, grad, dg, grad)
        for stratum in report["strata"].values():
            self.assertEqual(stratum["harmful"], 0)
        self.assertAlmostEqual(report["cosine_ce"], 1.)
        baseline = scores(z.detach(), y)
        updated = step_record(z.detach(), y, grad, "max", 1., baseline)
        self.assertEqual(updated["introduced"], 0)
        self.assertLess(updated["after"]["errors"], baseline["errors"])


if __name__ == "__main__":
    unittest.main()
