"""Regression checks for class identity, A/P swaps and multiclass gradients."""
import unittest
import numpy as np
import torch
from scripts.train_medsam3_multiclass import exclusive_log_probabilities, metrics, multiclass_losses
from scripts.medsam3_supervised_constraints import VolumeSupport


class MulticlassTests(unittest.TestCase):
    def test_swaps_are_errors_even_with_perfect_union(self):
        truth = np.array([0, 0, 1, 1, 1, 2, 2])
        pred = np.array([0, 1, 2, 2, 1, 1, 0])
        result = metrics(truth, pred)
        self.assertEqual(result['anterior_to_posterior'], 2)
        self.assertEqual(result['posterior_to_anterior'], 1)
        self.assertEqual(result['total_error_voxels'], 5)
        np.testing.assert_equal(result['confusion_true_rows_predicted_columns'], [[1, 1, 0], [0, 1, 2], [1, 1, 0]])
        swapped = np.where(truth == 1, 2, np.where(truth == 2, 1, 0))
        self.assertEqual(metrics(truth, swapped)['ap_mean_dice'], 0)
        np.testing.assert_equal(truth > 0, swapped > 0)

    def test_projection_preserves_both_classes_and_background(self):
        logits = torch.tensor([[-12., 12., -12.], [-12., -12., 12.]], requires_grad=True)
        fields = [(torch.nn.functional.logsigmoid(v), torch.nn.functional.logsigmoid(-v)) for v in logits]
        log_p = exclusive_log_probabilities(fields)
        torch.testing.assert_close(log_p.exp().sum(0), torch.ones(3))
        self.assertEqual(log_p.argmax(0).tolist(), [0, 1, 2])
        loss = -log_p[[0, 2, 1], [0, 1, 2]].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(1) > 0).all())

    def test_constraints_detect_internal_ap_swaps(self):
        labels = torch.zeros((3, 9, 9), dtype=torch.long)
        labels[:, 2:7, 2:4] = 1
        labels[:, 2:7, 4:7] = 2
        supports = [VolumeSupport.build(labels == c) for c in range(3)]
        correct = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()*.97+.01
        swapped = correct[:, [0, 2, 1]].clone().requires_grad_()
        good = multiclass_losses(correct[0].log(), correct[1].log(), supports, 0)
        bad = multiclass_losses(swapped[0].log(), swapped[1].log(), supports, 0)
        self.assertGreater(float(bad[0]), float(good[0]))
        self.assertGreater(float(bad[1]), float(good[1]))
        sum(bad).backward()
        self.assertTrue(torch.isfinite(swapped.grad).all())
        self.assertGreater(float(swapped.grad.abs().sum()), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
