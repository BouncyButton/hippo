import unittest
import torch
import torch.nn.functional as F
from scripts.colab_hfh.medsam_constraints_deterministic import strict_empty_point_grid
from scripts.colab_hfh.medsam_constraints_policy import STRICT_CONTROL, STRICT_RECIPES, make_protocol


class StrictTests(unittest.TestCase):
    def test_empty_and_nonempty_outputs_and_gradients_match_native(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        native = F.grid_sample
        torch.use_deterministic_algorithms(True)
        try:
            for n in (0, 3):
                value = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
                grid = torch.zeros(2, n, 1, 2, dtype=torch.float64, requires_grad=True)
                expected = native(value,grid,align_corners=False)
                grads = torch.autograd.grad(expected.sum(),(value,grid))
                with strict_empty_point_grid():
                    actual = F.grid_sample(value,grid,align_corners=False)
                    torch.testing.assert_close(actual, expected, rtol=0,atol=0)
                    got = torch.autograd.grad(actual.sum(),(value,grid))
                    for a,b in zip(got,grads):torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertIs(F.grid_sample,native)
        finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)

    def test_new_controls_and_constraints_have_explicit_numerical_binding(self):
        p=dict(arm='dice',updates=200,accumulation=2)
        order=[['a',0]]*400;history=[dict(lr=.00001)]*200
        control=make_protocol(p,order,history,STRICT_CONTROL,'dice','old')
        self.assertEqual(control['training_numerics'],'strict_empty_grid_v1')
        fresh={**control,'arm':'dice'}
        band=make_protocol(fresh,order,history,STRICT_RECIPES[0],'dice_bands','fresh')
        self.assertEqual(band['baseline_archive_sha256'],'fresh')
        self.assertEqual(band['constraint_recipe'],STRICT_RECIPES[0])
        self.assertEqual(band['training_numerics'],control['training_numerics'])
        with self.assertRaises(ValueError):make_protocol(p,order,history,STRICT_RECIPES[0],'dice','old')


if __name__=='__main__':unittest.main()
