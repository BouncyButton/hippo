import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.colab_hfh.medsam_constraints_schedule import constraint_ramp, delayed_source, install


class ScheduleTests(unittest.TestCase):
    def protocol(self, delay=100):
        return dict(updates=200, constraint_warmup_updates=20,
                    constraint_recipe=dict(delay=delay))

    def test_exact_activation_boundaries(self):
        p = self.protocol()
        self.assertTrue(all(constraint_ramp(i, p) == 0 for i in range(100)))
        self.assertEqual(constraint_ramp(100, p), .05)
        self.assertEqual(constraint_ramp(118, p), .95)
        self.assertTrue(all(constraint_ramp(i, p) == 1 for i in range(119, 200)))

    def test_legacy_function_is_untouched(self):
        marker = lambda: None
        module = SimpleNamespace(train=marker)
        install(module, self.protocol(0))
        self.assertIs(module.train, marker)

    def test_invalid_schedule_fails(self):
        for delay in (-1, 200, .5):
            with self.assertRaises(ValueError): constraint_ramp(0, self.protocol(delay))

    def test_only_ramp_and_auxiliary_bypass_change(self):
        path = Path(__file__).parents[1]/'cross_dataset_campaign/foundation.py'
        source = path.read_text()
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        original = ast.get_source_segment(source, node)
        adapted = delayed_source(original)
        ast.parse(adapted)
        changed = [(a,b) for a,b in zip(original.splitlines(),adapted.splitlines()) if a!=b]
        self.assertEqual(len(changed), 2)
        self.assertIn("if ramp == 0 else", adapted)
        with self.assertRaises(ValueError): delayed_source(adapted)
        full = delayed_source(original, full_native_graph=True)
        changed = [(a,b) for a,b in zip(original.splitlines(), full.splitlines()) if a!=b]
        self.assertEqual(len(changed), 1)
        self.assertNotIn('if ramp == 0 else', full)


if __name__ == '__main__': unittest.main()
