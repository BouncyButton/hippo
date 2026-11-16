import unittest
from mni_runtime import submission_flags, lane_command


class FixedProfileTests(unittest.TestCase):
    def test_exact_allocation_and_worker_binding(self):
        flags = submission_flags(dict(required_gpu_profile='balanced', workers=2))
        for expected in ('--nodes=1', '--ntasks=2', '--cpus-per-task=4', '--mem=80G',
                         '--gres=gpu:4g.40gb:2', '--export=ALL,HIPPO_GPU_PROFILE=balanced'):
            self.assertIn(expected, flags)
        base = ['srun', '--gres=gpu:4g.40gb:1', '--gpus-per-task=4g.40gb:1']
        self.assertIn('--gres=gpu:4g.40gb:1', lane_command(base, 0, 'balanced'))
        self.assertIn('--gres=gpu:4g.40gb:1', lane_command(base, 1, 'balanced'))

    def test_rejects_unapproved_profile_and_worker_count(self):
        for profile, workers in [('dual_full',2), ('mixed',2), ('balanced',3), (None,2)]:
            with self.assertRaises(ValueError):
                submission_flags(dict(required_gpu_profile=profile, workers=workers))


if __name__ == '__main__':
    unittest.main()
