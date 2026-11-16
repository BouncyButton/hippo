"""Reproduce legal Slurm GPU reuse across sequential, equal-type steps."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class BalancedProbeTests(unittest.TestCase):
    def exercise(self, runtime, duplicate=False):
        commands = []
        def read(path):
            return json.loads(path.read_text())
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        def run(command, **kwargs):
            commands.append(command)
            tasks = int(next(v.split('=')[1] for v in command if v.startswith('--ntasks=')))
            folder = Path(command[command.index('--probe-directory')+1])
            for rank in range(tasks):
                # A new sequential step can get GPU 0 again; a two-rank step
                # must hold separate GPUs for its two simultaneous tasks.
                device = 0 if duplicate else rank
                write(folder/f'{rank}.json', dict(status='passed', visible_devices=1,
                    slurm_job_id='testjob', device_uuid=f'gpu-{device}', total_memory=40*1024**3,
                    host='node', rank=str(rank)))
        source = Path(os.environ.get('HIPPO_PROBE_POOL_SOURCE',
            str(HERE.parents[2]/'scripts/cross_dataset_campaign/adni_foundation_full_pool.py')))
        namespace = dict(os=os, read=read, write=write, digest=lambda p:'sha',
                         subprocess=SimpleNamespace(run=run), OPS=HERE)
        prefix = function(source, 'step_prefix', namespace)
        original = function(source, 'verify_gpu_probe', namespace)
        namespace.update(pool=SimpleNamespace(step_prefix=prefix), balanced_gpu_probe=original,
                         lane_command=lambda command,lane:command)
        probe = function(runtime, 'gpu_probe', namespace)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,
                SLURM_JOB_ID='testjob', HIPPO_GPU_PROFILE='balanced'):
            stage = Path(tmp)
            probe(stage, {'foundation_python':'python'})
            proof = read(stage/'gpu_allocation.json')
        return commands, proof

    def test_old_sequential_probe_reproduces_observed_failure(self):
        with self.assertRaisesRegex(ValueError, 'GPU isolation failed'):
            self.exercise(HERE.parent/'v5/mni_runtime.py')

    def test_two_rank_probe_holds_two_distinct_devices(self):
        commands, proof = self.exercise(HERE/'mni_runtime.py')
        self.assertEqual(len(commands), 1)
        self.assertIn('--ntasks=2', commands[0])
        self.assertIn('--gres=gpu:4g.40gb:2', commands[0])
        self.assertIn('--gpus-per-task=4g.40gb:1', commands[0])
        self.assertEqual({r['device_uuid'] for r in proof['workers']}, {'gpu-0','gpu-1'})
        self.assertEqual(proof['gpu_profile'], 'balanced')

    def test_actual_duplicate_assignment_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'Two distinct isolated GPUs'):
            self.exercise(HERE/'mni_runtime.py', duplicate=True)


if __name__ == '__main__':
    unittest.main()
