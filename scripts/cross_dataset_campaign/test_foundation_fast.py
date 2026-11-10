"""Numeric cache/calibration tests and actual support preflight."""
import argparse
import os
from pathlib import Path
import tempfile
import time
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.cross_dataset_campaign import foundation_fast as fast
from scripts.cross_dataset_campaign import foundation_checks_job as job
from scripts.cross_dataset_campaign.common import read, write


class OptimizationTests(unittest.TestCase):
    def test_validation_restores_sampler_even_on_failed_check(self):
        native = torch.nn.functional.grid_sample
        flags = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
        def fail(*args):
            self.assertIsNot(torch.nn.functional.grid_sample, native)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            raise ValueError('test comparison failure')
        with patch.object(fast, '_runtime_validation', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'test comparison failure'):
                fast.runtime_validation(None, None, None, None, None, None)
        self.assertIs(torch.nn.functional.grid_sample, native)
        self.assertEqual(flags, (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()))

    def test_validation_child_environment_does_not_change_training_environment(self):
        before = dict(os.environ)
        child = fast.validation_environment()
        self.assertEqual(child['HIPPO_FOUNDATION_VALIDATE_ONLY'], '1')
        self.assertEqual(child['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual(dict(os.environ), before)
        child['CUBLAS_WORKSPACE_CONFIG'] = 'changed'
        self.assertEqual(dict(os.environ), before)

    def test_validation_restores_sampler_after_success(self):
        native = torch.nn.functional.grid_sample
        flags = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
        with patch.object(fast, '_runtime_validation', return_value='checked'):
            self.assertEqual(fast.runtime_validation(None, None, None, None, None, None), 'checked')
        self.assertIs(torch.nn.functional.grid_sample, native)
        self.assertEqual(flags, (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()))

    def test_cache_is_exact_isolated_bounded_and_preserves_rng(self):
        calls=[]
        def transform(x):
            calls.append(1)
            return (x.float()/255-.5)*2
        cache=fast.TransformCache(transform, max_bytes=64)
        x=torch.arange(8,dtype=torch.uint8).reshape(1,2,4)
        state=torch.get_rng_state().clone()
        expected=transform(x)
        self.assertTrue(torch.equal(cache(x),expected))
        cached=cache(x);cached.zero_()
        self.assertTrue(torch.equal(cache(x),expected))
        self.assertEqual(len(calls),2)
        for n in range(10):cache(x+n)
        self.assertLessEqual(cache.bytes,64)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        cache.clear();self.assertEqual(cache.bytes,0)

    def test_gram_matches_original_arithmetic_exactly(self):
        generator=torch.Generator().manual_seed(5)
        gradients=[[torch.randn(3,7,generator=generator),None,torch.randn(13,generator=generator)] for _ in range(3)]
        gradients[1][1]=torch.randn(2,generator=generator)
        expected=[[sum(float((a.detach().double()*b.detach().double()).sum())
            for a,b in zip(left,right) if a is not None and b is not None)
            for right in gradients] for left in gradients]
        self.assertEqual(fast.gram_matrix(gradients),expected)

    def test_submit_reuses_owned_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);write(p/'submissions/123.json',dict(job_id='123'))
            with patch.object(job.controller,'queue',return_value=[['123','PENDING','foundation']]):
                self.assertEqual(job.submit(p),dict(job_id='123'))

    def test_pending_replacement_never_cancels_unrelated_job(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);previous=p/'previous';stage=p/'next'
            write(previous/'control.json',dict(pause=False))
            write(previous/'successor.json',dict(job_id='123',manifest_sha256='old'))
            write(stage/'control.json',dict(pause=False))
            write(stage/'preflight.json',dict(status='passed',manifest_sha256='new'))
            plan=dict(predecessor=dict(manifest_sha256='old'),storage=dict(minimum_free_gib=6))
            with patch.object(job.chain,'verify',return_value=plan),patch.object(job.chain,'check_predecessor',return_value=(previous,{})),patch.object(job.chain,'digest',return_value='new'),patch.object(job.controller,'quota',return_value=dict(free_gib=50)),patch.object(job.controller,'queue',return_value=[['999','PENDING','unrelated']]),patch.object(job.chain.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'unrelated queued'):
                    job.chain.submit(stage)
                run.assert_not_called()

    def test_manual_pause_prevents_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);previous=p/'previous';stage=p/'next'
            write(previous/'control.json',dict(pause=True));write(stage/'control.json',dict(pause=False))
            write(stage/'preflight.json',dict(status='passed',manifest_sha256='new'))
            plan=dict(predecessor=dict(manifest_sha256='old'),storage=dict(minimum_free_gib=6))
            with patch.object(job.chain,'verify',return_value=plan),patch.object(job.chain,'check_predecessor',return_value=(previous,{})),patch.object(job.chain,'digest',return_value='new'),patch.object(job.controller,'quota',return_value=dict(free_gib=50)),patch.object(job.controller,'queue',return_value=[]),patch.object(job.chain.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'paused'):
                    job.chain.submit(stage)
                run.assert_not_called()


def support_check(stage):
    results=[]
    for dataset in job.DATASETS:
        p=read(stage/'protocols'/dataset/'protocol.json')
        p={**p,'arm':'dice_bands_edge'}
        slabs=fast.REFERENCE_SUPPORT(fast.original.Volumes(p),p,None)
        schedule=slabs.schedule(p)
        if len(slabs.inventory)!=5 or len(schedule)!=400 or len(slabs.calibration_schedule(4))!=20:
            raise ValueError('Foundation geometry/sampling mismatch')
        if schedule!=slabs.schedule(p):raise ValueError('Sampling is not deterministic')
        results.append(dict(dataset=dataset,inventory=slabs.inventory,slab_samples=len(schedule),calibration_samples=20))
    write(stage/'support_preflight.json',dict(status='passed',datasets=results))
    p=read(stage/'protocols/ADNI/protocol.json')
    sys.path.insert(0,p['upstream'])
    from sam3.model.sam3_image_processor import Sam3Processor
    from torchvision.transforms.v2.functional import to_image
    from PIL import Image
    processor=Sam3Processor(None,device='cpu')
    value=to_image(Image.fromarray(np.random.default_rng(0).integers(0,256,(96,96),dtype=np.uint8)).convert('RGB'))
    cache=fast.TransformCache(processor.transform)
    expected=processor.transform(value)
    if not torch.equal(cache(value),expected) or not torch.equal(cache(value),expected):
        raise ValueError('Actual SAM3 transformed pixels differ')
    times={}
    for name,fn in [('uncached',processor.transform),('cached',cache)]:
        samples=[]
        for _ in range(12):
            start=time.perf_counter();result=fn(value);samples.append(time.perf_counter()-start)
            if not torch.equal(result,expected):raise ValueError('Preprocessing changed')
        times[name]=float(np.median(samples))
    write(stage/'preprocessing_benchmark.json',dict(status='passed',seconds_per_plane=times,
        speedup=times['uncached']/times['cached'],resolution=processor.resolution,
        scope='CPU preprocessing cache-hit microbenchmark, not full fine-tuning speedup',tensors_exact=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path);args=parser.parse_args()
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(OptimizationTests)
    # The training/optimizer/checkpoint implementation is intentionally unchanged.
    from scripts.cross_dataset_campaign.test_resume import Resume
    suite.addTest(Resume('test_foundation_resume_preserves_adapter_optimizer_and_sampling'))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    if args.stage:support_check(args.stage)


if __name__=='__main__':main()
