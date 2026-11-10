"""Check four-class cache equivalence, immutable recovery and experiment scope."""
import importlib.util
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.colab_medsam3 import runtime
from scripts.colab_mni_medsam3 import optimized, supervisor
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.colab_medsam3.test_fast_eval import Model, Transform

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'reports/medsam3_mni_colab_two_subjects_20261108/bundle'


def three_prompts(model,processor,pixels,protocol=None,labels=None,criterion=None):
    fields=[]
    for text in ('CA1 CA2 CA3','subiculum','CA4 dentate gyrus'):
        value=processor.transform(torch.from_numpy(pixels.copy()))
        image=model.backbone.forward_image(value)
        text_feature=model.backbone.forward_text([text],device='cpu')
        fields.append(image['vision_features']+text_feature['language_features'])
    return torch.stack([torch.zeros_like(fields[0]),*fields]).log_softmax(0),None


class OptimizedTests(unittest.TestCase):
    def test_three_prompt_cache_exact_with_one_image_encode(self):
        spec=importlib.util.spec_from_file_location('mni_frozen_cache',BUNDLE/'source/scripts/colab_medsam3/fast_eval.py')
        fe=importlib.util.module_from_spec(spec);spec.loader.exec_module(fe)
        model=Model().eval();processor=SimpleNamespace(transform=Transform())
        cache=fe.FeatureReuse(model,processor,three_prompts)
        pixels=np.arange(16,dtype=np.uint8).reshape(4,4)
        with torch.inference_mode():
            for image in (pixels,pixels+10):
                expected=three_prompts(model,processor,image)[0]
                actual=cache(image)[0]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(cache.counts['image_encodes'],2)
        self.assertEqual(cache.counts['image_hits'],4)
        self.assertEqual(cache.counts['text_encodes'],3)
        self.assertEqual(cache.counts['text_hits'],3)

    def test_mni_audit_bridge_checks_complete_volume_and_restores_hooks(self):
        model=Model().eval();processor=SimpleNamespace(transform=Transform())
        pixels=np.arange(16,dtype=np.uint8).reshape(4,4)
        protocol=dict(classes=['background','CA1_3','subiculum','CA4_DG'])
        volume=SimpleNamespace(load=lambda case,labels=True:(np.stack([pixels.T]*3,axis=2),None))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'checkpoint_latest.pt').write_bytes(b'checkpoint');write(output/'protocol.json',protocol)
            def reference_evaluate(m,pr,vol,p,out):
                records=[]
                with torch.inference_mode():
                    for case in ('a','b'):
                        image,_=vol.load(case,labels=False)
                        for z in range(image.shape[2]):
                            result=optimized.original.forward_plane(m,pr,image[:,:,z].T,p)[0]
                            torch.testing.assert_close(result,three_prompts(m,pr,image[:,:,z].T,p)[0],rtol=0,atol=0)
                        records.append(dict(case_name=case))
                return records
            old_eval=optimized.original.evaluate
            try:
                with patch.object(optimized.original,'forward_plane',three_prompts),patch.object(optimized.matched,'evaluate',reference_evaluate),patch.dict('os.environ',MNI_MANIFEST_SHA='test'):
                    optimized.install_evaluation()
                    records=optimized.original.evaluate(model,processor,volume,protocol,output)
                    self.assertEqual(len(records),2)
                    receipt=read(output/'evaluation_acceleration.json')
                    self.assertEqual(receipt['full_volume']['checked'],3)
                    self.assertTrue(receipt['prediction_agreement'])
                    self.assertTrue(receipt['rng_agreement'])
                    self.assertIs(optimized.original.forward_plane,three_prompts)
            finally:optimized.original.evaluate=old_eval

    def test_async_recovery_is_immutable_and_restorable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'work';output.mkdir();recovery=root/'recovery'
            for name in ('protocol.json','checkpoint_ledger.json','calibration.json'):write(output/name,dict(step=5))
            checkpoint=output/'checkpoint_latest.pt';checkpoint.write_bytes(b'first checkpoint')
            writer=runtime.RecoveryWriter(output,recovery,interval=0)
            writer.capture()
            replacement=output/'replacement';replacement.write_bytes(b'new checkpoint');replacement.replace(checkpoint)
            writer.close()
            restored=root/'restored';runtime.restore(restored,recovery)
            self.assertEqual((restored/'checkpoint_latest.pt').read_bytes(),b'first checkpoint')
            record=read(recovery/'latest.json')
            self.assertEqual(digest(recovery/record['file']),record['sha256'])

    def test_recipe_changes_only_support_and_metadata(self):
        original=read(ROOT/'reports/medsam3_mni_colab_pilot_20261108/bundle/protocol.json')
        current=read(BUNDLE/'protocol.json')
        for name in ('seed','fold','updates','accumulation','learning_rate','final_learning_rate','warmup_updates','prompts','classes','augmentation','evaluation_cases'):
            self.assertEqual(original[name],current[name],name)
        self.assertTrue(set(original['train_cases'])<set(current['train_cases']))
        self.assertEqual(len(current['train_cases']),4)
        self.assertEqual(len(read(BUNDLE/'plan.json')['tasks']),1)
        for name,sha in read(BUNDLE/'manifest.json').items():self.assertEqual(digest(BUNDLE/name),sha,name)

    def test_completed_evaluation_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'protocol.json',{});write(p/'evaluation.json',{})
            write(p/'completion.json',dict(status='complete',evaluation_sha256=digest(p/'evaluation.json'),protocol_sha256=digest(p/'protocol.json')))
            self.assertTrue(supervisor.verify_completed(p))
            write(p/'evaluation.json',dict(changed=True))
            with self.assertRaises(ValueError):supervisor.verify_completed(p)


if __name__=='__main__':unittest.main()
