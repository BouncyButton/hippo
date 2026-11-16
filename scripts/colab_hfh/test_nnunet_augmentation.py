import ast
import json
from pathlib import Path
import unittest
import zlib

import torch
from scripts.colab_hfh.nnunet_augmentation import augment,shift_pair,tensor_sha,FLOPS_PER_UPDATE


class AugmentationTests(unittest.TestCase):
    def sample(self):
        image=torch.zeros(1,1,64,64,64);image[:,:,8:56,16:48,8:56]=.5
        label=torch.zeros_like(image,dtype=torch.long);label[:,:,25:39,25:39,25:39]=1
        return image,label

    def test_paired_views_rng_and_no_mutation(self):
        image,label=self.sample();before=[tensor_sha(x) for x in (image,label)]
        global_rng=torch.get_rng_state().clone()
        left=torch.Generator().manual_seed(2000);right=torch.Generator().manual_seed(2000)
        changes=0
        for _ in range(20):
            x,y,a=augment(image,label,left);xx,yy,b=augment(image,label,right)
            self.assertTrue(torch.equal(x,xx) and torch.equal(y,yy));self.assertEqual(a,b)
            self.assertEqual(int(y.sum()),int(label.sum()))
            self.assertEqual(set(y.unique().tolist()),{0,1})
            self.assertFalse(bool(y[:,:,:,:16,:].any()) or bool(y[:,:,:,48:,:].any()))
            changes+=a['image_sha256']!=before[0]
        self.assertGreater(changes,0)
        self.assertEqual(before,[tensor_sha(x) for x in (image,label)])
        self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
        saved=left.get_state();x,y,a=augment(image,label,left);left.set_state(saved)
        xx,yy,b=augment(image,label,left);self.assertEqual(a,b)

    def test_alignment_and_clipping(self):
        _,label=self.sample();image=label.float()
        x,y,ok=shift_pair(image,label,(1,0,-1))
        self.assertTrue(ok);self.assertTrue(torch.equal(x,y.float()))
        label[:,:,0,25,25]=1;image=label.float()
        x,y,ok=shift_pair(image,label,(-1,0,0))
        self.assertFalse(ok);self.assertTrue(torch.equal(x,image) and torch.equal(y,label))

    def test_cap_and_isolated_worker_composition(self):
        from scripts.colab_hfh.nnunet_seed0_aug import make_pack,augmented_archive
        from scripts.colab_hfh.nnunet_seed0_aug_worker import patched_run_source,source
        from scripts.colab_hfh.constraint_schedule import training_flops
        root=Path(__file__).resolve().parents[2]
        packed=root/'reports/hfh_colab_exploration_20261111/nnunet_warm_ready_v1/package.zlib'
        bundle=json.loads(zlib.decompress(packed.read_bytes()))
        for arm,steps in [('dice',90),('dice_bands',89)]:
            p=make_pack(bundle['base'],bundle['plan'],arm,Path(__file__).parent)['protocols']['nnunet']['protocol.json']
            s=p['models']['nnunet']['objectives'][arm]
            self.assertEqual(s['max_steps'],steps)
            self.assertLessEqual(training_flops(steps,s),p['budget_flops'])
            self.assertGreater(training_flops(steps+1,s),p['budget_flops'])
            self.assertEqual(s['augmentation_flops_per_update'],FLOPS_PER_UPDATE)
            self.assertEqual(p['seeds'],[0]);self.assertEqual(p['folds'],[0])
        filename=root/'scripts/cross_dataset_campaign/capped_core.py'
        original=filename.read_text()
        run=next(x for x in ast.parse(original).body if isinstance(x,ast.FunctionDef) and x.name=='run')
        patched=patched_run_source(ast.get_source_segment(original,run))
        compile(patched,'augmented_run','exec');compile(source(),'augmented_worker','exec')
        self.assertEqual(patched.count('bind_augmented_case('),1)
        self.assertNotIn('check_no_augmentation(protocol)',patched)
        self.assertTrue(callable(augmented_archive()))
        from types import SimpleNamespace
        from scripts.colab_hfh.nnunet_seed0_aug_worker import install
        import inspect
        ns={};exec(compile(ast.Module(body=[run],type_ignores=[]),str(filename),'exec'),ns)
        trainer=SimpleNamespace(run=ns['run'],calibrate=lambda *a,**k:None,
                                base=SimpleNamespace(),__file__=str(filename))
        getsource=inspect.getsource
        install(trainer,p,'dice_bands')
        self.assertIs(inspect.getsource,getsource)
        self.assertIn('bind_augmented_case',trainer.run.__code__.co_names)
        self.assertIn('validate_policy',trainer.run.__code__.co_names)
        self.assertIn('finish',trainer.run.__code__.co_names)


if __name__=='__main__':unittest.main()
