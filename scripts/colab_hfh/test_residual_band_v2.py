import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from scripts.colab_hfh.residual_band import ResidualBandLoss
from scripts.colab_hfh.residual_band_v2 import TrainingProbe


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv3d(1,2,1,bias=False)
        with torch.no_grad():self.conv.weight[:,0,0,0,0]=torch.tensor([-1.,1.])
    def forward(self,x):return [self.conv(x)]


def criterion(outputs,targets):return F.cross_entropy(outputs[0],targets[0][:,0])


class RampTests(unittest.TestCase):
    def test_activation_ramp_and_diagnostics_do_not_change_optimizer(self):
        torch.set_num_threads(2)
        truth=np.zeros((64,64,64),dtype=np.uint8);truth[20:44,22:42,20:44]=1
        labels=torch.tensor(truth[None,None]).long()
        pred=truth.copy();pred[20,25:29,25:29]=0;pred[19,30:34,30:34]=1
        good=torch.tensor(np.where(pred,1.,-1.)[None,None]).float()
        bad=torch.full_like(good,-1)
        model=Toy();reference=Toy()
        opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
        refopt=torch.optim.SGD(reference.parameters(),lr=.001,momentum=.9)
        with TemporaryDirectory() as folder:
            output=Path(folder);probe=TrainingProbe(None,output,{})
            probe.model=model
            for attempt,image,expected in [(46,bad,0.),(47,good,1/9),(48,bad,0.),(49,good,2/9)]:
                (output/'budget_ledger.json').write_text(json.dumps(dict(charged_attempts=attempt)))
                opt.zero_grad();refopt.zero_grad()
                outputs=model(image)
                loss=probe(outputs,labels,'dice_bands',{'bands':.01},min(1.,(attempt-45)/9),criterion=criterion)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                refout=reference(image)
                ref_loss=criterion(refout,[labels])+expected*(.01*ResidualBandLoss()(refout[0],labels).loss)
                torch.testing.assert_close(loss,ref_loss,rtol=0,atol=0)
                loss.backward();ref_loss.backward();opt.step();refopt.step()
                rng=torch.get_rng_state().clone()
                probe.finish(image,labels,'cpu',False)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
                row=json.loads(probe.path.read_text().splitlines()[-1])
                self.assertEqual(row['applied_scale'],expected)
                if expected==0:self.assertEqual(row['logit_gradients']['auxiliary_over_native'],0)
                else:self.assertGreater(row['logit_gradients']['auxiliary_over_native'],0)
            rows=[json.loads(x) for x in probe.path.read_text().splitlines()]
            self.assertEqual([r['active_update_count'] for r in rows],[0,1,1,2])
            self.assertIn('parameter_gradients',rows[1])
            s=rows[1]['same_crop_post_update']
            for k in ('fp','fn'):
                self.assertEqual(s[k+'_after']-s[k+'_before'],s[k+'_introduced']['total']-s[k+'_corrected']['total'])
            resumed=TrainingProbe(None,output,{})
            self.assertEqual(resumed.active_count,2)
            with self.assertRaisesRegex(ValueError,'Repeated'):
                resumed(model(good),labels,'dice_bands',{'bands':.01},1.,criterion=criterion)


if __name__=='__main__':unittest.main()
