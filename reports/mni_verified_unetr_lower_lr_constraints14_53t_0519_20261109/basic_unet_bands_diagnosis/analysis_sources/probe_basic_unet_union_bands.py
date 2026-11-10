"""CPU algebra probe of the frozen union-bands objective, not a training replay."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NAME='mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109'
STAGE=ROOT/'experiments'/NAME
sys.path[:0]=[str(STAGE/'source'),str(STAGE/'source/scripts')]

import torch
from scripts.mni_constraints import objectives


def main():
    torch.set_num_threads(2)
    assert Path(objectives.__file__).resolve().is_relative_to(STAGE/'source')
    manifest=json.loads((STAGE/'manifest.json').read_text())
    relative=str(Path(objectives.__file__).relative_to(STAGE))
    assert hashlib.sha256(Path(objectives.__file__).read_bytes()).hexdigest()==manifest[relative]
    y=torch.zeros((1,1,12,12,12),dtype=torch.long)
    y[:,:,3:9,3:9,3:9]=1;y[:,:,3:5,3:9,3:9]=3;y[:,:,5:7,3:9,3:9]=2
    z=torch.full((1,4,12,12,12),-2.)
    z[:,0]=3.
    for c in (1,2,3):
        z[:,0][y[:,0]==c]=-2.;z[:,c][y[:,0]==c]=2.
    voxel=(0,3,4,4)  # A true CA4/DG voxel in the inner contour band.
    z[0,1,3,4,4]=2.;z[0,3,3,4,4]=-4.
    z.requires_grad_(True)
    loss=objectives.term('bands',z,y)
    gradient=torch.autograd.grad(loss,z)[0]
    g=gradient[0,:,3,4,4]
    swapped=z.detach()[:,[0,3,2,1]].requires_grad_(True)
    swapped_loss=objectives.term('bands',swapped,y)
    base=objectives.base_loss(z,y,'basic_unet16')
    swapped_base=objectives.base_loss(swapped,y,'basic_unet16')
    assert torch.allclose(loss,swapped_loss,atol=1e-7,rtol=0)
    assert not torch.allclose(base,swapped_base,atol=1e-5,rtol=0)
    assert g[1]<g[3]<0  # Bands boosts the already dominant, incorrect CA1 logit more.
    interior=gradient[0,:,5,5,5]
    assert torch.count_nonzero(interior)==0
    recovery=[]
    for ca4_logit in (-2.,-6.,-10.):
        test=z.detach().clone()
        test[0,3,3,4,4]=ca4_logit
        test.requires_grad_(True)
        dice_gradient=torch.autograd.grad(objectives.base_loss(test,y,'basic_unet16'),test)[0][0,3,3,4,4]
        ce=torch.nn.functional.cross_entropy(test,y[:,0])
        ce_gradient=torch.autograd.grad(ce,test)[0][0,3,3,4,4]
        recovery.append(dict(ca4_logit=ca4_logit,ca4_probability=float(test.softmax(1)[0,3,3,4,4]),
                             dice_logit_gradient=float(dice_gradient),ce_logit_gradient=float(ce_gradient)))
    assert abs(recovery[-1]['dice_logit_gradient'])<abs(recovery[0]['dice_logit_gradient'])/1000
    assert abs(recovery[-1]['ce_logit_gradient'])>abs(recovery[0]['ce_logit_gradient'])
    result=dict(scope='Synthetic CPU logit-gradient algebra using frozen objective. Not model-parameter gradients, Adam updates, or proof of a successful training fix.',
        objective_file=str(Path(objectives.__file__)),objective_sha256=manifest[relative],torch=str(torch.__version__),
        bands_loss=float(loss),bands_loss_after_swapping_ca1_ca4_logits=float(swapped_loss),
        dice_loss=float(base),dice_loss_after_swapping_ca1_ca4_logits=float(swapped_base),
        true_ca4_inner_voxel_probabilities=z.detach().softmax(1)[0,:,3,4,4].tolist(),
        true_ca4_inner_voxel_bands_logit_gradient=g.tolist(),
        wrong_ca1_to_true_ca4_gradient_magnitude_ratio=float(g[1]/g[3]),
        interior_logit_gradient=interior.tolist(),low_probability_recovery_probe=recovery,status='passed')
    out=ROOT/'reports'/NAME/'basic_unet_bands_diagnosis/union_loss_probe.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
