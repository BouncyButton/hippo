"""CPU checks of frozen band loss against an independent grouped BCE formula."""
import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss, build_boundary_bands

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'experiments/mni_constraint_target_search_f0s0_20261107'
OUT=ROOT/'reports/mni_bands_implementation_audit_20261107'


def main():
    assert Path(inspect.getfile(OuterBoundaryBandLoss)).resolve() == STAGE/'source/thesis/new_constraints/bands/outer_boundary.py'
    torch.set_num_threads(4)
    frame=pd.read_pickle(STAGE/'data/MNI.pkl',compression='gzip')
    names=json.loads((STAGE/'protocols/nnunet_dice_bands_t60/splits_10patients.json').read_text())[0]['train']
    source={str(r.subject_id):r for r in frame.itertuples()}
    records=[]
    loss4=OuterBoundaryBandLoss(foreground_class_ids=(1,2,3),complement_class_ids=(0,))
    for name in names:
        native=np.asarray(source[name].label_data)
        before=(np.array([64]*3)-native.shape)//2
        after=np.array([64]*3)-native.shape-before
        y=torch.tensor(np.pad(native,tuple(zip(before,after))),dtype=torch.long)[None,None]
        inner,outer=build_boundary_bands(y>0,steps=2)
        assert inner.any() and outer.any() and not (inner&outer).any()
        records.append(dict(case=name,foreground=int((y>0).sum()),inner=int(inner.sum()),outer=int(outer.sum()),
            inner_fraction=float(inner.sum()/(y>0).sum()),
            per_class_inner_fraction={str(c):float((inner&(y==c)).sum()/(y==c).sum()) for c in (1,2,3)}))
        if len(records)>1:
            continue
        torch.manual_seed(314)
        z=torch.randn((1,4,64,64,64),requires_grad=True)
        actual=loss4(z,y).loss
        q=z.softmax(1)[:,1:].sum(1)
        bce=F.binary_cross_entropy(q,(y[:,0]>0).float(),reduction='none')
        manual=.5*((bce*inner[:,0]).sum()/(inner.sum()+1e-6)+(bce*outer[:,0]).sum()/(outer.sum()+1e-6))
        assert torch.allclose(actual,manual,atol=1e-6,rtol=1e-6)
        g=torch.autograd.grad(actual,z,retain_graph=True)[0]
        mg=torch.autograd.grad(manual,z,retain_graph=True)[0]
        assert torch.allclose(g,mg,atol=1e-9,rtol=1e-4)
        band=(inner|outer).expand_as(g)
        assert torch.count_nonzero(g[~band])==0 and torch.count_nonzero(g[band])>0
        grouped2=torch.cat([z[:,:1],torch.logsumexp(z[:,1:],1,keepdim=True)],1)
        binary=OuterBoundaryBandLoss(foreground_class_ids=(1,),complement_class_ids=(0,))(grouped2,(y>0).long()).loss
        grouped3=torch.cat([z[:,:1],torch.logsumexp(z[:,1:3],1,keepdim=True),z[:,3:]],1)
        label3=torch.where(y==3,2,torch.where(y>0,1,0))
        three=OuterBoundaryBandLoss()(grouped3,label3).loss
        assert torch.allclose(actual,binary,atol=1e-6) and torch.allclose(actual,three,atol=1e-6)
        swapped=loss4(z[:,[0,2,1,3]],y).loss
        assert torch.allclose(actual,swapped,atol=1e-6)
        assert not torch.equal(z.argmax(1),z[:,[0,2,1,3]].argmax(1))
        for k in range(4):
            for flip in (False,True):
                a,b=torch.rot90(z,k,(2,3)),torch.rot90(y,k,(2,3))
                if flip:a,b=torch.flip(a,(2,)),torch.flip(b,(2,))
                assert torch.allclose(loss4(a,b).loss,actual,atol=1e-6)
        checks=dict(frozen_source_import_verified=True,manual_balanced_union_bce=True,manual_gradient_match=True,
            nonzero_band_gradients=True,zero_gradients_outside_band=True,
            binary_three_and_four_class_grouped_equivalence=True,
            foreground_label_swap_invariance=True,eight_paired_D4_transforms=True,
            cpu_torch=torch.__version__)
    result=dict(status='passed',checks=checks,training_cases=records,
        note='Independent FP32 CPU loss/gradient checks on real fold0 training labels; no model training, validation fitting, checkpoint access, or GPU allocation.')
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'numerical_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',checks=checks,inner_fraction_range=[min(r['inner_fraction'] for r in records),max(r['inner_fraction'] for r in records)])))


if __name__=='__main__':main()
