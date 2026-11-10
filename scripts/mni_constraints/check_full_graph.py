"""Validate composed FLOPs against a real full-network CPU backward."""
import torch
from scripts.mni_constraints import objectives as o, profile as audit
from scripts.unetr_subfields import baseline_runtime
from scripts.cross_dataset_campaign.common import read, write


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    recipe=read(audit.RECIPES)['basic_unet16']
    p=recipe['protocol']
    architecture=read(audit.ROOT/'experiments'/recipe['stage']/'protocols/model/nnunet_architecture.json')
    model=baseline_runtime.build_model('basic_unet16',architecture).train()
    _,optimizer=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER('unet',model,p)
    image=torch.randn(1,1,64,64,64);label=audit.target('ellipsoid')
    sdf=o.signed_distance_maps(label)
    settings=read(audit.ROOT/'docs/experiments/depth_swinunetr_20261006/protocol.json')['geometry']
    depth=o.geometry_weights(label,torch.eye(4).numpy(),settings)
    arm='dice_bands_edge_sdf_depth'
    def step():
        outputs=model(image)
        loss=o.training_loss(outputs,label,arm,dict(bands=.04,edge=.1,sdf=.1,depth=.1),1.,sdf,None,depth,.2,identity='basic_unet16')
        loss.backward();optimizer.step()
        assert torch.isfinite(loss)
    measured,_=audit.audit_call(step)
    expected=read(audit.REPORT/'profiles/basic_unet16.json')['costs'][arm]['update_flops']
    assert measured['flops']==expected,(measured['flops'],expected)
    write(audit.REPORT/'full_network_audit.json',dict(status='passed',model='basic_unet16',arm=arm,
        full_network_cpu_flops=measured['flops'],composed_flops=expected,audit=measured))
    print('Full network audit exactly matches composed cost:',expected,flush=True)


if __name__=='__main__':main()
