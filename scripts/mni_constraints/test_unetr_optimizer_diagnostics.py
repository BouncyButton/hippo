"""Check virtual AdamW steps and observational probes against actual training."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import torch

from scripts.mni_constraints import objectives
from scripts.mni_constraints.unetr_optimizer_diagnostics import UnetrOptimizerDiagnostics,virtual_delta
from scripts.cross_dataset_campaign.common import read


def run():
    torch.set_num_threads(1);torch.manual_seed(531)
    p=torch.nn.Parameter(torch.randn(3,4));optimizer=torch.optim.AdamW([p],lr=.002,weight_decay=.03,foreach=False)
    for _ in range(5):
        p.grad=torch.randn_like(p)
        before=p.detach().clone();state=deepcopy(optimizer.state_dict())
        delta=virtual_delta(p,p.grad,optimizer.state.get(p,{}),optimizer.param_groups[0])
        assert torch.equal(p,before)
        for key,value in state['state'].get(0,{}).items():assert torch.equal(value,optimizer.state[p][key])
        optimizer.step();assert torch.allclose(p-before,delta,atol=2e-7,rtol=2e-5)
    model=torch.nn.Sequential(torch.nn.Conv3d(1,5,3,padding=1),torch.nn.GELU(),
                              torch.nn.Dropout3d(.1),torch.nn.Conv3d(5,4,1))
    reference=deepcopy(model)
    a=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=1e-5,foreach=False)
    b=torch.optim.AdamW(reference.parameters(),lr=.0006,weight_decay=1e-5,foreach=False)
    image=torch.randn(1,1,6,6,6);label=torch.zeros(1,1,6,6,6,dtype=torch.long)
    label[:,:,1:5,1:5,1:3]=1;label[:,:,1:5,1:5,3:4]=2;label[:,:,1:5,1:5,4:5]=3
    sdf=objectives.signed_distance_maps(label);depth=torch.ones(6,6,6)/216
    with TemporaryDirectory(prefix='unetr-adam-probe-') as temporary:
        probe=UnetrOptimizerDiagnostics(model,a,Path(temporary),dict(prefix_updates=1,max_steps=4));probe.steps={1,2,3,4}
        for step,arm in enumerate(('dice','dice_bands','dice_bands_edge_sdf_depth','dice_bands'),1):
            weights={n:.02 for n in objectives.TERMS if n in arm}
            target=label.clone()
            if step==4:target[target==3]=2  # A valid crop with the CA4 class absent.
            rng=torch.get_rng_state().clone();a.zero_grad(set_to_none=True);z=model(image)
            loss=objectives.training_loss(z,target,arm,weights,.5,sdf,None,depth,.5,identity='unetr_vit_b16')
            probe.observe(z,target,arm,weights,.5,sdf,depth,.5,loss,step,'synthetic',False)
            loss.backward();probe.before_step(.0006);a.step();probe.after_step()
            after_rng=torch.get_rng_state().clone()
            torch.set_rng_state(rng);b.zero_grad(set_to_none=True);zr=reference(image)
            lr=objectives.training_loss(zr,target,arm,weights,.5,sdf,None,depth,.5,identity='unetr_vit_b16')
            lr.backward();b.step()
            assert torch.equal(torch.get_rng_state(),after_rng)
            assert float(loss)==float(lr)
            for actual,expected in zip(model.parameters(),reference.parameters()):assert torch.equal(actual,expected)
            for pa,pb in zip(model.parameters(),reference.parameters()):
                for key,value in a.state[pa].items():assert torch.equal(value,b.state[pb][key])
        rows=read(Path(temporary)/'unetr_optimizer_diagnostics.json')['rows']
        assert len(rows)==4 and rows[0]['adam']['auxiliary_induced_update_norm']==0
        assert any(r['adam']['auxiliary_induced_update_norm']>0 for r in rows[1:])
        assert max(r['adam']['virtual_vs_actual_relative_error'] for r in rows)<2e-4
        assert rows[-1]['CA4_target_voxels']==0 and rows[-1]['parameter_gradients']['CA4_auxiliary_cosine'] is None
    return dict(status='passed',scope='Synthetic CPU test, not experiment evidence.',
        checks=['virtual AdamW agrees with actual optimizer across moment histories',
                'virtual step leaves parameters and state unchanged','probe preserves dropout RNG',
                'instrumented training parameters, optimizer states, and losses exactly match reference',
                'zero-auxiliary step has zero induced update','nonzero constraints have measured update effects'])


if __name__=='__main__':
    import json
    print(json.dumps(run()))
