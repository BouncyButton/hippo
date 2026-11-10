"""Profile native MNI baselines with the established dense arithmetic audit."""
from pathlib import Path
from copy import deepcopy
import gc
import torch
import monai
from torch.utils.flop_counter import FlopCounterMode
from scripts.unetr_subfields import profile, baseline_runtime
from scripts.unetr_subfields.profile_dice_ce import supplement
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.capped_depth_delay.objectives import training_loss

NAME='mni_other_baselines_20261106'


def measure(identity, classes):
    architecture=read(ROOT/'experiments/nnunet_pair95_20261005/baseline/nnunet_architecture.json')
    architecture['kwargs']['num_classes']=classes
    spec=baseline_runtime.MODELS[identity]
    protocol=dict(initial_lrs={spec['alias']:spec['lr']},adamw_weight_decay=1e-5,sgd_weight_decay=3e-5,nnunet_sgd_momentum=.95)
    torch.manual_seed(0)
    model=baseline_runtime.build_model(identity,architecture,classes).to('meta').train()
    with torch.device('meta'):
        image=torch.randn((1,1,64,64,64));target=torch.zeros((1,1,64,64,64),dtype=torch.long)
    criterion,optimizer=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER(spec['alias'],model,protocol)
    native=criterion if identity=='nnunet' else None
    if native:
        for parameter in model.parameters():optimizer.state[parameter]['momentum_buffer']=torch.zeros_like(parameter)
    profile.reference.supplementary_flops=supplement
    audit=profile.reference.OperatorAudit();stages={}
    with audit,FlopCounterMode(display=False,custom_mapping={torch.ops.aten.convolution_backward:profile.reference.conv_backward_flops}) as counter:
        for stage in ('forward','loss','backward','optimizer'):
            audit.stage=stage;before=counter.get_total_flops()
            if stage=='forward':outputs=model(image)
            elif stage=='loss':loss=training_loss(outputs,target,'dice',{},.2,criterion=native)
            elif stage=='backward':loss.backward()
            else:
                if native:torch.nn.utils.clip_grad_norm_(model.parameters(),12,foreach=False,error_if_nonfinite=False)
                optimizer.step()
            stages[stage]=counter.get_total_flops()-before
    result=dict(model=identity,classes=classes,parameters=sum(p.numel() for p in model.parameters()),
        update_flops=sum(stages.values())+sum(audit.supplement.values()),core_flops_per_stage=stages,
        supplementary_per_stage=audit.supplement,operator_audit=audit.ops,input_shape=[1,1,64,64,64],
        output_shapes=[list(o.shape) for o in outputs] if isinstance(outputs,(list,tuple)) else [list(outputs.shape)],
        runtime=dict(torch=torch.__version__,monai=monai.__version__,device='meta'),recipe=spec,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/unetr_subfields/baseline_runtime.py',ROOT/'scripts/unetr_subfields/profile_dice_ce.py']},
        accounting='Historical MAC=2 dense forward/backward plus scalar/loss/optimizer estimates, including fused CE. nnUNet includes deep supervision, steady-state SGD momentum and gradient clipping. Validation, preprocessing and I/O excluded.')
    del model,optimizer,outputs,loss;gc.collect();return result


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__=='1.6.0'
    torch.set_num_threads(4)
    output=ROOT/'reports'/NAME/'profiles'
    control=profile.measure(3,(64,64,64));assert control['update_flops']==130061273028
    write(output/'historical_unetr_control.json',dict(**control,status='exact_match'))
    for identity in baseline_runtime.MODELS:
        result=measure(identity,4);write(output/(identity+'.json'),result)
        print(identity,result['update_flops'],result['parameters'],flush=True)

if __name__=='__main__':main()
