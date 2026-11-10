"""Observational UNETR gradient and same-state AdamW counterfactual probes."""
from contextlib import nullcontext
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from monai.losses import DiceLoss

from scripts.cross_dataset_campaign.common import read,write
from scripts.mni_constraints import objectives
from scripts.unetr_subfields.geo_recipe import LOSS_ARGUMENTS


def dot(left,right):
    values=[(a.detach().float()*b.detach().float()).sum() for a,b in zip(left,right)
            if a is not None and b is not None]
    return float(torch.stack(values).sum()) if values else 0.


def norm(values):
    value=dot(values,values)
    assert math.isfinite(value),'Nonfinite diagnostic norm'
    return math.sqrt(max(0.,value))


def cosine(left,right):
    denominator=norm(left)*norm(right)
    return dot(left,right)/denominator if denominator else None


@torch.no_grad()
def virtual_delta(parameter,gradient,state,group):
    """One non-mutating AdamW step using the existing moments, including decay."""
    if gradient is None:return None
    assert not any(group.get(k,False) for k in ('amsgrad','maximize','capturable','differentiable','fused'))
    beta1,beta2=group['betas'];step=int(state.get('step',0))+1
    first=state['exp_avg'].clone() if 'exp_avg' in state else torch.zeros_like(parameter)
    second=state['exp_avg_sq'].clone() if 'exp_avg_sq' in state else torch.zeros_like(parameter)
    first.lerp_(gradient,1-beta1)
    second.mul_(beta2).addcmul_(gradient,gradient,value=1-beta2)
    denominator=second.sqrt().div_(math.sqrt(1-beta2**step)).add_(group['eps'])
    after=parameter.detach().clone().mul_(1-group['lr']*group['weight_decay'])
    after.addcdiv_(first,denominator,value=-group['lr']/(1-beta1**step))
    return after-parameter.detach()


def parameter_group(name):
    if name.startswith('vit.patch_embedding'):return 'patch_embedding'
    if name.startswith('vit.'):return 'transformer'
    if name.startswith('encoder'):return 'encoders'
    if name.startswith('decoder'):return 'decoders'
    if name.startswith('out'):return 'output_head'
    return 'other'


class UnetrOptimizerDiagnostics:
    def __init__(self,model,optimizer,output,schedule):
        assert isinstance(optimizer,torch.optim.AdamW) and len(optimizer.param_groups)==1
        assert optimizer.param_groups[0]['foreach'] is False
        self.named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        self.params=[p for _,p in self.named];self.optimizer=optimizer
        self.path=output/'unetr_optimizer_diagnostics.json'
        self.rows=read(self.path)['rows'] if self.path.exists() else []
        prefix=schedule['prefix_updates'];maximum=schedule['max_steps']
        self.steps={n for n in (1,41,82,102,144,205,267,328,349,369,389,409,
                    prefix,prefix+1,prefix+17,prefix+33,prefix+66) if 1<=n<=maximum}
        self.row=None;self.gradients={};self.before=None

    def observe(self,outputs,labels,arm,weights,scale,sdf,depth,sdf_scale,loss,step,case,amp):
        self.row=None
        if step not in self.steps:return
        started=time.perf_counter();cpu=torch.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state_all() if amp else []
        assert all(p.grad is None for p in self.params)
        with FlopCounterMode(display=False) as counter:
            with torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext():
                base=objectives.base_loss(outputs,labels,'unetr_vit_b16')
                terms={n:weights[n]*(sdf_scale if n=='sdf' else scale)*
                       objectives.term(n,outputs,labels,sdf,depth)
                       for n in objectives.TERMS if n in arm}
                auxiliary=sum(terms.values()) if terms else outputs.float().sum()*0.
                dice_arguments={k:v for k,v in LOSS_ARGUMENTS.items() if k not in ('lambda_dice','lambda_ce')}
                per_class_dice=DiceLoss(**dice_arguments,reduction='none')(outputs.float(),labels).flatten(1)
                voxel_ce=F.cross_entropy(outputs.float(),labels[:,0].long(),reduction='none')
                assert abs(float((per_class_dice.mean()+voxel_ce.mean()-base).detach()))<2e-5
                ca4=per_class_dice[:,2].mean()/3+(voxel_ce*(labels[:,0]==3)).mean()
            discrepancy=abs(float((base+auxiliary).detach())-float(loss.detach()))
            assert discrepancy<2e-5,discrepancy
            self.gradients={};logits={}
            for name,value in (('base',base),('auxiliary',auxiliary)):
                logits[name]=torch.autograd.grad(value,outputs,retain_graph=True)[0].detach()
                self.gradients[name]=tuple(None if g is None else g.detach() for g in
                    torch.autograd.grad(value,self.params,retain_graph=True,allow_unused=True))
            self.gradients['CA4_base_component']=tuple(None if g is None else g.detach() for g in
                torch.autograd.grad(ca4,self.params,retain_graph=True,allow_unused=True))
        assert all(p.grad is None for p in self.params)
        assert torch.equal(cpu,torch.get_rng_state())
        if amp:assert all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
        base_gradient,aux_gradient=self.gradients['base'],self.gradients['auxiliary']
        self.row=dict(step=step,case_name=case,active_arm=arm,scale=scale,sdf_scale=sdf_scale,
            base_loss=float(base.detach()),auxiliary_loss=float(auxiliary.detach()),
            weighted_term_losses={n:float(v.detach()) for n,v in terms.items()},loss_decomposition_error=discrepancy,
            logit_gradients=dict(base_norm=norm([logits['base']]),auxiliary_norm=norm([logits['auxiliary']]),
                auxiliary_over_base=norm([logits['auxiliary']])/max(norm([logits['base']]),1e-30),
                cosine=cosine([logits['base']],[logits['auxiliary']]),dtype=str(logits['base'].dtype)),
            parameter_gradients=dict(base_norm=norm(base_gradient),auxiliary_norm=norm(aux_gradient),
                auxiliary_over_base=norm(aux_gradient)/max(norm(base_gradient),1e-30),cosine=cosine(base_gradient,aux_gradient),
                CA4_component_norm=norm(self.gradients['CA4_base_component']),
                CA4_auxiliary_cosine=cosine(self.gradients['CA4_base_component'],aux_gradient)),
            diagnostic_profiled_flops=counter.get_total_flops(),diagnostic_seconds=time.perf_counter()-started,
            extra_parameter_gradient_calls=3,extra_logit_gradient_calls=2)
        # A local logit-space direction, not the actual network/Adam update.
        z=outputs.detach().float();truth=labels[:,0].long();prediction=z.argmax(1)
        wrong=z.clone().scatter_(1,truth[:,None],float('-inf')).argmax(1)
        auxiliary_z=logits['auxiliary'].float()
        margin_delta=(auxiliary_z.gather(1,wrong[:,None])-auxiliary_z.gather(1,truth[:,None]))[:,0]
        active=auxiliary_z.abs().sum(1)>0
        regions={}
        for cls in (1,2,3):
            mask=(truth==cls)&(prediction>0)&(prediction!=truth)&active
            count=int(mask.sum());values=margin_delta[mask]
            regions[str(cls)]=dict(affected_wrong_foreground_voxels=count,
                mean_true_vs_best_wrong_margin_direction=float(values.mean()) if count else None,
                fraction_margin_worsening=float((values<0).float().mean()) if count else None)
        self.row['wrong_foreground_margin_probe']=dict(
            scope='Direction from a hypothetical negative auxiliary LOGIT gradient, not a parameter/Adam step. Negative margin direction reinforces the best wrong class.',
            by_true_class=regions)
        self.row['CA4_component_definition']='One-third of the actual foreground squared-Dice CA4 loss plus full-volume-normalized CE on true CA4 voxels; other-class CE gradients are not included.'
        self.row['CA4_target_voxels']=int((truth==3).sum())
        self.row['diagnostic_seconds']=time.perf_counter()-started

    @torch.no_grad()
    def before_step(self,lr):
        if self.row is None:return
        started=time.perf_counter();group=self.optimizer.param_groups[0]
        self.before=[p.detach().clone() for p in self.params]
        self.virtual_actual=[virtual_delta(p,p.grad,self.optimizer.state.get(p,{}),group) for p in self.params]
        self.virtual_base=[virtual_delta(p,g,self.optimizer.state.get(p,{}),group)
                           for p,g in zip(self.params,self.gradients['base'])]
        self.contribution=[None if a is None else a-(torch.zeros_like(a) if b is None else b)
                           for a,b in zip(self.virtual_actual,self.virtual_base)]
        reconstructed=[None if a is None else a+(torch.zeros_like(a) if b is None else b)
                       for a,b in zip(self.gradients['base'],self.gradients['auxiliary'])]
        self.row.update(learning_rate=lr,actual_gradient_norm=norm([p.grad for p in self.params]),
            reconstructed_vs_actual_gradient_cosine=cosine(reconstructed,[p.grad for p in self.params]),
            adam=dict(betas=list(group['betas']),eps=group['eps'],weight_decay=group['weight_decay'],
                virtual_total_update_norm=norm(self.virtual_actual),virtual_base_update_norm=norm(self.virtual_base),
                auxiliary_induced_update_norm=norm(self.contribution),
                auxiliary_induced_over_base_update=norm(self.contribution)/max(norm(self.virtual_base),1e-30),
                total_vs_base_update_cosine=cosine(self.virtual_actual,self.virtual_base)))
        groups={}
        for label in sorted({parameter_group(n) for n,_ in self.named}):
            indices=[i for i,(n,_) in enumerate(self.named) if parameter_group(n)==label]
            subset=lambda values:[values[i] for i in indices]
            gb,ga=subset(self.gradients['base']),subset(self.gradients['auxiliary'])
            base_delta,extra_delta=subset(self.virtual_base),subset(self.contribution)
            groups[label]=dict(base_gradient_norm=norm(gb),auxiliary_gradient_norm=norm(ga),
                gradient_cosine=cosine(gb,ga),auxiliary_induced_over_base_update=norm(extra_delta)/max(norm(base_delta),1e-30))
        self.row['parameter_groups']=groups
        self.row['diagnostic_seconds']+=time.perf_counter()-started

    @torch.no_grad()
    def after_step(self):
        if self.row is None:return
        started=time.perf_counter();actual=[p.detach()-old for p,old in zip(self.params,self.before)]
        error=[a-(torch.zeros_like(a) if b is None else b) for a,b in zip(actual,self.virtual_actual)]
        self.row['adam'].update(actual_update_norm=norm(actual),
            virtual_vs_actual_relative_error=norm(error)/max(norm(actual),1e-30))
        self.row['first_order_loss_changes']={name:dict(actual_update=dot(g,actual),
            same_state_base_only_update=dot(g,self.virtual_base),
            auxiliary_induced_change=dot(g,self.contribution)) for name,g in self.gradients.items()}
        self.row['diagnostic_seconds']+=time.perf_counter()-started
        self.rows=[r for r in self.rows if r['step']<self.row['step']]+[self.row]
        write(self.path,dict(status='recording',optimizer_updates_added=0,rows=self.rows,probe_steps=sorted(self.steps),
            scope='Actual training forward and untouched optimizer step; observational extra autograd calls preserve RNG and .grad. Counterfactual base-only Adam step uses the current constrained-run moments, not another training history.',
            compute_accounting='Optimization budget unchanged; extra diagnostic operator FLOPs and wall time reported separately. Counter excludes unsupported elementwise/reduction and virtual-update bookkeeping operations; not a cap-matched performance trial.',
            total_diagnostic_profiled_flops=sum(r['diagnostic_profiled_flops'] for r in self.rows),
            total_diagnostic_seconds=sum(r['diagnostic_seconds'] for r in self.rows)))
        self.row=None;self.before=None;self.gradients={}
        self.virtual_actual=self.virtual_base=self.contribution=None


def instrument_source(source):
    replacements=[
        ('    native_criterion = criterion if args.model == "nnunet" else None\n',
         '    native_criterion = criterion if args.model == "nnunet" else None\n'
         '    from scripts.mni_constraints.unetr_optimizer_diagnostics import UnetrOptimizerDiagnostics\n'
         '    diagnostics = UnetrOptimizerDiagnostics(model, optimizer, args.output, schedule)\n'),
        ('        loss.backward()\n',
         '        diagnostics.observe(logits, label, active_arm, weights, scale, sdf, depth,\n'
         '                            sdf_scale_for_step(charged, schedule) if "sdf" in active_arm else 0.,\n'
         '                            loss, step + 1, batch["case_name"], amp)\n'
         '        loss.backward()\n'),
        ('        optimizer.step()\n',
         '        diagnostics.before_step(lr)\n'
         '        optimizer.step()\n'
         '        diagnostics.after_step()\n'),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source
