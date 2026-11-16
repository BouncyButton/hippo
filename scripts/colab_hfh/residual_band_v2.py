"""Activation-count ramp and read-only diagnostics of the actual training loss."""
import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from scripts.colab_hfh.residual_band import ResidualBandLoss,geometry,selection

VERSION='hfh_nnunet_conditional_residual_band_v2'
PROBE_UPDATES=(60,75,90)


def gradient_stats(native,inner,outer,auxiliary,combined):
    vectors={k:v.detach().float() for k,v in dict(native=native,fn=inner,fp=outer,
        auxiliary=auxiliary,combined=combined,effective_auxiliary=combined-native).items()}
    norm={k:float(v.square().sum().sqrt()) for k,v in vectors.items()}
    def cosine(a,b):
        return float((vectors[a]*vectors[b]).sum())/(norm[a]*norm[b]) if norm[a]*norm[b]>0 else None
    return dict(l2=norm,auxiliary_over_native=norm['auxiliary']/max(norm['native'],1e-30),
        effective_auxiliary_over_native=norm['effective_auxiliary']/max(norm['native'],1e-30),
        fn_over_native=norm['fn']/max(norm['native'],1e-30),fp_over_native=norm['fp']/max(norm['native'],1e-30),
        cosine_auxiliary_native=cosine('auxiliary','native'),cosine_fn_fp=cosine('fn','fp'),
        fp_gradient_squared_share=norm['fp']**2/max(norm['fp']**2+norm['fn']**2,1e-30))


def parameter_stats(gradients,factor):
    # Native includes every deep-supervision head. FN/FP terms are unweighted
    # boundary BCE gradients; apply the actual coefficient and active ramp here.
    def dot(a,b):
        return sum(float((x.float()*y.float()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    n,i,o=(gradients[k] for k in ('native','fn','fp'))
    nn,ii,oo=dot(n,n),dot(i,i),dot(o,o)
    ni,no,io=dot(n,i),dot(n,o),dot(i,o)
    aa=factor*factor*(ii+oo+2*io)
    return dict(native_l2=math.sqrt(nn),weighted_fn_l2=factor*math.sqrt(ii),weighted_fp_l2=factor*math.sqrt(oo),
        auxiliary_over_native=math.sqrt(max(aa,0)/max(nn,1e-30)),
        cosine_fn_fp=io/math.sqrt(ii*oo) if ii*oo>0 else None,
        cosine_auxiliary_native=factor*(ni+no)/math.sqrt(aa*nn) if aa*nn>0 else None,
        auxiliary_directional_fn_loss_change=-factor*(ii+io),
        auxiliary_directional_fp_loss_change=-factor*(oo+io),
        interpretation='First-order unit gradient-descent directions, not optimizer-step outcomes; negative means loss decreases')


class TrainingProbe:
    def __init__(self,original,output,protocol):
        self.original,self.output,self.protocol=original,output,protocol
        self.path=output/'residual_band_trace.jsonl'
        self.previous=[json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
        self.active_count=sum(r['selected_errors']>0 for r in self.previous)
        self.last_attempt=max((r['charged_attempt'] for r in self.previous),default=0)
        self.model=None;self.pending=None

    def __call__(self,outputs,labels,arm,weights,scale,target_sdf=None,criterion=None,depth_weights=None,sdf_scale=0.):
        if arm=='dice':
            return self.original(outputs,labels,arm,weights,scale,target_sdf,criterion,depth_weights,sdf_scale)
        from scripts.nnunet_band_objective import native_loss
        if arm!='dice_bands' or criterion is None or self.pending is not None:
            raise ValueError('Unexpected conditional training call')
        attempt=json.loads((self.output/'budget_ledger.json').read_text())['charged_attempts']
        if attempt<=self.last_attempt:raise ValueError('Repeated charged training attempt')
        z=outputs[0]
        # This is exactly the previous auxiliary formula. Diagnostics below do
        # not choose support, change coefficients, or contribute to training.
        captured=[]
        ResidualBandLoss.sink=captured.append
        try:auxiliary=ResidualBandLoss()(z,labels).loss
        finally:ResidualBandLoss.sink=None
        if len(captured)!=1:raise ValueError('Missing conditional decision')
        row=captured[0];active=row['selected_errors']>0
        first_active=active and self.active_count==0
        self.active_count+=int(active)
        applied=min(1.,self.active_count/9) if active else 0.
        factor=weights['bands']*applied
        native=native_loss(outputs,labels,criterion)
        # Preserve v1's multiplication order when its ramp matches this one.
        weighted_auxiliary=applied*(weights['bands']*auxiliary)
        loss=native+weighted_auxiliary
        annotation=labels.detach().cpu().numpy()[0,0].astype('uint8').tobytes()
        truth,roi,inner_shells,outer_shells=geometry(annotation)
        pred=z.detach().float().argmax(1)[0].cpu().numpy().astype(bool)
        support,_=selection(truth,pred,roi,inner_shells,outer_shells)
        with torch.autocast(z.device.type,enabled=False):
            odds=z[:,1].float()-z[:,0].float()
            target=torch.as_tensor(truth.astype('float32'),device=z.device)
            element=F.binary_cross_entropy_with_logits(odds[0],target,reduction='none')/int(truth.sum())
            fn=(element*torch.as_tensor(support&truth,device=z.device)).sum()
            fp=(element*torch.as_tensor(support&~truth,device=z.device)).sum()
        terms=(native,applied*(weights['bands']*fn),applied*(weights['bands']*fp),weighted_auxiliary,loss)
        gs=[torch.autograd.grad(term,z,retain_graph=True)[0] for term in terms]
        row.update(charged_attempt=attempt,active_update_count=self.active_count,scheduled_scale=float(scale),
            applied_scale=applied,coefficient=weights['bands'],weighted_coefficient=factor,
            loss_native=float(native.detach()),loss_fn=float(fn.detach()),loss_fp=float(fp.detach()),
            loss_auxiliary=float(auxiliary.detach()),weighted_loss_auxiliary=float(weighted_auxiliary.detach()),
            logit_gradients=gradient_stats(*gs),
            gradient_space='Actual training-forward output logits; includes BF16 cast effects; native includes full-resolution DS contribution')
        self.pending=dict(row=row)
        if first_active or attempt in PROBE_UPDATES:
            if self.model is None:raise ValueError('Model missing from diagnostic binding')
            params=[p for p in self.model.parameters() if p.requires_grad]
            gradients={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)
                for k,v in [('native',native),('fn',fn),('fp',fp)]}
            row['parameter_gradients']=parameter_stats(gradients,factor)
            self.pending.update(params=params,before=[p.detach().clone() for p in params],gradients=gradients,
                prediction=pred,truth=truth,roi=roi,support=support)
        return loss

    def finish(self,image,label,device,amp):
        if self.pending is None:return
        data=self.pending;row=data['row']
        if 'params' in data:
            with torch.no_grad():
                delta=[p.detach()-old for p,old in zip(data['params'],data['before'])]
                row['optimizer_step_directional_loss_change']={k:sum(float((g.float()*d.float()).sum())
                    for g,d in zip(values,delta) if g is not None) for k,values in data['gradients'].items()}
                row['optimizer_delta_l2']=math.sqrt(sum(float(d.float().square().sum()) for d in delta))
                # Preserve training mode, RNG and buffers. This diagnostic uses
                # the same crop after the real optimizer step, without updates.
                buffers={k:v.detach().clone() for k,v in self.model.named_buffers()}
                devices=[image.device.index] if image.is_cuda else []
                try:
                    with torch.random.fork_rng(devices=devices):
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):raw=self.model(image)
                finally:
                    for k,v in self.model.named_buffers():v.copy_(buffers[k])
                post=(raw[0] if isinstance(raw,(tuple,list)) else raw).float().argmax(1)[0].cpu().numpy().astype(bool)
                before=data['prediction'];truth=data['truth'];roi=data['roi'];support=data['support']
                events=dict(fp_corrected=before&~post&~truth&roi,fp_introduced=~before&post&~truth&roi,
                    fn_corrected=~before&post&truth,fn_introduced=before&~post&truth)
                row['same_crop_post_update']={k:dict(total=int(mask.sum()),inside_selected_support=int((mask&support).sum()),
                    outside_selected_support=int((mask&~support).sum())) for k,mask in events.items()}
                row['same_crop_post_update'].update(fp_before=int((before&~truth&roi).sum()),
                    fp_after=int((post&~truth&roi).sum()),fn_before=int((~before&truth).sum()),fn_after=int((~post&truth).sum()),
                    interpretation='Actual total optimizer-step change, including native loss and momentum; not an isolated BAND intervention')
        with self.path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.last_attempt=row['charged_attempt'];self.pending=None


def bind_trace(original,output,protocol):return TrainingProbe(original,output,protocol)
