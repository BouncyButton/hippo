"""Opt-in HFH hypothesis: unchanged four losses, mask-routed auxiliary gradients.

This is an experimental optimizer variant, not a validated improvement. Existing
campaigns do not import it. Calibration and the online cap are on raw gradients,
not on AdamW-preconditioned updates or validation performance.
"""
from copy import deepcopy
import hashlib
import inspect
import itertools
from pathlib import Path

import numpy as np
import torch

from scripts.medsam3_supervised_constraints import semantic_log_fields
from scripts.colab_hfh import medsam_gain_replication as gain
from scripts.colab_hfh import medsam_augmentation_components as component
from scripts.colab_hfh import medsam_intensity_augmentation as augmentation
from scripts.colab_hfh.medsam_band_recalibration import calibration_schedule

TERMS = ('bands', 'edge', 'sdf', 'depth')
ARMS = tuple('dice_' + '_'.join(c) for n in range(1, 5)
             for c in itertools.combinations(TERMS, n))
FORMULATION = 'hfh_joint_constraints_v1'
DEFAULT = dict(formulation=FORMULATION, auxiliary_field='semantic',
               median_target=.01, p95_cap=.05, online_block_cap=.05,
               bootstrap_replicates=1000, bootstrap_seed=20261113)


def active_terms(arm):
    if arm not in ARMS:
        raise ValueError('Expected a nonempty, explicitly declared constraint arm')
    return tuple(k for k in TERMS if k in arm.split('_'))


def check_protocol(protocol):
    gain.check_protocol(protocol)
    if protocol.get('joint_constraint_source_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Candidate source differs from the prepared protocol')
    cfg = protocol.get('joint_constraints', {})
    if cfg.keys() != DEFAULT.keys() or cfg['formulation'] != FORMULATION:
        raise ValueError('Incomplete or unknown joint constraint recipe')
    if cfg['auxiliary_field'] not in ('semantic', 'mask_grad'):
        raise ValueError('Unknown auxiliary gradient routing')
    if not (0 < cfg['median_target'] <= cfg['p95_cap'] <= .05):
        raise ValueError('Calibration budget exceeds the declared diagnostic range')
    if not 0 < cfg['online_block_cap'] <= .05:
        raise ValueError('Online cap must be in (0, .05]')
    if cfg['bootstrap_replicates'] != 1000 or cfg['bootstrap_seed'] != 20261113:
        raise ValueError('Stability diagnostic changed')
    if protocol['fold'] not in (1, 2) or protocol['seed'] != 0:
        raise ValueError('Only the requested fold1/fold2 seed0 comparison is prepared')
    if protocol['constraint_warmup_updates'] != 20 or protocol.get('learning_rate_horizon_updates') != 200:
        raise ValueError('Preserve original ramp and LR horizon')
    active_terms(protocol['requested_arms'][0])
    if len(protocol['requested_arms']) != 1:
        raise ValueError('One arm per protocol')
    if protocol.get('constraint_recipe', {}).get('delay', 0) or protocol['constraint_recipe'].get('inner_weight', .5) != .5:
        raise ValueError('No simultaneous BAND geometry/side weighting/schedule ablation')
    train = protocol['train_cases']
    val = protocol['evaluation_cases']
    subjects = {c.rsplit('_h', 1)[0] for c in train}
    if len(train) != 10 or len(set(train)) != 10 or len(subjects) != 5:
        raise ValueError('Expected five subjects with two crops each')
    if any(sum(c.rsplit('_h', 1)[0] == s for c in train) != 2 for s in subjects):
        raise ValueError('Unequal subject support')
    if subjects & {c.rsplit('_h', 1)[0] for c in val}:
        raise ValueError('Subject leakage')


def auxiliary_fields(output, shape):
    """Identical semantic values/winners; no direct score or presence gradient.

    Shared trainable parameters can still change confidence indirectly. This is
    not the earlier native-matched raw-mask formulation: inference values remain
    exactly those of the original semantic union.
    """
    detached = {**output, 'pred_logits': output['pred_logits'].detach(),
                'presence_logit_dec': output['presence_logit_dec'].detach()}
    return semantic_log_fields(detached, shape)


def calibrated_weights(grams, cfg):
    """Independent native reference, then a cap on the actual combined vector.

    Divide the nominal budget among active terms rather than giving each term
    the entire budget. Keep zero gradients. Never normalize a term against a
    previously added constraint; record cross terms in the joint Gram norm.
    """
    g = np.asarray(grams, dtype=np.float64)
    if g.ndim != 3 or g.shape[1] != g.shape[2] or not 2 <= g.shape[1] <= 5:
        raise ValueError('Expected native plus one to four auxiliary gradients')
    if not np.isfinite(g).all() or not np.allclose(g, g.transpose(0, 2, 1), atol=1e-8):
        raise ValueError('Invalid gradient Gram matrix')
    diagonal = np.diagonal(g, axis1=1, axis2=2)
    if (diagonal[:, 0] <= 0).any() or (diagonal[:, 1:] < 0).any():
        raise ValueError('Invalid native/auxiliary gradient norms')
    scale = np.maximum(diagonal.max(1), 1.)
    if (np.linalg.eigvalsh(g)[:, 0] < -1e-7 * scale).any():
        raise ValueError('Gram matrix is not positive semidefinite')
    ratios = np.sqrt(diagonal[:, 1:] / diagonal[:, :1])
    n = ratios.shape[1]
    med, tail = np.quantile(ratios, [.5, .95], axis=0)
    if (tail <= 0).any():
        raise ValueError('An active term has insufficient nonzero calibration support')
    w = np.minimum(np.divide(cfg['median_target']/n, med, out=np.full(n, np.inf), where=med > 0),
                   cfg['p95_cap']/n/tail)
    combined = np.sqrt(np.maximum(np.einsum('i,nij,j->n', w, g[:, 1:, 1:], w), 0.)/diagonal[:, 0])
    m, q = np.quantile(combined, [.5, .95])
    joint_scale = min(1., cfg['median_target']/m if m else 1., cfg['p95_cap']/q if q else 1.)
    return w*joint_scale


def calibration_summary(rows, terms, cfg):
    grams = np.asarray([r['gram'] for r in rows], dtype=float)
    if grams.shape != (len(rows), len(terms)+1, len(terms)+1):
        raise ValueError('Calibration term order/Gram shape differs')
    w = calibrated_weights(grams, cfg)
    def describe(weights, subset):
        raw = np.sqrt(np.diagonal(subset, axis1=1, axis2=2)[:, 1:] / subset[:, 0, 0, None])
        joint = np.sqrt(np.maximum(np.einsum('i,nij,j->n', weights, subset[:, 1:, 1:], weights), 0.)/subset[:, 0, 0])
        quantiles = lambda x: dict(median=float(np.median(x)), p95=float(np.quantile(x, .95)), maximum=float(np.max(x)))
        return dict(joint=quantiles(joint), terms={t:{**quantiles(raw[:, i]*weights[i]),
            'zero_gradients':int((raw[:, i] == 0).sum())} for i,t in enumerate(terms)})
    subjects = sorted({r['subject'] for r in rows})
    ids = [np.array([i for i,r in enumerate(rows) if r['subject']==s]) for s in subjects]
    if len(subjects) != 5 or any(len(x) != 40 for x in ids):
        raise ValueError('Require 200 equal-subject calibration slabs')
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    boot, undefined = [], 0
    for _ in range(cfg['bootstrap_replicates']):
        ii = np.concatenate([ids[i] for i in rng.integers(0, 5, 5)])
        try:
            boot.append(calibrated_weights(grams[ii], cfg))
        except ValueError:
            undefined += 1
    loo = {}
    for subject in subjects:
        try:
            value = calibrated_weights(grams[[r['subject'] != subject for r in rows]], cfg)
            loo[subject] = dict(zip(terms, map(float,value)))
        except ValueError:
            loo[subject] = None
    return dict(status='complete', names=['native', *terms], weights=dict(zip(terms,map(float,w))),
        statistics=describe(w, grams), rows=rows, policy=deepcopy(cfg),
        categories={c:describe(w, grams[[r['slab_category']==c for r in rows]])
                    for c in sorted({r['slab_category'] for r in rows})},
        stability=dict(subjects=subjects, undefined_bootstraps=undefined,
            weight_p025_p50_p975={t:np.quantile(np.asarray(boot)[:,i],[.025,.5,.975]).tolist()
                for i,t in enumerate(terms)} if boot else {}, leave_one_subject_out=loo,
            used_to_select_weights=False),
        scope='Augmented training slabs only; no optimizer updates; raw gradient norms')


def block_name(name):
    """Group A/B adapter tensors belonging to the same LoRA module."""
    return name.split('.lora_', 1)[0]


class GradientAccumulator:
    """Accumulate native/combined auxiliary gradients, cap after accumulation.

    No cancellation-blind per-term clipping, no projection, no extra loss. Each
    LoRA block's auxiliary norm is bounded relative to its native norm. Native
    zero blocks receive zero auxiliary gradient. The final ordinary norm clip
    and AdamW step remain the responsibility of the original training loop.
    """
    def __init__(self, parameters, names, cap):
        if len(parameters) != len(names) or len(set(names)) != len(names) or not 0 < cap <= .05:
            raise ValueError('Invalid adapter parameters or cap')
        self.parameters, self.names, self.cap = list(parameters), list(names), cap
        self.groups = {}
        for i,name in enumerate(names):
            self.groups.setdefault(block_name(name), []).append(i)
        self.reset()

    def reset(self):
        self.native = [None]*len(self.parameters)
        self.auxiliary = [None]*len(self.parameters)

    def add(self, native, auxiliary):
        gn = torch.autograd.grad(native, self.parameters, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(auxiliary, self.parameters, allow_unused=True)
        for destination, source in ((self.native,gn),(self.auxiliary,ga)):
            for i,g in enumerate(source):
                if g is None:
                    continue
                if not torch.isfinite(g).all():
                    raise FloatingPointError('Nonfinite component gradient')
                if destination[i] is None:
                    destination[i] = g.detach().clone()
                else:
                    destination[i].add_(g.detach())

    def finish(self):
        # Reductions stay on device; one compact readback per update.
        device = self.parameters[0].device
        def norm2(values, ids):
            return sum((values[i].double().square().sum() for i in ids if values[i] is not None),
                       torch.zeros((),device=device,dtype=torch.float64))
        rows, diagnostics = [], {}
        for block, ids in self.groups.items():
            n2, a2 = norm2(self.native,ids), norm2(self.auxiliary,ids)
            factor = torch.where(a2>0, (self.cap*torch.sqrt(n2/a2.clamp_min(1e-300))).clamp(max=1.), torch.ones_like(a2))
            dot = sum(((self.native[i].double()*self.auxiliary[i].double()).sum() for i in ids
                       if self.native[i] is not None and self.auxiliary[i] is not None),torch.zeros_like(n2))
            rows.append(torch.stack((n2,a2,dot,factor)))
            for i in ids:
                n,a=self.native[i],self.auxiliary[i]
                self.parameters[i].grad = None if n is None and a is None else (
                    (torch.zeros_like(self.parameters[i]) if n is None else n)
                    + (0. if a is None else a*factor.to(a.dtype)))
        array=torch.stack(rows).detach().cpu().numpy()
        for block,(n2,a2,dot,factor) in zip(self.groups,array):
            diagnostics[block]=dict(native_norm=float(np.sqrt(n2)),auxiliary_norm=float(np.sqrt(a2)),
                cosine=float(dot/np.sqrt(n2*a2)) if n2*a2>0 else None, scale=float(factor),
                applied_ratio=float(factor*np.sqrt(a2/n2)) if n2>0 else 0.)
        native2,aux2=array[:,:2].sum(0)
        applied2=float((array[:,1]*array[:,3]**2).sum())
        return dict(blocks=diagnostics, raw_joint_ratio=float(np.sqrt(aux2/native2)) if native2>0 else None,
            applied_joint_ratio=float(np.sqrt(applied2/native2)) if native2>0 else 0.,
            capped_blocks=int((array[:,3]<1).sum()), norm_reference='Accumulated raw gradients before total clipping/AdamW')


def guarded_training_source(source):
    replacements = (
        ("    frozen_before, adapter_before =", "    hfh_mix = hfh_gradient_accumulator(model, parameters, protocol)\n    frozen_before, adapter_before ="),
        ('        optimizer.zero_grad(set_to_none=True)', '        optimizer.zero_grad(set_to_none=True)\n        hfh_mix.reset()'),
        ("            (total/protocol['accumulation']).backward()",
         "            native_graph=losses['native']+ramp*sum(0.*losses[k] for k in ('bands','edge'))\n"
         "            auxiliary=ramp*sum(calibration['weights'][k]*losses[k] for k in calibration['weights'])\n"
         "            hfh_mix.add(native_graph/protocol['accumulation'], auxiliary/protocol['accumulation'])"),
        ('        norm=torch.nn.utils.clip_grad_norm_',
         "        mix_report=hfh_mix.finish()\n"
         "        write(output/'gradient_diagnostics.json', dict(update=update+1, **mix_report))\n"
         "        norm=torch.nn.utils.clip_grad_norm_"),
        ('history.append(dict(update=update+1,lr=lr,ramp=ramp,gradient_norm=float(norm),',
         'history.append(dict(update=update+1,lr=lr,ramp=ramp,gradient_norm=float(norm), gradient_control=mix_report,'),
    )
    for old,new in replacements:
        if source.count(old) != 1:
            raise ValueError('Audited training loop changed: '+old)
        source=source.replace(old,new)
    return source


def gradient_accumulator(model, parameters, protocol):
    names={id(p):n for n,p in model.named_parameters()}
    return GradientAccumulator(parameters,[names[id(p)] for p in parameters],protocol['joint_constraints']['online_block_cap'])


def install(engine, protocol, output):
    check_protocol(protocol)
    if protocol['joint_constraints']['auxiliary_field']=='mask_grad':
        engine.semantic_log_fields=auxiliary_fields
    engine.hfh_gradient_accumulator=gradient_accumulator
    exec(compile(guarded_training_source(inspect.getsource(engine.train)),__file__,'exec'),engine.__dict__)
    gain.install(engine,protocol,output)
    engine.write(output/'joint_constraint_source.json',dict(source=inspect.getsource(inspect.getmodule(install)),
        status='experimental; no claimed effectiveness', recipe=protocol['joint_constraints'],
        gradient_semantics='Scalar total is diagnostic; applied gradient is blockwise native + capped joint auxiliary'))


def gradient_preflight(engine, model, criterion, parameters, slabs, protocol, output):
    """Fail before training unless zero-auxiliary split backward matches native.

    Run a whole accumulated first update twice, restoring RNG. This is a real
    model check when the worker is explicitly launched, not a CPU claim about
    CUDA/BF16 numerical equivalence. No optimizer is constructed or stepped.
    """
    from scripts.colab_mni_medsam3.optimized import rng_state, set_rng
    before=rng_state()
    gradients=[]
    try:
        for split in (False, True):
            set_rng(before)
            engine.reset_seed(protocol['seed'])
            model.zero_grad(set_to_none=True)
            mix=gradient_accumulator(model,parameters,protocol)
            for micro,(case,anchor) in enumerate(protocol['frozen_sample_schedule'][:protocol['accumulation']]):
                loss=engine.hfh_augmented_forward(model,criterion,slabs,case,anchor,0,micro)
                native=loss['native']+.05*sum(0.*loss[k] for k in ('bands','edge'))
                if split:
                    mix.add(native/protocol['accumulation'],sum(0.*loss[k] for k in TERMS)/protocol['accumulation'])
                else:
                    (native/protocol['accumulation']).backward()
            if split:
                mix.finish()
            gradients.append([None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters])
        exact=all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                  for a,b in zip(*gradients))
        engine.write(output/'joint_gradient_preflight.json',dict(status='passed' if exact else 'failed',
            exact_native_gradient_match=exact, optimizer_updates=0, accumulation=protocol['accumulation']))
        if not exact:
            raise ValueError('Split backward changed native gradients; do not train or claim a matched comparison')
    finally:
        model.zero_grad(set_to_none=True)
        set_rng(before)


def install_calibration(engine, protocol, output):
    check_protocol(protocol)
    engine.write(output/'execution_hardware.json',dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        bf16=torch.cuda.is_bf16_supported(),deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32))
    cfg=protocol['joint_constraints']
    terms=active_terms(protocol['requested_arms'][0])
    def calibrate(model, criterion, parameters, slabs, current, arm, destination):
        from scripts.cross_dataset_campaign.adni_foundation_full_runtime import native_training_check
        from scripts.colab_mni_medsam3.optimized import rng_state, set_rng
        if active_terms(arm)!=terms:
            raise ValueError('Worker/calibration term mismatch')
        native_training_check(model,criterion,parameters,slabs,current,destination)
        gradient_preflight(engine,model,criterion,parameters,slabs,current,output)
        schedule=calibration_schedule(protocol,{k:len(v) for k,v in slabs.indices.items()})
        names=['native',*terms]
        path=output/'calibration_partial.json'
        partial=engine.read(path) if path.exists() else dict(policy=cfg,names=names,rows=[])
        if partial.get('policy')!=cfg or partial.get('names')!=names:
            raise ValueError('Partial calibration policy differs')
        rows=partial['rows']
        if len(rows)>len(schedule) or any(any(r.get(k)!=v for k,v in s.items()) for r,s in zip(rows,schedule)):
            raise ValueError('Partial calibration schedule differs')
        before=rng_state(); adapter=engine.parameter_digest(model,True)
        cache=slabs.processor.transform; enabled=getattr(cache,'enabled',None)
        try:
            if enabled is not None:cache.enabled=False
            for entry in schedule[len(rows):]:
                if engine.stopping():raise SystemExit(75)
                engine.reset_seed(entry['model_seed'])
                case,anchor=entry['case'],entry['anchor'];ids=slabs.indices[case]
                fg=sum(int(np.count_nonzero(slabs.samples[ids[z]][1])) for z in (anchor,(anchor+1)%len(ids)))
                with augmentation.augmented_samples(slabs,case,anchor,entry['augmentation_seed']) as params:
                    loss=engine.forward_slab(model,criterion,slabs,case,anchor)
                    gradients=[torch.autograd.grad(loss[k],parameters,retain_graph=i<len(names)-1,allow_unused=True)
                               for i,k in enumerate(names)]
                    gram=[[sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(ga,gb)
                               if a is not None and b is not None) for gb in gradients] for ga in gradients]
                    rows.append(dict(**entry,gram=gram,augmentation=params,foreground_voxels=fg,
                        slab_category='empty' if fg==0 else 'foreground',losses={k:float(loss[k].detach()) for k in names}))
                    if not np.isfinite(gram).all() or not np.isfinite(list(rows[-1]['losses'].values())).all():
                        raise FloatingPointError('Nonfinite calibration')
                    engine.write(path,dict(policy=cfg,names=names,rows=rows))
                    del gradients,loss
        finally:
            set_rng(before)
            if enabled is not None:cache.enabled=enabled
        if engine.parameter_digest(model,True)!=adapter:
            raise ValueError('Calibration changed adapter parameters')
        result=calibration_summary(rows,terms,cfg)
        engine.write(output/'calibration_scope.json',dict(policy=cfg,names=names,anchors=schedule,
            train_cases=protocol['train_cases'],validation_cases=protocol['evaluation_cases'],optimizer_updates=0))
        return result
    # Apply the established logging preflight to this collector, without its
    # old saved-weight hook or the original 40-anchor assertion.
    engine.calibrate=calibrate
    source=inspect.getsource(component.install_calibration)
    needle='    original_aug.install_calibration(engine, protocol, output)'
    if source.count(needle)!=1:raise ValueError('Logging preflight hook changed')
    source=source.replace(needle,'    # Preserve the new joint collector.')
    namespace=dict(component.__dict__)
    exec(compile(source,__file__,'exec'),namespace)
    namespace['install_calibration'](engine,protocol,output)
