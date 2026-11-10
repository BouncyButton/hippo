"""Support-only SAM3 gradient diagnosis: ten native updates, no validation inference."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

NAMES = ('native', 'bands', 'edge')
STEPS = (0, 5, 10)
HEADS = ('pred_masks', 'pred_logits', 'presence_logit_dec')


@contextmanager
def preserve_probe_state(model):
    """Diagnostic backwards must not alter the subsequent training trajectory."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None: torch.cuda.set_rng_state_all(state[3])
        with torch.no_grad():
            for buffer, value in buffers: buffer.copy_(value)
        for module, training in modes: module.training = training


def tensor_stats(value):
    if value is None:
        return dict(connected=False, l2=0., rms=0., mean=0., sum=0.)
    value = value.detach().float()
    return dict(connected=True, l2=float(value.square().sum().sqrt()),
                rms=float(value.square().mean().sqrt()), mean=float(value.mean()), sum=float(value.sum()))


def gram_matrix(gradients):
    return [[sum(float((a.detach().double()*b.detach().double()).sum())
                 for a,b in zip(left,right) if a is not None and b is not None)
             for right in gradients] for left in gradients]


def ratio_metrics(gram, weights, ramp=1.):
    gram = np.asarray(gram, dtype=float)
    wb, we = weights['bands'], weights['edge']
    coeff = np.array([1., ramp*wb])
    base = float(coeff@gram[:2,:2]@coeff)
    if not np.isfinite(gram).all() or base <= 0 or gram[0,0] <= 0:
        raise ValueError('Invalid probe gradient matrix')
    edge_raw = float(np.sqrt(max(gram[2,2],0.)/base))
    cosine = ((gram[0,2]+ramp*wb*gram[1,2])/np.sqrt(base*gram[2,2])
              if gram[2,2]>0 else None)
    return dict(edge_over_native_bands=float(ramp*we*edge_raw),
                bands_over_native=float(ramp*wb*np.sqrt(max(gram[1,1],0.)/gram[0,0])),
                cosine_edge_native_bands=None if cosine is None else float(cosine),
                native_gradient_norm=float(np.sqrt(gram[0,0])),
                edge_gradient_norm=float(np.sqrt(max(gram[2,2],0.))))


def measure(original, model, criterion, parameters, slabs, case, anchor, seed, weights, step):
    """One unchanged forward and three backwards, including native output-head gradients."""
    captured = []
    semantic = original.semantic_log_fields
    def capture(output, shape):
        fields = semantic(output, shape)
        captured.append((output, fields))
        return fields
    with preserve_probe_state(model):
        original.reset_seed(seed)
        original.semantic_log_fields = capture
        try:
            losses = original.forward_slab(model,criterion,slabs,case,anchor)
            head_tensors, head_indices = [], []
            for plane,(output,_) in enumerate(captured):
                for name in HEADS:
                    if output[name].requires_grad:
                        head_tensors.append(output[name]); head_indices.append((plane,name))
            gradients, head_stats = [], {name:[] for name in NAMES}
            for index,name in enumerate(NAMES):
                all_grads = torch.autograd.grad(losses[name], [*parameters,*head_tensors],
                                                retain_graph=index<2, allow_unused=True)
                gradients.append(all_grads[:len(parameters)])
                head_stats[name] = [dict(plane=plane,head=head,**tensor_stats(grad))
                                   for (plane,head),grad in zip(head_indices,all_grads[len(parameters):])]
            gram = gram_matrix(gradients)
            confidences = []
            indices = slabs.indices[case]
            for plane,(output,fields) in enumerate(captured):
                z = (anchor+plane)%len(indices)
                truth = torch.as_tensor(slabs.samples[indices[z]][1],device=fields[0].device)>0
                probability = fields[0].detach().exp()
                confidence = dict(plane=plane,foreground_pixels=int(truth.sum()),
                    presence_probability=float(output['presence_logit_dec'].detach().float().sigmoid().mean()),
                    max_query_probability=float(output['pred_logits'].detach().float().sigmoid().max()),
                    mean_foreground_probability=float(probability[truth].mean()) if truth.any() else None,
                    mean_background_probability=float(probability[~truth].mean()) if (~truth).any() else None,
                    predicted_foreground_fraction=float((probability>.5).float().mean()))
                confidences.append(confidence)
            return dict(case=case,anchor=anchor,update=step,gram=gram,
                losses={name:float(losses[name].detach()) for name in NAMES},
                fixed_weight_full_strength=ratio_metrics(gram,weights),
                fixed_weight_next_update_ramp=ratio_metrics(gram,weights,min((step+1)/20.,1.)),
                head_gradients=head_stats,confidence=confidences)
        finally:
            original.semantic_log_fields = semantic


def summarize(rows, weights):
    """Recalibration suggestions use five diagnostic slabs, not the production 20-slab set."""
    result=[]
    for step in sorted({r['update'] for r in rows}):
        selected=[r for r in rows if r['update']==step]
        ratios=[r['fixed_weight_full_strength']['edge_over_native_bands'] for r in selected]
        cos=[r['fixed_weight_full_strength']['cosine_edge_native_bands'] for r in selected]
        raw=np.asarray(ratios)/weights['edge']
        candidate=min(.1/np.median(raw),.5/np.quantile(raw,.95)) if np.all(raw>0) else None
        result.append(dict(update=step,slabs=len(selected),edge_ratio_median=float(np.median(ratios)),
            edge_ratio_p95=float(np.quantile(ratios,.95)),edge_ratio_max=max(ratios),
            conflicting_slabs=sum(v is not None and v<0 for v in cos),
            candidate_edge_weight_with_fixed_bands=candidate,
            mean_foreground_probability=float(np.mean([c['mean_foreground_probability']
                for r in selected for c in r['confidence'] if c['mean_foreground_probability'] is not None])),
            mean_presence_probability=float(np.mean([c['presence_probability'] for r in selected for c in r['confidence']])),
            edge_presence_gradient_sum=float(np.mean([g['sum'] for r in selected
                for g in r['head_gradients']['edge'] if g['head']=='presence_logit_dec'])),
            edge_query_gradient_sum=float(np.mean([g['sum'] for r in selected
                for g in r['head_gradients']['edge'] if g['head']=='pred_logits']))))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--source-results',type=Path,required=True)
    parser.add_argument('--durable',type=Path,required=True)
    args=parser.parse_args(); root=args.bundle.resolve(); durable=args.durable.resolve()
    sys.path.insert(0,str(root))
    import runtime as rt
    rt.activate(root)
    import engine
    from parallel import SweepLock
    from scripts.cross_dataset_campaign.common import install_signals,stopping
    install_signals(); durable.mkdir(parents=True,exist_ok=True)
    with SweepLock(root):
        plan=rt.verify_bundle(root)
        bound=rt.binding(root,durable)
        prior=rt.read(args.source_results/'environment_binding.json')
        if prior != bound: raise ValueError('Use the same bundle and GPU/software as the completed run')
        shared=rt.fold_calibration(root,args.source_results,0)
        task=next(t for t in plan['tasks'] if t['scientific_arm']=='dice_bands_edge')
        if not rt.completed(root,args.source_results,task):
            raise ValueError('The source bands+edge run has no verified completed archive')
        expected_cal=rt.read(args.source_results/'runs'/task['id']/'calibration.json')
        if expected_cal['shared_calibration_sha256']!=rt.digest(args.source_results/'calibration/fold0/calibration.json'):
            raise ValueError('Completed weights and shared calibration differ')
        weights=expected_cal['weights']
        p=rt.protocol(root,task); p['updates']=10
        # The middle-left original calibration anchor from each of the five volumes.
        probes=[]
        for case in p['train_cases']:
            candidates=[(i,r) for i,r in enumerate(shared['rows']) if r['case']==case]
            if len(candidates)!=4: raise ValueError('Expected four original anchors per volume')
            index,row=candidates[1]
            probes.append(dict(case=case,anchor=row['anchor'],seed=p['calibration_seed']+index))
        identity=dict(binding=bound,diagnostic_sha256=rt.digest(Path(__file__)),
            calibration_sha256=rt.digest(args.source_results/'calibration/fold0/calibration.json'),
            weights=weights,probes=probes,updates=10,probe_steps=list(STEPS),protocol=p)
        if (durable/'identity.json').exists() and rt.read(durable/'identity.json')!=identity:
            raise ValueError('Diagnostic identity changed; preserve this folder and use a new destination')
        rt.write(durable/'identity.json',identity)
        if (durable/'completion.json').exists() and (durable/'archive_manifest.json').exists():
            archive=rt.read(durable/'archive_manifest.json')
            for name,sha in archive.items():
                if rt.digest(durable/rt.relative(name))!=sha:raise ValueError('Diagnostic archive changed')
            print('Diagnostic already complete; using saved measurements. No training repeated.',flush=True)
            return
        original,model,processor,parameters,criterion,volumes,slabs=engine.load(root,p)
        initial=dict(frozen=original.parameter_digest(model,False),adapter=original.parameter_digest(model,True))
        if initial!=shared['initialization'] or slabs.inventory!=shared['inventory']:
            raise ValueError('Initialization or five-volume support differs')
        original.reset_seed(p['seed']);model.train()
        optimizer=torch.optim.AdamW(parameters,lr=p['learning_rate'],weight_decay=.01)
        schedule=slabs.schedule(p)
        rt.write(durable/'slab_schedule.json',schedule)
        work=root/'work/edge_diagnostic_v1';work.mkdir(parents=True,exist_ok=True)
        checkpoint=work/'checkpoint.pt'; saved=durable/'checkpoint.pt'; marker=durable/'checkpoint.json'
        identity_hash=rt.digest(durable/'identity.json')
        if not checkpoint.exists() and saved.exists():
            if rt.digest(saved)!=rt.read(marker)['sha256']:raise ValueError('Recovery hash differs')
            rt.copy_verified(saved,checkpoint)
        step=0;history=[]
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['identity_sha256']!=identity_hash:raise ValueError('Checkpoint identity differs')
            model.load_state_dict(state['adapter'],strict=False);optimizer.load_state_dict(state['optimizer'])
            engine.set_rng(state['rng']);step=state['step'];history=state['history']
        if (durable/'charged.json').exists() and rt.read(durable/'charged.json')['step']>step:
            raise RuntimeError('VM lost diagnostic updates newer than recovery; refusing silent replay')
        def save(upload=False):
            original.save_checkpoint(checkpoint,dict(step=step,history=history,identity_sha256=identity_hash,
                adapter={n:v.detach().cpu() for n,v in model.state_dict().items() if original.is_adapter(n)},
                optimizer=optimizer.state_dict(),rng=engine.rng_state()))
            rt.write(durable/'training_history.json',history)
            if upload:
                rt.copy_verified(checkpoint,saved)
                rt.write(marker,dict(step=step,sha256=rt.digest(saved)))
        if not checkpoint.exists():save(upload=True)
        measurements=rt.read(durable/'measurements.json') if (durable/'measurements.json').exists() else []
        began=time.time()
        while step<=10:
            if stopping():
                save(upload=True);print('Stopped cleanly. Run this diagnostic cell again to resume.',flush=True);return
            if step in STEPS:
                before=original.parameter_digest(model,True)
                for probe in probes:
                    if any(r['update']==step and r['case']==probe['case'] for r in measurements):continue
                    if stopping():
                        save(upload=True);print('Probe stopped cleanly; rerun the diagnostic cell.',flush=True);return
                    row=measure(original,model,criterion,parameters,slabs,**probe,weights=weights,step=step)
                    row['adapter_sha256']=before
                    measurements.append(row);rt.write(durable/'measurements.json',measurements)
                    rt.write(durable/'summary.json',summarize(measurements,weights))
                    print(json.dumps(dict(phase='probe',update=step,case=probe['case'],
                        **row['fixed_weight_full_strength'])),flush=True)
                if original.parameter_digest(model,True)!=before:raise ValueError('Probe changed model parameters')
            if step==10:break
            rt.write(durable/'charged.json',dict(step=step+1))
            lr=original.lr_for_update(step,200,p['warmup_updates'],p['learning_rate'],p['final_learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);values=[]
            for micro in range(p['accumulation']):
                case,anchor=schedule[step*p['accumulation']+micro]
                losses=original.forward_slab(model,criterion,slabs,case,anchor)
                # Preserve the existing native baseline's full zero-weight auxiliary graph.
                total=losses['native']+0.*losses['bands']+0.*losses['edge']
                if not torch.isfinite(total):raise ValueError('Nonfinite native adaptation loss')
                (total/p['accumulation']).backward()
                values.append({n:float(losses[n].detach()) for n in NAMES})
                del losses,total
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            history.append(dict(update=step,lr=lr,gradient_norm=float(norm),
                **{n:float(np.mean([v[n] for v in values])) for n in NAMES}))
            save(upload=step in STEPS)
            print(json.dumps(dict(phase='native_adaptation',**history[-1])),flush=True)
        if len(measurements)!=15 or len(history)!=10:raise ValueError('Incomplete diagnostic')
        if original.parameter_digest(model,False)!=initial['frozen']:raise ValueError('Frozen base changed')
        rt.copy_verified(Path(__file__),durable/'diagnostic_source.py')
        rt.write(durable/'completion.json',dict(status='complete',native_updates=10,probe_backwards=45,
            probe_slabs=15,training_volumes=5,validation_volumes_evaluated=0,
            elapsed_seconds_this_session=time.time()-began,initial=initial,
            final_adapter_sha256=original.parameter_digest(model,True),
            interpretation='Ratios track a native-only adaptation trajectory. Full-strength ratios are hypothetical fixed-weight comparisons. Five-slab candidate weights are diagnostic, not production calibration. Head gradients are output-logit derivatives, not additive LoRA-gradient shares.'))
        files=['identity.json','slab_schedule.json','training_history.json','measurements.json',
               'summary.json','completion.json','diagnostic_source.py']
        archive={name:rt.digest(durable/name) for name in files};rt.write(durable/'archive_manifest.json',archive)
        for name,sha in archive.items():
            if rt.digest(durable/name)!=sha:raise IOError('Diagnostic archive readback failed')
        removed=[]
        for path in (checkpoint,saved,marker,durable/'charged.json'):
            if path.exists():removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
        rt.write(durable/'cleanup.json',dict(status='complete',removed=removed))
        print('Diagnosis complete: 10 native updates, 15 support probes, zero validation volumes.',flush=True)


if __name__=='__main__':
    main()
