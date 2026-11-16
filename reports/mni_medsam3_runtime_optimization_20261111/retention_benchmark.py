"""Real-model, disposable tests of activation retention and packed checkpoints."""
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import statistics
import time
import torch


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


@contextmanager
def retention(model, mode):
    import torch.utils.checkpoint as cp
    import sam3.model.text_encoder_ve as text_encoder
    reference=cp.checkpoint
    text_reference=text_encoder.checkpoint
    selected={id(module) for name,module in model.named_modules()
              if (name.startswith('backbone.language_backbone') if mode.startswith('text')
                  else not name.startswith('backbone.vision_backbone'))}
    if mode.startswith('vision8'):
        selected.update(id(model.backbone.vision_backbone.trunk.blocks[i]) for i in range(0,32,4))
    if mode in ('text_vision4','text_vision8'):
        stride=8 if mode=='text_vision4' else 4
        selected.update(id(model.backbone.vision_backbone.trunk.blocks[i]) for i in range(0,32,stride))
    def checkpoint(function,*args,**kwargs):
        owner=getattr(function,'__self__',None)
        if (id(function) in selected or id(owner) in selected) and set(kwargs)<={'use_reentrant'}:
            if kwargs.get('use_reentrant',False):raise ValueError('Reentrant checkpoint not audited')
            return function(*args)
        return reference(function,*args,**kwargs)
    cp.checkpoint=checkpoint
    text_encoder.checkpoint=checkpoint
    try:yield [dict(policy=mode,selected_modules=len(selected),architecture_flags_unchanged=True)]
    finally:
        cp.checkpoint=reference
        text_encoder.checkpoint=text_reference


def state():
    import random,numpy as np
    return (torch.get_rng_state(),torch.cuda.get_rng_state_all(),random.getstate(),np.random.get_state())


def restore(r):
    import random,numpy as np
    torch.set_rng_state(r[0]);torch.cuda.set_rng_state_all(r[1]);random.setstate(r[2]);np.random.set_state(r[3])


def equal(a,b):
    import numpy as np
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    return a==b


def run(output,engine,model,processor,criterion,parameters,slabs,p,weights,mode='candidates'):
    from contextlib import nullcontext, ExitStack
    from scripts.cross_dataset_campaign.foundation_fast import validation_sampling
    modes=(['reference','nonvision','vision8'] if mode=='candidates' else
           ['reference','text','text_vision4','text_vision8'] if mode=='selective' else
           ['reference','casts','nonvision_casts','vision8_casts'])
    receipt=dict(status='testing',timings={},equivalence={},flags=[])
    schedule=slabs.schedule(p)
    initial=[x.detach().cpu().clone() for x in parameters]
    engine.reset_seed(7319)
    saved_rng=state()
    @contextmanager
    def context(mode):
        with ExitStack() as stack:
            flags=[]
            if mode not in ('reference','casts'):
                flags=stack.enter_context(retention(model,mode))
            if 'casts' in mode:
                from frozen_casts import frozen_linear_cache
                stack.enter_context(frozen_linear_cache(model))
            yield flags
    def backward(index=0):
        model.zero_grad(set_to_none=True)
        for micro in range(p['accumulation']):
            case,anchor=schedule[index*p['accumulation']+micro]
            losses=engine.forward_slab(model,criterion,slabs,case,anchor)
            total=losses['native']+sum(weights[n]*losses[n] for n in weights)
            (total/p['accumulation']).backward()
        return {n:float(v.detach()) for n,v in losses.items()}
    # Lightweight synchronized phases avoid the very large CUDA operator trace.
    import cProfile,pstats,io
    profile=cProfile.Profile()
    phases=[]
    model.zero_grad(set_to_none=True)
    for micro in range(p['accumulation']):
        case,anchor=schedule[micro]
        torch.cuda.synchronize();began=time.perf_counter();profile.enable()
        losses=engine.forward_slab(model,criterion,slabs,case,anchor)
        torch.cuda.synchronize();profile.disable();forward=time.perf_counter()-began
        total=losses['native']+sum(weights[n]*losses[n] for n in weights)
        torch.cuda.synchronize();began=time.perf_counter();profile.enable()
        (total/p['accumulation']).backward()
        torch.cuda.synchronize();profile.disable();backward_seconds=time.perf_counter()-began
        phases.append(dict(forward_seconds=forward,backward_seconds=backward_seconds))
        del losses,total
    stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(65)
    (output/'python_profile.txt').write_text(stream.getvalue())
    receipt['synchronized_phases']=phases
    save(output/'candidate_result.json',receipt)
    # Alternate variants. All use the same initial model, samples and RNG.
    for mode in modes+list(reversed(modes)):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        restore(saved_rng)
        try:
            with context(mode) as flags:
                torch.cuda.synchronize();start=time.perf_counter()
                values=backward()
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                peak=torch.cuda.max_memory_allocated()
            row=dict(seconds=seconds,peak_bytes=peak,losses=values)
            receipt['timings'].setdefault(mode,[]).append(row)
            if flags:receipt['flags']=flags
            print('TIMING',mode,seconds,peak/1024**3,flush=True)
        except torch.OutOfMemoryError as e:
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            receipt['timings'].setdefault(mode,[]).append(dict(error='out_of_memory'))
            print('OOM',mode,flush=True)
        save(output/'candidate_result.json',receipt)
    # Use the project's established deterministic sampling audit solely here.
    # Production preserves the existing CUDA sampling and stochastic layers.
    reference=None
    baseline_median=statistics.median(x['seconds'] for x in receipt['timings']['reference'])
    for mode in modes:
        if any('error' in x for x in receipt['timings'][mode]):continue
        if mode!='reference' and statistics.median(x['seconds'] for x in receipt['timings'][mode])>=baseline_median/1.02:
            receipt['equivalence'][mode]=dict(status='not_promoted_no_measurable_speed_gain')
            continue
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();restore(saved_rng)
        with validation_sampling(), context(mode):
            case,anchor=slabs.calibration_schedule(4)[0]
            losses=engine.forward_slab(model,criterion,slabs,case,anchor)
            names=['native','bands','edge','sdf']
            values={n:float(losses[n].detach()) for n in names}
            gradients={}
            for i,n in enumerate(names):
                gs=torch.autograd.grad(losses[n],parameters,retain_graph=i<len(names)-1,allow_unused=True)
                gradients[n]=[None if g is None else g.detach().cpu() for g in gs]
                del gs
            rng=state()
            del losses
        if mode=='reference':
            reference=(values,gradients,rng)
            receipt['equivalence'][mode]=dict(status='reference')
        else:
            exact=equal(values,reference[0]) and equal(gradients,reference[1]) and equal(rng,reference[2])
            maximum=0.;allclose=True
            for n in names:
                for a,b in zip(reference[1][n],gradients[n]):
                    if (a is None)!=(b is None):allclose=False
                    elif a is not None:
                        maximum=max(maximum,float((a-b).abs().max()))
                        allclose=allclose and torch.allclose(a,b,rtol=1e-5,atol=1e-7)
            receipt['equivalence'][mode]=dict(exact=exact,losses_exact=equal(values,reference[0]),
                rng_exact=equal(rng,reference[2]),gradients_allclose=bool(allclose),max_gradient_difference=maximum)
            del gradients
        print('EQUIVALENCE',mode,receipt['equivalence'][mode],flush=True)
        save(output/'candidate_result.json',receipt)
    receipt.update(status='complete',campaign_updates=0,
        parameters_unchanged=all(torch.equal(a,b.detach().cpu()) for a,b in zip(initial,parameters)))
    save(output/'candidate_result.json',receipt)
