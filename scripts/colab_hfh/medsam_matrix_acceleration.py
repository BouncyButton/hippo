"""Exact initialization calibration reuse and checkpoint-local binary prompt caching."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time
import zipfile


def same_initialization(engine, model, slabs, p, output, z):
    def data(name):
        return json.loads(z.read('results/'+name+'.json'))
    manifest=data('archive_manifest')
    for name in ('adaptive_calibration_sdf_000','protocol','execution_hardware','training_summary','training_inventory'):
        content=z.read('results/'+name+'.json')
        if hashlib.sha256(content).hexdigest()!=manifest[name+'.json']:
            return False
    q=data('protocol')
    keys=('fold','seed','dataset_sha256','lora_sha256','base_sha256','classes','prompts','training_numerics',
          'train_cases','augmentation','initialization','adaptive_band','matrix_source_hashes','calibration_seed')
    return (q['arm']=='dice_sdf' and all(q[k]==p[k] for k in keys)
            and data('training_inventory')==engine.read(output/'training_inventory.json')
            and data('execution_hardware')==engine.read(output/'execution_hardware.json')
            and data('training_summary')['frozen_before']==engine.parameter_digest(model,False)
            and data('training_summary')['adapter_before']==engine.parameter_digest(model,True))


def shared_sdf(reference, band, engine, model, criterion, parameters, slabs, p, output, term, update):
    import torch
    from scripts.colab_hfh.medsam_intensity_augmentation import augmented_samples
    archive=output.parents[1]/'archives'/f"matrix125__dice_sdf__f{p['fold']}_s{p['seed']}.zip"
    path=output/'adaptive_calibration_sdf_000.json'
    def original():
        return reference(engine,model,criterion,parameters,slabs,p,output,term,update)
    if term!='sdf' or update or not archive.is_file() or path.exists():
        return original()
    with zipfile.ZipFile(archive) as z:
        if not same_initialization(engine,model,slabs,p,output,z):
            return original()
        saved=json.loads(z.read('results/adaptive_calibration_sdf_000.json'))
        source_protocol=hashlib.sha256(z.read('results/protocol.json')).hexdigest()
    schedule=band.calibration_schedule(p,{k:len(v) for k,v in slabs.indices.items()})
    rows=saved['rows']
    if (saved['identity']['protocol_sha256']!=source_protocol or len(rows)!=len(schedule)
            or any(any(r[k]!=v for k,v in entry.items()) for r,entry in zip(rows,schedule))):
        return original()
    # Recompute one real training slab at this arm's initialization before using the other 79.
    entry=schedule[0]
    with band.preserve_runtime(model,slabs):
        model.train()
        engine.reset_seed(entry['model_seed'])
        with augmented_samples(slabs,entry['case'],entry['anchor'],entry['augmentation_seed']) as params:
            losses=engine.forward_slab(model,criterion,slabs,entry['case'],entry['anchor'])
            norms=[]
            for name in ('native','sdf'):
                grads=torch.autograd.grad(losses[name],parameters,retain_graph=name=='native',allow_unused=True)
                norms.append(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))
    matched=(params==rows[0]['augmentation'] and norms==[rows[0]['native_norm2'],rows[0]['auxiliary_norm2']])
    if not matched:
        engine.write(output/'sdf_cache_receipt.json',dict(hit=False,reason='reference slab differed'))
        return original()
    summary=band.calibration_summary(rows,'sdf')
    if any(saved[k]!=v for k,v in summary.items()):
        raise ValueError('Saved SDF coefficient does not reconstruct')
    result=dict(summary,identity=dict(saved['identity'],protocol_sha256=engine.digest(output/'protocol.json')))
    engine.write(path,result)
    engine.write(output/'sdf_cache_receipt.json',dict(hit=True,source_archive=archive.name,
                 sha256=engine.digest(archive),recomputed_slabs=1,reused_slabs=len(rows)-1,
                 source_protocol_sha256=source_protocol))
    print('SDF_CACHE_HIT',archive.name,'79 of 80 calibration slabs reused',flush=True)
    return result


@contextmanager
def text_cache(model, stats):
    """The binary case has one prompt. Reuse text only within a single frozen-weight inference pass."""
    import torch
    from scripts.colab_medsam3.fast_eval import clone_tree, equal_state, replace_attribute
    backbone=model.backbone
    original=backbone.forward_text
    cache={}
    disabled=False
    def forward(captions,input_boxes=None,additional_text=None,device='cuda'):
        nonlocal disabled
        if (disabled or torch.is_grad_enabled() or model.training or input_boxes is not None
                or additional_text is not None or not isinstance(captions,(list,tuple))):
            return original(captions,input_boxes=input_boxes,additional_text=additional_text,device=device)
        key=(tuple(captions),str(device))
        if key in cache and stats['checked']:
            stats['hits']+=1
            return clone_tree(cache[key])
        from scripts.colab_mni_medsam3.optimized import rng_state
        before=rng_state()
        value=original(captions,device=device)
        if not equal_state(before,rng_state()):
            disabled=True
        elif key in cache:
            stats['checked']=equal_state(value,cache[key])
            disabled=not stats['checked']
        elif not cache:
            try:
                cache[key]=clone_tree(value)
            except TypeError:
                disabled=True
        return value
    with replace_attribute(backbone,'forward_text',forward):
        yield


def install():
    from scripts.colab_hfh import medsam_adaptive_band as band
    from scripts.colab_hfh import medsam_constraints_mask_only as spatial
    from scripts.colab_hfh.runtime import write
    # ADNI-style grouped host reads. Each per-parameter sum keeps the original
    # FP64 GPU reduction and the final Python summation order.
    import inspect
    import torch
    source=inspect.getsource(band.calibrate_term)
    old='sum(float(g.detach().double().square().sum()) for g in gs if g is not None)'
    def norm2(gs):
        scalars=[g.detach().double().square().sum() for g in gs if g is not None]
        return sum(torch.stack(scalars).tolist()) if scalars else 0.
    assert source.count(old)==1
    scope=dict(band.__dict__,matrix_norm2=norm2)
    exec(compile(source.replace(old,'matrix_norm2(gs)'),__file__,'exec'),scope)
    original=scope['calibrate_term']
    band.calibrate_term=lambda *args:shared_sdf(original,band,*args)
    probe=band.BandGate.probe
    def faster_probe(self,update):
        stats=dict(hits=0,checked=False)
        began=time.monotonic()
        with text_cache(self.model,stats):
            result=probe(self,update)
        write(self.output/f'prompt_cache_probe_{update:03d}.json',dict(**stats,seconds=time.monotonic()-began))
        return result
    band.BandGate.probe=faster_probe
    original_spatial=spatial.install_spatial_evaluation
    def faster_spatial(matched):
        original_spatial(matched)
        evaluator=matched.REFERENCE_EVALUATE
        def evaluate(model,processor,volumes,protocol,output):
            stats=dict(hits=0,checked=False)
            began=time.monotonic()
            with text_cache(model,stats):
                result=evaluator(model,processor,volumes,protocol,output)
            write(output/'prompt_cache_eval.json',dict(**stats,seconds=time.monotonic()-began))
            return result
        matched.REFERENCE_EVALUATE=evaluate
    spatial.install_spatial_evaluation=faster_spatial
    # Store the exact execution overlay alongside the existing per-arm model and loss sources.
    import sys
    if '--output' in sys.orig_argv:
        output=Path(sys.orig_argv[sys.orig_argv.index('--output')+1])
        source=Path(__file__).read_bytes()
        write(output/'matrix_acceleration.json',dict(source=source.decode(),sha256=hashlib.sha256(source).hexdigest()))
