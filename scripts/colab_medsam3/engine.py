"""Original MedSAM3 training recipe with freshly matched Colab constraints."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from runtime import (ARMS, RecoveryWriter, binding, check_ledger, copy_verified,
                     digest, fold_calibration, read, write)

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def stopped(deadline):
    return STOP or (deadline is not None and time.time() >= deadline)


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(), numpy=np.random.get_state())


def set_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python']); np.random.set_state(state['numpy'])


def save_pt(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); torch.save(state, temp); temp.replace(path)


class TransformCache:
    """Cache only deterministic CPU image transforms, never learned features."""
    def __init__(self, transform, maximum=1024**3):
        self.transform, self.maximum = transform, maximum
        self.entries, self.bytes, self.enabled = OrderedDict(), 0, True

    def __call__(self, value):
        if not self.enabled: return self.transform(value)
        key = (tuple(value.shape), str(value.dtype), hashlib.sha256(value.numpy().tobytes()).digest())
        if key in self.entries:
            self.entries.move_to_end(key); return self.entries[key].clone()
        result = self.transform(value)
        if result.device.type != 'cpu': raise ValueError('Transform cache must remain on CPU')
        cost = result.numel()*result.element_size()
        while self.entries and self.bytes+cost > self.maximum:
            _, old = self.entries.popitem(last=False); self.bytes -= old.numel()*old.element_size()
        if cost <= self.maximum:
            self.entries[key] = result.clone(); self.bytes += cost
        return result

    def clear(self):
        self.entries.clear(); self.bytes = 0


def prepare_slabs(original, root, p, processor):
    from scripts.depth_supervision.geometry import build_geometry, signed_distance
    slabs = original.MulticlassSlabs(root/'data', p['train_cases'], processor)
    slabs.sdf, slabs.depth = {}, {}
    geometry_audit = []
    for case in p['train_cases']:
        source, _ = original.load_image(root/'data', case)
        xyz = original.read_labels(root/'data', case, source)
        labels = xyz.transpose(2, 1, 0).copy()
        affine = source.affine.copy(); affine[:3, :3] = source.affine[:3, :3][:, ::-1]
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.allclose(spacing, 1., atol=1e-6): raise ValueError('Original MSD source grid must be 1 mm')
        sdf = np.stack([signed_distance(labels == c, tuple(spacing), 8.)/8. for c in (1, 2)])
        geometry = build_geometry(labels, affine, fields=False)
        slabs.sdf[case] = torch.from_numpy(sdf).float()
        slabs.depth[case] = torch.from_numpy(geometry['weights']).float()
        if not np.isclose(slabs.depth[case].sum(), 1., atol=1e-6): raise ValueError('Depth normalization differs')
        geometry_audit.append(dict(case=case, spacing=spacing.tolist(), affine_zyx=affine.tolist(),
            sdf_sha256=hashlib.sha256(sdf.tobytes()).hexdigest(),
            depth_sha256=hashlib.sha256(geometry['weights'].tobytes()).hexdigest(),
            depth_mass=float(geometry['weights'].sum()), roi_voxels=int(geometry['roi_voxels'])))
    slabs.geometry_audit = geometry_audit
    return slabs


def auxiliary(logp, truth, sdf, depth, slices):
    """Uniform anchors estimate full-volume SDF mean and normalized depth CE."""
    fg = torch.stack([truth == c for c in (1, 2)]).float()
    sdf_loss = ((logp[1:].exp()-fg)*sdf).mean()
    depth_loss = -(logp.gather(0, truth[None])[0]*depth).sum()*slices
    return sdf_loss, depth_loss


def forward(original, model, criterion, slabs, case, anchor):
    indices = slabs.indices[case]; planes, native = [], []
    for z in (anchor, (anchor+1) % len(indices)):
        pixels, labels = slabs.samples[indices[z]]
        logp, loss = original.forward_plane(model, slabs.processor, pixels, labels, criterion)
        planes.append(logp); native.append(loss)
    bands, edge = original.multiclass_losses(*planes, slabs.supports[case], anchor)
    truth = torch.as_tensor(slabs.samples[indices[anchor]][1].copy(), device=planes[0].device).long()
    sdf, depth = auxiliary(planes[0], truth, slabs.sdf[case][:, anchor].to(planes[0].device),
                          slabs.depth[case][anchor].to(planes[0].device), len(indices))
    return dict(native=torch.stack(native).mean(), bands=bands, edge=edge, sdf=sdf, depth=depth)


def gram_matrix(gradients):
    """Retain original FP64 dots/summation order with one device transfer."""
    terms, lengths = [], []
    for left in gradients:
        for right in gradients:
            values = [(a.detach().double()*b.detach().double()).sum()
                      for a,b in zip(left,right) if a is not None and b is not None]
            lengths.append(len(values)); terms.extend(values)
    values = torch.stack(terms).cpu().tolist() if terms else []
    result, offset = [], 0
    for size in lengths:
        result.append(sum(values[offset:offset+size])); offset += size
    n = len(gradients)
    return [result[i*n:(i+1)*n] for i in range(n)]


def weights_for(arm, rows, old=None):
    """Calibrate every term on this runtime, including bands and bands+edge.

    The historical calibration is provenance only. Sequential denominators use
    the newly calibrated preceding terms, with the original median/p95 rule.
    """
    names = ['native', 'bands', 'edge', 'sdf', 'depth']
    selected = [] if arm == 'baseline' else arm.split('_')
    coefficients = np.zeros(5); coefficients[0] = 1.
    weights, diagnostics = {}, {}
    for name in selected:
        index = names.index(name); ratios = []
        for row in rows:
            gram = np.asarray(row['gram']); denom = coefficients @ gram @ coefficients
            if denom > 0 and gram[index, index] > 0:
                ratios.append(float(np.sqrt(gram[index, index]/denom)))
        if len(ratios) < 5 or not np.isfinite(ratios).all(): raise ValueError('Insufficient calibration support')
        weight = min(.1/np.median(ratios), .5/np.quantile(ratios, .95))
        if not np.isfinite(weight) or weight <= 0: raise ValueError('Invalid calibrated weight')
        weights[name] = float(weight); coefficients[index] = weight
        diagnostics[name] = dict(weight=float(weight), valid_slabs=len(ratios),
            median=float(np.median(ratios)*weight), p95=float(np.quantile(ratios, .95)*weight),
            imported_original_coefficient=False)
        if diagnostics[name]['p95'] > .5 + 1e-12:
            raise ValueError('Fresh coefficient exceeds the original p95 cap')
    return dict(weights=weights, diagnostics=diagnostics, target=.1, p95_cap=.5,
                policy='fresh_runtime_sequential', matched_controls='fresh_colab')


def load(root, p):
    from scripts import train_medsam3_multiclass as original
    from scripts.train_medsam3_few_shot import is_adapter, parameter_digest
    original.reset_seed(p['seed'])
    model, processor = original.load_model(SimpleNamespace(upstream=root/'upstream',
        base_weights=root/'weights/sam3.pt', lora_weights=root/'weights/best_lora_weights.pt', device='cuda'))
    for n, parameter in model.named_parameters(): parameter.requires_grad_(is_adapter(n))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 18497664 or len(parameters) != 916:
        raise ValueError('Original LoRA architecture differs')
    old = read(root/'calibration'/f"fold{p['fold_index']}.json")
    for kind, flag in [('adapter_before', True), ('frozen_before', False)]:
        if parameter_digest(model, flag) != old['context'][kind]: raise ValueError('Original pretrained parameters differ: '+kind)
    processor.transform = TransformCache(processor.transform)
    criterion = original.native_loss(model)
    slabs = prepare_slabs(original, root, p, processor)
    context = original.calibration_context(p, slabs.inventory,
        parameter_digest(model, True), parameter_digest(model, False))
    context['multiclass'] = {k: p[k] for k in ('classes', 'class_prompts', 'semantic_projection', 'constraints')}
    if context != old['context']: raise ValueError('Original calibration/data recipe mismatch')
    model.train()
    return original, model, processor, criterion, parameters, slabs, old


def equivalence(original, model, processor, criterion, parameters, slabs, p):
    """Real GPU loss/gradient and optimizer checkpoint-resume checks."""
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('Strict equivalence requires the isolated deterministic audit process')
    case, anchor = slabs.calibration_schedule(4)[1]
    by_id = {id(value): name for name, value in model.named_parameters()}
    parameter_names = [by_id[id(value)] for value in parameters]
    processor.transform.enabled = False
    original.reset_seed(7319)
    losses = original.forward_slab(model, criterion, slabs, case, anchor)
    expected = [float(v.detach()) for v in losses]
    gradients = []
    for i, loss in enumerate(losses):
        g = torch.autograd.grad(loss, parameters, retain_graph=i<2, allow_unused=True)
        gradients.append([None if x is None else x.detach().cpu().clone() for x in g])
    del losses, g, loss
    for test in ('original_repeat', 'extension_cached'):
        processor.transform.enabled = test == 'extension_cached'
        original.reset_seed(7319)
        losses = (forward(original, model, criterion, slabs, case, anchor) if test == 'extension_cached'
                  else dict(zip(('native', 'bands', 'edge'),
                                original.forward_slab(model, criterion, slabs, case, anchor))))
        for i, name in enumerate(('native', 'bands', 'edge')):
            np.testing.assert_allclose(float(losses[name].detach()), expected[i], rtol=1e-6, atol=1e-7,
                                       err_msg=f'{test}: {name} loss differs')
            actual = torch.autograd.grad(losses[name], parameters, retain_graph=i<2, allow_unused=True)
            for parameter_name, a, b in zip(parameter_names, gradients[i], actual):
                label = f'{test}: {name} gradient for {parameter_name}'
                if (a is None) != (b is None): raise ValueError(label+' connectivity changed')
                if a is not None:
                    torch.testing.assert_close(a, b.detach().cpu(), rtol=1e-5, atol=1e-7,
                                               msg=lambda message: label+'\n'+message)
        del actual, losses
        print('PASSED strict equivalence:', test, flush=True)
    del gradients
    initial = [x.detach().cpu().clone() for x in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=.01)
    def update():
        optimizer.zero_grad(set_to_none=True)
        values = forward(original, model, criterion, slabs, case, anchor)
        total = values['native'] + .01*sum(values[k] for k in ('bands', 'edge', 'sdf', 'depth'))
        total.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True); optimizer.step()
    original.reset_seed(7319); update()
    state = dict(parameters=[x.detach().cpu().clone() for x in parameters],
                 optimizer=deepcopy(optimizer.state_dict()), rng=rng_state())
    update(); expected_params = [x.detach().cpu().clone() for x in parameters]
    expected_optimizer = deepcopy(optimizer.state_dict()); expected_rng = rng_state()
    with torch.no_grad():
        for x, y in zip(parameters, state['parameters']): x.copy_(y)
    optimizer.load_state_dict(state['optimizer']); set_rng(state['rng']); update()
    for x, y in zip(parameters, expected_params): torch.testing.assert_close(x.detach().cpu(), y, rtol=0, atol=0)
    for key, record in optimizer.state_dict()['state'].items():
        for field, value in record.items():
            torch.testing.assert_close(value, expected_optimizer['state'][key][field], rtol=0, atol=0)
    actual_rng = rng_state()
    if not torch.equal(actual_rng['torch'], expected_rng['torch']) or any(not torch.equal(a,b) for a,b in zip(actual_rng['cuda'], expected_rng['cuda'])):
        raise ValueError('Checkpoint resume RNG differs')
    with torch.no_grad():
        for x, y in zip(parameters, initial): x.copy_(y)
    optimizer.zero_grad(set_to_none=True); del optimizer, state, expected_params, expected_optimizer, initial
    torch.cuda.empty_cache()
    # Inference has no annotation inputs and is unchanged by deterministic transform caching.
    model.eval(); pixels = slabs.samples[0][0]
    with torch.inference_mode():
        processor.transform.enabled=False; a,_=original.forward_plane(model, processor, pixels)
        processor.transform.enabled=True; b,_=original.forward_plane(model, processor, pixels)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    model.train()
    return dict(status='passed', original_repeatability=True, original_loss_gradient_equivalence=True,
                optimizer_checkpoint_resume_exact=True, label_free_inference_exact=True,
                numerical_mode='isolated deterministic audit; original calibration/training modes unchanged',
                peak_gpu_bytes=torch.cuda.max_memory_allocated())


def equivalence_audit(root, durable, fold):
    """Only this process changes GPU algorithms; calibration starts separately."""
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Audit must start with CUBLAS_WORKSPACE_CONFIG=:4096:8')
    # Deterministic operators can invoke AOTAutograd in torch 2.12.
    # These independent loss gradients deliberately reuse the same graph.
    import torch._functorch.config as aot_config
    aot_config.donated_buffer = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('AUDIT SETTINGS: deterministic=True, donated_buffer=False', flush=True)
    value = binding(root, durable)
    p = read(root/'protocols'/f'fold{fold}_seed17.json')
    original, model, processor, criterion, parameters, slabs, old = load(root, p)
    from audit_numerics import audit_empty_point_grid
    with audit_empty_point_grid() as grid_audit:
        checks = equivalence(original, model, processor, criterion, parameters, slabs, p)
    checks['empty_point_grid'] = grid_audit
    checks['aot_donated_buffer'] = aot_config.donated_buffer
    from scripts.train_medsam3_few_shot import parameter_digest
    if (parameter_digest(model, True) != old['context']['adapter_before']
            or parameter_digest(model, False) != old['context']['frozen_before']):
        raise ValueError('Diagnostic updates did not restore the exact original model')
    write(durable/'calibration'/f'fold{fold}'/'equivalence.json',
          dict(status='passed', binding=value, checks=checks))


def ensure_equivalence_audit(root, durable, fold, value):
    path = durable/'calibration'/f'fold{fold}'/'equivalence.json'
    def valid(record):
        checks = record.get('checks', {})
        return (record.get('status') == 'passed' and record.get('binding') == value
                and all(checks.get(key) is True for key in (
                    'original_repeatability', 'original_loss_gradient_equivalence',
                    'optimizer_checkpoint_resume_exact', 'label_free_inference_exact')))
    if not path.exists() or not valid(read(path)):
        command = [sys.executable, str(root/'engine.py'), 'equivalence',
                   '--bundle', str(root), '--durable', str(durable), '--fold', str(fold)]
        print('Running isolated deterministic GPU audit for fold', fold, flush=True)
        with tempfile.TemporaryDirectory(prefix='medsam3_audit_') as cache:
            env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', TORCHINDUCTOR_CACHE_DIR=cache)
            subprocess.run(command, env=env, check=True)
    record = read(path)
    if not valid(record): raise ValueError('Missing or invalid deterministic equivalence receipt')
    return record['checks']


def preflight(root, durable, fold):
    from reuse_preflight import validate_rows, verify_reused_evidence
    value = binding(root, durable); folder = durable/'calibration'/f'fold{fold}'
    verify_reused_evidence(folder, value)
    receipt = folder/'receipt.json'
    if receipt.exists() and read(receipt).get('status') == 'passed' and read(receipt).get('binding') == value:
        if digest(folder/'calibration.json') != read(receipt)['calibration_sha256']: raise ValueError('Calibration changed')
        print('Reusing fold calibration', fold, flush=True); return
    # No model is allocated in the calibration process until the audit exits.
    # Its deterministic kernels, cuBLAS workspace and diagnostic updates cannot
    # leak into the original calibration or scientific training process.
    check = ensure_equivalence_audit(root, durable, fold, value)
    p = read(root/'protocols'/f'fold{fold}_seed17.json')
    original, model, processor, criterion, parameters, slabs, old = load(root,p)
    names = ['native', 'bands', 'edge', 'sdf', 'depth']; rows=[]
    original.reset_seed(p['calibration_seed'])
    partial = folder/'partial.pt'
    if partial.exists():
        state = torch.load(partial,map_location='cpu',weights_only=False)
        if state['binding'] != value: raise ValueError('Calibration recovery environment changed')
        rows=state['rows']; set_rng(state['rng'])
    elif (folder/'measurements.json').exists():
        saved = read(folder/'measurements.json')
        if saved['binding'] != value: raise ValueError('Saved measurements environment changed')
        rows = saved['rows']
        validate_rows(rows, slabs.calibration_schedule(4))
        print(f'Fold {fold}: reusing all {len(rows)} verified measurements; recalculating weights', flush=True)
    for index,(case,anchor) in enumerate(slabs.calibration_schedule(4)):
        if index < len(rows): continue
        if STOP: raise SystemExit(75)
        losses=forward(original,model,criterion,slabs,case,anchor)
        gradients=[torch.autograd.grad(losses[n],parameters,retain_graph=i<4,allow_unused=True) for i,n in enumerate(names)]
        gram=gram_matrix(gradients)
        if not np.isfinite(gram).all() or gram[0][0] <= 0: raise ValueError('Invalid gradient calibration')
        rows.append(dict(case=case,anchor=anchor,gram=gram,losses={k:float(v.detach()) for k,v in losses.items()}))
        save_pt(partial,dict(binding=value,rows=rows,rng=rng_state()))
        print(f'Fold {fold}: calibration {len(rows)}/20',flush=True)
        del gradients,losses
    validate_rows(rows, slabs.calibration_schedule(4))
    calibrations={arm:weights_for(arm,rows,old) for arm in ARMS}
    write(folder/'calibration.json',dict(arms=calibrations,rows=rows,original_calibration_sha256=digest(root/'calibration'/f'fold{fold}.json'),
        scope='Training support only, shared exactly across optimization seeds', geometry=slabs.geometry_audit,
        accounting='Calibration/preflight are separate from the original 200-update budget; FLOPs not measured'))
    write(receipt,dict(status='passed',binding=value,calibration_sha256=digest(folder/'calibration.json'),checks=check))
    partial.unlink(missing_ok=True)


def evaluate(original,model,processor,root,p,output):
    records=read(output/'evaluation.json')['records'] if (output/'evaluation.json').exists() else []
    processor.transform.clear();processor.transform.enabled=False
    model.eval().requires_grad_(False)
    for split,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])]:
        for case in cases:
            if any(r['case']==case and r['split']==split for r in records):continue
            if stopped(p.get('deadline')):raise SystemExit(75)
            source,image=original.load_image(root/'data',case)
            prediction=np.empty(image.shape,dtype=np.uint8)
            with torch.inference_mode():
                for z in range(image.shape[2]):
                    pixels=np.round(image[:,:,z].T*255).astype(np.uint8)
                    logp,_=original.forward_plane(model,processor,pixels)
                    probs=logp.exp()
                    if not torch.isfinite(probs).all() or not torch.allclose(probs.sum(0),torch.ones_like(probs[0]),atol=1e-5):
                        raise ValueError('Invalid class probabilities')
                    prediction[:,:,z]=probs.argmax(0).cpu().numpy().T
            truth=original.read_labels(root/'data',case,source)
            records.append(dict(case=case,split=split,**original.metrics(truth,prediction),
                prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest()))
            write(output/'evaluation.json',dict(records=records))
            print('EVALUATED',split,case,flush=True)
    return records


def train(root,durable,task,deadline):
    from scripts.train_medsam3_few_shot import is_adapter, parameter_digest, lr_for_update
    p=read(root/'protocols'/f"fold{task['fold']}_seed{task['seed']}.json")
    p['arm']=task['arm']; output=root/'work'/task['id']; recovery=durable/'recovery'/task['id']
    if (output/'protocol.json').exists() and read(output/'protocol.json')!=p:raise ValueError('Resume recipe changed')
    write(output/'protocol.json',p);check_ledger(output,recovery)
    all_cal=fold_calibration(root,durable,task['fold'])
    calibration=all_cal['arms'][task['arm']]
    write(output/'calibration.json',dict(**calibration,rows=all_cal['rows'],geometry=all_cal['geometry'],
        original_calibration_sha256=all_cal['original_calibration_sha256'],
        shared_calibration_sha256=digest(durable/'calibration'/f"fold{task['fold']}"/'calibration.json')))
    original,model,processor,criterion,parameters,slabs,old=load(root,p)
    write(output/'runtime_recipe.json',dict(augmentation='none',precision='native CUDA BF16 autocast; FP32 model/LoRA parameters',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        optimizer='AdamW',weight_decay=.01,gradient_clip_norm=1.,updates=200,accumulation=2,
        learning_rate=p['learning_rate'],final_learning_rate=p['final_learning_rate'],
        warmup_updates=20,constraint_warmup_updates=20,trainable_parameters=sum(v.numel() for v in parameters),
        classes=p['classes'],prompts=p['class_prompts'],native_loss_source_sha256=digest(root/'source/scripts/train_medsam3_few_shot.py'),
        image_transform_cache_max_bytes=processor.transform.maximum,learned_feature_cache=False,
        optimization_flop_cap=None))
    schedule=slabs.schedule(p);write(output/'slab_schedule.json',schedule)
    original.reset_seed(p['seed'])
    optimizer=torch.optim.AdamW(parameters,lr=p['learning_rate'],weight_decay=.01)
    latest=output/'checkpoint_latest.pt'; history=[];start=0;elapsed=0.;began=time.monotonic()
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['protocol_sha256']!=digest(output/'protocol.json'):raise ValueError('Checkpoint recipe differs')
        if (state['step']!=read(output/'checkpoint_ledger.json')['step']
                or len(state['history'])!=state['step']
                or [r['update'] for r in state['history']]!=list(range(1,state['step']+1))):
            raise ValueError('Checkpoint update ledger/history disagree')
        model.load_state_dict(state['adapter'],strict=False);optimizer.load_state_dict(state['optimizer']);set_rng(state['rng'])
        start=state['step'];history=state['history'];elapsed=state['elapsed_seconds']
    writer=RecoveryWriter(output,recovery)
    def commit(step):
        save_pt(latest,dict(step=step,adapter={n:v.detach().cpu() for n,v in model.state_dict().items() if is_adapter(n)},
            optimizer=optimizer.state_dict(),rng=rng_state(),history=history,
            protocol_sha256=digest(output/'protocol.json'),elapsed_seconds=elapsed+time.monotonic()-began))
        write(output/'checkpoint_ledger.json',dict(step=step,protocol_sha256=digest(output/'protocol.json')))
        write(output/'training_history.json',history);writer.capture()
    coherent=False
    try:
        if not latest.exists():commit(0);writer.capture(force=True)
        for update in range(start,p['updates']):
            if stopped(deadline):commit(update);coherent=True;raise SystemExit(75)
            ledger=dict(step=update+1,protocol_sha256=digest(output/'protocol.json'))
            write(recovery/'charged_ledger.json',ledger)
            if read(recovery/'charged_ledger.json')!=ledger:raise IOError('Drive charge readback failed')
            lr=lr_for_update(update,p['updates'],p['warmup_updates'],p['learning_rate'],p['final_learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            ramp=min((update+1)/p['constraint_warmup_updates'],1.)
            optimizer.zero_grad(set_to_none=True);rows=[]
            for micro in range(p['accumulation']):
                case,anchor=schedule[update*p['accumulation']+micro]
                losses=forward(original,model,criterion,slabs,case,anchor)
                total=losses['native']+ramp*sum(w*losses[k] for k,w in calibration['weights'].items())
                if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
                (total/p['accumulation']).backward()
                rows.append(dict(total=float(total.detach()),**{k:float(v.detach()) for k,v in losses.items()}))
                del total,losses
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            history.append(dict(update=update+1,lr=lr,ramp=ramp,gradient_norm=float(norm),
                **{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}))
            # Local checkpoint every update; Drive snapshots are asynchronous and bounded.
            commit(update+1)
            if update==0 or (update+1)%10==0:print(history[-1],flush=True)
        writer.capture(force=True);coherent=True
        after=parameter_digest(model,False);adapter=parameter_digest(model,True)
        if after!=old['context']['frozen_before'] or adapter==old['context']['adapter_before']:
            raise ValueError('Frozen base changed or adapter failed to learn')
        write(output/'training_summary.json',dict(updates=200,augmentation='none',frozen_before=after,frozen_after=after,
            adapter_before=old['context']['adapter_before'],adapter_after=adapter,weights=calibration['weights'],
            elapsed_seconds=elapsed+time.monotonic()-began,schedule_sha256=digest(output/'slab_schedule.json'),
            budget_policy='Original 200 optimizer updates, accumulation 2; not a FLOP cap',
            flops=None,calibration_accounted_separately=True,checkpoint_selection='final_update'))
        optimizer.zero_grad(set_to_none=True);del optimizer
        records=evaluate(original,model,processor,root,{**p,'deadline':deadline},output)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for key in ['native','total']:axes[0].plot([r['update'] for r in history],[r[key] for r in history],label=key)
        for key in calibration['weights']:axes[1].plot([r['update'] for r in history],[r[key] for r in history],label=key)
        for ax in axes:ax.set_xlabel('Optimizer update');ax.legend();ax.grid(alpha=.2)
        axes[0].set_title('Training objective');axes[1].set_title('Auxiliary losses')
        fig.tight_layout();fig.savefig(output/'curves.png',dpi=150);plt.close(fig)
        validation=[r for r in records if r['split']=='validation'];training=[r for r in records if r['split']=='training_support']
        write(output/'completion.json',dict(status='complete',updates=200,augmentation='none',
            validation_dice=float(np.mean([r['ap_mean_dice'] for r in validation])),training_dice=float(np.mean([r['ap_mean_dice'] for r in training])),
            foreground_fn=float(np.mean([r['per_class']['background']['fp'] for r in validation])),
            foreground_fp=float(np.mean([r['per_class']['background']['fn'] for r in validation])),
            ap_swaps=float(np.mean([r['anterior_to_posterior']+r['posterior_to_anterior'] for r in validation])),
            checkpoint_sha256=digest(latest),evaluation_sha256=digest(output/'evaluation.json'),
            budget='200 updates; not FLOP capped',primary_checkpoint='final_update'))
    finally:writer.close(force=coherent)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('preflight','equivalence','train'));parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--durable',type=Path,required=True);parser.add_argument('--fold',type=int);parser.add_argument('--task')
    parser.add_argument('--deadline',type=float);args=parser.parse_args()
    sys.path.insert(0,str(args.bundle/'source'));torch.set_num_threads(4)
    for name in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):signal.signal(name,request_stop)
    if args.command=='preflight':preflight(args.bundle,args.durable,args.fold)
    elif args.command=='equivalence':equivalence_audit(args.bundle,args.durable,args.fold)
    else:
        task=next(t for t in read(args.bundle/'plan.json')['tasks'] if t['id']==args.task)
        train(args.bundle,args.durable,task,args.deadline)


if __name__=='__main__':main()
