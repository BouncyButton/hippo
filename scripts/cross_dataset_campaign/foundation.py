"""Resumable volume-supervised SAM3/MedSAM3 LoRA, without augmentation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from scripts.cross_dataset_campaign.common import (read, write, digest, confusion_metrics,
    install_signals, stopping, check_no_augmentation)
from scripts.train_medsam3_few_shot import (TrainingSlices, move_to_device, native_loss,
    is_adapter, parameter_digest, lr_for_update, load_image)
from scripts.train_medsam3_constraint_few_shot import reset_seed, Slabs
from scripts.medsam3_supervised_constraints import VolumeSupport, sampled_losses, semantic_log_fields
from evaluation.medsam3_zero_shot import normalize_image


class Volumes:
    def __init__(self, protocol):
        self.protocol = protocol
        self.dataframe = None
        if protocol['dataset'] == 'ADNI':
            from scripts.train_compute_matched_baselines import _load_pkl_dataframe
            self.dataframe = _load_pkl_dataframe(Path(protocol['data']))

    def load(self, case, labels=True):
        p = self.protocol
        if p['dataset'] == 'ADNI':
            from scripts.adni_baselines.core import case_name
            selected = [r for r in self.dataframe.itertuples() if case_name(r) == case]
            if len(selected) != 1:
                raise ValueError('Missing/duplicate ADNI crop')
            row = selected[0]
            return normalize_image(np.asarray(row.image_data)), np.asarray(row.data, dtype=np.uint8) if labels else None
        if p['dataset'] == 'GeoFWI3D':
            import nibabel as nib
            item = p['manifest']['cases'][case]
            root = Path(p['data'])
            image = np.asarray(nib.load(root/item['image']).dataobj, dtype=np.float32)
            # Preserve the published, label-independent amplitude normalization.
            image = (image+1.)*.5
            label = np.asarray(nib.load(root/item['label']).dataobj, dtype=np.uint8) if labels else None
            return image, label
        from scripts.train_medsam3_multiclass import read_labels
        source, image = load_image(Path(p['data']), case)
        return image, read_labels(Path(p['data']), case, source) if labels else None


class SupportSlabs(Slabs):
    def __init__(self, volumes, protocol, processor):
        from scripts.depth_supervision.geometry import signed_distance, build_geometry, GeometryConfig
        self.processor, self.protocol = processor, protocol
        self.samples, self.inventory, self.indices, self.supports = [], [], {}, {}
        self.sdf, self.depth = {}, {}
        for case in protocol['train_cases']:
            image, label = volumes.load(case)
            if image.shape != label.shape or not np.isin(label, range(len(protocol['classes']))).all():
                raise ValueError('Invalid support volume')
            labels = torch.from_numpy(label.transpose(2,1,0).copy())
            self.indices[case] = []
            for z in range(image.shape[2]):
                self.indices[case].append(len(self.samples))
                self.samples.append((np.round(image[:,:,z].T*255).astype(np.uint8), label[:,:,z].T.copy()))
            ids = range(len(protocol['classes'])) if len(protocol['classes']) == 3 else (1,)
            self.supports[case] = {c: VolumeSupport.build(labels == c) for c in ids}
            self.sdf[case] = torch.from_numpy(np.stack([
                signed_distance(labels.numpy() == c, (1.,1.,1.), 8.)/8.
                for c in range(1, len(protocol['classes']))])).float() if 'sdf' in protocol['arm'] else torch.zeros((len(protocol['classes'])-1,*labels.shape))
            affine = np.eye(4)
            # Native sampled planes are Z,Y,X. Depth uses the source X axis.
            affine[:3,:3] = np.eye(3)[::-1]
            if 'depth' in protocol['arm']:
                geometry = build_geometry(labels.numpy(), affine, GeometryConfig(), fields=False)
                self.depth[case] = torch.from_numpy(geometry['weights']).float()
            else:
                self.depth[case] = torch.zeros(labels.shape)
            self.inventory.append(dict(case=case, shape=list(label.shape),
                image_array_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                label_array_sha256=hashlib.sha256(label.tobytes()).hexdigest(),
                class_voxels={name:int((label==i).sum()) for i,name in enumerate(protocol['classes'])}))

    def calibration_schedule(self, count):
        order = []
        for case, supports in self.supports.items():
            active = torch.stack([s.inner | s.outer for s in supports.values()]).any(0)
            anchors = active.flatten(1).any(1).nonzero().flatten().tolist()
            if len(anchors) < count:
                raise ValueError('Insufficient calibration anchors')
            order.extend((case, anchors[i]) for i in np.linspace(0,len(anchors)-1,count).round().astype(int))
        return order


def forward_plane(model, processor, pixels, protocol, labels=None, criterion=None):
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    fields, losses = [], []
    for class_id in range(1, len(protocol['classes'])):
        mask = np.zeros(pixels.shape, dtype=bool) if labels is None else labels == class_id
        view = SimpleNamespace(samples=[(pixels,mask,'',0)], transform=processor.transform, resolution=processor.resolution)
        point = TrainingSlices.__getitem__(view, 0)
        point.find_queries[0].query_text = protocol['prompts'][str(class_id)]
        batch = collate_fn_api([point], dict_key='input', with_seg_masks=True)['input']
        for target in batch.find_targets:
            if target.segments.numel() == 0:
                target.segments = torch.empty((0,processor.resolution,processor.resolution),dtype=torch.bool)
        batch = move_to_device(batch, 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch)
            if criterion is not None:
                losses.append(criterion(outputs,[model.back_convert(t) for t in batch.find_targets])[CORE_LOSS_KEY])
        fields.append(semantic_log_fields(outputs.output[-1][-1], pixels.shape))
    if len(fields) == 1:
        logp = torch.stack((fields[0][1], fields[0][0]))
    else:
        logp = torch.stack([torch.zeros_like(fields[0][0])] + [p-q for p,q in fields]).log_softmax(0)
    return logp, torch.stack(losses).mean() if losses else None


def forward_slab(model, criterion, slabs, case, anchor):
    indices = slabs.indices[case]
    planes, natives = [], []
    for z in (anchor, (anchor+1)%len(indices)):
        pixels, labels = slabs.samples[indices[z]]
        lp, native = forward_plane(model,slabs.processor,pixels,slabs.protocol,labels,criterion)
        planes.append(lp); natives.append(native)
    band, edge = [], []
    for c, support in slabs.supports[case].items():
        others = [i for i in range(len(planes[0])) if i != c]
        b,e = sampled_losses(planes[0][c].exp(),planes[1][c].exp(),support,anchor,
            log_probability=planes[0][c],log_complement=torch.logsumexp(planes[0][others],0))
        band.append(b); edge.append(e)
    truth = torch.as_tensor(slabs.samples[indices[anchor]][1].copy(),device='cuda').long()
    foreground = torch.stack([(truth==c) for c in range(1,len(planes[0]))]).float()
    sdf = ((planes[0][1:].exp()-foreground)*slabs.sdf[case][:,anchor].to('cuda')).mean()
    # Uniform slice sampling estimates the full-volume normalized depth CE.
    depth = -(planes[0].gather(0,truth[None])[0]*slabs.depth[case][anchor].to('cuda')).sum()*len(indices)
    return dict(native=torch.stack(natives).mean(),bands=torch.stack(band).mean(),
                edge=torch.stack(edge).mean(),sdf=sdf,depth=depth)


def load_foundation(protocol, base_weights):
    if digest(base_weights) != protocol['base_sha256']:
        raise ValueError('Base checkpoint differs')
    if protocol['dataset'] != 'GeoFWI3D':
        from evaluation.medsam3_zero_shot import load_model
        if digest(protocol['lora_weights']) != protocol['lora_sha256']:
            raise ValueError('Medical adapter differs')
        return load_model(SimpleNamespace(upstream=Path(protocol['upstream']), base_weights=Path(base_weights),
                         lora_weights=Path(protocol['lora_weights']),device='cuda'))
    import sys
    sys.path.insert(0, protocol['upstream'])
    import yaml
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from lora_layers import LoRAConfig, apply_lora_to_model
    from evaluation.medsam3_zero_shot import load_detector_state
    cfg = yaml.safe_load((Path(protocol['upstream'])/'configs/full_lora_config.yaml').read_text())['lora']
    cfg['dropout'] = 0.
    model = build_sam3_image_model(load_from_HF=False, bpe_path=str(Path(protocol['upstream'])/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'), device='cpu',compile=False,eval_mode=True)
    load_detector_state(model,torch.load(base_weights,map_location='cpu',weights_only=True))
    model = apply_lora_to_model(model,LoRAConfig(**cfg))
    # Start from SAM3, without loading the medical adapter.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith('.lora_B'):
                value.zero_()
    model.to('cuda').requires_grad_(False).eval()
    return model, Sam3Processor(model,device='cuda',confidence_threshold=.5)


def calibrate(model, criterion, parameters, slabs, protocol, arm, output):
    names = ['native'] + [k for k in ('bands','edge','sdf','depth') if k in arm]
    if len(names) == 1:
        return dict(weights={}, status='not_required')
    reset_seed(protocol['calibration_seed'])
    rows = []
    path = output/'calibration_partial.json'
    if path.exists():
        rows = read(path)['rows']
    for index,(case,anchor) in enumerate(slabs.calibration_schedule(protocol['calibration_slabs_per_volume'])):
        if index < len(rows):
            continue
        if stopping():
            raise SystemExit(75)
        reset_seed(protocol['calibration_seed']+index)
        losses = forward_slab(model,criterion,slabs,case,anchor)
        gradients = [torch.autograd.grad(losses[name],parameters,retain_graph=i<len(names)-1,allow_unused=True)
                     for i,name in enumerate(names)]
        gram = [[sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(ga,gb)
                     if a is not None and b is not None) for gb in gradients] for ga in gradients]
        if not np.isfinite(gram).all() or gram[0][0] <= 0:
            raise ValueError('Invalid calibration gradient')
        rows.append(dict(case=case,anchor=anchor,gram=gram,losses={k:float(losses[k].detach()) for k in names}))
        write(path,dict(names=names,rows=rows))
        del gradients, losses
    coefficients = [1.]
    statistics, weights = {}, {}
    for i,name in enumerate(names[1:],1):
        ratios = []
        for row in rows:
            gram = np.asarray(row['gram'])
            norm = np.asarray(coefficients) @ gram[:i,:i] @ np.asarray(coefficients)
            if norm > 0 and gram[i,i] > 0:
                ratios.append(np.sqrt(gram[i,i]/norm))
        if len(ratios)<5:
            raise ValueError('Too few nonzero calibration gradients')
        target = protocol.get('gradient_ratio_targets', {}).get(name, protocol['gradient_ratio_target'])
        cap = protocol.get('gradient_ratio_p95_caps', {}).get(name, protocol['gradient_ratio_p95_cap'])
        if not (0 < target <= cap):
            raise ValueError(f'Invalid gradient calibration target/cap for {name}')
        weight = min(target/np.median(ratios),cap/np.quantile(ratios,.95))
        weights[name]=float(weight); coefficients.append(weight)
        statistics[name]=dict(valid=len(ratios),target=target,p95_cap=cap,
                              weighted_median=float(np.median(ratios)*weight),
                              weighted_p95=float(np.quantile(ratios,.95)*weight))
    return dict(status='complete',weights=weights,statistics=statistics,rows=rows,names=names,
                scope='Training support only; sequential LoRA gradient norm calibration',
                target=protocol['gradient_ratio_target'],p95_cap=protocol['gradient_ratio_p95_cap'])


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def train(model, criterion, parameters, slabs, protocol, calibration, arm, output):
    schedule = slabs.schedule(protocol)
    write(output/'slab_schedule.json',schedule)
    reset_seed(protocol['seed'])
    model.train()
    optimizer = torch.optim.AdamW(parameters,lr=protocol['learning_rate'],weight_decay=.01)
    frozen_before, adapter_before = parameter_digest(model,False),parameter_digest(model,True)
    start, history = 0, []
    latest = output/'checkpoint_latest.pt'
    elapsed = 0.
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['protocol_sha256'] != digest(output/'protocol.json'):
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(state['adapter'],strict=False)
        optimizer.load_state_dict(state['optimizer'])
        start,history,elapsed = state['step'],state['history'],state['elapsed_seconds']
        if state['frozen_before'] != frozen_before or state['adapter_before'] != adapter_before:
            raise ValueError('Initial model changed')
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    began = time.monotonic()
    def commit(step):
        save_checkpoint(latest,dict(step=step,history=history,
            adapter={n:p.detach().cpu() for n,p in model.state_dict().items() if is_adapter(n)},
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),protocol_sha256=digest(output/'protocol.json'),
            frozen_before=frozen_before,adapter_before=adapter_before,elapsed_seconds=elapsed+time.monotonic()-began))
        write(output/'training_history.json',history)
    for update in range(start,protocol['updates']):
        if stopping():
            commit(update)
            raise SystemExit(75)
        horizon=protocol.get('learning_rate_horizon_updates',protocol['updates'])
        if horizon < protocol['updates']:
            raise ValueError('Learning-rate horizon cannot precede the stopping update')
        lr=lr_for_update(update,horizon,protocol['warmup_updates'],protocol['learning_rate'],protocol['final_learning_rate'])
        for group in optimizer.param_groups:
            group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        ramp=min((update+1)/protocol['constraint_warmup_updates'],1.)
        rows=[]
        for micro in range(protocol['accumulation']):
            case,anchor=schedule[update*protocol['accumulation']+micro]
            losses=forward_slab(model,criterion,slabs,case,anchor)
            total=losses['native']+ramp*sum(calibration['weights'][name]*losses[name] for name in calibration['weights'])
            if not torch.isfinite(total):
                raise ValueError('Nonfinite fine-tuning loss')
            (total/protocol['accumulation']).backward()
            rows.append({**{k:float(v.detach()) for k,v in losses.items()},'total':float(total.detach())})
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(update=update+1,lr=lr,ramp=ramp,gradient_norm=float(norm),
                            **{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}))
        if (update+1)%5==0 or update==0:
            commit(update+1)
            print(json.dumps(history[-1]),flush=True)
    commit(protocol['updates'])
    model.eval().requires_grad_(False)
    frozen_after,adapter_after=parameter_digest(model,False),parameter_digest(model,True)
    if frozen_before!=frozen_after or adapter_before==adapter_after:
        raise ValueError('Frozen base changed or adapter did not learn')
    write(output/'training_summary.json',dict(updates=protocol['updates'],frozen_before=frozen_before,
        frozen_after=frozen_after,adapter_before=adapter_before,adapter_after=adapter_after,
        schedule_sha256=digest(output/'slab_schedule.json'),elapsed_seconds=elapsed+time.monotonic()-began,
        augmentation='none',weights=calibration['weights'],
        learning_rate_horizon_updates=protocol.get('learning_rate_horizon_updates',protocol['updates'])))


def evaluate(model, processor, volumes, protocol, output):
    path=output/'evaluation.json'
    records=read(path)['records'] if path.exists() else []
    for split,cases in [('training_support',protocol['train_cases']),('validation',protocol['evaluation_cases'])]:
        for case in cases:
            if any(r['case_name']==case and r['split']==split for r in records):
                continue
            if stopping():
                raise SystemExit(75)
            image,_=volumes.load(case,labels=False)
            prediction=np.empty(image.shape,dtype=np.uint8)
            with torch.inference_mode():
                for z in range(image.shape[2]):
                    pixels=np.round(image[:,:,z].T*255).astype(np.uint8)
                    logp,_=forward_plane(model,processor,pixels,protocol)
                    probabilities=logp.exp()
                    if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(0),torch.ones_like(probabilities[0]),atol=1e-5):
                        raise ValueError('Invalid semantic probabilities')
                    prediction[:,:,z]=probabilities.argmax(0).cpu().numpy().T
            _,truth=volumes.load(case)
            n=len(protocol['classes'])
            matrix=np.bincount((truth.astype(np.int64)*n+prediction).ravel(),minlength=n*n).reshape(n,n).tolist()
            records.append(dict(case_name=case,split=split,**confusion_metrics(matrix,protocol['classes']),
                prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),shape=list(truth.shape)))
            write(path,dict(records=records))
            print(json.dumps(dict(evaluation_case=case,split=split,dice=records[-1]['macro_hard_dice'])),flush=True)
    return dict(records=records,mean_dice={s:float(np.mean([r['macro_hard_dice'] for r in records if r['split']==s]))
                for s in ('training_support','validation')})


def main():
    parser=argparse.ArgumentParser()
    for name in ('protocol','output','base-weights'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',required=True)
    args=parser.parse_args()
    protocol=read(args.protocol);check_no_augmentation(protocol);install_signals()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'protocol.json'
    actual={**protocol,'arm':args.arm}
    if path.exists() and read(path)!=actual:
        raise ValueError('Resume protocol changed')
    write(path,actual)
    torch.set_num_threads(4);reset_seed(protocol['seed'])
    model,processor=load_foundation(protocol,args.base_weights)
    for name,p in model.named_parameters():
        p.requires_grad_(is_adapter(name))
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=18497664:
        raise ValueError('Unexpected LoRA architecture')
    criterion=native_loss(model);volumes=Volumes(actual);slabs=SupportSlabs(volumes,actual,processor)
    write(args.output/'training_inventory.json',slabs.inventory)
    model.train()
    calibration_path=args.output/'calibration.json'
    calibration=read(calibration_path) if calibration_path.exists() else calibrate(model,criterion,parameters,slabs,protocol,args.arm,args.output)
    write(calibration_path,calibration)
    train(model,criterion,parameters,slabs,protocol,calibration,args.arm,args.output)
    result=evaluate(model,processor,volumes,protocol,args.output)
    write(args.output/'evaluation.json',result)
    write(args.output/'completion.json',dict(status='complete',dataset=protocol['dataset'],arm=args.arm,
        fold=protocol['fold'],seed=protocol['seed'],updates=protocol['updates'],
        validation_dice=result['mean_dice']['validation'],training_dice=result['mean_dice']['training_support'],
        primary_checkpoint='final_update',checkpoint_sha256=digest(args.output/'checkpoint_latest.pt'),
        evaluation_sha256=digest(args.output/'evaluation.json'),protocol_sha256=digest(path)))


if __name__=='__main__':
    main()
