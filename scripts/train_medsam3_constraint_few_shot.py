"""Matched five-volume MedSAM3 LoRA ablation with supervised training losses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.medsam3_zero_shot import load_model, infer_volume, save_json, sha256
from scripts.train_medsam3_few_shot import (TrainingSlices, validate_protocol, load_image,
    load_label, move_to_device, native_loss, is_adapter, parameter_digest,
    lr_for_update, save_predictions)
from scripts.medsam3_supervised_constraints import VolumeSupport, sampled_losses, semantic_log_fields
from scripts.medsam3_calibration_cache import calibration_context, load_calibration


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Slabs:
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices, self.supports = {}, {}
        for case in (row['case'] for row in dataset.inventory):
            indices = [i for i, sample in enumerate(dataset.samples) if sample[2] == case]
            assert [dataset.samples[i][3] for i in indices] == list(range(len(indices)))
            self.indices[case] = indices
            foreground = torch.from_numpy(np.stack([dataset.samples[i][1] for i in indices]))
            self.supports[case] = VolumeSupport.build(foreground)

    def schedule(self, protocol):
        rng = random.Random(protocol['seed'])
        cases = list(self.indices)
        order = []
        for _ in range(protocol['updates']*protocol['accumulation']):
            case = rng.choice(cases)
            order.append((case, rng.randrange(len(self.indices[case]))))
        return order

    def calibration_schedule(self, count):
        order = []
        for case, support in self.supports.items():
            active = (support.inner | support.outer).flatten(1).any(1).nonzero().flatten().tolist()
            if len(active) < count:
                raise ValueError('Insufficient labelled support for gradient calibration.')
            for index in np.linspace(0, len(active)-1, count).round().astype(int):
                order.append((case, active[index]))
        return order


def forward_slab(model, criterion, slabs, case, anchor):
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    dataset = slabs.dataset
    indices = slabs.indices[case]
    log_probabilities, native = [], []
    for z in (anchor, (anchor+1)%len(indices)):
        index = indices[z]
        batch = collate_fn_api([dataset[index]], dict_key='input', with_seg_masks=True)['input']
        for target in batch.find_targets:
            if target.segments.numel() == 0:
                target.segments = torch.empty((0, dataset.resolution, dataset.resolution), dtype=torch.bool)
        batch = move_to_device(batch, 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch)
            targets = [model.back_convert(t) for t in batch.find_targets]
            native.append(criterion(outputs, targets)[CORE_LOSS_KEY])
        log_probabilities.append(semantic_log_fields(outputs.output[-1][-1], dataset.samples[index][1].shape))
    band, edge = sampled_losses(*(p[0].exp() for p in log_probabilities), slabs.supports[case], anchor,
                                log_probability=log_probabilities[0][0],
                                log_complement=log_probabilities[0][1])
    return torch.stack(native).mean(), band, edge


def squared_norm(grads):
    return sum(float(g.detach().double().square().sum()) for g in grads if g is not None)


def calibrate(model, criterion, parameters, slabs, protocol, output):
    """Freeze coefficients using support-only LoRA gradient magnitudes."""
    reset_seed(protocol.get('calibration_seed', protocol['seed']))
    rows = []
    for case, anchor in slabs.calibration_schedule(protocol['calibration_slabs_per_volume']):
        losses = forward_slab(model, criterion, slabs, case, anchor)
        if not all(torch.isfinite(loss) for loss in losses):
            raise RuntimeError('Nonfinite calibration loss.')
        grads = [torch.autograd.grad(loss, parameters, retain_graph=i < 2, allow_unused=True)
                 for i, loss in enumerate(losses)]
        norms = [squared_norm(g) for g in grads]
        dot = sum(float((a.detach().double()*b.detach().double()).sum())
                  for a, b in zip(grads[0], grads[1]) if a is not None and b is not None)
        if not np.isfinite(norms+[dot]).all() or norms[0] <= 0:
            raise RuntimeError('Invalid native calibration gradient.')
        row = dict(case=case, anchor=anchor, native_sq=norms[0], bands_sq=norms[1],
                   edge_sq=norms[2], native_bands_dot=dot,
                   losses=[float(loss.detach()) for loss in losses])
        rows.append(row)
        del grads, losses
        print(json.dumps(dict(calibration=len(rows), **row)), flush=True)
    def coefficient(ratios):
        ratios = np.asarray(ratios)
        if len(ratios) < 5 or not np.isfinite(ratios).all() or (ratios <= 0).any():
            raise RuntimeError('Insufficient nonzero finite constraint gradients.')
        weight = min(protocol['gradient_ratio_target']/np.median(ratios),
                     protocol['gradient_ratio_p95_cap']/np.quantile(ratios, .95))
        return dict(weight=float(weight), valid_slabs=len(ratios),
                    raw_quantiles=np.quantile(ratios, [0, .5, .95, 1]).tolist(),
                    weighted_quantiles=np.quantile(weight*ratios, [0, .5, .95, 1]).tolist())
    bands = coefficient([np.sqrt(r['bands_sq']/r['native_sq']) for r in rows if r['bands_sq'] > 0])
    weight = bands['weight']
    ratios = []
    for row in rows:
        combined = row['native_sq']+weight**2*row['bands_sq']+2*weight*row['native_bands_dot']
        if not np.isfinite(combined) or combined <= 0:
            raise RuntimeError('Invalid native-plus-bands gradient norm.')
        if row['edge_sq'] > 0:
            ratios.append(np.sqrt(row['edge_sq']/combined))
    edge = coefficient(ratios)
    report = dict(bands=bands, edge=edge, slabs=rows, evaluation_labels_used=False,
                  policy='median ratio target with p95 cap; edge relative to native+bands')
    save_json(output/'calibration.json', report)
    return report


def train_arm(model, criterion, parameters, slabs, protocol, calibration, arm, output, schedule):
    reset_seed(protocol['seed'])
    model.train()
    frozen_before, adapter_before = parameter_digest(model, False), parameter_digest(model, True)
    optimizer = torch.optim.AdamW(parameters, lr=protocol['learning_rate'], weight_decay=.01)
    wb = calibration['bands']['weight'] if arm in ('bands', 'bands_edge') else 0.
    we = calibration['edge']['weight'] if arm == 'bands_edge' else 0.
    output.mkdir()
    exposures = {case: 0 for case in slabs.indices}
    start = time.monotonic()
    with (output/'training.jsonl').open('x') as log:
        for update in range(protocol['updates']):
            lr = lr_for_update(update, protocol['updates'], protocol['warmup_updates'],
                               protocol['learning_rate'], protocol['final_learning_rate'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            ramp = min((update+1)/protocol['constraint_warmup_updates'], 1.)
            optimizer.zero_grad(set_to_none=True)
            values = []
            for micro in range(protocol['accumulation']):
                case, anchor = schedule[update*protocol['accumulation']+micro]
                exposures[case] += 2
                native, bands, edge = forward_slab(model, criterion, slabs, case, anchor)
                total = native + ramp*(wb*bands + we*edge)
                if not all(torch.isfinite(v) for v in (native, bands, edge, total)):
                    raise RuntimeError('Nonfinite training objective.')
                (total/protocol['accumulation']).backward()
                values.append([float(v.detach()) for v in (native, bands, edge, total)])
                del native, bands, edge, total
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            means = np.mean(values, axis=0)
            row = dict(arm=arm, update=update+1, native=means[0], bands=means[1], edge=means[2],
                       total=means[3], bands_weight=ramp*wb, edge_weight=ramp*we, lr=lr,
                       gradient_norm=float(norm), elapsed_seconds=time.monotonic()-start)
            log.write(json.dumps(row)+'\n')
            log.flush()
            if update == 0 or (update+1)%10 == 0:
                print(json.dumps(row), flush=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval().requires_grad_(False)
    frozen_after, adapter_after = parameter_digest(model, False), parameter_digest(model, True)
    if frozen_before != frozen_after or adapter_before == adapter_after:
        raise RuntimeError('Frozen parameters changed or adapter failed to update.')
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items() if is_adapter(k)}
    path = output/'final_lora_weights.pt'
    torch.save(weights, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)
    report = dict(arm=arm, updates=protocol['updates'], slice_exposures=exposures,
                  frozen_before=frozen_before, frozen_after=frozen_after,
                  adapter_before=adapter_before, adapter_after=adapter_after,
                  final_adapter_sha256=sha256(path), bands_weight=wb, edge_weight=we,
                  trainable_parameters=sum(p.numel() for p in parameters),
                  schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest())
    save_json(output/'training_summary.json', report)
    torch.cuda.empty_cache()
    return report


def score(data, protocol, output):
    cases, arms = protocol['evaluation_cases'], protocol['arms']
    for case in cases:
        for arm in arms:
            for suffix in ('probability', 'mask'):
                if not (output/case/f'{arm}_{suffix}.nii.gz').is_file():
                    raise RuntimeError('All arms must finish predictions before references are opened.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores = {}
    for case in cases:
        source, image = load_image(data, case)
        truth = load_label(data, case, source)
        scores[case], masks = {}, {}
        for arm in arms:
            prediction = nib.load(output/case/f'{arm}_mask.nii.gz')
            if prediction.shape != source.shape or not np.allclose(prediction.affine, source.affine):
                raise RuntimeError('Prediction and reference grids differ.')
            hard = prediction.get_fdata() > .5
            masks[arm] = hard
            tp, fp, fn = int((hard & truth).sum()), int((hard & ~truth).sum()), int((~hard & truth).sum())
            scores[case][arm] = dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., tp=tp, fp=fp, fn=fn)
        if case not in protocol.get('plot_cases', cases):
            continue
        z = image.shape[2]//2
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, (name, mask) in zip(axes, [('Reference', truth)]+list(masks.items())):
            ax.imshow(image[:, :, z].T, cmap='gray', origin='lower')
            if mask[:, :, z].any():
                ax.contour(mask[:, :, z].T, levels=[.5], colors=['lime'], linewidths=.8)
            ax.set_title(name)
            ax.axis('off')
        fig.suptitle(f'{case}: canonical slice {z}')
        fig.tight_layout()
        fig.savefig(output/case/'comparison.png', dpi=150)
        plt.close(fig)
    return dict(per_case=scores, mean_dice={arm: float(np.mean([scores[c][arm]['dice'] for c in cases])) for arm in arms})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('upstream', 'base-weights', 'lora-weights', 'data', 'protocol', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--calibration-input', type=Path)
    parser.add_argument('--calibration-sha256')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    split = args.data/'splits_final.json'
    if sha256(split) != protocol['split_sha256']:
        raise ValueError('Dataset split changed.')
    folds = json.loads(split.read_text())
    fold_index = protocol.get('fold_index', 0)
    if not isinstance(fold_index, int) or not 0 <= fold_index < len(folds):
        raise ValueError('Invalid fold index.')
    validate_protocol(protocol, folds[fold_index])
    if protocol['arms'] != ['baseline', 'bands', 'bands_edge']:
        raise ValueError('Unexpected ablation arms.')
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'protocol.json', protocol)
    torch.set_num_threads(4)
    reset_seed(protocol['seed'])
    model, processor = load_model(SimpleNamespace(upstream=args.upstream, base_weights=args.base_weights,
                                                 lora_weights=args.lora_weights, device='cuda'))
    if model.supervise_joint_box_scores:
        raise RuntimeError('Projection requires separate query and presence scores.')
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_adapter(n)}
    for name, p in model.named_parameters():
        p.requires_grad_(is_adapter(name))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if len(initial) != 916 or sum(p.numel() for p in parameters) != 18497664:
        raise RuntimeError('Unexpected LoRA parameter set.')
    model.train()
    criterion = native_loss(model)
    dataset = TrainingSlices(args.data, protocol['train_cases'], processor)
    slabs = Slabs(dataset)
    schedule = slabs.schedule(protocol)
    save_json(args.output/'training_inventory.json', dataset.inventory)
    save_json(args.output/'slab_schedule.json', schedule)
    shared_source = protocol.get('calibration_source_run')
    reuse = bool(shared_source and shared_source != protocol['study_run'])
    if reuse != bool(args.calibration_input) or bool(args.calibration_input) != bool(args.calibration_sha256):
        raise ValueError('Shared calibration input must follow the frozen source-run policy.')
    context = calibration_context(protocol, dataset.inventory, parameter_digest(model, True),
                                  parameter_digest(model, False)) if shared_source else None
    if reuse:
        calibration = load_calibration(args.calibration_input, args.calibration_sha256, context, shared_source)
        save_json(args.output/'calibration.json', calibration)
    else:
        calibration = calibrate(model, criterion, parameters, slabs, protocol, args.output)
        if shared_source:
            calibration.update(context=context, source_run=shared_source)
            save_json(args.output/'calibration.json', calibration)
    training = {}
    for arm in protocol['arms']:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(is_adapter(name))
                if name in initial:
                    parameter.copy_(initial[name])
        training[arm] = train_arm(model, criterion, parameters, slabs, protocol, calibration,
                                  arm, args.output/arm, schedule)
        for case in protocol['evaluation_cases']:
            source, image = load_image(args.data, case)
            prediction, slices = infer_volume(image, processor, 'hippocampus')
            save_predictions(args.output, case, source, {arm: prediction})
            save_json(args.output/case/f'{arm}_slices.json', slices)
    if len({r['adapter_before'] for r in training.values()}) != 1:
        raise RuntimeError('Arms did not start from the same adapter.')
    scores = score(args.data, protocol, args.output)
    report = dict(status='complete', metric='3D whole-hippocampus union Dice', **scores,
                  training=training, calibration=calibration, protocol=protocol,
                  base_loading=model.medsam3_base_report,
                  output_hashes={str(f.relative_to(args.output)): sha256(f) for f in args.output.glob('*/*.nii.gz')},
                  evaluation_label_hashes={c: sha256(args.data/'labelsTr'/f'{c}.nii.gz') for c in protocol['evaluation_cases']},
                  limitations=['Existing dataset volume splits; patient identities unavailable.',
                               'One seed per run and a fixed 200-update budget.',
                               'SAM3 instance confidences projected to a semantic union for the original 3D spatial objectives.'])
    save_json(args.output/'results.json', report)
    print(json.dumps(dict(status='complete', mean_dice=report['mean_dice'])), flush=True)


if __name__ == '__main__':
    main()
