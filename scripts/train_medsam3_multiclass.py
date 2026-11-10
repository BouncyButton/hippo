"""Matched MedSAM3 LoRA baseline/bands/bands+edge with explicit BG/A/P labels."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.medsam3_zero_shot import load_model, save_json, sha256
from scripts.train_medsam3_few_shot import (TrainingSlices, load_image, move_to_device,
    native_loss, is_adapter, parameter_digest, validate_protocol)
from scripts.train_medsam3_constraint_few_shot import reset_seed, Slabs, calibrate, train_arm
from scripts.medsam3_supervised_constraints import VolumeSupport, sampled_losses, semantic_log_fields
from scripts.medsam3_calibration_cache import calibration_context, load_calibration

PROMPTS = {1: 'anterior hippocampus', 2: 'posterior hippocampus'}
CLASSES = ['background', 'anterior', 'posterior']


def exclusive_log_probabilities(fields):
    """Normalize two prompt odds against a background logit of zero."""
    logits = torch.stack([torch.zeros_like(fields[0][0])] + [p-q for p, q in fields])
    return torch.log_softmax(logits, dim=0)


def read_labels(data, case, source):
    label = nib.as_closest_canonical(nib.load(data/'labelsTr'/f'{case}.nii.gz'))
    if label.shape != source.shape or not np.allclose(label.affine, source.affine, atol=1e-4):
        raise ValueError(f'Image/label grid mismatch: {case}')
    values = label.get_fdata()
    if not np.isin(values, [0, 1, 2]).all():
        raise ValueError(f'Invalid BG/A/P labels: {case}')
    return values.astype(np.uint8)


def datapoint(pixels, labels, class_id, processor):
    # Reuse native SAM3 target construction, retaining class identity in the query.
    mask = np.zeros(pixels.shape, dtype=bool) if labels is None else labels == class_id
    view = SimpleNamespace(samples=[(pixels, mask, '', 0)], transform=processor.transform,
                           resolution=processor.resolution)
    point = TrainingSlices.__getitem__(view, 0)
    point.find_queries[0].query_text = PROMPTS[class_id]
    return point


class MulticlassSlabs(Slabs):
    def __init__(self, data, cases, processor):
        self.processor = processor
        self.samples, self.inventory = [], []
        self.indices, self.supports = {}, {}
        for case in cases:
            source, image = load_image(data, case)
            labels = read_labels(data, case, source)
            if set(np.unique(labels)) != {0, 1, 2}:
                raise ValueError(f'Training volume lacks a required class: {case}')
            self.indices[case] = []
            for z in range(image.shape[2]):
                self.indices[case].append(len(self.samples))
                self.samples.append((np.round(image[:, :, z].T*255).astype(np.uint8),
                                     labels[:, :, z].T.copy()))
            zyxt = torch.from_numpy(labels.transpose(2, 1, 0).copy())
            self.supports[case] = [VolumeSupport.build(zyxt == c) for c in range(3)]
            self.inventory.append(dict(case=case, shape=list(labels.shape),
                class_voxels={name: int((labels == c).sum()) for c, name in enumerate(CLASSES)},
                image_sha256=sha256(data/'imagesTr'/f'{case}_0000.nii.gz'),
                label_sha256=sha256(data/'labelsTr'/f'{case}.nii.gz')))

    def calibration_schedule(self, count):
        order = []
        for case, supports in self.supports.items():
            active = torch.stack([s.inner | s.outer for s in supports]).any(0)
            anchors = active.flatten(1).any(1).nonzero().flatten().tolist()
            if len(anchors) < count:
                raise ValueError('Insufficient calibration anchors')
            order.extend((case, anchors[i]) for i in np.linspace(0, len(anchors)-1, count).round().astype(int))
        return order


def forward_plane(model, processor, pixels, labels=None, criterion=None):
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    fields, losses = [], []
    for class_id in (1, 2):
        point = datapoint(pixels, labels, class_id, processor)
        batch = collate_fn_api([point], dict_key='input', with_seg_masks=True)['input']
        for target in batch.find_targets:
            if target.segments.numel() == 0:
                target.segments = torch.empty((0, processor.resolution, processor.resolution), dtype=torch.bool)
        batch = move_to_device(batch, 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch)
            if criterion is not None:
                losses.append(criterion(outputs, [model.back_convert(t) for t in batch.find_targets])[CORE_LOSS_KEY])
        fields.append(semantic_log_fields(outputs.output[-1][-1], pixels.shape))
    log_p = exclusive_log_probabilities(fields)
    return log_p, torch.stack(losses).mean() if losses else None


def multiclass_losses(log_p, next_log_p, supports, anchor):
    bands, edges = [], []
    for c, support in enumerate(supports):
        others = [i for i in range(3) if i != c]
        band, edge = sampled_losses(log_p[c].exp(), next_log_p[c].exp(), support, anchor,
            log_probability=log_p[c], log_complement=torch.logsumexp(log_p[others], dim=0))
        bands.append(band)
        edges.append(edge)
    return torch.stack(bands).mean(), torch.stack(edges).mean()


def forward_slab(model, criterion, slabs, case, anchor):
    indices = slabs.indices[case]
    planes, native = [], []
    for z in (anchor, (anchor+1) % len(indices)):
        pixels, labels = slabs.samples[indices[z]]
        log_p, loss = forward_plane(model, slabs.processor, pixels, labels, criterion)
        planes.append(log_p)
        native.append(loss)
    bands, edge = multiclass_losses(*planes, slabs.supports[case], anchor)
    return torch.stack(native).mean(), bands, edge


def metrics(truth, prediction):
    if truth.shape != prediction.shape or not np.isin(truth, [0, 1, 2]).all() or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError('Invalid multiclass segmentation')
    cm = np.bincount((truth.astype(np.int64)*3 + prediction.astype(np.int64)).ravel(), minlength=9).reshape(3, 3)
    per_class = {}
    for c, name in enumerate(CLASSES):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum()-tp); fn = int(cm[c, :].sum()-tp)
        per_class[name] = dict(tp=tp, fp=fp, fn=fn, dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
    return dict(confusion_true_rows_predicted_columns=cm.tolist(), per_class=per_class,
                ap_mean_dice=(per_class['anterior']['dice']+per_class['posterior']['dice'])/2,
                anterior_to_posterior=int(cm[1, 2]), posterior_to_anterior=int(cm[2, 1]),
                total_error_voxels=int(cm.sum()-np.trace(cm)))


def predict(model, processor, data, case, output, arm):
    source, image = load_image(data, case)  # No labels loaded during inference.
    probabilities = np.empty((*image.shape, 3), dtype=np.float32)
    with torch.inference_mode():
        for z in range(image.shape[2]):
            pixels = np.round(image[:, :, z].T*255).astype(np.uint8)
            log_p, _ = forward_plane(model, processor, pixels)
            probabilities[:, :, z, :] = log_p.exp().cpu().numpy().transpose(2, 1, 0)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1, atol=1e-5):
        raise RuntimeError('Invalid class probabilities')
    mask = probabilities.argmax(-1).astype(np.uint8)
    folder = output/case
    folder.mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(probabilities, source.affine), folder/f'{arm}_probabilities.nii.gz')
    nib.save(nib.Nifti1Image(mask, source.affine), folder/f'{arm}_mask.nii.gz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('upstream', 'base-weights', 'lora-weights', 'data', 'protocol', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--calibration-input', type=Path)
    parser.add_argument('--calibration-sha256')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if sha256(args.data/'splits_final.json') != protocol['split_sha256']:
        raise ValueError('Split hash mismatch')
    folds = json.loads((args.data/'splits_final.json').read_text())
    fold = protocol['fold_index']
    if not isinstance(fold, int) or not 0 <= fold < len(folds):
        raise ValueError('Invalid fold index')
    validate_protocol(protocol, folds[fold])
    if not isinstance(protocol['seed'], int) or protocol['seed'] < 0:
        raise ValueError('Invalid optimization seed')
    assert protocol['classes'] == CLASSES and protocol['arms'] == ['baseline', 'bands', 'bands_edge']
    assert sha256(args.base_weights) == protocol['base_sha256']
    for relative, digest in protocol['data_hashes'].items():
        if sha256(args.data/relative) != digest:
            raise ValueError(f'Data hash mismatch: {relative}')
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'protocol.json', protocol)
    torch.set_num_threads(4)
    reset_seed(protocol['seed'])
    model, processor = load_model(SimpleNamespace(upstream=args.upstream, base_weights=args.base_weights,
                                                 lora_weights=args.lora_weights, device='cuda'))
    assert not model.supervise_joint_box_scores
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_adapter(n)}
    for n, p in model.named_parameters():
        p.requires_grad_(is_adapter(n))
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert len(initial) == 916 and sum(p.numel() for p in parameters) == 18497664
    criterion = native_loss(model)
    slabs = MulticlassSlabs(args.data, protocol['train_cases'], processor)
    schedule = slabs.schedule(protocol)
    save_json(args.output/'training_inventory.json', slabs.inventory)
    save_json(args.output/'slab_schedule.json', schedule)
    # Exercise label-free eval before spending the training budget.
    model.eval()
    with torch.inference_mode():
        probe, _ = forward_plane(model, processor, slabs.samples[0][0])
        assert probe.shape[0] == 3 and torch.isfinite(probe).all()
    model.train()
    source_run = protocol.get('calibration_source_run')
    reuse = bool(source_run and source_run != protocol['study_run'])
    if reuse != bool(args.calibration_input) or bool(args.calibration_input) != bool(args.calibration_sha256):
        raise ValueError('Calibration reuse must match the frozen protocol')
    context = calibration_context(protocol, slabs.inventory, parameter_digest(model, True),
                                  parameter_digest(model, False))
    context['multiclass'] = {key: protocol[key] for key in
                           ('classes', 'class_prompts', 'semantic_projection', 'constraints')}
    if reuse:
        calibration = load_calibration(args.calibration_input, args.calibration_sha256, context, source_run)
    else:
        calibration = calibrate(model, criterion, parameters, slabs, protocol, args.output, forward_fn=forward_slab)
        if source_run:
            calibration.update(source_run=source_run, context=context)
    save_json(args.output/'calibration.json', calibration)
    training = {}
    for arm in protocol['arms']:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.requires_grad_(is_adapter(n))
                if n in initial:
                    p.copy_(initial[n])
        training[arm] = train_arm(model, criterion, parameters, slabs, protocol, calibration,
                                 arm, args.output/arm, schedule, forward_fn=forward_slab)
        for split, cases in [('train', protocol['train_cases']), ('validation', protocol['evaluation_cases'])]:
            for case in cases:
                print(json.dumps(dict(stage='inference', arm=arm, split=split, case=case)), flush=True)
                predict(model, processor, args.data, case, args.output, arm)
    assert len({r['adapter_before'] for r in training.values()}) == 1
    report = dict(status='complete', protocol=protocol, training=training, calibration=calibration,
                  probability_class_order=CLASSES, results={})
    for split, cases in [('train', protocol['train_cases']), ('validation', protocol['evaluation_cases'])]:
        scores = {}
        for case in cases:
            source, _ = load_image(args.data, case)
            truth = read_labels(args.data, case, source)
            scores[case] = {}
            for arm in protocol['arms']:
                pred = nib.load(args.output/case/f'{arm}_mask.nii.gz')
                assert pred.shape == source.shape and np.allclose(pred.affine, source.affine)
                scores[case][arm] = metrics(truth, pred.get_fdata())
        report['results'][split] = dict(per_case=scores, mean_ap_dice={
            arm: float(np.mean([scores[c][arm]['ap_mean_dice'] for c in cases])) for arm in protocol['arms']})
    report['output_hashes'] = {str(p.relative_to(args.output)): sha256(p) for p in args.output.glob('*/*.nii.gz')}
    save_json(args.output/'results.json', report)
    print(json.dumps(dict(status='complete', validation=report['results']['validation']['mean_ap_dice'])), flush=True)


if __name__ == '__main__':
    main()
