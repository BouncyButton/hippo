"""Five-volume LoRA adaptation followed by a locked, label-blind evaluation."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.medsam3_zero_shot import load_model, normalize_image, infer_volume, refine, save_json, sha256


def validate_protocol(protocol, split):
    train, evaluation = protocol['train_cases'], protocol['evaluation_cases']
    if len(train) != 5 or len(set(train)) != 5 or len(set(evaluation)) != len(evaluation):
        raise ValueError('Require five distinct training volumes and distinct evaluation volumes.')
    if not evaluation or set(train) & set(evaluation):
        raise ValueError('Training and evaluation overlap or empty evaluation.')
    if not set(train) <= set(split['train']) or not set(evaluation) <= set(split['val']):
        raise ValueError('Cases do not belong to their declared fold-0 splits.')
    if set(split['train']) & set(split['val']):
        raise ValueError('Source split overlaps.')
    if protocol['updates'] <= 0 or protocol['accumulation'] <= 0:
        raise ValueError('Training budget must be positive.')


def load_image(data, case):
    path = data/'imagesTr'/f'{case}_0000.nii.gz'
    volume = nib.as_closest_canonical(nib.load(path))
    return volume, normalize_image(volume.get_fdata(dtype=np.float32))


def load_label(data, case, image):
    label = nib.as_closest_canonical(nib.load(data/'labelsTr'/f'{case}.nii.gz'))
    if label.shape != image.shape or not np.allclose(label.affine, image.affine, atol=1e-4):
        raise ValueError(f'Mismatched image/label grids: {case}')
    values = label.get_fdata()
    if not np.isfinite(values).all() or not np.isin(values, [0, 1, 2]).all():
        raise ValueError(f'Invalid reference values: {case}')
    return values > 0


def target_box(mask):
    """Normalized CxCyWH with exclusive upper bounds; no box for empty slices."""
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
    return torch.tensor([(x0+x1)/(2*w), (y0+y1)/(2*h), (x1-x0)/w, (y1-y0)/h], dtype=torch.float32)


class TrainingSlices:
    """Native SAM3 targets; training masks supervise losses, never prompt inputs."""
    def __init__(self, data, cases, processor):
        self.transform = processor.transform
        self.resolution = processor.resolution
        self.samples = []
        self.inventory = []
        for case in cases:
            source, image = load_image(data, case)
            truth = load_label(data, case, source)
            for z in range(image.shape[2]):
                pixels = np.round(image[:, :, z].T*255).astype(np.uint8)
                self.samples.append((pixels, truth[:, :, z].T.copy(), case, z))
            self.inventory.append(dict(case=case, slices=image.shape[2],
                                       positive_slices=int(truth.any(axis=(0, 1)).sum()),
                                       image_sha256=sha256(data/'imagesTr'/f'{case}_0000.nii.gz'),
                                       label_sha256=sha256(data/'labelsTr'/f'{case}.nii.gz')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
        from torchvision.transforms.v2.functional import to_image
        pixels, mask, _, _ = self.samples[index]
        # Exactly the inference processor's input transform, without its no-grad forward.
        tensor = self.transform(to_image(PILImage.fromarray(pixels).convert('RGB')))
        box = target_box(mask)
        objects = []
        if box is not None:
            segment = F.interpolate(torch.from_numpy(mask.copy())[None, None].float(),
                                    size=(self.resolution, self.resolution), mode='nearest')[0, 0].bool()
            objects = [Object(bbox=box, area=float(box[2]*box[3]), object_id=0, segment=segment)]
        query = FindQueryLoaded(
            query_text='hippocampus', image_id=0, object_ids_output=[0] if objects else [],
            is_exhaustive=True, input_bbox=None, input_points=None,
            inference_metadata=InferenceMetadata(index, index, 1, pixels.shape, -1, -1))
        return Datapoint(find_queries=[query], images=[Image(tensor, objects, (self.resolution, self.resolution))])


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if is_dataclass(obj):
        for field in fields(obj):
            setattr(obj, field.name, move_to_device(getattr(obj, field.name), device))
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def native_loss(model):
    """Use the weights and native detection/presence/mask losses from upstream."""
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    matcher = BinaryHungarianMatcherV2(cost_class=2., cost_bbox=5., cost_giou=2., focal=True)
    model.matcher = matcher
    losses = [Boxes(weight_dict={'loss_bbox': 5., 'loss_giou': 2.}),
              IABCEMdetr(pos_weight=10., weight_dict={'loss_ce': 20., 'presence_loss': 20.},
                        pos_focal=False, alpha=.25, gamma=2, use_presence=True, pad_n_queries=200),
              Masks(weight_dict={'loss_mask': 200., 'loss_dice': 10.},
                    focal_alpha=.25, focal_gamma=2., compute_aux=False)]
    return Sam3LossWrapper(loss_fns_find=losses, matcher=matcher,
                          o2m_matcher=BinaryOneToManyMatcher(alpha=.3, threshold=.4, topk=4),
                          o2m_weight=2., use_o2m_matcher_on_o2m_aux=False,
                          normalization='local', normalize_by_valid_object_num=False)


def is_adapter(name):
    return name.endswith(('.lora_A', '.lora_B'))


def parameter_digest(model, adapter):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if is_adapter(name) == adapter:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def lr_for_update(index, total, warmup, peak, floor):
    if index < warmup:
        return peak * (index+1)/warmup
    fraction = (index-warmup)/max(total-warmup-1, 1)
    return floor + .5*(peak-floor)*(1+math.cos(math.pi*fraction))


def train_adapter(model, dataset, protocol, output):
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    torch.manual_seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    rng = random.Random(protocol['seed'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_adapter(name))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError('No LoRA parameters found.')
    frozen_before = parameter_digest(model, False)
    adapter_before = parameter_digest(model, True)
    criterion = native_loss(model)
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=protocol['learning_rate'], weight_decay=.01)
    order, cursor, exposures = [], 0, {c['case']: 0 for c in dataset.inventory}
    start = time.monotonic()
    trace = output/'training.jsonl'
    with trace.open('x') as log:
        for update in range(protocol['updates']):
            lr = lr_for_update(update, protocol['updates'], protocol['warmup_updates'],
                               protocol['learning_rate'], protocol['final_learning_rate'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_values = []
            for micro in range(protocol['accumulation']):
                if cursor == len(order):
                    order = list(range(len(dataset)))
                    rng.shuffle(order)
                    cursor = 0
                index = order[cursor]
                cursor += 1
                exposures[dataset.samples[index][2]] += 1
                batch = collate_fn_api([dataset[index]], dict_key='input', with_seg_masks=True)['input']
                # Native collator emits shape (0,) for all-negative batch masks.
                for target in batch.find_targets:
                    if target.segments.numel() == 0:
                        target.segments = torch.empty((0, dataset.resolution, dataset.resolution), dtype=torch.bool)
                batch = move_to_device(batch, 'cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    predictions = model(batch)
                    targets = [model.back_convert(t) for t in batch.find_targets]
                    losses = criterion(predictions, targets)
                    loss = losses[CORE_LOSS_KEY]
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at update {update+1}.')
                (loss/protocol['accumulation']).backward()
                loss_values.append(float(loss.detach()))
                del predictions, targets, losses, loss, batch
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            record = dict(update=update+1, loss=float(np.mean(loss_values)), lr=lr,
                          gradient_norm=float(norm), elapsed_seconds=time.monotonic()-start)
            log.write(json.dumps(record)+'\n')
            log.flush()
            if update == 0 or (update+1) % 10 == 0:
                print(json.dumps(record), flush=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, criterion
    model.eval().requires_grad_(False)
    frozen_after, adapter_after = parameter_digest(model, False), parameter_digest(model, True)
    if frozen_after != frozen_before or adapter_after == adapter_before:
        raise RuntimeError('Frozen backbone changed or adapter did not change.')
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items() if is_adapter(k)}
    path = output/'final_lora_weights.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(weights, temporary)
    temporary.replace(path)
    report = dict(updates=protocol['updates'], slice_exposures=exposures,
                  trainable_parameters=sum(p.numel() for p in parameters), adapter_tensors=len(weights),
                  frozen_before=frozen_before, frozen_after=frozen_after,
                  adapter_before=adapter_before, adapter_after=adapter_after,
                  final_adapter_sha256=sha256(path), training_volumes=dataset.inventory,
                  evaluation_labels_used_for_training=False, checkpoint_selection='final fixed update')
    save_json(output/'training_summary.json', report)
    torch.cuda.empty_cache()
    return report


def save_predictions(directory, case, source, predictions):
    directory = directory/case
    directory.mkdir(parents=True, exist_ok=True)
    for name, probability in predictions.items():
        for suffix, data in [('probability', probability), ('mask', (probability > .5).astype(np.uint8))]:
            path = directory/f'{name}_{suffix}.nii.gz'
            if path.exists():
                raise FileExistsError(path)
            nib.save(nib.Nifti1Image(data, source.affine), path)


def score_saved_predictions(data, cases, output):
    names = ('zero_shot', 'few_shot', 'few_shot_bands', 'few_shot_bands_edge')
    # Every output for every evaluation case must exist before any evaluation label is read.
    for case in cases:
        for name in names:
            for suffix in ('mask', 'probability'):
                if not (output/case/f'{name}_{suffix}.nii.gz').is_file():
                    raise RuntimeError('Evaluation attempted before all predictions were saved.')
    metrics = {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for case in cases:
        source, image = load_image(data, case)
        truth = load_label(data, case, source)
        metrics[case] = {}
        masks = {}
        for name in names:
            prediction = nib.load(output/case/f'{name}_mask.nii.gz')
            if prediction.shape != source.shape or not np.allclose(prediction.affine, source.affine):
                raise ValueError('Prediction grid mismatch.')
            hard = prediction.get_fdata() > .5
            masks[name] = hard
            tp, fp, fn = int((hard & truth).sum()), int((hard & ~truth).sum()), int((~hard & truth).sum())
            metrics[case][name] = dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., tp=tp, fp=fp, fn=fn)
        z = image.shape[2]//2
        fig, axes = plt.subplots(1, 5, figsize=(17, 4))
        for ax, (name, mask) in zip(axes, [('Reference', truth)]+list(masks.items())):
            ax.imshow(image[:, :, z].T, cmap='gray', origin='lower')
            if mask[:, :, z].any():
                ax.contour(mask[:, :, z].T, levels=[.5], colors=['lime'], linewidths=.8)
            ax.set_title(name.replace('_', ' '), fontsize=10)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(output/case/'comparison.png', dpi=150)
        plt.close(fig)
    return dict(per_case=metrics, mean_dice={name:float(np.mean([metrics[c][name]['dice'] for c in cases])) for name in names})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('upstream', 'base-weights', 'lora-weights', 'data', 'protocol', 'output'):
        p.add_argument('--'+flag, type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    split_path = args.data/'splits_final.json'
    if sha256(split_path) != protocol['split_sha256']:
        raise ValueError('Split file differs from frozen protocol.')
    validate_protocol(protocol, json.loads(split_path.read_text())[0])
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'protocol.json', protocol)
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    model, processor = load_model(SimpleNamespace(upstream=args.upstream, base_weights=args.base_weights,
                                                 lora_weights=args.lora_weights, device='cuda'))
    base_loading = model.medsam3_base_report
    # Store unadapted controls without opening any evaluation references.
    for case in protocol['evaluation_cases']:
        source, image = load_image(args.data, case)
        prediction, slices = infer_volume(image, processor, 'hippocampus')
        save_predictions(args.output, case, source, {'zero_shot': prediction})
        save_json(args.output/case/'zero_shot_slices.json', slices)
    dataset = TrainingSlices(args.data, protocol['train_cases'], processor)
    save_json(args.output/'training_inventory.json', dataset.inventory)
    training = train_adapter(model, dataset, protocol, args.output)
    for case in protocol['evaluation_cases']:
        source, image = load_image(args.data, case)
        baseline, slices = infer_volume(image, processor, 'hippocampus')
        predictions, diagnostics = {'few_shot': baseline}, {}
        for name, use_edge in [('few_shot_bands', False), ('few_shot_bands_edge', True)]:
            predictions[name], diagnostics[name] = refine(image, baseline, edge=use_edge, device='cuda')
        save_predictions(args.output, case, source, predictions)
        save_json(args.output/case/'few_shot_slices.json', slices)
        save_json(args.output/case/'refinement.json', diagnostics)
    del model, processor
    torch.cuda.empty_cache()
    scores = score_saved_predictions(args.data, protocol['evaluation_cases'], args.output)
    report = dict(status='complete', metric='3D whole-hippocampus union Dice', **scores,
                  training=training, base_loading=base_loading, protocol=protocol,
                  hashes=dict(base=sha256(args.base_weights), initial_adapter=sha256(args.lora_weights),
                              runner=sha256(__file__)),
                  output_hashes={str(f.relative_to(args.output)):sha256(f) for f in args.output.glob('*/*.nii.gz')},
                  evaluation_label_hashes={case:sha256(args.data/'labelsTr'/f'{case}.nii.gz') for case in protocol['evaluation_cases']},
                  limitations=['Five labelled volumes, not five slices; patient identities unavailable.',
                               'Five existing validation volumes, not an independent test population.',
                               'Constraints are unchanged mask-free inference refinements, not training losses.'])
    save_json(args.output/'results.json', report)
    print(json.dumps(dict(status='complete', mean_dice=report['mean_dice'])), flush=True)


if __name__ == '__main__':
    main()
