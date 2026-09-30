"""Frozen text-prompted MedSAM3 and explicitly mask-free boundary refinements.

The ground truth is opened only after all predictions have been persisted.
These image-driven inference objectives are NOT the supervised training losses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize_image(image):
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError('Expected a finite 3D image.')
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        raise ValueError('Image has no usable intensity range.')
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def refine(image, probability, *, edge=False, steps=80, device='cpu'):
    """Optimize only voxel logits in a fixed two-step prediction boundary band.

    Bands: balanced foreground/background image likelihood, fitted to frozen
    prediction seeds. Edge: image-weighted adjacent-voxel probability coherence.
    A soft baseline anchor and +/-2 logit bound limit deviation. No labels,
    learned parameters, external training examples, or metric selection enter.
    """
    if image.shape != probability.shape or image.ndim != 3:
        raise ValueError('Image and probabilities must be matching 3D arrays.')
    if not np.isfinite(image).all() or not np.isfinite(probability).all():
        raise ValueError('Nonfinite input.')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must lie in [0,1].')
    hard = probability > .5
    structure = ndimage.generate_binary_structure(3, 1)
    inner_core = ndimage.binary_erosion(hard, structure, iterations=2)
    dilated = ndimage.binary_dilation(hard, structure, iterations=2)
    inner, outer = hard & ~inner_core, dilated & ~hard
    support = inner | outer
    fg_seed = inner_core & (probability >= .8)
    bg_seed = ~dilated & (probability <= .2)
    info = dict(edge=edge, steps=steps, learning_rate=.05, band_weight=1.,
                edge_weight=.2 if edge else 0., max_logit_shift=2.,
                foreground_seeds=int(fg_seed.sum()), background_seeds=int(bg_seed.sum()),
                support_voxels=int(support.sum()), trace=[])
    if min(fg_seed.sum(), bg_seed.sum(), inner.sum(), outer.sum()) < 8:
        info['status'] = 'no_op_insufficient_prediction_seeds_or_band'
        return probability.copy(), info
    # Equal-prior Gaussian likelihood; variance floor prevents singular fits.
    mean_fg, mean_bg = float(image[fg_seed].mean()), float(image[bg_seed].mean())
    var_fg, var_bg = max(float(image[fg_seed].var()), .0025), max(float(image[bg_seed].var()), .0025)
    likelihood = .5 * np.log(var_bg / var_fg) + .5 * (
        (image - mean_bg) ** 2 / var_bg - (image - mean_fg) ** 2 / var_fg)
    target = torch.as_tensor(1 / (1 + np.exp(-np.clip(likelihood, -8, 8))), device=device)
    p0 = torch.as_tensor(probability.copy(), device=device).float()
    z0 = torch.logit(p0.clamp(1e-5, 1-1e-5))
    delta = torch.zeros_like(z0, requires_grad=True)
    masks = [torch.as_tensor(m, device=device) for m in (inner, outer, support)]
    im = torch.as_tensor(image, device=device)
    faces = []
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        valid = masks[2][a] & masks[2][b]
        weight = torch.exp(-(im[a]-im[b]).square() / (2 * .1**2))
        faces.append((a, b, valid, weight))
    optimizer = torch.optim.Adam([delta], lr=.05)
    for step in range(steps):
        optimizer.zero_grad()
        logits = z0 + delta * masks[2]
        p = logits.sigmoid()
        anchor = F.binary_cross_entropy_with_logits(logits[masks[2]], p0[masks[2]])
        region = .5 * sum(F.binary_cross_entropy_with_logits(logits[m], target[m]) for m in masks[:2])
        coherence = p.sum() * 0
        face_count = sum(int(valid.sum()) for _, _, valid, _ in faces)
        if edge and face_count:
            coherence = sum(((p[a]-p[b]).square()*weight)[valid].sum()
                            for a, b, valid, weight in faces) / face_count
        loss = anchor + region + (.2 * coherence if edge else 0)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-2, 2)
        if step % 10 == 0 or step == steps-1:
            info['trace'].append(dict(step=step+1, loss=float(loss.detach()),
                                     anchor=float(anchor.detach()), bands=float(region.detach()),
                                     edge=float(coherence.detach())))
    result = probability.copy()
    result[support] = (z0 + delta).detach().sigmoid().cpu().numpy()[support]
    info.update(status='complete', mean_fg=mean_fg, mean_bg=mean_bg,
                variance_fg=var_fg, variance_bg=var_bg,
                changed_voxels=int(((result > .5) != hard).sum()))
    return result, info


def load_detector_state(model, checkpoint):
    """Require all text-image weights; allow only the disabled SAM2 neck extras."""
    if isinstance(checkpoint.get('model'), dict):
        checkpoint = checkpoint['model']
    detector = {k.removeprefix('detector.'): v for k, v in checkpoint.items() if k.startswith('detector.')}
    expected = set(model.state_dict())
    extras = set(detector) - expected
    if any(not k.startswith('backbone.vision_backbone.sam2_convs.') for k in extras):
        raise RuntimeError(f'Unexpected active detector keys: {sorted(extras)}')
    model.load_state_dict({k: v for k, v in detector.items() if k in expected}, strict=True)
    return dict(loaded_detector_tensors=len(expected), ignored_disabled_interactive_tensors=sorted(extras))


def load_model(args):
    sys.path.insert(0, str(args.upstream.resolve()))
    import yaml
    from sam3.model_builder import build_sam3_image_model
    from lora_layers import LoRAConfig, apply_lora_to_model
    from sam3.model.sam3_image_processor import Sam3Processor
    cfg = yaml.safe_load((args.upstream / 'configs/full_lora_config.yaml').read_text())['lora']
    cfg['dropout'] = 0.
    # Upstream's convenience loader tolerates missing detector parameters.
    # Validate the complete base state strictly so a partial checkpoint cannot
    # silently produce a mostly randomly initialized "baseline".
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA inference, but no GPU is available.')
    model = build_sam3_image_model(
        load_from_HF=False,
        bpe_path=str(args.upstream / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
        device='cpu', compile=False, eval_mode=True)
    checkpoint = torch.load(args.base_weights, map_location='cpu', weights_only=True)
    base_report = load_detector_state(model, checkpoint)
    del checkpoint
    model = apply_lora_to_model(model, LoRAConfig(**cfg))
    weights = torch.load(args.lora_weights, map_location='cpu', weights_only=True)
    expected = {k for k in model.state_dict() if k.endswith(('.lora_A', '.lora_B'))}
    if set(weights) != expected:
        raise RuntimeError(f'Adapter mismatch: missing {len(expected-set(weights))}, unexpected {len(set(weights)-expected)}')
    model.load_state_dict(weights, strict=False)
    model.to(args.device).requires_grad_(False).eval()
    model.medsam3_base_report = base_report
    assert not any(p.requires_grad for p in model.parameters())
    return model, Sam3Processor(model, device=args.device, confidence_threshold=.5)


def infer_volume(image, processor, prompt):
    from PIL import Image
    from torchvision.ops import nms
    result = np.zeros_like(image, dtype=np.float32)
    diagnostics = []
    # Canonical RAS axial planes; transpose makes PIL rows Y and columns X.
    for z in range(image.shape[2]):
        pixels = np.round(image[:, :, z].T * 255).astype(np.uint8)
        pil = Image.fromarray(pixels).convert('RGB')
        with torch.inference_mode():
            state = processor.set_image(pil)
            state = processor.set_text_prompt(prompt, state)
            scores = state['scores']
            if len(scores):
                keep = nms(state['boxes'], scores, .5)
                # Upstream calls this masks_logits, but it contains sigmoid probabilities.
                masks = state['masks_logits'][keep, 0]
                result[:, :, z] = masks.amax(0).cpu().numpy().T
            diagnostics.append(dict(slice=z, detections=int(len(scores)),
                                    max_score=float(scores.max()) if len(scores) else None))
        print(json.dumps(diagnostics[-1]), flush=True)
    return result, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('upstream', 'base-weights', 'lora-weights', 'image', 'label', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--prompt', default='hippocampus')
    parser.add_argument('--device', choices=['cuda'], default='cuda',
                        help='Pinned upstream model construction requires CUDA.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    source = nib.load(args.image)
    canonical = nib.as_closest_canonical(source)
    image = normalize_image(canonical.get_fdata(dtype=np.float32))
    model, processor = load_model(args)
    base_report = getattr(model, 'medsam3_base_report', {})
    baseline, slices = infer_volume(image, processor, args.prompt)
    del processor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    predictions = {'medsam3': baseline}
    diagnostics = {}
    for name, use_edge in [('medsam3_bands', False), ('medsam3_bands_edge', True)]:
        predictions[name], diagnostics[name] = refine(image, baseline, edge=use_edge, device=args.device)
    # Commit all outputs BEFORE opening labels. Both refinements start at baseline.
    for name, probability in predictions.items():
        for suffix, data in [('probability', probability), ('mask', (probability > .5).astype(np.uint8))]:
            nib.save(nib.Nifti1Image(data, canonical.affine), args.output / f'{name}_{suffix}.nii.gz')
    save_json(args.output / 'refinement.json', diagnostics)
    save_json(args.output / 'slices.json', slices)
    labels = nib.as_closest_canonical(nib.load(args.label))
    if labels.shape != canonical.shape or not np.allclose(labels.affine, canonical.affine, atol=1e-4):
        raise ValueError('Reference mask and image grids do not match.')
    label_data = labels.get_fdata()
    if not np.isfinite(label_data).all() or not np.isin(label_data, [0, 1, 2]).all():
        raise ValueError('MSD reference must contain only finite labels 0, 1, and 2.')
    truth = label_data > 0
    metrics = {}
    for name, probability in predictions.items():
        hard = probability > .5
        tp, fp, fn = int((hard & truth).sum()), int((hard & ~truth).sum()), int((~hard & truth).sum())
        denom = int(hard.sum() + truth.sum())
        metrics[name] = dict(dice=2*tp/denom if denom else 1., tp=tp, fp=fp, fn=fn)
    commit = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    report = dict(status='complete', metrics=metrics, metric='3D whole-hippocampus union Dice',
                  prompt=args.prompt, ground_truth_used_for_prediction=False, model_updates=0,
                  case=args.image.name, upstream_commit=commit,
                  hashes={key: sha256(getattr(args, key)) for key in ('image', 'label', 'base_weights', 'lora_weights')},
                  limitations=['Single previously used development case, not population evidence.',
                               'Pretraining overlap with MSD is unknown; zero-shot means no adaptation training here.',
                               'Image-likelihood bands and image-weighted coherence are new inference adaptations.'],
                  torch_version=torch.__version__)
    report['base_loading'] = base_report
    report['runner_sha256'] = sha256(__file__)
    report['output_sha256'] = {p.name: sha256(p) for p in args.output.glob('*.nii.gz')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = image.shape[2] // 2  # Fixed image-only visualization plane.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (name, mask) in zip(axes, [('Reference', truth)] + [(n, p > .5) for n, p in predictions.items()]):
        ax.imshow(image[:, :, z].T, cmap='gray', origin='lower')
        if mask[:, :, z].any():
            ax.contour(mask[:, :, z].T, levels=[.5], colors=['lime'], linewidths=.8)
        ax.set_title(name.replace('medsam3', 'MedSAM3').replace('_', ' + '), fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=160)
    plt.close(fig)
    save_json(args.output / 'results.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
