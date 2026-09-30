"""Inference-only spatial coherence audit of the nine 5%-data comparisons.

Uses the original frozen source payloads and selected checkpoints. No fitting,
checkpoint selection, or mutation of existing experiment artifacts occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def geometry(truth):
    y = truth > 0
    cross = ndi.generate_binary_structure(3, 1)
    inner = y & ~ndi.binary_erosion(y, cross, iterations=2, border_value=0)
    outer = ndi.binary_dilation(y, cross, iterations=2, border_value=0) & ~y
    return y, inner, outer


def metrics(probability, pred, truth):
    """Uniform face means, split by GT relation to avoid rewarding smoothing.

    Hard predictions use foreground union of the three-class argmax. A pair
    is coherent if its endpoints agree, but can be coherently wrong; both
    correctness and disagreement are therefore recorded for same-side pairs.
    """
    y, inner, outer = geometry(truth)
    hard = pred > 0
    p = np.asarray(probability, dtype=np.float64)
    assert p.shape == y.shape and np.isfinite(p).all()
    support = inner | outer
    raw = {k: dict(n=0, soft_squared_error=0., soft_abs_difference=0.,
                   hard_squared_error=0, disagree=0, both_correct=0,
                   both_wrong=0, correct_transition=0, reversed_transition=0,
                   aligned_soft_jump=0.) for k in ('all', 'inner', 'outer', 'cross')}
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        valid = support[a] & support[b]
        dy = y[a].astype(np.int8) - y[b].astype(np.int8)
        dp = p[a] - p[b]
        dh = hard[a].astype(np.int8) - hard[b].astype(np.int8)
        masks = dict(all=valid, inner=inner[a] & inner[b],
                     outer=outer[a] & outer[b], cross=valid & (dy != 0))
        for key, mask in masks.items():
            r = raw[key]
            r['n'] += int(mask.sum())
            r['soft_squared_error'] += float(np.square(dp[mask] - dy[mask]).sum())
            r['soft_abs_difference'] += float(np.abs(dp[mask]).sum())
            r['hard_squared_error'] += int(np.square(dh[mask] - dy[mask]).sum())
            r['disagree'] += int(((dh != 0) & mask).sum())
            r['both_correct'] += int(((hard[a] == y[a]) & (hard[b] == y[b]) & mask).sum())
            r['both_wrong'] += int(((hard[a] != y[a]) & (hard[b] != y[b]) & mask).sum())
            r['correct_transition'] += int(((dh == dy) & (dy != 0) & mask).sum())
            r['reversed_transition'] += int(((dh == -dy) & (dy != 0) & mask).sum())
            r['aligned_soft_jump'] += float((dp[mask] * dy[mask]).sum())
    result = {}
    for key, r in raw.items():
        result[key + '_face_count'] = r['n']
        for field, v in r.items():
            if field != 'n':
                result[key + '_' + field] = v / r['n'] if r['n'] else None
    for key, mask in dict(all=np.ones_like(y), inner=inner, outer=outer).items():
        result[key + '_fp'] = int((hard & ~y & mask).sum())
        result[key + '_fn'] = int((~hard & y & mask).sum())
        result[key + '_voxel_count'] = int(mask.sum())
    result['predicted_foreground_voxels'] = int(hard.sum())
    result['gt_foreground_voxels'] = int(y.sum())
    result['union_dice'] = float(2 * (hard & y).sum() / max(int(hard.sum() + y.sum()), 1))
    result['macro_dice'] = float(np.mean([2 * ((pred == c) & (truth == c)).sum() /
        max(int((pred == c).sum() + (truth == c).sum()), 1) for c in (1, 2)]))
    cross = ndi.generate_binary_structure(3, 1)
    components, n = ndi.label(hard, cross)
    sizes = np.bincount(components.ravel())[1:]
    result['foreground_components_6'] = int(n)
    result['foreground_outside_largest_6'] = int(sizes.sum() - sizes.max()) if n else 0
    result['foreground_outside_largest_fraction_6'] = result['foreground_outside_largest_6'] / max(int(hard.sum()), 1)
    result['foreground_singletons_6'] = int((sizes == 1).sum())
    result['gt_components_6'] = int(ndi.label(y, cross)[1])
    result['enclosed_background_voxels_6'] = int((ndi.binary_fill_holes(hard, cross) & ~hard).sum())
    return result


def volume_match(p, count):
    """Exactly match another model's foreground volume using probability ranks.

    This descriptive control uses no GT to select a threshold. It changes the
    decision rule from the original three-class argmax, so is not a new model.
    """
    flat = p.ravel()
    mask = np.zeros(flat.size, dtype=np.uint8)
    if count:
        ids = np.argpartition(flat, flat.size - count)[flat.size - count:]
        mask[ids] = 1
    return mask.reshape(p.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.utils.data import DataLoader
    frozen = args.base / 'edge_lowdata_folds12_20260924_01/source'
    sys.path[:0] = [str(frozen), str(frozen / 'scripts')]
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from train_edge_lowdata import data_arguments
    device = torch.device('cuda')
    assert torch.cuda.is_available()
    bindings, rows = [], []
    for fold in range(3):
        for seed in range(3):
            root = args.base / (f'edge_lowdata_20260924_01/seed{seed}' if fold == 0 else
                               f'edge_lowdata_folds12_20260924_01/fold{fold}/seed{seed}')
            references = json.loads((root / 'reference_bindings.json').read_text())
            historical = json.loads((root / 'validation/audit/checkpoints.json').read_text())
            dirs = {m: Path(references[m]['directory']) for m in ('dice', 'bands')}
            dirs['edge'] = root / 'run'
            config = json.loads((dirs['edge'] / 'config.json').read_text())
            for k in ('pkl', 'splits_json'):
                assert sha(config[k]) == config[k + '_sha256']
            trainer.seed_everything(seed)
            train, val, _, nt, nv = trainer.build_data(data_arguments(config), torch.Generator().manual_seed(seed))
            assert (nt, nv) == (10, 52)
            loaders = dict(train=DataLoader(train.dataset, batch_size=1, shuffle=False), validation=val)
            baseline = {}
            for method, directory in dirs.items():
                cfg = json.loads((directory / 'config.json').read_text())
                assert (cfg['run']['fold'], cfg['run']['seed'], cfg['run']['epochs']) == (fold, seed, 75)
                for k in ('pkl_sha256', 'splits_json_sha256'):
                    assert cfg[k] == config[k]
                weight = directory / 'checkpoint_best.pt'
                weight_sha = sha(weight)
                assert weight_sha == historical[method]['checkpoint_sha256']
                checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
                assert checkpoint['epoch'] == historical[method]['selected_epoch']
                assert cfg['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py'] == sha(frozen / 'baselines/swin_unetr/swin_unetr.py')
                model = trainer.build_swinunetr(tuple(cfg['run']['spatial_size']), 3, device,
                                               drop_rate=0, activation_checkpointing=False)
                model.load_state_dict(checkpoint['model'], strict=True)
                model.eval()
                bindings.append(dict(fold=fold, seed=seed, method=method, path=str(weight),
                                     sha256=weight_sha, selected_epoch=checkpoint['epoch'],
                                     config_sha256=sha(directory / 'config.json')))
                del checkpoint
                for split, loader in loaders.items():
                    old = json.loads((root / split / 'audit/cases.json').read_text())[method]
                    old = {r['case_name']: r for r in old}
                    for batch in loader:
                        name = str(batch['case_name'][0])
                        truth = batch['label'][0, 0].numpy().astype(np.uint8)
                        with torch.inference_mode(), torch.autocast('cuda', enabled=cfg['run']['amp']):
                            logits = model(batch['image'].to(device))
                        probs = logits[0].float().softmax(0).cpu().numpy()
                        p, pred = probs[1:].sum(0), probs.argmax(0).astype(np.uint8)
                        m = metrics(p, pred, truth)
                        prior = old[name]['metrics']['macro_dice']
                        # Record, rather than hide, small CUDA/AMP inference variation.
                        m['macro_dice_minus_historical_audit'] = m['macro_dice'] - prior
                        row = dict(fold=fold, seed=seed, method=method, split=split, case_name=name, metrics=m)
                        rows.append(row)
                        if method == 'dice':
                            baseline[split, name] = (p.copy(), sha_array(truth))
                        if method == 'edge':
                            bp, thash = baseline[split, name]
                            assert thash == sha_array(truth)
                            matched = volume_match(bp, int((pred > 0).sum()))
                            cm = metrics(bp, matched, truth)
                            cm.pop('macro_dice')  # Binary diagnostic has no A/P labels.
                            rows.append(dict(fold=fold, seed=seed, method='dice_volume_matched',
                                             split=split, case_name=name, metrics=cm))
                    print(json.dumps(dict(fold=fold, seed=seed, method=method, split=split,
                                          completed_cases=len(loader.dataset))), flush=True)
                del model
                torch.cuda.empty_cache()
                save(args.output / 'cases.json', rows)
                save(args.output / 'bindings.json', bindings)
    save(args.output / 'completion.json', dict(status='complete', job_id=os.getenv('SLURM_JOB_ID'),
         model_count=len(bindings), record_count=len(rows), script_sha256=sha(__file__),
         frozen_source=str(frozen), torch_version=torch.__version__))
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_edge_coherence.py')),
                    str(args.output)], check=True)


def sha_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


if __name__ == '__main__':
    main()
