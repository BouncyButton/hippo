"""Inference-only supplement; frozen selected checkpoints are never modified."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--home', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    new = args.home/'pcgrad_50cases_seed0_20260929_01'
    old = args.home/'pcgrad_edge_20260929_01'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for base in (new, old):
        assert base != out and base not in out.parents
        manifest = read(base/'source/PAYLOAD.json')['files']
        for name, expected in manifest.items():
            assert sha(base/'source'/name) == expected, name
        manifests[base.name] = manifest
    # All shared inference implementations must be byte-identical.
    for name, expected in manifests[old.name].items():
        assert manifests[new.name][name] == expected, name
    sys.path[:0] = [str(new/'source'), str(new/'source/scripts')]
    import torch
    import monai
    import numpy as np
    from scipy import ndimage as ndi
    from torch.utils.data import DataLoader
    from train_separated_edge import loaders as old_loaders, build_model, objectives
    from train_pcgrad_50cases import loaders as new_loaders
    from audit_edge_coherence import metrics
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    runtime = dict(torch=torch.__version__, monai=monai.__version__, numpy=np.__version__,
                   gpu=torch.cuda.get_device_name(), job_id=os.getenv('SLURM_JOB_ID'))
    verification = dict(runtime=runtime, source_files_verified={k:len(v) for k,v in manifests.items()}, runs={})
    for label, base, arm, get_loaders in [('pcgrad50', new, 'pcgrad', new_loaders),
            ('pcgrad10', old, 'pcgrad', old_loaders), ('sum10', old, 'sum', old_loaders)]:
        folder = base/'fold0/seed0'/arm
        config, done = read(folder/'config.json'), read(folder/'completion.json')
        assert done['status'] == 'complete'
        assert config['runtime_provenance']['torch'] == torch.__version__
        assert config['runtime_provenance']['monai'] == monai.__version__
        assert config['runtime_provenance']['numpy'] == np.__version__
        for key in ('pkl', 'splits_json'):
            assert sha(config[key]) == config[key+'_sha256']
        cp_hash = sha(folder/'checkpoint_best.pt')
        assert cp_hash == done['checkpoint_sha256']
        assert sha(folder/'selected_cases.json') == done['selected_cases_sha256']
        trainer.seed_everything(0)
        train, val, _, _ = get_loaders(config)
        cp = torch.load(folder/'checkpoint_best.pt', map_location='cpu', weights_only=True)
        assert cp['epoch'] == done['selected_epoch'] and cp['run'] == done['run']
        model = build_model(config, device)
        model.load_state_dict(cp['model'], strict=True)
        model.eval()
        del cp
        _, objective = objectives(config)
        bands = objective.bands
        assert bands.inner_focal_gamma == bands.outer_focal_gamma == bands.degree_alpha == 0
        assert bands.adherence_threshold == .9 and bands.steps == 2
        original = {(r['split'],r['case_name']):r['metrics'] for r in read(folder/'selected_cases.json')}
        rows = []
        for split, loader in [('train',DataLoader(train.dataset,batch_size=1,shuffle=False)),('validation',val)]:
            for batch in loader:
                with torch.inference_mode(), torch.autocast('cuda',enabled=config['run']['amp']):
                    logits = model(batch['image'].to(device))
                with torch.inference_mode():
                    result = bands(logits,batch['label'].to(device))
                p = logits[0].float().softmax(0).cpu().numpy()
                truth = batch['label'][0,0].numpy().astype(np.uint8)
                pred = p.argmax(0).astype(np.uint8)
                fresh = metrics(p[1:].sum(0),pred,truth)
                name = str(batch['case_name'][0])
                b = {k:float(result.details[k][0]) for k in ('inner_loss','outer_loss','case_loss')}
                b.update(truth=float(result.truth[0]),inner_truth=float(result.value[0,0]),
                    outer_truth=float(result.value[0,1]),confidence_adherent=bool(result.details['confidence_adherent'][0]))
                # Distances are in the evaluation grid's voxel units, not millimetres.
                struct = ndi.generate_binary_structure(3,1)
                a, y = pred>0, truth>0
                sa = a & ~ndi.binary_erosion(a,struct,border_value=0)
                sy = y & ~ndi.binary_erosion(y,struct,border_value=0)
                assert sa.any() and sy.any()
                distances = np.concatenate([ndi.distance_transform_edt(~sy)[sa],ndi.distance_transform_edt(~sa)[sy]])
                surface = dict(symmetric_mean_surface_distance_voxels=float(distances.mean()),
                    symmetric_surface_distance_p95_voxels=float(np.percentile(distances,95)))
                old_metrics = original[(split,name)]
                rows.append(dict(split=split,case_name=name,bands=b,surface=surface,
                    recomputed_metrics=fresh,drift={k:fresh[k]-old_metrics[k] for k in fresh}))
        assert {(r['split'],r['case_name']) for r in rows} == set(original)
        assert sha(folder/'checkpoint_best.pt') == cp_hash
        verification['runs'][label] = dict(checkpoint_sha256=cp_hash,
            selected_epoch=done['selected_epoch'],selected_cases_sha256=done['selected_cases_sha256'],
            max_drift={k:max(abs(r['drift'][k]) for r in rows) for k in rows[0]['drift']})
        save(out/(label+'.json'),dict(label=label,selected_epoch=done['selected_epoch'],cases=rows))
        print(json.dumps(dict(label=label,cases=len(rows),max_dice_drift=verification['runs'][label]['max_drift']['macro_dice'])),flush=True)
        del model, objective, bands
        torch.cuda.empty_cache()
    verification.update(status='complete',training=False,outputs={p.name:sha(p) for p in out.glob('*.json')})
    save(out/'VERIFICATION.json',verification)

if __name__ == '__main__':
    main()
