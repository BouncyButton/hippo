"""Inference-only spatial audit of the preserved HFH fixed125 pair."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from scipy import ndimage as ndi

RUNS = {
    'baseline': ('u125_strict_native__dice__f0_s0', '3d0e332535d1ea1346e2f546431e82f3c106acfad342a866f2f9cd7fcc4784a1'),
    'band': ('u125_strict_gentle_t001_r20__dice_bands__f0_s0', '4c4c9bdef6dc80076c3d05825fe6b39532cfcaa38dee23ed3e5e1f49d2fb5330'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def spatial(truth, prediction):
    """Exact two-step six-connected BAND and separate physical EDT diagnostics."""
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    structure = ndi.generate_binary_structure(3, 1)
    inner = truth & ~ndi.binary_erosion(truth, structure, iterations=2)
    outer = ndi.binary_dilation(truth, structure, iterations=2) & ~truth
    regions = dict(inner=inner, core=truth & ~inner, outer=outer,
                   distant_background=~truth & ~outer)
    fp, fn = prediction & ~truth, truth & ~prediction
    l1 = ndi.distance_transform_cdt(~truth, metric='taxicab')
    mm = ndi.distance_transform_edt(~truth, sampling=(1., 1., 2.))
    components, count = ndi.label(prediction, structure)
    overlaps = np.unique(components[truth]); overlaps = overlaps[overlaps != 0]
    detached = prediction & ~np.isin(components, overlaps)
    empty = np.broadcast_to(~truth.any(axis=(0, 1))[None, None, :], truth.shape)
    row = dict(tp=int((truth & prediction).sum()), fp=int(fp.sum()), fn=int(fn.sum()),
               gt_voxels=int(truth.sum()), predicted_voxels=int(prediction.sum()),
               region_voxels={k: int(v.sum()) for k, v in regions.items()},
               errors_by_region={k: int(((fp | fn) & v).sum()) for k, v in regions.items()},
               fp_l1_steps={k: int((fp & m).sum()) for k, m in {
                   '1': l1 == 1, '2': l1 == 2, '3_4': (l1 > 2) & (l1 <= 4), 'over4': l1 > 4}.items()},
               fp_mm={k: int((fp & m).sum()) for k, m in {
                   'le1': mm <= 1, '1_2': (mm > 1) & (mm <= 2),
                   '2_4': (mm > 2) & (mm <= 4), 'over4': mm > 4}.items()},
               detached_component_fp=int((fp & detached).sum()),
               fp_on_empty_slices=int((fp & empty).sum()),
               predicted_components=int(count))
    row['dice'] = 2 * row['tp'] / max(2 * row['tp'] + row['fp'] + row['fn'], 1)
    return row, regions


def paired(truth, a, b):
    ra, regions = spatial(truth, a)
    rb, _ = spatial(truth, b)
    truth, a, b = truth.astype(bool), a.astype(bool), b.astype(bool)
    changes = dict(new_fp=~truth & ~a & b, removed_fp=~truth & a & ~b,
                   recovered_fn=truth & ~a & b, new_fn=truth & a & ~b)
    return dict(baseline=ra, band=rb,
                changes={k: dict(total=int(v.sum()), **{r: int((v & m).sum()) for r, m in regions.items()})
                         for k, v in changes.items()})


def main():
    import torch
    import torch.nn.functional as F
    from scripts.cross_dataset_campaign import foundation as engine, foundation_msd as matched
    from scripts.colab_hfh.data import Volumes
    from scripts.colab_hfh.medsam_constraints_deterministic import install
    from scripts.colab_hfh.medsam_recovery import copy_verified
    from scripts.train_medsam3_few_shot import parameter_digest, is_adapter

    root = Path('/content/hippo_hfh')
    out = root/'medsam_spatial125_v1'
    out.mkdir(exist_ok=True)
    remote = Path('/content/hfh_storage/results_medsam3_fixed125_v1/spatial_audit')
    started = time.time()
    status = dict(status='starting', optimizer_updates=0, records=[], started_at=started)

    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2) + '\n')

    def publish():
        status['observed_at'] = time.time()
        write('status.json', status)
        copy_verified(out/'status.json', remote/'status.json')

    try:
        torch.set_num_threads(4)
        protocols = {}
        for arm, (run, expected) in RUNS.items():
            source = root/'medsam_fixed125_v1/runs'/run
            done = json.loads((source/'completion.json').read_text())
            protocol = json.loads((source/'protocol.json').read_text())
            checkpoint = source/'checkpoint_latest.pt'
            if done['updates'] != 125 or done['checkpoint_sha256'] != expected or digest(checkpoint) != expected:
                raise ValueError('Preserved checkpoint identity differs: ' + arm)
            if digest(source/'protocol.json') != done['protocol_sha256']:
                raise ValueError('Protocol differs')
            if digest(source/'evaluation.json') != done['evaluation_sha256']:
                raise ValueError('Archived evaluation differs')
            protocols[arm] = protocol
        for key in ('train_cases', 'evaluation_cases', 'frozen_sample_schedule', 'frozen_learning_rates'):
            if protocols['baseline'][key] != protocols['band'][key]:
                raise ValueError('Pair mismatch: '+key)
        publish()
        for arm, (run, expected) in RUNS.items():
            source = root/'medsam_fixed125_v1/runs'/run
            protocol = protocols[arm]
            install(protocol)
            engine.reset_seed(protocol['seed'])
            status.update(status='loading', arm=arm)
            publish()
            model, processor = matched.load_foundation(protocol, root/'sam3.pt')
            processor.transform.clear(); processor.transform.enabled = False
            state = torch.load(source/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
            if state['step'] != 125 or state['protocol_sha256'] != digest(source/'protocol.json'):
                raise ValueError('Checkpoint step/protocol differs')
            if set(state['adapter']) != {n for n in model.state_dict() if is_adapter(n)}:
                raise ValueError('Adapter key set differs')
            model.load_state_dict(state['adapter'], strict=False)
            model.eval().requires_grad_(False)
            summary = json.loads((source/'training_summary.json').read_text())
            if parameter_digest(model, False) != summary['frozen_after'] or parameter_digest(model, True) != summary['adapter_after']:
                raise ValueError('Loaded model differs from final archived weights')
            torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            del state
            volumes = Volumes(protocol)
            reference = json.loads((source/'evaluation.json').read_text())['records']
            fields = {}
            original_fields = engine.semantic_log_fields

            def capture(output, shape):
                logs = original_fields(output, shape)
                masks = F.interpolate(output['pred_masks'][0, :, None].float(), size=shape,
                                      mode='bilinear', align_corners=False)[:, 0]
                scores = output['pred_logits'][0].float().reshape(-1)
                presence = output['presence_logit_dec'][0].float().reshape(-1)
                winner = (F.logsigmoid(masks) + F.logsigmoid(scores)[:, None, None]
                          + F.logsigmoid(presence)).argmax(0)
                fields.update(mask=masks.gather(0, winner[None])[0].sigmoid().cpu().numpy().T,
                              score=scores[winner].sigmoid().cpu().numpy().T,
                              presence=float(presence.sigmoid().item()))
                return logs

            engine.semantic_log_fields = capture
            try:
                for index, ref in enumerate(reference):
                    case = ref['case_name']
                    path = out/(arm+'__'+case+'.npz')
                    if path.exists():
                        with np.load(path) as saved:
                            if hashlib.sha256(saved['prediction'].tobytes()).hexdigest() != ref['prediction_sha256']:
                                raise ValueError('Existing audit prediction differs')
                    else:
                        image, truth = volumes.load(case)
                        arrays = {k: np.empty(image.shape, dtype=np.float32) for k in ('probability', 'mask', 'score', 'presence')}
                        prediction = np.empty(image.shape, dtype=np.uint8)
                        with torch.inference_mode():
                            for z in range(image.shape[2]):
                                pixels = np.round(image[:, :, z].T*255).astype(np.uint8)
                                logp, _ = engine.forward_plane(model, processor, pixels, protocol)
                                prediction[:, :, z] = logp.exp().argmax(0).cpu().numpy().T
                                arrays['probability'][:, :, z] = logp[1].exp().cpu().numpy().T
                                for key in ('mask', 'score', 'presence'):
                                    arrays[key][:, :, z] = fields[key]
                        if hashlib.sha256(truth.tobytes()).hexdigest() != ref['truth_sha256']:
                            raise ValueError('Truth hash differs')
                        if hashlib.sha256(prediction.tobytes()).hexdigest() != ref['prediction_sha256']:
                            raise ValueError('Prediction hash differs: '+arm+' '+case)
                        np.savez_compressed(path, truth=truth, prediction=prediction, **arrays)
                    status.update(status='inference', arm=arm, case=case, completed_in_arm=index+1,
                                  prediction_hash_verified=True)
                    print(json.dumps({k:v for k,v in status.items() if k!='records'}), flush=True)
                    publish()
            finally:
                engine.semantic_log_fields = original_fields
            if digest(source/'checkpoint_latest.pt') != expected:
                raise ValueError('Source checkpoint changed during read-only audit')
            del model, processor
            torch.cuda.empty_cache()
        reference = json.loads((root/'medsam_fixed125_v1/runs'/RUNS['baseline'][0]/'evaluation.json').read_text())['records']
        for ref in reference:
            case = ref['case_name']
            with np.load(out/('baseline__'+case+'.npz')) as aa, np.load(out/('band__'+case+'.npz')) as bb:
                truth, a, b = aa['truth'], aa['prediction'], bb['prediction']
                if not np.array_equal(truth, bb['truth']):
                    raise ValueError('Paired truth differs')
                row = dict(case=case, split=ref['split'], **paired(truth, a, b))
                new_fp = (truth == 0) & (a == 0) & (b == 1)
                row['new_fp_fields'] = {arm: {k: float(data[k][new_fp].mean()) if new_fp.any() else None
                    for k in ('probability', 'mask', 'score', 'presence')} for arm, data in [('baseline', aa), ('band', bb)]}
                status['records'].append(row)
        status.update(status='complete', seconds=time.time()-started,
                      all_prediction_hashes_verified=True, source_checkpoints_unchanged=True,
                      axis_order='source X,Z,Y; physical spacing1,1,2mm',
                      band='two six-connected morphological steps; not2mm EDT',
                      scope='fold0 seed0 native125 vs gentleBAND125; exploratory')
        write('analysis.json', status)
        archive = out/'spatial_audit.zip'
        files = list(out.glob('*.npz'))+[out/'analysis.json', Path(__file__)]
        manifest = {p.name:digest(p) for p in files}
        write('manifest.json', manifest)
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in files+[out/'manifest.json']:
                z.write(p, p.name)
        copy_verified(archive, remote/archive.name)
        copy_verified(out/'analysis.json', remote/'analysis.json')
        publish()
    except Exception as error:
        status.update(status='error', error=repr(error))
        publish()
        raise


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    main()
