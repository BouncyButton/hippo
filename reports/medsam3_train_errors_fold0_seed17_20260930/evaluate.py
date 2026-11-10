"""Post-training inference on the five support volumes; never updates weights."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path('/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928')
DATA = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD')
RUN = ROOT / 'results/fold0_seed17'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    original = json.loads((RUN / 'results.json').read_text())
    inventory = json.loads((RUN / 'training_inventory.json').read_text())
    with tempfile.TemporaryDirectory(prefix='medsam3-train-errors-', dir='/dev/shm') as temp:
        scratch = Path(temp)
        base = scratch / 'sam3.pt'
        for key in ['TMPDIR', 'MPLCONFIGDIR', 'HF_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR']:
            folder = scratch / key
            folder.mkdir()
            os.environ[key] = str(folder)
        write(OUT / 'status.json', dict(status='waiting_for_base', destination=str(base),
              ready=str(scratch / 'ready'), job_id=os.environ.get('SLURM_JOB_ID')))
        deadline = time.monotonic() + 1200
        while not (scratch / 'ready').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Base checkpoint transfer timed out')
            time.sleep(2)

        import nibabel as nib
        import numpy as np
        import torch
        from evaluation.medsam3_zero_shot import load_model, infer_volume, sha256
        from scripts.train_medsam3_few_shot import load_image, save_predictions

        assert sha256(base) == original['protocol']['base_sha256']
        for item in inventory:
            case = item['case']
            assert sha256(DATA / 'imagesTr' / f'{case}_0000.nii.gz') == item['image_sha256']
            assert sha256(DATA / 'labelsTr' / f'{case}.nii.gz') == item['label_sha256']
        report = dict(status='running', split='training_support', fold=0, seed=17,
                      metric='3D whole-hippocampus union Dice', model_updates=0,
                      source_results_sha256=sha256(RUN / 'results.json'),
                      per_case={}, adapters={}, validation_reproduction={},
                      error_map_labels={'0': 'correct', '1': 'false_positive',
                                        '2': 'missed_anterior', '3': 'missed_posterior'},
                      ap_swaps=None, ap_swaps_note='Undefined: the model predicts binary union only.')
        model = processor = None
        for arm in original['protocol']['arms']:
            adapter = RUN / arm / 'final_lora_weights.pt'
            digest = sha256(adapter)
            assert digest == original['training'][arm]['final_adapter_sha256']
            report['adapters'][arm] = digest
            if model is None:
                args = SimpleNamespace(upstream=ROOT / 'MedSAM3', base_weights=base,
                                       lora_weights=adapter, device='cuda')
                model, processor = load_model(args)
            else:
                weights = torch.load(adapter, map_location='cpu', weights_only=True)
                expected = {k for k in model.state_dict() if k.endswith(('.lora_A', '.lora_B'))}
                assert set(weights) == expected
                model.load_state_dict(weights, strict=False)
                model.eval().requires_grad_(False)
            # Reproduce one original validation mask per arm before scoring training data.
            check_case = 'hippocampus_164'
            source, image = load_image(DATA, check_case)
            probability, _ = infer_volume(image, processor, 'hippocampus')
            saved = nib.load(RUN / check_case / f'{arm}_mask.nii.gz')
            assert saved.shape == source.shape and np.allclose(saved.affine, source.affine)
            differences = int(np.count_nonzero((probability > .5) != (saved.get_fdata() > .5)))
            report['validation_reproduction'][arm] = dict(case=check_case, differing_voxels=differences)
            if differences:
                write(OUT / 'reproduction_failure.json', report)
                raise RuntimeError(f'{arm}: saved validation mask differs at {differences} voxels')
            for case in original['protocol']['train_cases']:
                write(OUT / 'status.json', dict(status='inference', arm=arm, case=case))
                source, image = load_image(DATA, case)
                probability, slices = infer_volume(image, processor, 'hippocampus')
                save_predictions(OUT, case, source, {arm: probability})
                write(OUT / case / f'{arm}_slices.json', slices)
                labels = nib.as_closest_canonical(nib.load(DATA / 'labelsTr' / f'{case}.nii.gz'))
                assert labels.shape == source.shape and np.allclose(labels.affine, source.affine)
                truth = labels.get_fdata()
                assert np.isin(truth, [0, 1, 2]).all()
                hard = nib.load(OUT / case / f'{arm}_mask.nii.gz').get_fdata() > .5
                tp = int((hard & (truth > 0)).sum())
                fp = int((hard & (truth == 0)).sum())
                fn_a = int((~hard & (truth == 1)).sum())
                fn_p = int((~hard & (truth == 2)).sum())
                fn = fn_a + fn_p
                report['per_case'].setdefault(case, {})[arm] = dict(
                    dice=2*tp/(2*tp+fp+fn), tp=tp, fp=fp, fn=fn,
                    fn_anterior=fn_a, fn_posterior=fn_p,
                    anterior_voxels=int((truth == 1).sum()), posterior_voxels=int((truth == 2).sum()))
                errors = np.zeros(hard.shape, dtype=np.uint8)
                errors[hard & (truth == 0)] = 1
                errors[~hard & (truth == 1)] = 2
                errors[~hard & (truth == 2)] = 3
                nib.save(nib.Nifti1Image(errors, source.affine), OUT / case / f'{arm}_errors.nii.gz')
                write(OUT / 'results.json', report)
        report['mean_dice'] = {arm: float(np.mean([v[arm]['dice'] for v in report['per_case'].values()]))
                               for arm in original['protocol']['arms']}
        report['totals'] = {arm: {k: sum(v[arm][k] for v in report['per_case'].values())
                                 for k in ['tp', 'fp', 'fn', 'fn_anterior', 'fn_posterior']}
                            for arm in original['protocol']['arms']}
        report['output_hashes'] = {str(p.relative_to(OUT)): sha256(p) for p in OUT.glob('*/*.nii.gz')}
        report['status'] = 'complete'
        write(OUT / 'results.json', report)
        write(OUT / 'status.json', dict(status='complete'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(OUT / 'status.json', dict(status='failed', error=repr(exc)))
        raise
