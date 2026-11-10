"""Read-only spatial diagnosis of the failed and successful width-48 pilots."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.pilot_root / 'source'
    sys.path[:0] = [str(source / 'scripts'), str(source)]
    import torch
    import train_compute_matched_baselines as base
    from train_unet48_budget_pilot import KWARGS

    torch.set_num_threads(4)
    device = torch.device('cuda')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    for name, sha in json.loads((args.pilot_root / 'runtime_origin_manifest.json').read_text()).items():
        assert digest(source / name) == sha, name
    coordinates = torch.arange(64, device=device)
    d, h, w = torch.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    parity = ((d % 2) * 4 + (h % 2) * 2 + (w % 2)).reshape(-1)
    report = {'training_updates_performed': 0, 'diagnostic_source_sha256': digest(__file__),
              'hypothesis': 'Most false posterior voxels arise in zero-intensity background and follow a stride-2 decoder pattern.',
              'parity_index': '4*(D%2)+2*(H%2)+(W%2)', 'runs': []}
    dataframe = base._load_pkl_dataframe(args.pkl)
    for index in [6, 7]:
        run = args.pilot_root / 'pilots' / f'candidate{index:02d}' / 'run'
        done = json.loads((run / 'completion.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        assert digest(run / 'checkpoint_latest.pt') == done['checkpoint_latest_sha256']
        assert digest(args.pkl) == config['dataset_sha256']
        cp = torch.load(run / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
        model = base.UNet(**KWARGS).to(device)
        model.load_state_dict(cp['model'])
        model.eval()
        records = []
        with torch.no_grad():
            for split_name, names in [('train', config['train_cases']), ('validation', config['validation_cases'])]:
                patients = base.PreparedPatients(dataframe, names)
                for item in patients:
                    image = item['image'][None].to(device)
                    truth = item['label'].to(device).reshape(-1).long()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(image)
                    probabilities = logits.float().softmax(1)[0].reshape(3, -1)
                    predicted = probabilities.argmax(0)
                    background = truth == 0
                    zero_background = background & (image.reshape(-1).abs() < 1e-6)
                    posterior_fp = background & (predicted == 2)
                    confusion = torch.bincount(truth * 3 + predicted, minlength=9).reshape(3, 3)
                    parity_bg = torch.bincount(parity[zero_background], minlength=8)
                    parity_fp = torch.bincount(parity[zero_background & posterior_fp], minlength=8)
                    records.append({'split': split_name, 'case_name': item['case_name'],
                                    'truth_by_prediction_confusion': confusion.tolist(),
                                    'zero_intensity_background_voxels': int(zero_background.sum()),
                                    'posterior_false_positives': int(posterior_fp.sum()),
                                    'posterior_false_positives_zero_intensity': int((posterior_fp & zero_background).sum()),
                                    'zero_background_by_parity': parity_bg.tolist(),
                                    'posterior_false_positives_zero_background_by_parity': parity_fp.tolist(),
                                    'posterior_probability_zero_background_mean': float(probabilities[2, zero_background].mean()),
                                    'posterior_minus_background_logit_zero_background_quantiles': torch.quantile(
                                        (logits[0, 2].float() - logits[0, 0].float()).reshape(-1)[zero_background],
                                        torch.tensor([0., .1, .5, .9, 1.], device=device)).tolist()})
            empty = torch.zeros(1, 1, 64, 64, 64, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                empty_logits = model(empty)
            empty_prediction = empty_logits.float().argmax(1).reshape(-1)
            empty_counts = torch.bincount(empty_prediction, minlength=3).tolist()
            empty_parity = torch.bincount(parity[empty_prediction == 2], minlength=8).tolist()
        row = {'candidate': index, 'max_updates': config['max_steps'],
               'checkpoint_sha256': done['checkpoint_latest_sha256'], 'completion_sha256': digest(run / 'completion.json'),
               'cases': records, 'empty_input_prediction_class_counts': empty_counts,
               'empty_input_posterior_voxels_by_parity': empty_parity,
               'optimizer_betas': cp['optimizer']['param_groups'][0]['betas']}
        report['runs'].append(row)
        print(json.dumps({'candidate': index, 'empty_counts': empty_counts, 'empty_posterior_parity': empty_parity,
                          'first_case': records[0]}), flush=True)
        del model, cp
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('SPATIAL_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
