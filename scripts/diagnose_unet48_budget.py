"""Inspect a completed width-48 pilot without modifying weights or artifacts."""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    source = args.pilot_root / 'source'
    sys.path.insert(0, str(source / 'scripts'))
    sys.path.insert(0, str(source))
    import torch
    import train_compute_matched_baselines as base
    from monai.losses import DiceLoss
    from train_unet48_budget_pilot import KWARGS

    torch.set_num_threads(4)
    device = torch.device(args.device)
    amp = device.type == 'cuda'
    run = args.pilot_root / 'pilots/candidate03/run'
    done = json.loads((run / 'completion.json').read_text())
    assert digest(run / 'checkpoint_latest.pt') == done['checkpoint_latest_sha256']
    config = json.loads((run / 'config.json').read_text())
    assert digest(args.pkl) == config['dataset_sha256']
    for name, sha in json.loads((args.pilot_root / 'runtime_origin_manifest.json').read_text()).items():
        assert digest(source / name) == sha, name
    cp = torch.load(run / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    assert cp['configuration'] == config
    model = base.UNet(**KWARGS).to(device)
    model.load_state_dict(cp['model'])
    dataframe = base._load_pkl_dataframe(args.pkl)
    train = base.PreparedPatients(dataframe, config['train_cases'])
    validation = base.PreparedPatients(dataframe, config['validation_cases'])
    report = {'checkpoint_sha256': digest(run / 'checkpoint_latest.pt'),
              'completion_sha256': digest(run / 'completion.json'),
              'dataset_sha256': config['dataset_sha256'], 'checkpoint_updates': cp['step'],
              'diagnostic_source_sha256': digest(__file__), 'torch': torch.__version__,
              'purpose': 'Read-only evaluation; diagnostic compute is separate from candidate training budgets.',
              'training_updates_performed': 0, 'model_structure': str(model)}
    model.eval()
    with torch.no_grad():
        report['train_evaluation'] = base.evaluate(model, train, device, amp)
        report['validation_evaluation'] = base.evaluate(model, validation, device, amp)
        details = []
        for item in train:
            image, label = item['image'][None].to(device), item['label'][None].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
                logits = model(image)
            probs = logits.float().softmax(1)
            truth = label[:, 0].long()
            prediction = probs.argmax(1)
            counts = torch.bincount((truth * 3 + prediction).reshape(-1), minlength=9).reshape(3, 3)
            conditional_probs = [probs.movedim(1, -1)[truth == c].mean(0).tolist() for c in range(3)]
            details.append({'case_name': item['case_name'], 'truth_by_prediction_confusion': counts.tolist(),
                            'probabilities_by_true_class': conditional_probs,
                            'logit_min': logits.float().amin((0, 2, 3, 4)).tolist(),
                            'logit_max': logits.float().amax((0, 2, 3, 4)).tolist()})
        report['training_case_details'] = details
        item = train[0]
        image, label = item['image'][None].to(device), item['label'][None].to(device)
        fp32 = model(image).float()
        with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
            mixed = model(image)
        report['precision_forward_comparison'] = {
            'case_name': item['case_name'], 'mean_abs_logit_difference': float((fp32 - mixed.float()).abs().mean()),
            'max_abs_logit_difference': float((fp32 - mixed.float()).abs().max()),
            'argmax_disagreement_fraction': float((fp32.argmax(1) != mixed.argmax(1)).float().mean())}

    # Compare only the loss's derivatives with respect to identical logits, so
    # this isolates loss precision without taking an optimizer step.
    gradients = []
    for loss_precision in ('runtime', 'float32'):
        logits = mixed.detach().clone().requires_grad_(True)
        criterion = DiceLoss(to_onehot_y=True, softmax=True, reduction='none')
        with torch.autocast('cuda', dtype=torch.bfloat16) if amp and loss_precision == 'runtime' else nullcontext():
            per_class = criterion(logits if loss_precision == 'runtime' else logits.float(), label)
            loss = per_class.mean()
        gradient, = torch.autograd.grad(loss, logits)
        gradient = gradient.float()
        gradients.append({'precision': loss_precision, 'loss': float(loss.detach()),
                          'per_class_loss': per_class.detach().float().reshape(-1).tolist(),
                          'gradient_abs_sum_per_class': gradient.abs().sum((0, 2, 3, 4)).tolist(),
                          'gradient_zero_fraction_per_class': (gradient == 0).float().mean((0, 2, 3, 4)).tolist(),
                          'gradient_positive_sum_per_class': gradient.clamp(min=0).sum((0, 2, 3, 4)).tolist(),
                          'gradient_negative_sum_per_class': gradient.clamp(max=0).sum((0, 2, 3, 4)).tolist()})
    report['loss_precision_comparison'] = gradients
    report['weight_statistics'] = [{'name': name, 'shape': list(p.shape),
                                    'minimum': float(p.detach().min()), 'maximum': float(p.detach().max()),
                                    'norm': float(p.detach().norm())} for name, p in model.named_parameters()
                                   if name.startswith('model.2')]
    assert digest(run / 'checkpoint_latest.pt') == report['checkpoint_sha256']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('train_evaluation', 'validation_evaluation',
                                                 'precision_forward_comparison', 'loss_precision_comparison')}), flush=True)


if __name__ == '__main__':
    main()
