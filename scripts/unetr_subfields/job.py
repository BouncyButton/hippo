"""Reuse the verified per-run archive/cleanup controller for this 18-run plan."""
import argparse
from pathlib import Path
import sys

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign import controller
from scripts.unetr_subfields.contracts import NAME, CAP, DATASETS, tasks, subject, baseline_schedule


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        if digest(stage / name) != sha:
            raise ValueError('Frozen campaign changed: ' + name)
    plan = read(stage / 'plan.json')
    if plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 18 or plan['augmentation'] != 'none':
        raise ValueError('Campaign scope differs')
    for dataset, spec in DATASETS.items():
        folder = stage / 'protocols' / dataset
        p = read(folder / 'protocol.json')
        profile = read(folder / 'profile.json')
        if p['budget_flops'] != CAP or p['requested_arm_order'] != ['dice'] or p['seeds'] != [0, 1, 2]:
            raise ValueError('Budget/seed/objective differs')
        if p['models']['unet']['objectives']['dice'] != baseline_schedule(profile['update_flops'], 2 * spec['train_subjects']):
            raise ValueError('Update schedule differs from profile')
        if digest(stage / 'data' / f'{dataset}.pkl') != p['pkl_sha256']:
            raise ValueError('Prepared dataset differs')
        folds = read(folder / 'splits_10patients.json')
        for fold in folds:
            train, val = fold['train'], fold['val']
            train_subjects, val_subjects = {subject(v) for v in train}, {subject(v) for v in val}
            if train_subjects & val_subjects or len(train_subjects) != spec['train_subjects']:
                raise ValueError('Training support or subject separation differs')
            if len(train) != 2 * spec['train_subjects'] or len(val) != (10 if dataset == 'MNI' else 2):
                raise ValueError('Wrong crop counts')
    return plan


def command_for(stage, plan, task, base_path):
    folder = stage / 'protocols' / task['protocol']
    p = read(folder / 'protocol.json')
    return [plan['conventional_python'], '-m', 'scripts.unetr_subfields.core',
            '--model', 'unet', '--objective', 'dice', '--fold', str(task['fold']),
            '--seed', str(task['seed']), '--pkl', p['data'], '--protocol-dir', str(folder),
            '--output', str(stage / 'runs' / task['id']), '--resume']


def preflight(stage, local=False):
    plan = verify(stage)
    import monai
    import torch
    import pandas as pd
    from scripts.unetr_subfields.core import PreparedSubfields, build_model
    if monai.__version__ != '1.6.0':
        raise ValueError('Use the measured MONAI version')
    if not local and not str(torch.__version__).startswith('2.12.1'):
        raise ValueError('Historical cluster training runtime changed')
    checked = []
    for dataset, spec in DATASETS.items():
        p = read(stage / 'protocols' / dataset / 'protocol.json')
        path = stage / 'data' / f'{dataset}.pkl' if local else Path(p['data'])
        if digest(path) != p['pkl_sha256']:
            raise ValueError('Runtime data differ')
        frame = pd.read_pickle(path, compression='gzip')
        # Check every stored case, including all selected training/validation folds.
        prepared = PreparedSubfields(frame, frame.subject_id.tolist(), spec)
        with torch.device('meta'):
            model = build_model(len(spec['classes']), spec['spatial_size'])
            output = model(torch.zeros((1, 1, *spec['spatial_size'])))
        if tuple(output.shape) != (1, len(spec['classes']), *spec['spatial_size']):
            raise ValueError('Unexpected model output shape')
        if sum(p.numel() for p in model.parameters()) != read(stage / 'protocols' / dataset / 'profile.json')['parameters']:
            raise ValueError('Model parameter count differs from profile')
        checked.append(dict(dataset=dataset, prepared_cases=len(prepared), output_shape=list(output.shape)))
    write(stage / ('local_preflight.json' if local else 'preflight.json'), dict(
        status='local_passed_not_submission_ready' if local else 'passed',
        manifest_sha256=digest(stage / 'manifest.json'), optimizer_updates=0,
        torch=torch.__version__, monai=monai.__version__, checks=checked))
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify', 'preflight', 'submit', 'run'])
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    stage = args.stage.resolve()
    controller.verify = verify
    controller.command_for = command_for
    if args.command == 'verify':
        print(verify(stage)['name'])
    elif args.command == 'preflight':
        print(preflight(stage, args.local))
    elif args.command == 'submit':
        print(controller.submit(stage))
    else:
        controller.run(stage)


if __name__ == '__main__':
    main()
