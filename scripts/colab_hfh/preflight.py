"""Actual GPU graph/dependency checks before starting the baseline search."""
import argparse
import gc
from pathlib import Path
import torch
from scripts.colab_hfh.data import PreparedPatients
from scripts.colab_hfh.policy import MODELS
from scripts.colab_hfh.runtime import read, write, digest
from scripts.cross_dataset_campaign import conventional as exp
from scripts.adni_constraints.objectives import training_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from scripts.colab_adni.runtime import environment
    binding = dict(manifest=digest(args.bundle/'manifest.json'), environment=environment())
    if args.output.exists() and read(args.output).get('binding') == binding:
        print('Reusing same-source/GPU/software baseline preflight', flush=True)
        return
    reports = []
    for identity in MODELS:
        folder = args.bundle/'protocols'/identity
        p = read(folder/'protocol.json')
        exp.base.seed_everything(0)
        model = exp.build_model(p, read(folder/'nnunet_architecture.json')).cuda()
        if sum(v.numel() for v in model.parameters()) != p['models'][p['trainer_alias']]['parameters']:
            raise ValueError('Model parameter count differs from inherited FLOP profile')
        data = PreparedPatients(read(args.bundle/'data_index.json'), read(folder/'splits_10patients.json')[0]['train'])
        x, y = data[0]['image'][None].cuda(), data[0]['label'][None].cuda()
        criterion, optimizer = exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], model, p)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(x)
            loss = training_loss(prediction, y, 'dice', {}, .2,
                                 criterion=criterion if identity == 'nnunet' else None)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite preflight loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        if norm <= 0:
            raise ValueError('No training gradients')
        reports.append(dict(model=identity, loss=float(loss.detach()), gradient_norm=float(norm),
            optimizer=type(optimizer).__name__, updates_executed=0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print('GPU PREFLIGHT PASSED', reports[-1], flush=True)
        del model, data, x, y, criterion, optimizer, prediction, loss, norm
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    write(args.output, dict(status='passed', binding=binding, reports=reports,
        scope='Setup forward/backward, no optimizer updates; separate from experiment FLOP caps'))


if __name__ == '__main__':
    main()
