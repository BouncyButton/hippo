"""MSD-style native MedSAM3 training with the established resumable controller.

Use the original full native/bands/edge graph and original optimizer loop.
Only deterministic CPU image transforms are cached. GPU operations, precision,
sampling and the scientific recipe retain their original settings.
"""
import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from scripts.cross_dataset_campaign import foundation as original
from scripts.cross_dataset_campaign.common import digest, write
from scripts.cross_dataset_campaign.foundation_fast import TransformCache

REFERENCE_LOAD = original.load_foundation
REFERENCE_CALIBRATE = original.calibrate
REFERENCE_EVALUATE = original.evaluate


def configure_retained_backward():
    # The upstream native losses contain @torch.compile functions. Calibration
    # differentiates multiple losses through one graph, as in the MSD trainer.
    # Newer AOTAutograd must retain these buffers until the last backward call.
    import torch._functorch.config as config
    if hasattr(config, 'donated_buffer'):
        config.donated_buffer = False


def load_foundation(protocol, base_weights):
    model, processor = REFERENCE_LOAD(protocol, base_weights)
    processor.transform = TransformCache(processor.transform)
    return model, processor


def native_training_check(model, criterion, parameters, slabs, protocol, output):
    """Check the actual original losses and repeated backward, without updates."""
    states = (torch.get_rng_state(), torch.cuda.get_rng_state_all(),
              random.getstate(), np.random.get_state())
    checks = []
    try:
        original.reset_seed(protocol['calibration_seed'])
        case, anchor = slabs.calibration_schedule(protocol['calibration_slabs_per_volume'])[0]
        losses = original.forward_slab(model, criterion, slabs, case, anchor)
        names = ('native', 'bands', 'edge') + tuple(k for k in ('sdf','depth') if k in slabs.protocol['arm'])
        for i, name in enumerate(names):
            if not torch.isfinite(losses[name]):
                raise ValueError('Nonfinite original MSD-style loss: '+name)
            gradients = torch.autograd.grad(losses[name], parameters,
                retain_graph=i < len(names)-1, allow_unused=True)
            norm = 0.
            for gradient in gradients:
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise ValueError('Nonfinite original MSD-style gradient: '+name)
                    norm += float(gradient.detach().double().square().sum())
            if name == 'native' and norm <= 0:
                raise ValueError('Native loss has no trainable gradient')
            checks.append(dict(loss=name, value=float(losses[name].detach()),
                               squared_gradient_norm=norm, status='passed'))
            del gradients
        del losses
    finally:
        torch.set_rng_state(states[0]); torch.cuda.set_rng_state_all(states[1])
        random.setstate(states[2]); np.random.set_state(states[3])
    write(output/'runtime_validation.json', dict(status='passed',
        method='Original MSD-style native objective: finite losses and repeated calibration backward; no optimizer updates',
        checks=checks, source_sha256=digest(__file__),
        reference_engine_sha256=digest(original.__file__),
        protocol_sha256=digest(output/'protocol.json'), torch=torch.__version__,
        training_runtime_restored=True, rng_restored=True,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        donated_buffer=False, optimized_gpu_loss_graph=False))
    print('Original losses and all active auxiliary backward checks passed.', flush=True)


def calibrate(model, criterion, parameters, slabs, protocol, arm, output):
    native_training_check(model, criterion, parameters, slabs, protocol, output)
    result = REFERENCE_CALIBRATE(model, criterion, parameters, slabs, protocol, arm, output)
    # The successful MSD train_arm keeps both terms in the objective even when
    # their coefficient is zero. Preserve that full graph for matched baselines.
    result['weights'] = {'bands': 0., 'edge': 0., **result['weights']}
    return result


def evaluate(model, processor, volumes, protocol, output):
    cache = processor.transform
    write(output/'runtime_cache.json', dict(hits=cache.hits, misses=cache.misses,
        retained_bytes=cache.bytes, maximum_bytes=cache.max_bytes,
        policy='Exact CPU image transforms; original MSD-style full loss graph; released before evaluation'))
    cache.clear(); cache.enabled = False
    return REFERENCE_EVALUATE(model, processor, volumes, protocol, output)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    configure_retained_backward()
    write(args.output/'runtime_training_environment.json', dict(
        python=sys.executable, torch=torch.__version__, donated_buffer=False,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        training_path='Original MSD-style full native/bands/edge objective with resumable original optimizer loop'))
    original.load_foundation = load_foundation
    original.calibrate = calibrate
    original.evaluate = evaluate
    original.main()


def gpu_preflight():
    """Exercise every new loss on the real GPU, with zero optimizer updates."""
    import time
    from scripts.cross_dataset_campaign.common import read
    parser=argparse.ArgumentParser()
    for name in ('protocol','output','base-weights'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',required=True)
    args=parser.parse_args();started=time.time()
    p={**read(args.protocol),'arm':args.arm};args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'protocol.json',p)
    if p['arm']!='dice_bands_edge_sdf_depth':raise ValueError('Full objective preflight required')
    configure_retained_backward();torch.set_num_threads(4);original.reset_seed(p['seed'])
    model,processor=load_foundation(p,args.base_weights)
    for name,parameter in model.named_parameters():parameter.requires_grad_(original.is_adapter(name))
    parameters=[v for v in model.parameters() if v.requires_grad]
    before=original.parameter_digest(model,True)
    criterion=original.native_loss(model);volumes=original.Volumes(p)
    slabs=original.SupportSlabs(volumes,p,processor);model.train()
    result=calibrate(model,criterion,parameters,slabs,p,p['arm'],args.output)
    write(args.output/'calibration.json',result)
    after=original.parameter_digest(model,True)
    if before!=after or set(result['statistics'])!={'bands','edge','sdf','depth'}:
        raise ValueError('GPU preflight changed parameters or omitted an auxiliary loss')
    if any(r['valid']<5 or r['weighted_p95']>.50000001 for r in result['statistics'].values()):
        raise ValueError('GPU calibration failed')
    write(args.output/'complete.json',dict(status='passed',optimizer_updates=0,adapter_before=before,
        adapter_after=after,protocol_sha256=digest(args.protocol),elapsed_seconds=time.time()-started,
        calibration_pairs=len(result['rows']),source_sha256=digest(__file__)))


if __name__=='__main__':main()
