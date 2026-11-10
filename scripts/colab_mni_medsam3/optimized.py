"""MNI bridge to MSD's audited inference cache and asynchronous recovery."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts.colab_medsam3.fast_eval import EvaluationAudit, replace_attribute
from scripts.colab_medsam3.runtime import RecoveryWriter, copy_verified, restore
from scripts.colab_mni_medsam3.train import SupportSlabs
from scripts.colab_mni_medsam3.data import Volumes
from scripts.cross_dataset_campaign import foundation as original
from scripts.cross_dataset_campaign import foundation_msd as matched
from scripts.cross_dataset_campaign.common import digest, read, write


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(), numpy=np.random.get_state())


def set_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python']); np.random.set_state(state['numpy'])


def install_evaluation():
    """Reuse the exact MSD audit on the generic four-class volume evaluator."""
    reference_plane = original.forward_plane
    def evaluate(model, processor, volumes, protocol, output):
        bridge = SimpleNamespace(rng_state=rng_state, set_rng=set_rng, read=read, write=write)
        def forward(m, pr, pixels, labels=None, criterion=None):
            return reference_plane(m, pr, pixels, protocol, labels, criterion)
        identity = dict(checkpoint_sha256=digest(output/'checkpoint_latest.pt'),
            protocol_sha256=digest(output/'protocol.json'),
            source_manifest_sha256=os.environ['MNI_MANIFEST_SHA'])
        audit = EvaluationAudit(bridge, SimpleNamespace(forward_plane=forward), model,
            processor, output, output/'evaluation_acceleration.json', identity)
        load = volumes.load
        def tracked_load(case, labels=True):
            value = load(case, labels=labels)
            if not labels:
                audit.begin_volume(case, value[0].shape[2])
            return value
        def plane(m, pr, pixels, current_protocol, labels=None, criterion=None):
            if current_protocol != protocol:
                raise ValueError('Evaluation protocol changed')
            return audit.plane(m, pr, pixels, labels, criterion)
        try:
            with replace_attribute(volumes, 'load', tracked_load), replace_attribute(original, 'forward_plane', plane):
                return matched.evaluate(model, processor, volumes, protocol, output)
        finally:
            audit.fast.clear()
            audit.save()
    original.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser()
    for name in ('protocol','output','base-weights','durable'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    output, durable = args.output, args.durable
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    recovery = durable/'recovery'
    if not (output/'checkpoint_latest.pt').exists():
        restore(output, recovery)
    # A hard stop can leave stale sidecars. The optimizer checkpoint is authoritative.
    checkpoint = output/'checkpoint_latest.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        expected = {**read(args.protocol), 'arm':'dice'}
        if read(output/'protocol.json') != expected or state['protocol_sha256'] != digest(output/'protocol.json'):
            raise ValueError('Recovery protocol mismatch')
        if state['step'] != len(state['history']):
            raise ValueError('Recovery history is incomplete')
        write(output/'training_history.json',state['history'])
        if state['step'] < expected['updates']:
            for name in ('training_summary.json','evaluation.json','completion.json','evaluation_acceleration.json'):
                if (output/name).exists():
                    raise ValueError('Evaluation metadata precedes completed training; preserve for inspection')
        del state
    original.Volumes = Volumes
    original.SupportSlabs = SupportSlabs
    original.load_foundation = matched.load_foundation
    original.calibrate = matched.calibrate
    matched.configure_retained_backward()
    install_evaluation()
    writer = RecoveryWriter(output, recovery, interval=180)
    save = original.save_checkpoint
    def commit(path, state):
        save(path, state)
        write(output/'checkpoint_ledger.json',dict(step=state['step'], protocol_sha256=state['protocol_sha256']))
        writer.capture()
    original.save_checkpoint = commit
    install = original.install_signals
    def signals():
        install()
        write(output/'supervisor_ready.json',dict(pid=os.getpid()))
    original.install_signals = signals
    write(output/'environment.json',dict(python=sys.version, torch=torch.__version__,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        source_manifest_sha256=os.environ['MNI_MANIFEST_SHA'],
        training='Unchanged native-loss trainer; exact CPU transform cache',
        recovery='MSD immutable snapshot uploader, every180seconds and clean stop',
        evaluation='MSD full-volume numerical/speed audit with per-volume sentinels and automatic fallback'))
    sys.argv = [sys.argv[0], '--protocol',str(args.protocol),'--output',str(output),
                '--base-weights',str(args.base_weights),'--arm','dice']
    began = time.monotonic()
    try:
        if not (output/'completion.json').exists():
            original.main()
    finally:
        writer.close(force=True)
        write(output/'worker_timing.json',dict(elapsed_seconds=time.monotonic()-began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        for p in output.iterdir():
            if p.is_file() and p.suffix == '.json':
                copy_verified(p,durable/p.name)
        if checkpoint.exists():
            # Final/clean-stop checkpoint retained until the notebook verifies all results.
            copy_verified(checkpoint,durable/checkpoint.name)


if __name__ == '__main__':
    # The controller waits for supervisor_ready before requesting a clean stop.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    main()
