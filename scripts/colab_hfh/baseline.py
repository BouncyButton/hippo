"""HFH binding to the existing FLOP-capped mixed-loss ADNI trainer."""
import argparse
import os
from pathlib import Path
import sys
import time
from functools import partial

from scripts.colab_hfh.data import PreparedPatients
from scripts.cross_dataset_campaign import conventional as exp
from scripts.adni_constraints import objectives as binary
from scripts.colab_adni.runtime import RecoveryWriter
from scripts.colab_hfh.runtime import read, write, digest, restore_snapshot, publish_metadata


def evaluate(model, patients, device, amp):
    """Discard synthetic padding at evaluation; retain the shared physical crop."""
    import torch
    class CropOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, image):
            result = self.inner(image)
            if isinstance(result, (list, tuple)):
                result = result[0]
            return result[:, :, :, 16:48, :]
    cropped = [{**item, 'label':item['label'][:, :, 16:48, :]} for item in patients]
    return exp.evaluate(CropOutput(), cropped, device, amp, classes=['background', 'hippocampus'])


def main():
    parser = argparse.ArgumentParser()
    for name in ('protocol-dir', 'output', 'durable', 'pkl'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    p = read(args.protocol_dir/'protocol.json')
    args.model, args.objective, args.threads, args.resume = p['trainer_alias'], 'dice', 4, True
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.durable/'invocation.json'
    if not (args.output/'budget_ledger.json').exists():
        if marker.exists() and read(marker)['status'] != 'clean':
            raise RuntimeError('Abrupt VM loss: newer charged work may be missing. Baseline cap recovery audit required; no silent replay.')
        restore_snapshot(args.output, args.durable/'recovery')
    write(marker, dict(status='active', pid=os.getpid(), at=time.time()))
    exp.configure(p, 'dice')
    binary.bind()
    exp.base.PreparedPatients = PreparedPatients
    exp.base.evaluate = evaluate
    exp.base._load_pkl_dataframe = read
    exp.base.build_loss_and_optimizer = exp.ORIGINAL_LOSS_OPTIMIZER
    exp.trainer.training_loss = partial(binary.training_loss, policy=p['sdf_policy']['dice'])
    builder = exp.base.build_model
    def build(name, architecture):
        model = builder(name, architecture)
        write(args.output/'initialization.json', dict(model_state_sha256=exp.trainer.model_digest(model)))
        return model
    exp.base.build_model = build
    write(args.output/'runtime_recipe.json', dict(dataset='HFH', model=p['model_identity'],
        loss=p['training_loss'], augmentation='none', final_budget_primary=True,
        protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        recovery='Local charge-before-update; 180-second immutable snapshots; remote unclean resume fails closed'))
    writer = RecoveryWriter(args.output, args.durable/'recovery', interval=180)
    save = exp.base.save_checkpoint
    last_best = None
    def commit(path, state):
        nonlocal last_best
        if path.name == 'checkpoint_best.pt' and path.exists() and last_best == state['step']:
            return
        save(path, state)
        if path.name == 'checkpoint_best.pt':
            last_best = state['step']
        if path.name == 'checkpoint_latest.pt':
            writer.capture()
    exp.base.save_checkpoint = commit
    install = exp.trainer.install_signals
    def signals():
        install()
        write(args.output/'supervisor_ready.json', dict(pid=os.getpid()))
    exp.trainer.install_signals = signals
    started, clean = time.monotonic(), False
    try:
        exp.trainer.run(args)
        clean = True
    except SystemExit as error:
        clean = error.code == 75
        raise
    finally:
        writer.close(force=True)
        import torch
        write(args.output/'worker_timing.json', dict(seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0))
        publish_metadata(args.output, args.durable)
        if clean:
            write(marker, dict(status='clean', at=time.time()))


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    main()
