"""Run the frozen trainer with storage hooks and exact portable CPU initialization."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from runtime import RecoveryWriter, read, write, install_initialization
from scripts.cross_dataset_campaign import msd_mixed_losses as scientific


def main():
    install_initialization(Path(__file__).parent, scientific.experiment)
    if '--benchmark' in sys.argv:
        sys.argv.remove('--benchmark')
        from scripts.cross_dataset_campaign import benchmark_msd_runtime
        benchmark_msd_runtime.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    args, _ = parser.parse_known_args()
    p = read(args.protocol_dir/'protocol.json')
    if p['augmentation'] != 'none':
        raise ValueError('Augmentation is forbidden')
    durable = Path(os.environ['HIPPO_COLAB_RECOVERY'])
    if (durable/'charged_ledger.json').exists():
        local = args.output/'budget_ledger.json'
        if not local.exists() or read(local) != read(durable/'charged_ledger.json'):
            raise RuntimeError('Durable and local compute ledgers differ; budget audit required before resume')
    writer = RecoveryWriter(args.output, durable)
    base = scientific.experiment.base
    original_json, original_save = base.atomic_json, base.save_checkpoint

    def atomic_json(path, value):
        if Path(path).name == 'budget_ledger.json':
            # A durable charge precedes the optimizer update, including on VM loss.
            write(durable/'charged_ledger.json', value)
            if read(durable/'charged_ledger.json') != value:
                raise IOError('Drive ledger readback failed')
        return original_json(path, value)

    def save_checkpoint(path, value):
        original_save(path, value)
        if Path(path).name == 'checkpoint_latest.pt':
            writer.capture()

    base.atomic_json, base.save_checkpoint = atomic_json, save_checkpoint
    coherent_exit = False
    try:
        scientific.main()
        coherent_exit = True
    except SystemExit as exc:
        coherent_exit = exc.code in (0, 75, None)
        raise
    finally:
        writer.close(force=coherent_exit)


if __name__ == '__main__':
    main()
