"""Adapt the frozen ADNI native-loss trainer to four-class MNI volumes."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

from scripts.colab_mni_medsam3.data import Volumes
from scripts.cross_dataset_campaign import foundation as original
from scripts.cross_dataset_campaign import foundation_msd as matched
from scripts.cross_dataset_campaign.common import digest, read, write


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.uploading')
    shutil.copyfile(source, temp)
    if digest(source) != digest(temp):
        raise IOError('Recovery copy differs: ' + str(source))
    temp.replace(destination)


class SupportSlabs(original.SupportSlabs):
    def __init__(self, volumes, protocol, processor):
        import torch
        from scripts.medsam3_supervised_constraints import VolumeSupport
        super().__init__(volumes, protocol, processor)
        # The older binary/three-class dispatcher otherwise includes class 1 only.
        for case in protocol['train_cases']:
            _, label = volumes.load(case)
            labels = torch.from_numpy(label.transpose(2, 1, 0).copy())
            self.supports[case] = {c: VolumeSupport.build(labels == c) for c in range(4)}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    durable = Path(os.environ['MNI_DURABLE'])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    write(output/'environment.json', dict(python=sys.version,
        packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        training='Frozen ADNI native detection/presence/mask objective; three MNI foreground prompts',
        source_manifest_sha256=os.environ['MNI_MANIFEST_SHA']))
    original.Volumes = Volumes
    original.SupportSlabs = SupportSlabs
    original.load_foundation = matched.load_foundation
    original.calibrate = matched.calibrate
    original.evaluate = matched.evaluate
    matched.configure_retained_backward()
    save = original.save_checkpoint

    def durable_checkpoint(path, state):
        save(path, state)
        # Store the exact RNG/optimizer/sampling state, and a matching history.
        if state['step'] % 25 == 0 or state['step'] == 1:
            copy_verified(path, durable/path.name)
            write(durable/'training_history.json', state['history'])
            for name in ('protocol.json', 'calibration.json', 'environment.json',
                         'training_inventory.json', 'slab_schedule.json', 'runtime_validation.json'):
                if (output/name).exists():
                    copy_verified(output/name, durable/name)

    original.save_checkpoint = durable_checkpoint
    try:
        original.main()
    finally:
        # A requested clean stop and completed evaluation both get a full copy.
        for path in output.iterdir():
            if path.is_file() and path.suffix in ('.json', '.pt'):
                copy_verified(path, durable/path.name)


if __name__ == '__main__':
    main()
