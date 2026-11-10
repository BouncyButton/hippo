"""Reuse the tested plane/unwrapped branches and read-only frozen atlas inputs."""
import json
from pathlib import Path
import numpy as np
from scripts.depth_supervision.utils import digest, atomic_json
from scripts.unwrapped_swin.models import DualViewSwin
from scripts.unwrapped_swin.evaluate import evaluate as evaluate_branch


class FrozenAtlas:
    """Read existing inputs only; this experiment never constructs new charts."""
    def __init__(self, root, protocol, fraction):
        self.directory = Path(root)/'atlas'/fraction
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        if not self.manifest['complete'] or self.manifest['protocol_sha256'] != protocol['atlas_protocol_sha256']:
            raise ValueError('Incomplete or mismatched frozen view geometry')
        self.verified = set()

    def get(self, item):
        path = self.directory/(item['case_name']+'.npz')
        if path.name not in self.verified:
            if digest(path) != self.manifest['artifacts'][path.name]:
                raise ValueError('Frozen atlas changed: '+str(path))
            self.verified.add(path.name)
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}


def freeze_branch_controls(root, protocol):
    """Bind already completed Dice-only branch controls after predecessor ends."""
    files = {}
    for fraction, arms in protocol['view_pairs'].items():
        for arm in arms:
            entry_id = f'{fraction}_dice_ablation_s0_{arm}_none'
            folder = Path(protocol['branch_baseline_root'])/'runs'/entry_id
            path = folder/'completion.json'
            record = json.loads(path.read_text())
            identity = record['identity']
            assert identity['protocol_sha256'] == protocol['branch_baseline_identity']
            assert identity['run']['objective'] == 'dice' and identity['run']['seed'] == 0
            assert identity['run']['arm'] == arm and identity['run']['gate'] == 'none'
            assert identity['train'] == protocol['splits'][fraction]['train']
            assert identity['val'] == protocol['splits'][fraction]['val']
            files[str(path)] = digest(path)
            for name, expected in record['artifacts'].items():
                if name == 'best.pt':
                    continue
                path = folder/name
                if digest(path) != expected:
                    raise ValueError('Completed branch control changed: '+str(path))
                files[str(path)] = expected
    destination = Path(root)/'branch_controls.json'
    value = dict(files=files, completed_controls_retrained=False)
    if destination.exists() and json.loads(destination.read_text()) != value:
        raise ValueError('Branch baseline references changed since training began')
    atomic_json(destination, value)
    return value
