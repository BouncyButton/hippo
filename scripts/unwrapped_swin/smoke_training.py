"""Exercise real optimization, checkpoint resume and completion on one train case."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients
from scripts.depth_supervision.utils import atomic_json
from . import train


def main():
    root = Path(__file__).resolve().parents[2]
    p = json.loads((root / 'docs/experiments/unwrapped_swin_20261006/protocol.json').read_text())
    p = copy.deepcopy(p)
    name = p['splits']['5pct']['train'][0]
    frame = _load_pkl_dataframe(str(root / 'datasets/Dataset101_MSD/msd_hippocampus_full.pkl'))
    patients = Patients(frame, [name], p['native_geometry'])
    p['splits']['5pct'] = dict(train=[name], val=[name])  # Deliberate smoke-only self-evaluation.
    p['max_epochs'] = 2; p['guide_epochs'] = 1
    p['early_stopping']['min_epochs'] = 2
    p['identity_sha'] = 'smoke-only-not-a-scientific-run'
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='hippo-unfold-smoke-') as temporary:
        output = Path(temporary)
        with patch.object(train, '_load_pkl_dataframe', return_value=None), patch.object(train, 'Patients', return_value=patients):
            train.run(p, output, '5pct', 'dice', stop_after=1, device_override='cpu')
            run_dir = output / 'runs/5pct/dice'
            first = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=False)
            assert first['updates'] == 1 and len(first['history']) == 1
            # Exercise recovery of an interruption between latest and best publication.
            (run_dir / 'best.pt').unlink()
            p['native_training_identity_sha'] = p['identity_sha']
            p['identity_sha'] = 'repaired-campaign-same-native-training'
            result = train.run(p, output, '5pct', 'dice', device_override='cpu')
            assert result['completed_epochs'] == 2 and result['optimizer_updates'] == 2
            assert not (run_dir / 'latest.pt').exists()
            assert train.run(p, output, '5pct', 'dice', device_override='cpu') == result
            assert set(np.load(run_dir / 'fixed_epoch_guide_predictions.npz').files) == {name}
    record = dict(passed=True, resumed_updates=2, best_publication_recovery=True,
                  completion_idempotent=True, fixed_epoch_export=True,
                  resume_across_geometry_repair=True,
                  code_sha256=__import__('hashlib').sha256(Path(train.__file__).read_bytes()).hexdigest())
    atomic_json(root / 'evaluation_output/unwrapped_swin_campaign/training_smoke.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
