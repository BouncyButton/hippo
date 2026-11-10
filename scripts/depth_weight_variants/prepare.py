"""Freeze four requested follow-up runs against the completed depth campaign."""
import copy
import json
from pathlib import Path

from .losses import ARMS, SETTINGS
from .utils import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'docs/experiments/depth_weight_variants_20261006'


def main():
    parent_path = ROOT / 'docs/experiments/depth_swinunetr_20261006/protocol.json'
    parent = json.loads(parent_path.read_text())
    p = copy.deepcopy(parent)
    p.update(name='depth_weight_variants_fold0_seed0_20261006', arms=list(ARMS),
             weight_variants=SETTINGS,
             parent_protocol_file_sha256=digest(parent_path),
             comparison_root='/mnt/beegfsstudents/home/3160552/depth_swinunetr_fold0_seed0_20261006_01',
             depth_objective='Dice + one depth auxiliary: either partial class-balanced CE or focal gamma=1 CE',
             permitted_auxiliaries=['depth CE with partial class balancing', 'depth focal CE, gamma=1'])
    p.pop('sdf_objective')
    p.pop('layer_sdf')
    sources = sorted((ROOT / 'scripts/depth_weight_variants').glob('*.py'))
    sources += [ROOT / 'baselines/swin_unetr/swin_unetr.py', ROOT / 'scripts/depth_weight_variants/run.sh']
    # Keep the data/model helper and geometry identical to the original campaign.
    assert digest(ROOT / 'baselines/swin_unetr/swin_unetr.py') == parent['source_sha256']['baselines/swin_unetr/swin_unetr.py']
    for name in ('geometry.py', 'utils.py'):
        assert digest(ROOT/'scripts/depth_weight_variants'/name) == parent['source_sha256']['scripts/depth_supervision/'+name]
    p['source_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    atomic_json(DIRECTORY/'protocol.json', p)
    print(json.dumps(dict(protocol=str(DIRECTORY/'protocol.json'), runs=4, arms=p['arms'],
                          full_train=208, low_train=10, validation=52)))


if __name__ == '__main__':
    main()
