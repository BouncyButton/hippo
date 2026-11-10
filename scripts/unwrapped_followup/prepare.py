"""Freeze the three authorized follow-up questions and their matched controls."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/experiments/unwrapped_followup_20261007'
NAME = 'unwrapped_followup_fold0_20261007_01'
PRIOR = '/mnt/beegfsstudents/home/3160552/unwrapped_swin_fold0_seed0_20261006_02'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entries():
    result = []
    pairs = {'5pct': ['plane_position','unfold_position'], 'full': ['plane_attention','unfold_attention']}
    for fraction, arms in pairs.items():
        def add(group, seed, arm, objective, gate='none'):
            result.append(dict(id=f'{fraction}_{group}_s{seed}_{arm}_{gate}', group=group,
                               fraction=fraction, seed=seed, arm=arm, objective=objective, gate=gate))
        for arm in arms:
            add('dice_ablation',0,arm,'dice')
        for seed in (1,2):
            for arm in ['depth',*arms]:
                add('replication',seed,arm,'depth')
        for arm in arms:
            for gate in ('entropy','mean_entropy'):
                add('uncertainty',0,arm,'depth',gate)
    return result, pairs


def build():
    previous_path=ROOT/'docs/experiments/unwrapped_swin_20261006_repair/protocol.json'
    old=json.loads(previous_path.read_text())
    keep=['dataset','fold','max_epochs','baseline_loss','model','feature_size','spatial_size','batch_size',
          'drop_rate','activation_checkpointing','pretrained','training_augmentation','precision',
          'learning_rate','weight_decay','step_size','lr_gamma','early_stopping','selection','geometry',
          'depth_objective','plane','required_monai','splits','low_data_fraction_actual','pkl_sha256',
          'uncompressed_pickle_sha256','native_geometry','pkl','fixed_depth_coefficients','required_dependencies']
    p={key:copy.deepcopy(old[key]) for key in keep}
    runs,pairs=entries()
    p.update(name=NAME, prior_root=PRIOR, prior_protocol_sha256=sha(previous_path),
             atlas_protocol_sha256=old['identity_sha'], runs=runs, main_runs=len(runs), pairs=pairs,
             scientific_questions=['Remove depth loss while keeping the branch architecture fixed.',
                'Replicate the selected matched pairs with segmentor seeds 1 and 2 and native depth-only controls.',
                'Test spatial uncertainty residual gates against ordinary fusion and a mean-entropy gate control.'],
             preparation='Reuse the completed seed-0 out-of-fold guide geometry, atlas files and depth coefficients; no new guide fits or completed seed-0 reruns.',
             seed_scope='Only segmentor initialization and case order vary. Frozen seed-0 guide geometry and coefficients are shared across seeds, so this is conditional segmentor replication.',
             gate_definition='Detached native-path softmax entropy divided by log(3), multiplying the final 24-channel fusion residual before the shared output head. No extra parameters or loss.',
             gate_control='The same detached entropy averaged over the whole native input grid and broadcast to every voxel. This controls average residual strength, not local weighting.',
             gate_caveat='The native-only path shares the final trained classifier; entropy is a confidence proxy, not calibrated uncertainty from an independently supervised head.',
             diagnostics='Before training, evaluate the four selected seed-0 branches normally, with mapped features zeroed but support/fusion retained, and with only their native path. Compare introduced missing tissue by GT depth and distance from GT surface; annotations are diagnostic only.',
             permitted_losses=['Dice','Dice + original frozen depth-balanced CE'],
             resources=dict(partition='stud',gpu='4g.40gb:1',cpus=4,memory='24G',time='23:50:00'),
             minimum_free_gib=6.,
             storage_budget='24 best checkpoints about 1.8 GiB; one active optimizer checkpoint and atomic copies under 0.7 GiB; prediction/metric/diagnostic headroom under 1.5 GiB. Reserve 6 GiB. Shared frozen atlas files are not duplicated.')
    sources={}
    for name,expected in old['source_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Previous implementation changed: '+name)
        sources[name]=expected
    for path in sorted((ROOT/'scripts/unwrapped_followup').glob('*')):
        if path.suffix not in ('.py','.sh'):continue
        if path.suffix=='.py':ast.parse(path.read_text(),str(path))
        sources[str(path.relative_to(ROOT))]=sha(path)
    p['source_sha256']=sources
    p['identity_sha']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    return p


if __name__=='__main__':
    p=build()
    print(json.dumps(dict(protocol=str(DOC/'protocol.json'),identity=p['identity_sha'],runs=p['main_runs'])))
