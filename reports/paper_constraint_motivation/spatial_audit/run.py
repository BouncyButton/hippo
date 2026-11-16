"""Supplementary few-label 3D U-Net diagnostic. Execute from the project root."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.nets import BasicUNet

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spatial_metrics import counts, FACE

OUT = Path(__file__).resolve().parent
NAMES = ['MSD', 'MNI', 'ADNI', 'HFH']
HFH = ROOT / 'reports/hfh_colab_exploration_20261111/baseline_v2/bundle/data'
MANIFEST = json.loads((ROOT / 'reports/full_data_baselines_20261115/manifest.json').read_text())


def recipe(name, fold):
    r = next(v for v in MANIFEST.values() if v['dataset'] == name and
             v['model'] == 'basic_unet16' and v['partition'] == fold)
    selected = []
    for key in ['archived_protocol', 'archived_validation_source']:
        desc = r[key]
        raw = (ROOT / desc['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == desc['sha256']
        selected.append(json.loads(raw))
    return selected[0], selected[1][fold]


def load(name, case):
    if name == 'HFH':
        with np.load(HFH / (case + '.npz')) as f:
            x, y = f['image'], f['label']
    else:
        d = next((ROOT / 'datasets').glob('Dataset*_' + name))
        x = np.asarray(nib.load(d / 'imagesTr' / (case + '_0000.nii.gz')).dataobj)
        y = np.asarray(nib.load(d / 'labelsTr' / (case + '.nii.gz')).dataobj)
    x = x.astype('float32')
    y = y.astype('uint8')
    assert x.shape == y.shape and max(x.shape) <= 64 and np.isfinite(x).all()
    nz = x != 0
    x[nz] = (x[nz] - x[nz].mean()) / max(x[nz].std(), 1e-8)
    pad = [((64 - n)//2, (65 - n)//2) for n in x.shape]
    native = tuple(slice(a, a + n) for (a, _), n in zip(pad, x.shape))
    return torch.from_numpy(np.pad(x, pad))[None, None], torch.from_numpy(np.pad(y, pad))[None, None], native


def train(name, fold, seed, checkpoints):
    protocol, split = recipe(name, fold)
    assert not set(split['train']) & set(split['val'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    classes = protocol['num_classes']
    net = BasicUNet(spatial_dims=3, in_channels=1, out_channels=classes,
                    features=(16,16,32,64,128,16), upsample='deconv')
    loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
    opt = torch.optim.AdamW(net.parameters(), lr=protocol['initial_lrs']['unet'],
                            weight_decay=protocol['adamw_weight_decay'])
    training = [load(name, c) for c in split['train']]
    order = []
    for t in range(1,max(checkpoints)+1):
        if not order:
            order = list(rng.permutation(len(training)))
        x,y,_ = training[order.pop()]
        net.train()
        opt.zero_grad()
        objective = loss(net(x), y.long())
        objective.backward()
        opt.step()
        if t not in checkpoints:
            continue
        rows=[]
        net.eval()
        with torch.inference_mode():
            for case in split['val']:
                x,y,crop = load(name,case)
                labels=y[0,0].numpy()[crop]
                pred=net(x).argmax(1)[0].numpy()[crop]
                affine=np.array(protocol['native_geometry'][case]['model_affine'])
                rows.append(dict(dataset=name,fold=fold,seed=seed,step=t,case=case,
                                 **counts(labels,pred,affine,classes)))
        dest=OUT / f'{name}_f{fold}_s{seed}_u{t}.csv'
        temp=dest.with_suffix('.tmp')
        with temp.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(dest)
        print(dest.name,len(rows),'Dice',np.mean([r['dice'] for r in rows]),flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--seed',type=int,default=1741)
    parser.add_argument('--steps',nargs='+',type=int,default=[5,50])
    parser.add_argument('--datasets',nargs='+',default=NAMES)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    for fold in args.folds:
        for name in args.datasets:
            if not all((OUT/f'{name}_f{fold}_s{args.seed}_u{s}.csv').exists() for s in args.steps):
                train(name,fold,args.seed,args.steps)
