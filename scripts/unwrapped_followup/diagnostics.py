"""Read-only checkpoint ablations and localization of introduced errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients, GeometryCache, autocast
from scripts.depth_supervision.utils import atomic_json, digest
from scripts.unwrapped_swin.cache import AtlasCache
from scripts.unwrapped_swin.evaluate import evaluate
from .models import FollowupSwin, uncertainty_gate
from .train import atlas_protocol


class View(torch.nn.Module):
    def __init__(self, model, condition):
        super().__init__(); self.model=model; self.condition=condition

    def forward(self,x,atlas=None):
        return self.model(x,atlas,native_only=self.condition=='native_only',suppress_aux=self.condition=='zero_mapped')


def run(p, root, fraction, arm):
    directory=root/'diagnostics'/fraction/arm
    directory.mkdir(parents=True,exist_ok=True)
    prior=Path(p['prior_root'])/'runs'/fraction/arm
    record=json.loads((prior/'completion.json').read_text())
    checkpoint=prior/'best.pt'
    if digest(checkpoint)!=record['artifacts']['best.pt']:
        raise ValueError('Prior checkpoint changed')
    done=directory/'completion.json'
    if done.exists():
        d=json.loads(done.read_text())
        assert d['protocol_sha256']==p['identity_sha']
        assert d['checkpoint_sha256']==digest(checkpoint)
        for name,expected in d['artifacts'].items():assert digest(directory/name)==expected
        return d
    patients=Patients(_load_pkl_dataframe(p['pkl']),p['splits'][fraction]['val'],p['native_geometry'])
    cache=GeometryCache(root/'loss_geometry',p['geometry'])
    atlas=AtlasCache(root/'atlas'/fraction,atlas_protocol(p))
    device=torch.device('cuda')
    model=FollowupSwin(device,arm)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state['model']);del state
    results={}
    for condition in ('normal','zero_mapped','native_only'):
        results[condition]=evaluate(View(model,condition),patients,device,geometry_cache=cache,
            atlas_cache=atlas,detailed=True,predictions_path=directory/(condition+'.npz'))
        atomic_json(directory/(condition+'.json'),results[condition])
    with np.load(directory/'normal.npz') as final, np.load(directory/'native_only.npz') as native:
        bins={}; case_rows=[]
        model.eval()
        for item in patients.items:
            name=item['case_name'];target=item['label'][0].numpy(); fg=target>0
            a,b=native[name],final[name]
            created_fn=fg&(a>0)&(b==0)
            corrected_fn=fg&(a==0)&(b>0)
            surface=fg&~ndimage.binary_erosion(fg,structure=ndimage.generate_binary_structure(3,1))
            distance=ndimage.distance_transform_edt(~surface,sampling=np.linalg.norm(item['affine'][:3,:3],axis=0))
            with torch.no_grad(),autocast(device):
                native_logits=model(item['image'][None].to(device),native_only=True)
            entropy=uncertainty_gate(native_logits,'entropy')[0,0].cpu().numpy()
            depth=cache.get(item,fields=False)['bin_ids']
            row=dict(case=name,introduced_missed=int(created_fn.sum()),recovered_missed=int(corrected_fn.sum()),
                     introduced_missed_within_2mm=int((created_fn&(distance<=2)).sum()),
                     introduced_missed_beyond_2mm=int((created_fn&(distance>2)).sum()),
                     introduced_missed_native_entropy_mean=float(entropy[created_fn].mean()) if created_fn.any() else None)
            case_rows.append(row)
            for k in np.unique(depth[created_fn|corrected_fn]):
                sel=depth==k
                v=bins.setdefault(str(int(k)),dict(introduced_missed=0,recovered_missed=0))
                v['introduced_missed']+=int((created_fn&sel).sum());v['recovered_missed']+=int((corrected_fn&sel).sum())
    audit=dict(reference='Native-only path of the same jointly trained checkpoint; not an independently trained baseline.',
               foreground_recovery_is_not_necessarily_correct_AP_label=True,cases=case_rows,depth_bands=bins,
               normal_dice_delta_from_saved=results['normal']['summary']['macro_dice']-record['validation']['macro_dice'])
    if abs(audit['normal_dice_delta_from_saved'])>1e-4:
        raise ValueError('Reloaded checkpoint differs materially from saved validation')
    atomic_json(directory/'introduced_errors.json',audit)
    names=[condition+suffix for condition in ('normal','zero_mapped','native_only') for suffix in ('.json','.npz')]+['introduced_errors.json']
    d=dict(protocol_sha256=p['identity_sha'],checkpoint_sha256=digest(checkpoint),
           summaries={k:v['summary'] for k,v in results.items()},artifacts={n:digest(directory/n) for n in names})
    atomic_json(done,d)
    print(json.dumps(d),flush=True)
    return d


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--fraction',required=True);parser.add_argument('--arm',required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    run(json.loads((args.root/'protocol.json').read_text()),args.root,args.fraction,args.arm)
