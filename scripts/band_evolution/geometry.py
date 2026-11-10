"""GT boundary samples with independent side eligibility and explicit reductions."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from scripts.band_chamfer.geometry import Targets as PreviousTargets, edge_grid, surface_points
from scripts.edge_ablation.objectives import valid_field


def move(value, device):
    """Transfer nested scale/side samples, including sequence-held tensors."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def pair_geometry(y, valid, spacing, spec):
    if not np.allclose(spacing, 1.):
        raise ValueError('Pair offsets require verified native 1 mm spacing')
    shape=np.asarray(y.shape)
    foreground=y>0
    groups=[dict(inside=[],outside=[]) for _ in spec['distances']]
    steps=spec.get('endpoint_steps', [[(int(d)-1)//2]*2 for d in spec['distances']])
    if len(steps)!=len(groups) or any(min(s)<0 for s in steps):
        raise ValueError('Invalid endpoint offsets')
    support=spec.get('support','all')
    common=(int(support.removeprefix('common'))-1)//2 if support.startswith('common') else None
    def eligibility(anchors,direction,offsets):
        okay=np.ones(len(anchors),bool)
        for offset in offsets:
            coords=anchors+offset*direction
            bounded=((coords>=0)&(coords<shape)).all(1)
            clipped=np.clip(coords,0,shape-1)
            okay &= bounded & valid[tuple(clipped.T)] & (foreground[tuple(clipped.T)] == (offset<=0))
        return okay
    for axis in range(3):
        for sign in (-1,1):
            direction=np.zeros(3,int);direction[axis]=sign
            anchors=np.argwhere(foreground&valid)
            anchors=anchors[eligibility(anchors,direction,(0,1))]
            if common is not None:
                anchors=anchors[eligibility(anchors,direction,range(-common,common+2))]
            for group,(inside_steps,outside_steps) in zip(groups,steps):
                a=eligibility(anchors,direction,range(-inside_steps,1))
                b=eligibility(anchors,direction,range(1,outside_steps+2))
                keep=(a,b) if support=='separate' else (a&b,a&b)
                for side,offset,okay in zip(('inside','outside'),(-inside_steps,1+outside_steps),keep):
                    base=anchors[okay]
                    points=base+offset*direction
                    group[side].append((np.ravel_multi_index(points.T,y.shape),
                        np.ravel_multi_index(base.T,y.shape),y[tuple(base.T)]))
    out=[]
    for group in groups:
        values={}
        for side in ('inside','outside'):
            index,anchors,classes=[np.concatenate([part[k] for part in group[side]]) for k in range(3)]
            weights=np.ones(len(index),np.float32)
            reduction=spec.get('reduction','pairs')
            if len(index) and reduction in ('unique_voxel','anchor'):
                keys=index if reduction=='unique_voxel' else anchors
                _,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
                weights=1/counts[inverse].astype(np.float32)
            elif len(index) and reduction=='class_balanced':
                for c in np.unique(classes):
                    use=classes==c;weights[use]=1/use.sum()
            elif reduction not in ('pairs','unique_voxel','anchor','class_balanced'):
                raise ValueError('Unknown reduction')
            if len(weights):weights/=weights.sum()
            values[side]=dict(index=torch.from_numpy(index),weight=torch.from_numpy(weights))
        out.append(values)
    return out


class Targets:
    def __init__(self,root,protocol,entry):
        self.protocol,self.entry=protocol,entry
        base=dict(entry,pair=None)
        if entry['band']['kind']=='regional':base['band']=dict(kind='canonical')
        self.delegate=PreviousTargets(root,protocol,base)
        self.cached={}

    def get(self,item,device):
        target=self.delegate.get(item,device)
        name=item['case_name']
        if name not in self.cached:
            labels=item['label'][0].long().numpy()
            valid=valid_field(labels.shape,self.protocol['native_geometry'][name]['native_shape'])
            extra={}
            spec=self.entry.get('pair')
            if spec:
                if spec.get('support')=='original':
                    from scripts.band_chamfer.geometry import pair_geometry as old_pairs
                    extra['legacy_pair']=old_pairs(labels,valid,np.ones(3),spec)
                else:
                    extra['pairs']=pair_geometry(labels,valid,np.ones(3),spec)
                self.delegate.stats[name]['pair_counts']=(
                    {str(d):int((extra['legacy_pair']['scale']==k).sum()) for k,d in enumerate(spec['distances'])}
                    if 'legacy_pair' in extra else {str(k):{side:len(g[side]['index']) for side in g}
                    for k,g in enumerate(extra['pairs'])})
            if self.entry['band'].get('class_balanced'):
                base=self.delegate.cached[name]
                grid=base['grid']
                flat=(labels>0).ravel()
                crossing=flat[grid['i'].numpy()]!=flat[grid['j'].numpy()]
                ii,jj=grid['i'].numpy()[crossing],grid['j'].numpy()[crossing]
                inside=np.where(flat[ii],ii,jj)
                nearest=cKDTree(base['gt'].numpy()).query(np.stack(np.indices(labels.shape),-1).reshape(-1,3),workers=1)[1]
                extra['nearest_class']=torch.from_numpy(labels.ravel()[inside][nearest].reshape(labels.shape).astype(np.int64))
            self.cached[name]=extra
        target.update(move(self.cached[name],device))
        return target

    def statistics(self):
        return self.delegate.statistics()
