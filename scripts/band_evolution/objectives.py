"""Independent scale/side objectives and physical regional band alternatives."""
import math
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from scripts.band_chamfer import objectives as old
from .geometry import Targets


def reduce_error(error,weight,spec):
    if not len(error):return error.sum()*0
    weight=weight/weight.sum()
    average=(error*weight).sum()
    fraction=spec.get('hard_fraction')
    if fraction:
        selected=error.topk(max(1,math.ceil(len(error)*fraction))).indices
        w=weight[selected];average=.5*(average+(error[selected]*w).sum()/w.sum())
    return average


def point_error(score,inside,spec):
    probability=score.sigmoid()
    error=1-probability if inside else probability
    penalty=spec.get('penalty','brier')
    if penalty=='bce':return F.softplus(-score if inside else score)
    error=F.relu(error-spec.get('tolerance',0.))
    if penalty=='huber':
        delta=spec.get('delta',.25)
        return torch.where(error<=delta,.5*error.square()/delta,error-.5*delta)
    return error.pow(spec.get('power',2.))


def independent_loss(score,pairs,spec):
    flat=score.reshape(-1)
    sides=[]
    inside_weight=spec.get('inside_weight',.5)
    for side,side_weight in (('inside',inside_weight),('outside',1-inside_weight)):
        scales=spec.get(side+'_weights',spec['weights'])
        values=[]
        for group,scale_weight in zip(pairs,scales):
            if not len(group[side]['index']) or scale_weight<=0:continue
            error=point_error(flat[group[side]['index']],side=='inside',spec)
            values.append((scale_weight,reduce_error(error,group[side]['weight'],spec)))
        if values and side_weight>0:
            sides.append((side_weight,sum(w*v for w,v in values)/sum(w for w,_ in values)))
    return sum(w*v for w,v in sides)/sum(w for w,_ in sides) if sides else score.sum()*0


def regional_band(score,spec,target):
    fg,valid,distance=[target[k] for k in ('foreground','valid','distance')]
    parts=[]
    for inside,width,side_weight in ((True,spec['inner'],spec.get('inside_weight',.5)),
                                   (False,spec['outer'],1-spec.get('inside_weight',.5))):
        error=point_error(score,inside,spec)
        mask=valid & (fg if inside else ~fg) & (distance<=width)
        classes=[mask]
        if spec.get('class_balanced'):
            classes=[mask & (target['nearest_class']==c) for c in (1,2)]
        class_values=[]
        for class_mask in classes:
            lower=0.;shell_values=[]
            shells=spec.get('shells',[width]);weights=spec.get('shell_weights',[1.])
            for upper,shell_weight in zip(shells,weights):
                use=class_mask & (distance>lower) & (distance<=upper);lower=upper
                if not use.any() or shell_weight<=0:continue
                weight=torch.ones_like(error[use])
                if spec.get('decay_mm'):weight=torch.exp(-distance[use]/spec['decay_mm'])
                shell_values.append((shell_weight,reduce_error(error[use],weight,spec)))
            if shell_values:class_values.append(sum(w*v for w,v in shell_values)/sum(w for w,_ in shell_values))
        if class_values:parts.append((side_weight,torch.stack(class_values).mean()))
    return sum(w*v for w,v in parts)/sum(w for w,_ in parts) if parts else score.sum()*0


def local_mass(probability,spec,target):
    """Compare local foreground fractions, normalizing away padding."""
    k=spec['kernel'];valid=target['valid'].float()[None,None]
    def pool(x):return F.avg_pool3d(x,k,stride=1,padding=k//2,count_include_pad=True)
    denominator=pool(valid).clamp_min(1e-6)
    predicted=pool(probability[None,None]*valid)/denominator
    truth=pool(target['foreground'].float()[None,None]*valid)/denominator
    error=(predicted-truth).square()[0,0]
    mask=target['valid'] & (target['distance']<=3)
    return old.weighted_sides(error,mask & target['foreground'],mask & ~target['foreground'])


def components(logits,labels,entry,target,active=True):
    values=dict(dice=DiceLoss(to_onehot_y=True,softmax=True)(logits.float(),labels))
    if not active:return values
    score=old.foreground_score(logits)
    values['bands']=(regional_band(score,entry['band'],target) if entry['band']['kind']=='regional'
                     else old.band_loss(logits,labels,entry['band'],target))
    if entry['pair']:
        values['pair']=(old.pair_loss(score.sigmoid(),target['legacy_pair'],entry['pair'])
                        if 'legacy_pair' in target else independent_loss(score,target['pairs'],entry['pair']))
    if entry['surface']:
        values['surface']=(local_mass(score.sigmoid(),entry['surface'],target) if entry['surface']['kind']=='local_mass'
                           else old.surface_loss(score.sigmoid(),entry['surface'],target))
    return values


def select_weights(rows,entry,settings):
    result=old.select_weights(rows,entry,settings)
    budget=entry.get('extra_budget',1.)
    if budget!=1.:
        adjusted=dict(settings,target_ratio=settings['target_ratio']*budget)
        # Bands stay exactly as calibrated above; only the extra target changes.
        from scripts.compute_constraint_objectives import calibrated_weight
        gkey='pair:pair'
        b=result['coefficients']['bands']
        ratios=[]
        for row in rows:
            g=row['gradient_gram'];base=g['dice:dice']+2*b*g['dice:bands']+b*b*g['bands:bands']
            ratios.append(math.sqrt(g[gkey]/base))
        coefficient,stats=calibrated_weight(ratios,adjusted['target_ratio'],settings['cap_ratio'])
        result['coefficients']['pair']=coefficient
        result['statistics']['extra_total']=stats
    return result
