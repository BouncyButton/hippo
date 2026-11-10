"""Anatomically restricted hard-example, neighborhood and A/P supervision."""
from copy import deepcopy
import math
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from scripts.band_evolution.objectives import independent_loss
from scripts.band_chamfer import objectives as previous
from scripts.compute_constraint_objectives import calibrated_weight


def point_error(score, inside, spec):
    probability = score.sigmoid()
    residual = 1 - probability if inside else probability
    kind = spec.get('penalty', 'brier')
    if kind == 'bce':
        return F.softplus(-score if inside else score)
    if kind == 'hybrid':
        a = spec.get('bce_share', .5)
        return (1-a)*residual.square() + a*F.softplus(-score if inside else score)
    residual = F.relu(residual-spec.get('tolerance', 0.))
    if kind == 'huber':
        delta = spec.get('delta', .25)
        return torch.where(residual <= delta, .5*residual.square()/delta, residual-.5*delta)
    return residual.pow(spec.get('power', 2.))


def reduce_error(error, weight, spec, ranking=None):
    """Keep the full mean; selection weights never backpropagate through ranking."""
    if not len(error):
        return error.sum()*0
    weight = weight/weight.sum()
    mean = (error*weight).sum()
    alpha = spec.get('hard_mix', .5)
    if spec.get('smooth_power') is not None:
        rank = error.detach() if ranking is None else ranking.detach()
        focus = (rank.clamp_min(1e-8)/rank.mean().clamp_min(1e-8)).pow(spec['smooth_power']).clamp_max(20.)
        w = weight*focus
        return (1-alpha)*mean + alpha*(error*w).sum()/w.sum()
    q = spec.get('hard_fraction')
    if not q:
        return mean
    rank = error.detach() if ranking is None else ranking.detach()
    selected = rank.topk(max(1, math.ceil(len(error)*q))).indices
    w = weight[selected]
    focused = (error[selected]*w).sum()/w.sum()
    return .5*(mean+focused) if alpha == .5 else (1-alpha)*mean + alpha*focused


def box_sum(value, kernel):
    """Separable box sum, including zero padding explicitly."""
    x = value[None, None]
    for axis in range(3):
        k = [1, 1, 1]; k[axis] = kernel
        padding = [0, 0, 0]; padding[axis] = kernel//2
        x = F.avg_pool3d(x, tuple(k), stride=1, padding=tuple(padding), count_include_pad=True)*kernel
    return x[0, 0]


def neighborhood_rank(error, mask, kernel):
    # Average errors, never signed residuals or predicted class fractions.
    support = mask.to(error.dtype)
    return box_sum(error.detach()*support, kernel)/box_sum(support, kernel).clamp_min(1)


def tile_focus(error, mask, weight, spec):
    tile = spec['tile']
    pads = [(tile-n % tile) % tile for n in error.shape]
    padding = (0, pads[2], 0, pads[1], 0, pads[0])
    def pack(x):
        x = F.pad(x, padding)
        d,h,w = x.shape
        return x.reshape(d//tile,tile,h//tile,tile,w//tile,tile).permute(0,2,4,1,3,5).reshape(-1,tile**3)
    e,m,w = pack(error),pack(mask),pack(weight*mask)
    counts = m.sum(1)
    indices = e.detach().masked_fill(~m, -torch.inf).argsort(1, descending=True)
    keep = torch.arange(e.shape[1],device=e.device)[None] < (counts*spec['hard_fraction']).ceil()[:,None]
    selected_weights = w.gather(1, indices)*keep
    selected_errors = e.gather(1, indices)
    means = (selected_errors*selected_weights).sum(1)/selected_weights.sum(1).clamp_min(1e-12)
    mass = w.sum(1)
    focused = (means*mass).sum()/mass.sum().clamp_min(1e-12)
    ordinary = (error[mask]*weight[mask]).sum()/weight[mask].sum()
    alpha = spec.get('hard_mix', .5)
    return (1-alpha)*ordinary + alpha*focused


def regional_band(score, spec, target):
    if spec.get('multiscale'):
        parts = []
        for width, share in spec['multiscale']:
            sub = dict(spec, inner=width, outer=width)
            sub.pop('multiscale')
            parts.append((share, regional_band(score, sub, target)))
        return sum(w*v for w,v in parts)/sum(w for w,_ in parts)
    fg, valid, distance = [target[k] for k in ('foreground','valid','distance')]
    values = []
    for inside, width, mass, name in ((True,spec['inner'],spec.get('inside_weight',.5),'inside'),
                                     (False,spec['outer'],1-spec.get('inside_weight',.5),'outside')):
        error = point_error(score, inside, spec)
        mask = valid & (fg if inside else ~fg) & (distance > 0) & (distance <= width)
        if not mask.any() or mass <= 0:
            continue
        side = dict(spec)
        if name+'_fraction' in spec:
            side['hard_fraction'] = spec[name+'_fraction']
        weights = torch.ones_like(error)
        if spec.get('decay_mm'):
            weights = torch.exp(-distance/spec['decay_mm'])
        if spec.get('tile'):
            value = tile_focus(error, mask, weights, side)
        elif spec.get('shells'):
            parts = []; lower = 0.
            for upper, share in zip(spec['shells'],spec['shell_weights']):
                use = mask & (distance > lower) & (distance <= upper); lower = upper
                if use.any():
                    parts.append((share,reduce_error(error[use],weights[use],side)))
            value = sum(w*v for w,v in parts)/sum(w for w,_ in parts)
        else:
            rank = None
            if spec.get('neighborhood'):
                ranks = [neighborhood_rank(error,mask,k) for k in spec['neighborhood']]
                rank = torch.stack(ranks).mean(0)
                share = spec.get('point_rank_share',0.)
                rank = (1-share)*rank + share*error.detach()
                rank = rank[mask]
            value = reduce_error(error[mask],weights[mask],side,rank)
        values.append((mass,value))
    return sum(w*v for w,v in values)/sum(w for w,_ in values) if values else score.sum()*0


def focused_ap(logits, spec, target):
    score = logits.float()[0,1]-logits.float()[0,2]
    valid = target['support'] & (target['distance'] <= spec.get('width_mm',float('inf')))
    values = []
    for anterior in (True,False):
        mask = valid & (target['anterior'] if anterior else ~target['anterior'])
        if mask.any():
            error = point_error(score,anterior,spec)[mask]
            values.append(reduce_error(error,torch.ones_like(error),spec))
    return torch.stack(values).mean() if values else score.sum()*0


def at_epoch(entry, epoch):
    schedule = entry.get('focus_schedule')
    if not schedule:
        return entry
    result = deepcopy(entry)
    t = min(1.,max(0.,(epoch-5)/(schedule['end_epoch']-5)))
    for key, start in schedule['start'].items():
        result['band'][key] = start + t*(entry['band'][key]-start)
    return result


def components(logits, labels, entry, target, active=True):
    values = dict(dice=DiceLoss(to_onehot_y=True,softmax=True)(logits.float(),labels))
    if not active or entry['band']['kind'] == 'none':
        return values
    score = previous.foreground_score(logits)
    values['bands'] = (regional_band(score,entry['band'],target) if entry['band']['kind']=='regional'
                       else previous.band_loss(logits,labels,entry['band'],target))
    if entry['pair']:
        values['pair'] = (previous.pair_loss(score.sigmoid(),target['legacy_pair'],entry['pair'])
                          if 'legacy_pair' in target else independent_loss(score,target['pairs'],entry['pair']))
    if entry['surface']:
        values['surface'] = previous.surface_loss(score.sigmoid(),entry['surface'],target)
    if entry.get('ap'):
        values['ap'] = focused_ap(logits,entry['ap'],target['ap'])
    return values


def select_weights(rows, entry, settings):
    if entry['band']['kind'] == 'none':
        return dict(coefficients={},statistics={},cases=rows,checkpoint_epoch=settings['prefix_epochs'])
    statistics = {}
    ratios = [math.sqrt(r['gradient_gram']['bands:bands']/r['gradient_gram']['dice:dice']) for r in rows]
    band,statistics['bands'] = calibrated_weight(ratios,settings['target_ratio']*entry.get('band_budget',1.),settings['cap_ratio'])
    weights = dict(bands=band)
    extras = [k for k in ('pair','surface','ap') if entry.get(k)]
    if extras:
        base = []
        for row in rows:
            g = row['gradient_gram']
            norm = g['dice:dice']+2*band*g['dice:bands']+band*band*g['bands:bands']
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('Degenerate calibration base')
            base.append(norm)
        normalized = {}
        ratio = settings['target_ratio']*entry.get('extra_budget',1.)
        for key in extras:
            ratios = [math.sqrt(r['gradient_gram'][key+':'+key]/b) for r,b in zip(rows,base)]
            normalized[key],statistics[key+'_initial'] = calibrated_weight(ratios,ratio,settings['cap_ratio'])
            if entry.get('extra_shares'):
                normalized[key] *= entry['extra_shares'][key]
        if len(extras)==1:
            weights.update(normalized);statistics['extra_total']=statistics[extras[0]+'_initial']
        else:
            ratios=[]
            for row,b in zip(rows,base):
                g=row['gradient_gram']
                norm=sum(normalized[a]*normalized[c]*g[a+':'+c] for a in extras for c in extras)
                ratios.append(math.sqrt(max(norm,0.)/b))
            joint,statistics['extra_total']=calibrated_weight(ratios,ratio,settings['cap_ratio'])
            weights.update({k:v*joint for k,v in normalized.items()})
    return dict(coefficients=weights,statistics=statistics,cases=rows,
                basis='Training-only gradient calibration; fixed band budget then one joint extra budget',
                checkpoint_epoch=settings['prefix_epochs'])
