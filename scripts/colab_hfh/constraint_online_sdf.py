"""Optional training-only SDF strength tracking on detached FP32 logits.

This is a separate recipe, not a change to the canonical SDF definition or
native model loss. The detached probe never differentiates through the model.
"""
import json
import math

PROBE_FLOPS_PER_LOGIT = 512
PROBE_SCALAR_ALLOWANCE = 65536


def probe_allowance():
    """Conservative arithmetic allowance for the fixed B1,C2,64-cube probe."""
    return PROBE_FLOPS_PER_LOGIT * 2 * 64**3 + PROBE_SCALAR_ALLOWANCE


def measure(logits, labels, sdf):
    """Measure native and unweighted SDF gradients without model/RNG changes."""
    import torch
    from monai.losses import DiceLoss
    from scripts.adni_constraints.objectives import sdf_loss
    z = logits.detach().float().requires_grad_(True)
    with torch.autocast(z.device.type, enabled=False):
        native = DiceLoss(to_onehot_y=True, softmax=True)(z, labels)
        auxiliary = sdf_loss(z, labels, sdf)
        native_gradient = torch.autograd.grad(native, z)[0]
        auxiliary_gradient = torch.autograd.grad(auxiliary, z)[0]
        native_rms = float(native_gradient.square().mean().sqrt())
        auxiliary_rms = float(auxiliary_gradient.square().mean().sqrt())
    if not all(math.isfinite(x) and x > 0 for x in (native_rms, auxiliary_rms)):
        raise ValueError('Degenerate online SDF probe')
    return native_rms, auxiliary_rms


def coefficient(native_rms, auxiliary_rms, anchor_weight, target, max_multiplier=64.):
    if not all(math.isfinite(x) and x > 0 for x in
               (native_rms, auxiliary_rms, anchor_weight, target, max_multiplier)):
        raise ValueError('Positive finite online SDF inputs required')
    requested = target * native_rms / auxiliary_rms
    weight = min(requested, anchor_weight * max_multiplier)
    return weight, weight * auxiliary_rms / native_rms, requested > weight


def validate_trace(output, protocol, schedule, anchor_weight):
    rows=[json.loads(line) for line in (output/'online_sdf_strength.jsonl').read_text().splitlines()]
    expected=list(range(schedule['sdf_prefix_updates']+1,schedule['max_steps']+1))
    if [r['charged_attempt'] for r in rows] != expected:
        raise ValueError('Online SDF trace has missing or replayed charged updates')
    for row in rows:
        weight,ratio,clipped=coefficient(row['native_rms'],row['unweighted_sdf_rms'],
            anchor_weight,protocol['online_sdf_gradient_ratio'],protocol['online_sdf_max_multiplier'])
        ramp=min(1.,(row['charged_attempt']-schedule['sdf_prefix_updates'])/schedule['sdf_warmup_updates'])
        values=dict(anchor_weight=anchor_weight,target=protocol['online_sdf_gradient_ratio'],
            effective_weight=weight,achieved_unramped_ratio=ratio,sdf_scale=ramp,
            achieved_ramped_ratio=ratio*ramp)
        if any(not math.isclose(row[k],v,rel_tol=1e-9,abs_tol=1e-12) for k,v in values.items()):
            raise ValueError('Online SDF strength differs from the declared training rule')
        if (row['multiplier_limited'] != clipped or row['probe_flops_allowance'] != probe_allowance()
                or schedule['online_sdf_probe_flops_per_active_update'] != probe_allowance()
                or ratio > protocol['online_sdf_gradient_ratio']+1e-9):
            raise ValueError('Online SDF cap or arithmetic accounting differs')
    return dict(active_updates=len(rows),target=protocol['online_sdf_gradient_ratio'],
        multiplier_limited_updates=sum(r['multiplier_limited'] for r in rows),
        minimum_unramped_ratio=min(r['achieved_unramped_ratio'] for r in rows),
        maximum_unramped_ratio=max(r['achieved_unramped_ratio'] for r in rows),
        probe_optimization_flops_allowance=len(rows)*probe_allowance(),
        accounting='Already included in optimization_flops; do not add twice')


class OnlineSDF:
    def __init__(self, original, protocol, output):
        if protocol['model_identity'] != 'swinunetr' or protocol['active_objective'] != 'dice_sdf':
            raise ValueError('Online SDF pilot is restricted to SwinUNETR, SDF alone')
        self.original, self.output = original, output
        self.target = protocol['online_sdf_gradient_ratio']
        self.max_multiplier = protocol['online_sdf_max_multiplier']

    def __call__(self, outputs, labels, arm, weights, scale, target_sdf=None,
                 criterion=None, depth_weights=None, sdf_scale=0.):
        if arm == 'dice':
            return self.original(outputs, labels, arm, weights, scale, target_sdf,
                                 criterion, depth_weights, sdf_scale)
        if arm != 'dice_sdf' or criterion is not None or scale != 1. or not 0 < sdf_scale <= 1.:
            raise ValueError('Online SDF pilot received an unexpected active loss')
        if tuple(outputs.shape)!=(1,2,64,64,64) or tuple(labels.shape)!=(1,1,64,64,64):
            raise ValueError('Online SDF arithmetic allowance requires the frozen HFH grid')
        native_rms, auxiliary_rms = measure(outputs, labels, target_sdf)
        value, ratio, clipped = coefficient(native_rms, auxiliary_rms, weights['sdf'],
                                            self.target, self.max_multiplier)
        ledger = json.loads((self.output/'budget_ledger.json').read_text())
        row = dict(charged_attempt=ledger['charged_attempts'], native_rms=native_rms,
            unweighted_sdf_rms=auxiliary_rms, anchor_weight=weights['sdf'], effective_weight=value,
            target=self.target, achieved_unramped_ratio=ratio, sdf_scale=sdf_scale,
            achieved_ramped_ratio=ratio*sdf_scale, multiplier_limited=clipped,
            probe_flops_allowance=probe_allowance())
        with (self.output/'online_sdf_strength.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        return self.original(outputs, labels, arm, {**weights, 'sdf': value}, scale,
                             target_sdf, criterion, depth_weights, sdf_scale)
