"""Audit auxiliary arithmetic at the logits; network backward is unchanged."""
import argparse
import gc
from functools import partial
import json
import torch
import monai
from torch.utils.flop_counter import FlopCounterMode
from scripts.unetr_subfields import profile as network, profile_baselines, profile_geo_recipe
from scripts.unetr_subfields import geo_recipe, log_squared_dice, baseline_runtime
from scripts.cross_dataset_campaign.common import ARMS, read, write, digest
from scripts.unetr_subfields.contracts import ROOT
from scripts.mni_constraints import objectives

REPORT = ROOT / 'reports/mni_constraints14_cap8_20261107'
RECIPES = ROOT / 'reports/mni_baseline_rescue_20261106/final_recipes.json'


def supplement(func, args, kwargs, result):
    name = str(func)
    # Native depth uses an unreduced unweighted CE, unlike the mean baseline CE.
    if name in ('aten.nll_loss_forward.default', 'aten.nll_loss2d_forward.default') and args[3] == 0:
        return args[1].numel(), 'ce_nll_none_estimate'
    if name in ('aten.nll_loss_backward.default', 'aten.nll_loss2d_backward.default') and args[4] == 0:
        return args[2].numel(), 'ce_nll_none_estimate'
    value, category = profile_geo_recipe.supplement(func, args, kwargs, result)
    if category != 'no_supplement':
        return value, category
    if name == 'aten.logsumexp.default':
        # Stable logsumexp: subtract max, exp, sum, log, add max.
        return 3 * args[0].numel() + result.numel(), 'grouped_logsumexp_estimate'
    return value, category


def target(geometry):
    x, y, z = torch.meshgrid(*[torch.arange(64)]*3, indexing='ij')
    fg = (((x-31.5)/9)**2 + ((y-31.5)/15)**2 + ((z-31.5)/7)**2 < 1) if geometry == 'ellipsoid' else ((x>=22)&(x<42)&(y>=16)&(y<48)&(z>=25)&(z<39))
    return (fg.long() * (1 + (y>=29).long() + (y>=36).long()))[None, None]


def audit_call(call):
    network.reference.supplementary_flops = supplement
    audit = network.reference.OperatorAudit()
    audit.stage = 'logit_objective'
    with audit, FlopCounterMode(display=False, custom_mapping={torch.ops.aten.convolution_backward: network.reference.conv_backward_flops}) as counter:
        result = call()
    return dict(flops=counter.get_total_flops()+sum(audit.supplement.values()),
                core_flops=counter.get_total_flops(), supplementary=audit.supplement, operators=audit.ops), result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--model', required=True)
    args = parser.parse_args(); identity = args.model
    assert torch.__version__.startswith('2.2.2') and monai.__version__ == '1.6.0'
    torch.set_num_threads(4)
    recipe = read(RECIPES)[identity]; protocol = recipe['protocol']
    origin = ROOT/'experiments'/recipe['stage']/'protocols/model'
    prior = read(origin/'profile.json')
    if identity == 'unetr_vit_b16':
        network.build_model, network.model_kwargs = geo_recipe.build_model, geo_recipe.model_kwargs
        network.training_loss, network.intermediate_supplement = geo_recipe.training_loss, profile_geo_recipe.supplement
        control = network.measure(4, (64,64,64))
    else:
        if identity == 'swinunetr':
            profile_baselines.training_loss = log_squared_dice.training_loss
        control = profile_baselines.measure(identity, 4)
    assert control['update_flops'] == prior['update_flops'], (identity, control['update_flops'], prior['update_flops'])
    assert control['parameters'] == prior['parameters']
    architecture = read(origin/'nnunet_architecture.json')
    model = (geo_recipe.build_model(4,(64,64,64)) if identity == 'unetr_vit_b16' else baseline_runtime.build_model(identity, architecture)).to('meta').eval()
    def forward():
        with torch.no_grad():
            return model(torch.zeros((1,1,64,64,64), device='meta'))
    forward_row, out = audit_call(forward)
    shapes = [list(o.shape) for o in out] if isinstance(out,(tuple,list)) else [list(out.shape)]
    del model, out; gc.collect()
    criterion = None
    if identity == 'nnunet':
        criterion, _ = baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER('nnunet', torch.nn.Linear(1,1), protocol)
    settings = read(ROOT/'docs/experiments/depth_swinunetr_20261006/protocol.json')['geometry']
    costs, rows = {}, []
    baseline_logit = {}
    for arm in (*ARMS, 'dice_edge', 'dice_edge_sdf'):
        measured = []
        for geometry in ('ellipsoid','box'):
            label = target(geometry)
            sdf = objectives.signed_distance_maps(label) if 'sdf' in arm else None
            depth = objectives.geometry_weights(label, torch.eye(4).numpy(), settings) if 'depth' in arm else None
            for calibration in ([False] if arm == 'dice' else [False,True]):
                torch.manual_seed(0)
                outputs = [torch.randn(shape, requires_grad=True) for shape in shapes]
                if identity != 'nnunet': outputs = outputs[0]
                def loss_call():
                    if calibration:
                        return objectives.calibration_case(outputs,label,arm,sdf,criterion,depth,identity=identity)
                    loss = objectives.training_loss(outputs,label,arm,dict(bands=.04,edge=.1,sdf=.1,depth=.1),1.,sdf,criterion,depth,.2,identity=identity)
                    loss.backward()
                    assert torch.isfinite(loss)
                    return float(loss)
                row, _ = audit_call(loss_call)
                row.update(arm=arm, geometry=geometry, calibration=calibration)
                if arm == 'dice': baseline_logit[geometry] = row['flops']
                row['composed_flops'] = (forward_row['flops']+row['flops'] if calibration else
                    prior['update_flops']+row['flops']-baseline_logit[geometry])
                rows.append(row); measured.append(row)
                del outputs
        training = {r['composed_flops'] for r in measured if not r['calibration']}
        calibration = {r['composed_flops'] for r in measured if r['calibration']}
        assert len(training) == 1 and len(calibration) <= 1, 'Geometry-dependent arithmetic'
        costs[arm] = dict(update_flops=training.pop(), calibration_flops=10*calibration.pop()+4096 if calibration else 0)
        print(identity, arm, costs[arm], flush=True)
        write(REPORT/'profiles'/f'{identity}.partial.json',dict(costs=costs,profiles=rows))
    write(REPORT/'profiles'/f'{identity}.json',dict(status='profiled',model=identity,parameters=prior['parameters'],costs=costs,
        baseline_control=control,baseline_profile_sha256=digest(origin/'profile.json'),forward_calibration=forward_row,
        profiles=rows,output_shapes=shapes,objective_sha256=digest(ROOT/'scripts/mni_constraints/objectives.py'),
        runtime=dict(torch=torch.__version__,monai=monai.__version__),
        accounting='Exact reproduced baseline update plus measured FP32 full objective/logit-backward minus baseline objective/logit-backward. Shared network backward and optimizer unchanged; one combined backward. Calibration is eval-mode forward plus detached-logit gradient Gram matrix for all10 training crops, with4096 host selection FLOPs. Historical dense MAC=2/scalar estimates; target preprocessing/validation/IO excluded. Native depth unreduced CE and four-class grouped logsumexp explicitly counted.'))


if __name__ == '__main__': main()
