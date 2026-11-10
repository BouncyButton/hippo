"""CPU-only value, gradient, support and differential arithmetic audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/geofwi3d_unetr_lr5_frame_bands_20261019'
PARENT = ROOT / 'reports/geofwi3d_unetr_lr3_bands_20261019/bundle'
BUNDLE = OUT / 'bundle'
sys.path[:0] = [str(BUNDLE), str(ROOT), str(ROOT / 'scripts'), str(Path(__file__).parent)]
import numpy as np
import nibabel as nib
import torch
import rescue_objective as recipe
from profile_ce1_bands import full, supplement
from scripts.adni_constraints.audit_correction import correct_row


def load(folder, name):
    path = folder / 'source/thesis/new_constraints/bands/outer_boundary.py'
    spec = importlib.util.spec_from_file_location('thesis.new_constraints.bands.' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module, labels, calibration):
    recipe.bands = lambda: module.OuterBoundaryBandLoss(
        foreground_class_ids=(1,), complement_class_ids=(0,))
    z = torch.randn(1, 2, 96, 96, 96, requires_grad=True)
    operator = full.audit.reference.OperatorAudit()
    operator.stage = 'calibration_logit_gradients' if calibration else 'loss_and_logit_backward'
    with operator, full.audit.FlopCounterMode(display=False) as counter:
        if calibration:
            recipe.calibration_case(z, labels, 'dice_bands')
        else:
            recipe.training_loss(z, labels, 'dice_bands', {'bands': .04}, 1.).backward()
    return correct_row(dict(flops=counter.get_total_flops() + sum(operator.supplement.values()),
        supplementary_per_stage=operator.supplement, operator_audit=operator.ops))


def main():
    assert torch.__version__.startswith('2.2.2') and full.audit.monai.__version__ == '1.6.0'
    torch.set_num_threads(2)
    torch.manual_seed(0)
    full.audit.reference.supplementary_flops = supplement
    old, new = load(PARENT, 'audit_old'), load(BUNDLE, 'audit_frame2')
    mask = torch.zeros(1, 1, 96, 96, 96, dtype=torch.bool)
    mask[:, :, :65, 20:70, 24:80] = True
    interior = torch.zeros_like(mask)
    interior[:, :, 2:-2, 2:-2, 2:-2] = True
    old_support = old.build_boundary_bands(mask)
    support = new.build_boundary_bands(mask)
    assert all(torch.equal(a & interior, b) for a, b in zip(old_support, support))
    loss_fn = new.OuterBoundaryBandLoss(foreground_class_ids=(1,), complement_class_ids=(0,))
    z = torch.randn(1, 2, 96, 96, 96, requires_grad=True)
    loss = loss_fn(z, mask.long()).loss
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z[:, 1:2]-z[:, 0:1], mask.float(), reduction='none')
    reference = .5 * sum((bce*s).sum()/(s.sum()+1e-6) for s in support)
    assert torch.allclose(loss, reference, atol=1e-6, rtol=1e-6)
    gradient, = torch.autograd.grad(loss, z, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference, z)
    assert torch.allclose(gradient, reference_gradient, atol=1e-10, rtol=1e-5)
    assert torch.count_nonzero(gradient.masked_select(~interior.expand_as(z))) == 0
    centered = torch.zeros_like(mask); centered[:, :, 20:70, 20:70, 24:80] = True
    assert all(torch.equal(a,b) for a,b in zip(old.build_boundary_bands(centered),new.build_boundary_bands(centered)))
    split = json.loads((BUNDLE/'splits.json').read_text())[0]
    supports = []
    for name in split['train']:
        labels = torch.from_numpy(np.asarray(nib.load(BUNDLE/'data/labelsTr'/f'{name}.nii.gz').dataobj).copy()).bool()[None,None]
        inner, outer = new.build_boundary_bands(labels)
        assert inner.any() and outer.any()
        assert not (inner & ~interior).any() and not (outer & ~interior).any()
        supports.append(dict(case=name,inner=int(inner.sum()),outer=int(outer.sum())))
    rows = []
    for geometry, labels in [('edge_touching',mask.long()), ('centered',centered.long())]:
        for calibration in (False,True):
            a,b = profile(old,labels,calibration),profile(new,labels,calibration)
            assert a['flops'] == b['flops']
            rows.append(dict(geometry=geometry,calibration=calibration,old=a,new=b,delta_flops=b['flops']-a['flops']))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(status='passed',frame_width=2,gradient_in_frame_is_zero=True,
        unchanged_centered_support=True,manual_binary_loss_and_gradient_match=True,
        valid_training_cases=supports,profiles=rows,profile_method='Differential native-96-cubed DiceCE+BANDS loss/logit-backward and calibration. Model forward/backward/AdamW unchanged; inherit original full graph costs plus measured zero delta. Fixed boolean mask operations excluded under established accounting.',
        parent_profile_sha256=digest(PARENT/'rescue_profile.json'),
        old_loss_sha256=digest(PARENT/'source/thesis/new_constraints/bands/outer_boundary.py'),
        new_loss_sha256=digest(BUNDLE/'source/thesis/new_constraints/bands/outer_boundary.py'),
        runtime=dict(torch=torch.__version__,monai=full.audit.monai.__version__),
        script_sha256=digest(Path(__file__)))
    (OUT/'frame_cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('profiles','valid_training_cases')},indent=2))


if __name__ == '__main__':
    main()
