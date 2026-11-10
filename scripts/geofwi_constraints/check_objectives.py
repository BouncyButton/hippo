"""Zero-optimizer-update checks for the extended objective and XY geometry."""
import json
from pathlib import Path
import sys
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(os.environ.get('HIPPO_OBJECTIVE_CHECK_STAGE', ROOT / "experiments/geofwi3d_constraints_3fold_3seed_20261020"))
bundle = STAGE / "bundles" / sys.argv[1]
sys.path[:0] = [str(bundle), str(bundle/"source"), str(bundle/"source/scripts"),
                str(bundle/"source/baselines/nnUNet")]
import worker
import torch
assert Path(worker.__file__).resolve() == (bundle/"worker.py").resolve()
from scripts.adni_constraints import objectives as binary
from scripts.cross_dataset_campaign.common import ARMS
from scripts.cross_dataset_campaign import conventional as exp
from functools import partial
assert Path(exp.__file__).resolve().is_relative_to(bundle.resolve())

torch.set_num_threads(2)
model = "unetr_vit_b16" if sys.argv[1]=="unetr" else sys.argv[2]
p = json.loads((STAGE/f"protocols/GeoFWI3D_{model}_dice_bands_edge_sdf_depth_f0s0/protocol.json").read_text())
output = STAGE / "checks" / model
output.mkdir(parents=True, exist_ok=True)
worker.configure(p, output)
if model == "unetr_vit_b16":
    import constraint_objective as objective
    loss_fn, cal_fn = objective.training_loss, objective.calibration_case
else:
    loss_fn, cal_fn = binary.training_loss, binary.calibration_case
torch.manual_seed(18)
coords = torch.meshgrid(*(torch.arange(96) for _ in range(3)), indexing="ij")
y = (((coords[0]-40)/24)**2 + ((coords[1]-43)/21)**2 + ((coords[2]-51)/35)**2 < 1)[None,None].long()
sdf = binary.signed_distance_maps(y)
depth = binary.geometry_weights(y, np.eye(4), p["depth_geometry"])
criterion = None
if model == "nnunet":
    # Only the native criterion is needed; no model forward or optimizer update.
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    weights = np.array([1., .5, .25, 0.])
    criterion = DeepSupervisionWrapper(DC_and_CE_loss(
        {"batch_dice": True,"smooth":1e-5,"do_bg":False,"ddp":False}, {},weight_ce=1,weight_dice=1,
        ignore_label=None,dice_class=MemoryEfficientSoftDiceLoss),weights/weights.sum())

records = []
for arm in ARMS[1:]:
    z = torch.randn(1,2,96,96,96,requires_grad=True)
    outputs = [z] + [torch.nn.functional.interpolate(z,size=(n,n,n),mode="trilinear",align_corners=False)
                     for n in (48,24,12)] if criterion is not None else z
    row = cal_fn(outputs,y,arm,sdf,criterion,depth)
    selected, statistics = exp.trainer.select_weights([row],arm)
    assert set(selected)==set(arm.split("_")[1:]) and all(np.isfinite(v) and v>0 for v in selected.values())
    value = loss_fn(outputs,y,arm,selected,1.,sdf,criterion,depth,1.,policy=p["sdf_policy"][arm])
    gradient = torch.autograd.grad(value,z)[0]
    assert torch.isfinite(value) and torch.isfinite(gradient).all() and gradient.square().sum()>0
    records.append(dict(arm=arm,loss=float(value),weights=selected))

if model == "unetr_vit_b16":
    from xy_augmentation import augment, transform_id
    image = torch.stack([c.float() for c in coords])[None]
    seen = set()
    for update in range(100):
        idx = transform_id(0,update)
        if idx in seen: continue
        seen.add(idx)
        xx = torch.cat([augment(image[:,i:i+1],y,0,update)[0] for i in range(3)],dim=1)
        _, yy = augment(image[:,:1],y,0,update)
        affine = objective.xy_affine(np.eye(4),0,update,96)
        predicted = np.einsum("ij,jxyz->ixyz",affine[:3,:3],image[0].numpy()) + affine[:3,3,None,None,None]
        assert np.array_equal(predicted,xx[0].numpy()), idx
        actual_depth = binary.geometry_weights(yy,affine,p["depth_geometry"])
        expected_depth,_ = augment(depth[None,None],y,0,update)
        assert torch.allclose(actual_depth,expected_depth[0,0],atol=1e-8,rtol=1e-4),idx
    assert len(seen)==8
    import rescue_objective
    for arm in ("dice","dice_bands"):
        z=torch.randn(1,2,96,96,96,requires_grad=True)
        args=(z,y,arm,{"bands":.04} if arm=="dice_bands" else {},.7)
        a=rescue_objective.training_loss(*args)
        b=objective.training_loss(*args)
        assert torch.equal(a,b)
        assert torch.equal(torch.autograd.grad(a,z)[0],torch.autograd.grad(b,z)[0])
    records.append(dict(xy_coordinate_and_depth_equivariance=8,exact_dice_bands_loss_and_gradient=True))
worker.io.write(output/"objective_checks.json",dict(status="passed",model=model,optimizer_updates=0,checks=records))
print(json.dumps(dict(status="passed",model=model,arms=13,optimizer_updates=0)))
