"""Validate frozen training annotations for constraint geometry on CPU."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/"experiments/geofwi3d_constraints_3fold_3seed_20261020"
bundle=STAGE/"bundles/unetr"
sys.path[:0]=[str(bundle),str(bundle/"source"),str(bundle/"source/scripts"),str(bundle/"source/baselines/nnUNet")]
import worker
from scripts.adni_constraints import objectives as binary
from scripts.cross_dataset_campaign import conventional as exp
splits=worker.io.read(STAGE/"splits.json")
names=sorted({c for s in splits for c in s["train"]})
data=exp.SeismicDataset(dict(dataset_root=str(STAGE/"data"),manifest=str(STAGE/"data/fold_0_manifest.json")),names)
torch.set_num_threads(2)
rows=[]
for item in data:
    labels=item["label"][None]
    points=np.argwhere(labels[0,0].numpy()>0).astype(float)
    centered=points-points.mean(0)
    values,axes=np.linalg.eigh(centered.T@centered/len(points))
    axis=axes[:,-1]
    fallback=np.linalg.norm(np.eye(3)[0]-axis[0]*axis)<.1
    band=binary.bands()(torch.zeros((1,2,96,96,96)),labels)
    assert bool(band.details["valid"].all()),item["case_name"]
    depth=binary.geometry_weights(labels,np.eye(4),dict(band_voxels=2.,roi_margin_voxels=8.,sparse_band_voxels=64))
    assert torch.isfinite(depth).all() and torch.isclose(depth.sum(),torch.tensor(1.))
    if fallback:
        from xy_augmentation import augment,transform_id
        from constraint_objective import xy_affine
        seen=set()
        for update in range(100):
            idx=transform_id(0,update)
            if idx in seen: continue
            seen.add(idx)
            expected,yy=augment(depth[None,None],labels,0,update)
            actual=binary.geometry_weights(yy,xy_affine(np.eye(4),0,update,96),
                         dict(band_voxels=2.,roi_margin_voxels=8.,sparse_band_voxels=64))
            assert torch.allclose(actual,expected[0,0],rtol=1e-4,atol=1e-8),(item["case_name"],idx)
        assert len(seen)==8
    rows.append(dict(case=item["case_name"],foreground_voxels=int(labels.sum()),
                     normalized_depth=True,corrected_bands_valid=True,depth_axis_fallback=bool(fallback)))
worker.io.write(STAGE/"checks/geometry.json",dict(status="passed",unique_training_cases=len(rows),rows=rows))
print(json.dumps(dict(status="passed",unique_training_cases=len(rows),
                      depth_axis_fallback_cases=[r["case"] for r in rows if r["depth_axis_fallback"]])))
