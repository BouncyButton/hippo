"""Read-only decomposition of saved step-head masks on the pilot DEV cohort."""

import json
import sys
from pathlib import Path

import numpy as np

from baselines.swin_unetr.swin_unetr import (
    _build_monai_dataset_from_pkl,
    _load_pkl_dataframe,
)


def dice(pred, truth):
    return 2.0 * np.count_nonzero(pred & truth) / (np.count_nonzero(pred) + np.count_nonzero(truth))


root = Path(sys.argv[1])
pkl = Path(sys.argv[2])
names = set(json.loads((root / "split.json").read_text())["outer_dev"])
ds = _build_monai_dataset_from_pkl(_load_pkl_dataframe(pkl), "MSD", 3, spatial_size=(64, 64, 64))
keys = ("seg", "head", "oracle_interval", "true_presence_head_edges", "head_presence_true_edges", "seg_presence_head_edges", "head_presence_seg_edges")
thresholds = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
results = {a: {key: [] for key in keys} | {f"threshold_{v:.1f}": [] for v in thresholds} for a in "CD"}
totals = {"gt_voxels": 0, "oracle_extra": 0, "columns_with_gaps": 0, "occupied_columns": 0}
for i, record in enumerate(ds.data):
    name = record["case_name"]
    if name not in names:
        continue
    truth = ds[i]["label"][0].numpy() > 0
    occ = truth.any(-1)
    z = np.arange(truth.shape[-1])[None, None, :]
    low = np.where(truth, z, truth.shape[-1]).min(-1)
    high = np.where(truth, z, -1).max(-1)
    oracle = occ[..., None] & (z >= low[..., None]) & (z <= high[..., None])
    totals["gt_voxels"] += int(truth.sum())
    totals["oracle_extra"] += int((oracle & ~truth).sum())
    totals["columns_with_gaps"] += int(((oracle & ~truth).any(-1) & occ).sum())
    totals["occupied_columns"] += int(occ.sum())
    for arm in "CD":
        with np.load(root / arm / "probabilities" / f"{name}.npz") as saved:
            seg = saved["hard_mask"] > 0
            head = saved["head_mask"] > 0
            pres_prob = saved["head_presence_probability"].astype(np.float32)
            pres = pres_prob >= 0.5
            edge = saved["head_edge_probability"].astype(np.float32)
        means = (edge * np.arange(edge.shape[-1])).sum(-1)
        hlow = np.rint(np.minimum(means[0], means[1])).astype(int)
        hhigh = np.rint(np.maximum(means[0], means[1])).astype(int)
        head_interval = (z >= hlow[..., None]) & (z <= hhigh[..., None])
        assert np.count_nonzero((pres[..., None] & head_interval) != head) < 200
        true_presence_head_edges = occ[..., None] & head_interval
        head_presence_true_edges = (pres & occ)[..., None] & oracle | (pres & ~occ)[..., None] & head_interval
        seg_occ = seg.any(-1)
        seg_low = np.where(seg, z, seg.shape[-1]).min(-1)
        seg_high = np.where(seg, z, -1).max(-1)
        seg_interval = (z >= seg_low[..., None]) & (z <= seg_high[..., None])
        seg_presence_head_edges = seg_occ[..., None] & head_interval
        head_presence_seg_edges = ((pres & seg_occ)[..., None] & seg_interval) | ((pres & ~seg_occ)[..., None] & head_interval)
        values = (seg, head, oracle, true_presence_head_edges, head_presence_true_edges, seg_presence_head_edges, head_presence_seg_edges)
        for key, pred in zip(keys, values):
            results[arm][key].append(dice(pred, truth))
        for threshold in thresholds:
            pred = (pres_prob >= threshold)[..., None] & head_interval
            results[arm][f"threshold_{threshold:.1f}"].append(dice(pred, truth))
assert len(results["C"]["seg"]) == 52
print("GT", json.dumps(totals, sort_keys=True))
for arm in "CD":
    print(arm, json.dumps({key: float(np.mean(scores)) for key, scores in results[arm].items()}, sort_keys=True))
