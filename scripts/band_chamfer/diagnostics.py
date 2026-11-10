"""Final-checkpoint A/P cut and interface diagnostics; never select checkpoints."""
import numpy as np
from scipy.spatial import cKDTree
import torch
from .geometry import edge_grid, surface_points, valid_field


def best_cut(labels):
    if not (labels == 1).any() or not (labels == 2).any():
        return None
    anterior = (labels == 1).sum((0, 2)); posterior = (labels == 2).sum((0, 2))
    errors = anterior.cumsum()[:-1]+posterior[::-1].cumsum()[::-1][1:]
    occupied = np.flatnonzero((anterior+posterior) > 0)
    candidates = np.arange(occupied[0], occupied[-1])
    return int(candidates[np.argmin(errors[candidates])]+1)


def add_ap_metrics(result, patients, predictions_path, protocol):
    predictions = np.load(predictions_path)
    items = {item['case_name']: item for item in patients.items}
    for row in result['cases']:
        item = items[row['case_name']]
        truth = item['label'][0].numpy().astype(np.uint8)
        pred = predictions[row['case_name']]
        valid = valid_field(truth.shape, protocol['native_geometry'][row['case_name']]['native_shape'])
        gt_grid = edge_grid((truth > 0) & valid, np.ones(3))
        pred_grid = edge_grid((pred > 0) & valid, np.ones(3))
        gt_points = surface_points(torch.tensor(truth == 1, dtype=torch.float32), gt_grid).numpy()
        pred_points = surface_points(torch.tensor(pred == 1, dtype=torch.float32), pred_grid).numpy()
        gt_cut, pred_cut = best_cut(truth), best_cut(pred)
        row['ap_best_plane_error_mm'] = float(abs(pred_cut-gt_cut)) if gt_cut is not None and pred_cut is not None else None
        row['ap_gt_plane_disagreement'] = int(((truth > 0) & (truth != np.where(np.arange(truth.shape[1])[None, :, None] >= gt_cut, 1, 2))).sum()) if gt_cut is not None else None
        row['ap_interface_chamfer_mm'] = None
        row['ap_interface_hd95_mm'] = None
        if len(gt_points) and len(pred_points):
            a = cKDTree(gt_points).query(pred_points, workers=1)[0]
            b = cKDTree(pred_points).query(gt_points, workers=1)[0]
            row['ap_interface_chamfer_mm'] = float(.5*(a.mean()+b.mean()))
            row['ap_interface_hd95_mm'] = float(np.quantile(np.r_[a, b], .95))
        swaps = np.argwhere((truth > 0) & (pred > 0) & (truth != pred))
        if len(gt_points):
            distance = cKDTree(gt_points).query(swaps, workers=1)[0] if len(swaps) else np.empty(0)
            row['ap_swaps_near3mm'] = int((distance <= 3).sum())
            row['ap_swaps_far3mm'] = int((distance > 3).sum())
        else:
            row['ap_swaps_near3mm'] = row['ap_swaps_far3mm'] = None
    predictions.close()
    for key in ('ap_best_plane_error_mm', 'ap_interface_chamfer_mm', 'ap_interface_hd95_mm'):
        values = [row[key] for row in result['cases'] if row[key] is not None]
        result['summary'][key] = float(np.mean(values)) if values else None
        result['summary'][key+'_undefined_cases'] = len(result['cases'])-len(values)
    for key in ('ap_swaps_near3mm', 'ap_swaps_far3mm'):
        result['summary'][key] = sum(row[key] for row in result['cases'] if row[key] is not None)
    return result
