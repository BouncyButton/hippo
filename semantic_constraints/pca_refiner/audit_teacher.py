"""Check whether the eight-case oracle has unusually weak gains versus all 208."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch

from semantic_constraints.pca_refiner.fit_eight import prepare_training
from semantic_constraints.spline_refinement.train import signed_distance, dice, write_json
from semantic_constraints.spectral_shape.audit import SIZE, shift_into_canvas, band_fit


def summarize(rows):
    gain = np.array([r["gain"] for r in rows])
    return dict(cases=len(rows), mean_baseline_dice=float(np.mean([r["baseline_dice"] for r in rows])),
        mean_oracle_dice=float(np.mean([r["oracle_dice"] for r in rows])), mean_gain=float(gain.mean()),
        median_gain=float(np.median(gain)), gain_quantiles=np.quantile(gain, [0, .25, .5, .75, 1]).tolist(),
        improved=int((gain > 1e-10).sum()), worsened=int((gain < -1e-10).sum()),
        unchanged=int((abs(gain) <= 1e-10).sum()), gain_at_least_0p1_percentage_points=int((gain >= .001).sum()),
        gain_at_least_1_percentage_point=int((gain >= .01).sum()),
        fixed=sum(r["fixed"] for r in rows), introduced=sum(r["introduced"] for r in rows))


def main():
    parser = argparse.ArgumentParser()
    for key in ("checkpoint", "pkl", "splits", "basis", "protocol", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    selected = set(json.loads(args.protocol.read_text())["selected"])
    basis = torch.load(args.basis, map_location=device, weights_only=False)
    records, provenance = prepare_training(args, device)
    rows = []
    for record in records:
        original = record["prediction"] > 0
        shift = np.rint((SIZE-1)/2 - np.argwhere(original).mean(0)).astype(int)
        p = shift_into_canvas(original, shift)
        g = shift_into_canvas(record["truth"] > 0, shift)
        support = ndi.binary_dilation(p, iterations=2) & ~ndi.binary_erosion(p, iterations=2)
        source = np.clip(signed_distance(p), -8, 8)
        residual = np.clip(signed_distance(g), -8, 8) - source
        idx = torch.from_numpy(np.flatnonzero(support)).to(device)
        design = basis["modes"][:, idx].T
        target = torch.from_numpy(residual[support]).to(device)
        delta, coefficients = band_fit(design, target, basis["error_mean"][idx])
        mask = p.copy()
        mask[support] = source[support] + delta.cpu().numpy() < 0
        baseline, oracle = dice(p, g), dice(mask, g)
        rows.append(dict(case=record["name"], selected=record["name"] in selected,
            baseline_dice=baseline, oracle_dice=oracle, gain=oracle-baseline,
            fixed=int(((p != g) & (mask == g)).sum()), introduced=int(((p == g) & (mask != g)).sum()),
            outside_band_changes=int(((p != mask) & ~support).sum())))
    result = dict(all_training=summarize(rows), selected_eight=summarize([r for r in rows if r["selected"]]),
                  other_200=summarize([r for r in rows if not r["selected"]]), cases=rows,
                  provenance=provenance, purpose="Post-hoc teacher-strength diagnostic, no network training or validation use")
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "provenance")}), flush=True)


if __name__ == "__main__":
    main()
