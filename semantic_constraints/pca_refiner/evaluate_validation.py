"""Evaluate a frozen coefficient regressor on fold-0 validation."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
from monai.networks.nets import SwinUNETR

from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
from semantic_constraints.pca_refiner.fit_eight import CoefficientRegressor, translate_image
from semantic_constraints.spectral_shape.audit import SIZE, shift_into_canvas
from semantic_constraints.spline_refinement.train import (
    EXPECTED_CHECKPOINT, EXPECTED_DATA, EXPECTED_SPLITS, sha256, signed_distance,
    dice, surface_distances, restore_labels, write_json,
)


@torch.inference_mode()
def refine(image, prediction, probability, model, basis, device):
    """Predict a correction using image and baseline outputs only; no annotation."""
    foreground = prediction > 0
    assert foreground.any()
    shift = np.rint((SIZE - 1) / 2 - np.argwhere(foreground).mean(0)).astype(int)
    original = shift_into_canvas(prediction, shift)
    foreground = original > 0
    source = np.clip(signed_distance(foreground), -8, 8)
    inner = foreground & ~ndi.binary_erosion(foreground, iterations=2)
    outer = ndi.binary_dilation(foreground, iterations=2) & ~foreground
    support = inner | outer
    features = np.stack((translate_image(image, shift), translate_image(probability, shift),
                         np.clip(source / 4, -1, 1), inner, outer)).astype(np.float32)
    normalized = model(torch.from_numpy(features)[None].to(device))[0]
    coefficients = normalized / basis["common_scale"]
    delta = (coefficients @ basis["modes"] + basis["error_mean"]).cpu().numpy().reshape(source.shape)
    assert np.isfinite(delta).all()
    mask = np.where(support, source + delta < 0, foreground)
    labels = restore_labels(mask, original)
    assert np.array_equal(labels[~support], original[~support])
    retained = foreground & mask
    assert np.array_equal(labels[retained], original[retained])
    return dict(prediction=original, refined=labels, support=support, shift=shift,
                coefficients=coefficients.cpu().numpy())


def score(name, data, truth):
    original, corrected = data["prediction"] > 0, data["refined"] > 0
    row = dict(case=name, baseline_dice=dice(original, truth), refined_dice=dice(corrected, truth),
               fixed=int(((original != truth) & (corrected == truth)).sum()),
               introduced=int(((original == truth) & (corrected != truth)).sum()),
               changed=int((original != corrected).sum()),
               outside_band_changes=int(((original != corrected) & ~data["support"]).sum()))
    row["gain"] = row["refined_dice"] - row["baseline_dice"]
    for label, mask in (("baseline", original), ("refined", corrected)):
        row[f"{label}_assd_mm"], row[f"{label}_hd95_mm"] = surface_distances(mask, truth)
        row[f"{label}_components"] = int(ndi.label(mask)[1])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "pkl", "splits", "refiner", "basis", "protocol", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases").mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    hashes = {key: sha256(getattr(args, key)) for key in
              ("checkpoint", "pkl", "splits", "refiner", "basis", "protocol")}
    assert hashes["checkpoint"] == EXPECTED_CHECKPOINT
    assert hashes["pkl"] == EXPECTED_DATA and hashes["splits"] == EXPECTED_SPLITS
    fold = json.loads(args.splits.read_text())[0]
    protocol = json.loads(args.protocol.read_text())
    assert fold == protocol["split"]
    assert len(fold["val"]) == 52 and len(fold["train"]) == 208
    assert set(protocol["selected"]) <= set(fold["train"])
    assert not set(fold["train"]) & set(fold["val"])
    assert protocol["validation_used"] is False
    dataset = _build_monai_dataset_from_pkl(_load_pkl_dataframe(args.pkl), "MSD", 3)
    options = dict(in_channels=1, out_channels=3, use_checkpoint=False, drop_rate=0.)
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        options["img_size"] = (64, 64, 64)
    baseline = SwinUNETR(**options).to(device).eval().requires_grad_(False)
    baseline_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    baseline.load_state_dict(baseline_state.get("model", baseline_state), strict=True)
    del baseline_state
    model = CoefficientRegressor().to(device).eval().requires_grad_(False)
    state = torch.load(args.refiner, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    basis = torch.load(args.basis, map_location=device, weights_only=False)
    assert basis["modes"].shape == (32, SIZE ** 3)
    rows = []
    with torch.inference_mode():
        for index, item in enumerate(dataset.data):
            name = item["case_name"]
            if name not in fold["val"]:
                continue
            sample = dataset[index]
            image = sample["image"].as_tensor().float()
            logits = baseline(image[None].to(device)).float()
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            probability = logits.softmax(1)[0, 1:].sum(0).cpu().numpy()
            data = refine(image[0].numpy(), prediction, probability, model, basis, device)
            # Annotation is accessed for scoring only after the output is fixed.
            labels = sample["label"].as_tensor()[0].numpy().astype(np.uint8)
            truth = shift_into_canvas(labels, data["shift"])
            rows.append(score(name, data, truth > 0))
            np.savez_compressed(args.output / "cases" / f"{name}.npz", **data, truth=truth)
            print("VALIDATION", len(rows), "/52", json.dumps(rows[-1]), flush=True)
    assert {r["case"] for r in rows} == set(fold["val"])
    differences = np.array([r["gain"] for r in rows])
    rng = np.random.default_rng(0)
    bootstrap = rng.choice(differences, size=(10000, len(rows)), replace=True).mean(1)
    summary = {key: float(np.mean([r[key] for r in rows])) for key in (
        "baseline_dice", "refined_dice", "baseline_assd_mm", "refined_assd_mm",
        "baseline_hd95_mm", "refined_hd95_mm")}
    assert abs(summary["baseline_dice"] - 0.9009775183944705) < 1e-10
    summary.update(cases=52, mean_dice_gain=float(differences.mean()),
        paired_bootstrap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist(),
        improved=int((differences > 1e-10).sum()), worsened=int((differences < -1e-10).sum()),
        unchanged=int((np.abs(differences) <= 1e-10).sum()),
        fixed=sum(r["fixed"] for r in rows), introduced=sum(r["introduced"] for r in rows),
        outside_band_changes=sum(r["outside_band_changes"] for r in rows))
    # Evaluation must not modify the frozen artifacts.
    for key in ("refiner", "basis", "protocol"):
        assert hashes[key] == sha256(getattr(args, key))
    result = dict(summary=summary, cases=rows, hashes=hashes, split=fold,
        refiner_training_cases=protocol["selected"], basis_training_cases=208,
        checkpoint_step=state["step"], validation_fitting=False, oracle_fitting=False,
        coefficient_scaling="unchanged training common_scale", threshold=0,
        checkpoint_selection="lowest training field MSE; fixed before validation",
        source_sha256=sha256(Path(__file__)), gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__)
    write_json(args.output / "results.json", result)
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
