"""Checks for gradient calibration, budget recovery and the four-run pilot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import train_compute_matched_constraints as trainer
import summarize_compute_matched_constraints as reporter
from compute_constraint_objectives import ARMS, calibration_case, select_weights, scheduled_costs, training_loss
from monai.losses import DiceLoss
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.signed_distance import signed_distance_maps, signed_distance_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def labels():
    y = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    y[:, :, 3:8, 4:12, 4:12] = 1
    y[:, :, 8:13, 4:12, 4:12] = 2
    return y


def test_calibration_norm_identity_and_safety_caps():
    z = torch.randn(1, 3, 16, 16, 16, generator=torch.Generator().manual_seed(37), requires_grad=True)
    y = labels()
    row = calibration_case(z, y, "dice_bands_edge")
    rows = [dict(row, case_name=str(i)) for i in range(10)]
    weights, stats = select_weights(rows, "dice_bands_edge")
    base_loss = DiceLoss(to_onehot_y=True, softmax=True)(z, y) + weights["bands"] * OuterBoundaryBandLoss()(z, y).loss
    actual = float(torch.autograd.grad(base_loss, z)[0].square().mean().sqrt())
    reconstructed = row["edge_rms"] / rows[0]["edge_over_base"]
    assert actual == pytest.approx(reconstructed, rel=2e-6)
    assert all(v["weighted_median_ratio"] <= .1 + 1e-8 and v["weighted_p95_ratio"] <= .5 + 1e-8 for v in stats.values())
    zero = torch.zeros(1, 3, 16, 16, 16, requires_grad=True)
    with pytest.raises(ValueError, match="valid inner"):
        calibration_case(zero, torch.zeros_like(y), "dice_bands")


def test_band_and_edge_change_gradients_without_extra_network_forward():
    z = torch.randn(1, 3, 16, 16, 16, generator=torch.Generator().manual_seed(3), requires_grad=True)
    y = labels()
    gradients = {}
    for arm in ARMS:
        loss = training_loss(z, y, arm, {"bands": .1, "edge": .1, "sdf": .1}, 1,
                             signed_distance_maps(y) if "sdf" in arm else None)
        gradients[arm] = torch.autograd.grad(loss, z)[0]
        assert torch.isfinite(gradients[arm]).all()
    assert all(not torch.equal(gradients["dice"], gradients[arm]) for arm in ARMS[1:])


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0., .1, -.1]))

    def forward(self, image):
        return self.logits[None, :, None, None, None].expand(image.shape[0], 3, *image.shape[2:])


def campaign(tmp_path):
    source = tmp_path / "source"
    protocol_dir = source / trainer.DEFAULT_PROTOCOL.relative_to(ROOT)
    shutil.copytree(ROOT / "experiments/compute_matched_constraints_20261004", protocol_dir)
    (source / "scripts").mkdir()
    for name in ("train_compute_matched_constraints.py", "train_compute_matched_baselines.py", "compute_constraint_objectives.py"):
        shutil.copy(ROOT / "scripts" / name, source / "scripts" / name)
    sdf_source = source / "thesis/new_constraints/signed_distance.py"
    sdf_source.parent.mkdir(parents=True)
    shutil.copy(ROOT / "thesis/new_constraints/signed_distance.py", sdf_source)
    names = [f"synthetic_{i:02}" for i in range(12)]
    split = {"train": names[:10], "val": names[10:]}
    splits_path = protocol_dir / "splits_10patients.json"
    splits_path.write_text(json.dumps([split] * 3))
    raw_label = labels()[0, 0].numpy()
    data = pd.DataFrame([{"subject_id": name, "image_data": np.ones_like(raw_label, dtype=np.float32),
                          "label_data": raw_label.copy()} for name in names])
    pkl = tmp_path / "synthetic.pkl.gz"
    data.to_pickle(pkl, compression="gzip")
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    protocol.update(budget_flops=100, pkl_sha256=trainer.base.digest(pkl),
                    splits_sha256=trainer.base.digest(splits_path), validation_patients_per_fold=2)
    protocol["models"]["unet"]["parameters"] = 3
    for arm in ARMS:
        schedule = scheduled_costs(100, 10, 12 if arm != "dice" else 10, 12 if arm != "dice" else 0, 1)
        schedule.update(flops_per_update=12 if arm != "dice" else 10, baseline_flops_per_update=10)
        protocol["models"]["unet"]["objectives"][arm] = schedule
    (protocol_dir / "protocol.json").write_text(json.dumps(protocol))
    manifest = {str(p.relative_to(source)): trainer.base.digest(p) for p in source.rglob("*") if p.is_file()}
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest))
    return protocol_dir, pkl


def args(root, protocol_dir, pkl, arm):
    return argparse.Namespace(model="unet", objective=arm, fold=0, seed=0, pkl=pkl,
        output=root / "unet" / arm / "fold0/seed0/run", protocol_dir=protocol_dir,
        device="cpu", threads=2, resume=True, smoke_steps=0)


def test_four_arm_campaign_aggregation_and_tampering(tmp_path):
    protocol_dir, pkl = campaign(tmp_path)
    with patch.object(trainer.base, "build_model", lambda *a: Tiny()):
        for arm in ARMS:
            trainer.run(args(tmp_path, protocol_dir, pkl, arm))
    result = reporter.summarize(tmp_path)
    assert len(result["runs"]) == 8
    for row in result["runs"]:
        assert row["total_gflops"] <= 100 / 1e9
    path = tmp_path / "unet/dice_bands_edge/fold0/seed0/run/calibration.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Changed artifact calibration"):
        reporter.summarize(tmp_path)


def test_interrupted_calibration_retry_stays_charged(tmp_path):
    protocol_dir, pkl = campaign(tmp_path)
    run_args = args(tmp_path, protocol_dir, pkl, "dice_bands_edge")
    with patch.object(trainer.base, "build_model", lambda *a: Tiny()):
        with patch.object(trainer, "calibrate", side_effect=RuntimeError("interrupted calibration")):
            with pytest.raises(RuntimeError, match="interrupted calibration"):
                trainer.run(run_args)
        trainer.run(run_args)
    done = json.loads((run_args.output / "completion.json").read_text())
    assert done["calibration_attempts"] == 2 and done["calibration_flops"] == 24
    assert done["consumed_total_flops"] <= 100 and done["unused_flops"] < 12
    assert done["training_attempts_charged"] < json.loads((protocol_dir / "protocol.json").read_text())["models"]["unet"]["objectives"]["dice_bands_edge"]["max_steps"]


def test_real_unet_prefix_calibration_and_constraint_update(tmp_path):
    protocol_dir, pkl = campaign(tmp_path)
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    protocol["models"]["unet"]["parameters"] = 4807482
    (protocol_dir / "protocol.json").write_text(json.dumps(protocol))
    run_args = args(tmp_path, protocol_dir, pkl, "dice_bands_edge_sdf")
    run_args.smoke_steps = 2
    trainer.run(run_args)
    done = json.loads((run_args.output / "completion.json").read_text())
    assert done["status"] == "smoke_complete" and done["updates"] == 2
    report = json.loads((run_args.output / "calibration.json").read_text())
    assert len(report["cases"]) == 10 and set(report["weights"]) == {"bands", "edge", "sdf"}


def test_classwise_sdf_sign_degenerate_classes_and_spatial_alignment():
    y = labels()
    phi = signed_distance_maps(y)
    assert phi.shape == (1, 2, 16, 16, 16) and torch.isfinite(phi).all()
    assert phi.abs().max() <= 1
    assert phi[0, 0, 5, 8, 8] < 0 < phi[0, 1, 5, 8, 8]
    assert phi[0, 1, 10, 8, 8] < 0 < phi[0, 0, 10, 8, 8]
    assert torch.equal(signed_distance_maps(torch.flip(y, (-1,))), torch.flip(phi, (-1,)))
    assert signed_distance_maps(torch.zeros_like(y)).eq(1).all()
    full = signed_distance_maps(torch.ones_like(y))
    assert full[:, 0].eq(-1).all() and full[:, 1].eq(1).all()


def test_sdf_full_volume_gradient_and_constant_subtraction():
    y = labels();phi = signed_distance_maps(y)
    z = torch.zeros(1, 3, 16, 16, 16, requires_grad=True)
    loss = signed_distance_loss(z, y, phi)
    g = torch.autograd.grad(loss, z)[0]
    # A distant background voxel still receives supervision, beyond any band.
    assert g[0, 1, 0, 0, 0] > 0 and g[0, 2, 0, 0, 0] > 0
    assert g[0, 1, 5, 8, 8] < 0 and g[0, 2, 5, 8, 8] > 0
    canonical = (z.softmax(1)[:, 1:] * phi).mean()
    assert torch.allclose(g, torch.autograd.grad(canonical, z)[0])
    good = torch.full_like(z, -30.).scatter(1, y, 30.)
    assert signed_distance_loss(good, y, phi) < loss
    assert float(signed_distance_loss(good, y, phi)) == pytest.approx(0, abs=1e-8)


def test_sdf_calibration_combined_gradient_identity():
    y = labels();phi = signed_distance_maps(y)
    z = torch.randn(1, 3, 16, 16, 16, generator=torch.Generator().manual_seed(11), requires_grad=True)
    row = calibration_case(z, y, "dice_bands_edge_sdf", phi)
    rows = [dict(row, case_name=str(i)) for i in range(10)]
    weights, stats = select_weights(rows, "dice_bands_edge_sdf")
    actual = torch.autograd.grad(training_loss(z, y, "dice_bands_edge", weights, 1), z)[0].square().mean().sqrt()
    reconstructed = row["sdf_rms"] / rows[0]["sdf_over_base"]
    assert float(actual) == pytest.approx(reconstructed, rel=2e-6)
    assert stats["sdf"]["weighted_median_ratio"] <= .1 + 1e-8
    assert stats["sdf"]["weighted_p95_ratio"] <= .5 + 1e-8


def test_startup_failure_is_logged_when_output_cannot_be_written(monkeypatch, capsys, tmp_path):
    argv = ["trainer", "--model", "unet", "--objective", "dice", "--fold", "0", "--seed", "0",
            "--pkl", str(tmp_path / "data.pkl"), "--output", str(tmp_path / "run")]
    monkeypatch.setattr(sys, "argv", argv)
    with patch.object(Path, "mkdir", side_effect=OSError(122, "Disk quota exceeded")), \
         patch.object(trainer.base, "atomic_json", side_effect=OSError(122, "Disk quota exceeded")):
        with pytest.raises(OSError, match="Disk quota exceeded"):
            trainer.main()
    captured = capsys.readouterr()
    assert "Disk quota exceeded" in captured.out and "Disk quota exceeded" in captured.err
