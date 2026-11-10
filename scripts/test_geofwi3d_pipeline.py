"""Scientific/provenance regression checks for low-data volume training."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import geofwi3d_data as data
import train_geofwi3d_swinunetr as trainer

torch.set_num_threads(2)


def test_campaign_can_freeze_preflight_already_inside_campaign(tmp_path, monkeypatch):
    import prepare_geofwi3d_campaign as builder
    root, campaign = tmp_path / "dataset", tmp_path / "campaign"
    root.mkdir()
    campaign.mkdir()
    data.write_json(root / "manifest.json", {"fixture": "campaign-copy-regression"})
    preflight = campaign / "preflight.json"
    data.write_json(preflight, {"state": "passed", "manifest_sha256": data.sha256(root / "manifest.json"),
                               "model_smoke": {"optimizer_updated_weights": True}})
    expected_hash = data.sha256(preflight)
    monkeypatch.setattr(sys, "argv", ["prepare_geofwi3d_campaign.py", "--root", str(root),
                                     "--campaign", str(campaign), "--preflight", str(preflight)])
    builder.main()
    frozen = json.loads((campaign / "source_manifest.json").read_text())["sha256"]
    assert frozen["preflight.json"] == expected_hash == data.sha256(preflight)
    assert not (campaign / "submission.json").exists()


def metadata_rows():
    # Same stratum sizes as the published archive-0 metadata, without networking.
    rows = []
    for index in range(1000):
        rows.append({"sample_index": str(index), "seed": str(index + 10000),
                     "yn_salt": str(int(index < 479)), "yn_fault": str(int(249 <= index < 479))})
    return rows


def test_low_data_splits_are_nested_disjoint_and_seed_reproducible():
    rows = metadata_rows()
    splits = data.make_splits(rows, 0)
    assert splits == data.make_splits(rows, 0)
    assert splits != data.make_splits(rows, 1)
    assert (len(splits["train_pool"]), len(splits["validation"])) == (383, 96)
    assert {key: len(ids) for key, ids in splits["training_subsets"].items()} == {"5pct": 19, "20pct": 77, "100pct": 383}
    assert set(splits["training_subsets"]["5pct"]) <= set(splits["training_subsets"]["20pct"]) <= set(splits["train_pool"])
    rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(ValueError, match="Repeated simulation seeds"):
        data.make_splits(rows, 0)


def test_labels_remove_background_nulls_but_keep_disconnected_physical_salt():
    rgt, vp = np.ones((12, 12, 12), dtype=np.float32), np.full((12, 12, 12), 2000, dtype=np.float32)
    rgt[3:8, 4:9, 2:7] = 0
    vp[3:8, 4:9, 2:7] = 4500
    rgt[10, 10, 10], vp[10, 10, 10] = 0, 4500  # genuinely detached salt voxel
    rgt[0, 0, 0] = 0  # geological-time origin, not salt
    mask, audit = data.salt_mask(rgt, vp, 4500)
    assert mask.sum() == 126 and mask[10, 10, 10] == 1 and mask[0, 0, 0] == 0
    assert audit["removed_null_rgt_background_voxels"] == 1
    assert np.array_equal(mask, vp == 4500)


def test_band_loss_ignores_cube_faces_and_only_supervises_near_true_interfaces():
    logits = torch.zeros(1, 2, 16, 16, 16, requires_grad=True)
    labels = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    labels[:, :, :10, 4:12, 4:12] = 1
    loss = trainer.band_loss(logits, labels)
    loss.backward()
    assert float(loss) == pytest.approx(np.log(2), rel=1e-6)
    assert logits.grad[:, :, :2].abs().sum() == 0  # artificial observation boundary
    assert logits.grad[:, :, 9, 6, 6].abs().sum() > 0  # real salt/rock interface
    assert logits.grad[:, :, 5, 7, 7].abs().sum() == 0  # deep interior
    assert logits.grad[:, :, 14, 7, 7].abs().sum() == 0  # far background
    full = torch.ones_like(labels)
    assert trainer.band_loss(logits, full) == 0  # a full cube has no observed geological interface


def test_validation_uses_mean_foreground_dice_per_whole_volume():
    class Model(torch.nn.Module):
        def forward(self, image):
            return torch.cat([torch.zeros_like(image), image - .5], dim=1)
    big, small = np.zeros(data.SHAPE, np.uint8), np.zeros(data.SHAPE, np.uint8)
    big[:10, :10, :10] = 1
    small[:2, :2, :2] = 1
    arrays = {"big": (big.astype(np.float32), big), "small": (np.zeros(data.SHAPE, np.float32), small)}
    class Store:
        def get(self, name):
            return arrays[name]
    metrics, rows = trainer.evaluate(Model(), Store(), ["big", "small"], torch.device("cpu"), False)
    assert metrics["mean_dice"] == .5  # neither voxel pooling nor background accuracy
    assert [row["dice"] for row in rows] == [1.0, 0.0]
    assert metrics["validation_volumes"] == 2


def test_sampler_visits_all_volumes_and_resume_preserves_next_sample():
    first = [trainer.sample_index(i, 19, 7) for i in range(19)]
    second = [trainer.sample_index(i, 19, 7) for i in range(19, 38)]
    assert sorted(first) == sorted(second) == list(range(19))
    assert first != second
    full = [trainer.sample_index(i, 19, 7) for i in range(53)]
    resumed = [trainer.sample_index(i, 19, 7) for i in range(26, 53)]
    assert full[26:] == resumed


def test_short_budget_preserves_long_schedule_prefix():
    full = {"warmup_updates": 100, "max_updates": 5000}
    short = {"warmup_updates": 100, "max_updates": 500, "scheduler_updates": 5000}
    assert [trainer.learning_rate_factor(i, short) for i in range(500)] == [
        trainer.learning_rate_factor(i, full) for i in range(500)]
    assert trainer.learning_rate_factor(499, short) > .9
    assert trainer.learning_rate_factor(499, {"warmup_updates": 100, "max_updates": 500}) < .011


def test_binary_sdf_sign_transform_equivariance_and_gradient_direction():
    mask = np.zeros((16, 16, 16), np.uint8)
    mask[3:10, 4:12, 2:9] = 1
    field = trainer.signed_distance_map(mask)
    assert field.min() < 0 < field.max() <= 1
    assert np.all(trainer.signed_distance_map(np.zeros_like(mask)) == 1)
    assert np.all(trainer.signed_distance_map(np.ones_like(mask)) == -1)
    for position in range(12):
        _, target = trainer.augmented(mask, mask, 0, position, True)
        transformed, _ = trainer.augmented(field, field, 0, position, True)
        assert np.allclose(transformed, trainer.signed_distance_map(target))
    labels = torch.from_numpy(mask[None, None].astype(np.int64))
    phi = torch.from_numpy(field[None, None])
    logits = torch.zeros(1, 2, 16, 16, 16, requires_grad=True)
    loss = trainer.signed_distance_loss(logits, labels, phi)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert gradient[0, 1, 0, 0, 0] > 0  # distant background supervision
    assert gradient[0, 1, 5, 7, 5] < 0  # raise salt probability inside
    canonical = (logits.softmax(1)[:, 1:2] * phi).mean()
    assert torch.allclose(gradient, torch.autograd.grad(canonical, logits)[0])
    perfect = torch.full_like(logits, -30).scatter(1, labels, 30)
    assert trainer.signed_distance_loss(perfect, labels, phi) == pytest.approx(0, abs=1e-8)


def test_signed_face_edge_correct_orientation_and_crop_exclusion():
    labels = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    labels[:, :, :9, 4:12, 4:12] = 1
    supports = trainer.band_support(labels)
    perfect = torch.full((1, 2, 16, 16, 16), -30.).scatter(1, labels, 30.)
    support = supports[0] | supports[1]
    correct = trainer.signed_face_edge_loss(perfect, labels, support)
    inverted = trainer.signed_face_edge_loss(perfect.flip(1), labels, support)
    assert correct == pytest.approx(0, abs=1e-8) and inverted > .01
    logits = torch.zeros_like(perfect, requires_grad=True)
    trainer.signed_face_edge_loss(logits, labels, support).backward()
    assert logits.grad[:, :, :2].abs().sum() == 0
    assert logits.grad[:, :, 8:10, 6, 6].abs().sum() > 0
    empty = torch.zeros_like(support)
    zero = trainer.signed_face_edge_loss(logits, labels, empty)
    assert zero == 0 and torch.isfinite(torch.autograd.grad(zero, logits)[0]).all()


@pytest.mark.parametrize("method", ["sdf", "bands_sdf", "bands_edge", "bands_sdf_edge"])
def test_requested_loss_combinations_train_native_volume(tmp_path, monkeypatch, method):
    ids = ["train", "validation"]
    manifest = {"cases": {name: {"seed": str(i)} for i, name in enumerate(ids)},
                "splits": {"train_pool": ids[:1], "validation": ids[1:], "test_set": None,
                           "training_subsets": {name: ids[:1] for name in data.FRACTIONS}}}
    data.write_json(tmp_path / "manifest.json", manifest)
    image = np.zeros(data.SHAPE, np.float32)
    labels = np.zeros(data.SHAPE, np.uint8)
    labels[24:60, 24:60, 24:60] = 1
    image[:] = labels * 1.5 - .5
    sdf_requests = []
    class Store:
        def __init__(self, *args):
            pass
        def get(self, name):
            return image, labels
        def get_sdf(self, name, clip):
            sdf_requests.append(name)
            return trainer.signed_distance_map(labels, clip)
    monkeypatch.setattr(trainer, "CaseStore", Store)
    monkeypatch.setattr(trainer, "make_model", lambda *_: torch.nn.Conv3d(1, 2, 1))
    policy = {"model_seed": 0, "activation_checkpointing": False, "learning_rate": 1e-4,
              "weight_decay": 1e-5, "amp": False, "cache_volumes": True, "batch_size": 1,
              "augmentation": True, "warmup_updates": 1, "max_updates": 2, "min_updates": 2,
              "validate_every": 1, "early_stopping_patience": 10, "min_delta": .0005,
              "band_weight": .1, "band_warmup_updates": 1, "sdf_weight": .1,
              "sdf_warmup_updates": 1, "edge_weight": .1, "edge_warmup_updates": 1}
    output = tmp_path / method
    result = trainer.train_run(tmp_path, manifest, policy, output, "5pct", method, torch.device("cpu"))
    assert result["state"] == "complete" and result["stopped_update"] == 2
    assert set(sdf_requests) == ({"train"} if "sdf" in method else set())
    history = list(csv.DictReader((output / "history.csv").open()))
    for term in method.split("_"):
        assert float(history[0][f"mean_{term}_loss"]) > 0


def test_checkpoint_resume_reproduces_uninterrupted_training(tmp_path, monkeypatch):
    ids = ["a", "b", "c", "d"]
    manifest = {"cases": {name: {"seed": str(i)} for i, name in enumerate(ids)},
                "splits": {"train_pool": ids[:2], "validation": ids[2:], "test_set": None,
                           "training_subsets": {"5pct": ids[:1], "20pct": ids[:1], "100pct": ids[:2]}}}
    data.write_json(tmp_path / "manifest.json", manifest)
    rng = np.random.default_rng(11)
    arrays = {}
    for name in ids:
        image = rng.normal(size=data.SHAPE).astype(np.float32)
        label = (image > .5).astype(np.uint8)
        arrays[name] = image, label
    class Store:
        def __init__(self, *args):
            pass
        def get(self, name):
            return arrays[name]
    monkeypatch.setattr(trainer, "CaseStore", Store)
    monkeypatch.setattr(trainer, "make_model", lambda *_: torch.nn.Conv3d(1, 2, 1))
    policy = {"model_seed": 0, "activation_checkpointing": False, "learning_rate": 1e-4,
              "weight_decay": 1e-5, "amp": False, "cache_volumes": True, "batch_size": 1,
              "augmentation": True, "warmup_updates": 1, "max_updates": 4, "min_updates": 0,
              "validate_every": 2, "early_stopping_patience": 10, "min_delta": .0005,
              "band_weight": .1, "band_warmup_updates": 1}
    full = tmp_path / "full"
    interrupted = tmp_path / "interrupted"
    trainer.train_run(tmp_path, manifest, policy, full, "100pct", "baseline", torch.device("cpu"))
    original = trainer.atomic_checkpoint
    def crash_after_latest(path, payload):
        original(path, payload)
        if path.name == "latest.pt":
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(trainer, "atomic_checkpoint", crash_after_latest)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.train_run(tmp_path, manifest, policy, interrupted, "100pct", "baseline", torch.device("cpu"))
    monkeypatch.setattr(trainer, "atomic_checkpoint", original)
    resumed = trainer.train_run(tmp_path, manifest, policy, interrupted, "100pct", "baseline", torch.device("cpu"))
    a, b = [torch.load(folder / "latest.pt", map_location="cpu", weights_only=False) for folder in [full, interrupted]]
    assert a["update"] == b["update"] == 4
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert resumed["validation"]["mean_dice"] == json.loads((full / "completion.json").read_text())["validation"]["mean_dice"]
    rows = list(csv.DictReader((interrupted / "validation_per_image.csv").open()))
    assert {row["case_id"] for row in rows} == set(ids[2:])
    changed = {**policy, "learning_rate": 2e-4}
    with pytest.raises(RuntimeError, match="Run inputs or source changed"):
        trainer.train_run(tmp_path, manifest, changed, interrupted, "100pct", "baseline", torch.device("cpu"))
