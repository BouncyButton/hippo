"""Check split preservation, quota limits and complete matched-cell aggregation."""
import copy
import csv
import json
import os
from pathlib import Path

import pytest

from geofwi3d_data import sha256, write_json
from geofwi3d_replication import METHODS, aggregate, fold_splits, quota_available, verify_run


def pilot_manifest():
    default = Path(__file__).parents[1] / "datasets/Dataset203_GeoFWI3D/manifest.json"
    return json.loads(Path(os.environ.get("GEOFWI3D_TEST_MANIFEST", str(default))).read_text())


def test_anchor_fold_and_disjoint_stratified_partitions():
    original = pilot_manifest()
    unchanged = copy.deepcopy(original)
    folds = fold_splits(original)
    assert original == unchanged
    assert folds[0] == original["splits"]
    assert folds == fold_splits(original)
    validation = [set(fold["validation"]) for fold in folds]
    assert set.union(*validation) == set(original["cases"])
    assert list(map(len, validation)) == [96, 96, 96, 96, 95]
    for i, fold in enumerate(folds):
        assert len(fold["training_subsets"]["5pct"]) == 19
        assert sum(original["cases"][k]["fault_present"] for k in fold["validation"]) == 46
        for other in validation[:i]:
            assert not validation[i] & other


def test_quota_uses_finite_user_and_tighter_group_limit():
    unlimited = "1009 1009 group pool_home 10.09TiB/∞ 39.91M/∞\n"
    user = "3160552 1395 user pool_home 90.62GiB/93.13GiB 187.13k/∞\n"
    free, inodes = quota_available(unlimited + user)
    assert free / 1024**3 == pytest.approx(2.51)
    assert inodes is None
    group = "1009 1009 group pool_home 20GiB/21GiB 10k/11k\n"
    free, inodes = quota_available(group + user)
    assert free == 1024**3
    assert inodes == 1000
    with pytest.raises(RuntimeError):
        quota_available(unlimited)
    with pytest.raises(RuntimeError):
        quota_available("quota unavailable")


def fake_run(directory, manifest, method, seed, dice=.8):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "run_config.json", {
        "training_case_ids": manifest["splits"]["training_subsets"]["5pct"],
        "validation_case_ids": manifest["splits"]["validation"], "protocol": {"model_seed": seed}})
    with (directory / "validation_per_image.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "dice", "intersection", "predicted_voxels", "target_voxels"])
        for case in manifest["splits"]["validation"]:
            writer.writerow([case, dice, int(dice * 10), 10, 10])
    record = {"state": "complete", "method": method, "seed": seed, "stopped_update": 500,
              "validation": {"mean_dice": dice}, "artifact_sha256": {
                  name: sha256(directory / name) for name in ["run_config.json", "validation_per_image.csv"]}}
    write_json(directory / "completion.json", record)


def test_exported_records_reject_missing_cases_and_corruption(tmp_path):
    manifest = pilot_manifest()
    fake_run(tmp_path, manifest, "bands", 2)
    verify_run(tmp_path, manifest, "bands", 2, checkpoints=False)
    with pytest.raises(RuntimeError, match="identity"):
        verify_run(tmp_path, manifest, "bands", 1, checkpoints=False)
    csv_path = tmp_path / "validation_per_image.csv"
    csv_path.write_text(csv_path.read_text() + csv_path.read_text().splitlines()[1] + "\n")
    with pytest.raises(RuntimeError, match="Changed artifact"):
        verify_run(tmp_path, manifest, "bands", 2, checkpoints=False)
    record = json.loads((tmp_path / "completion.json").read_text())
    record["artifact_sha256"][csv_path.name] = sha256(csv_path)
    write_json(tmp_path / "completion.json", record)
    with pytest.raises(RuntimeError, match="duplicated validation"):
        verify_run(tmp_path, manifest, "bands", 2, checkpoints=False)


def test_aggregation_requires_all_cells_and_pairs_by_fold_seed(tmp_path):
    original = pilot_manifest()
    refs, tasks = [], []
    for fold in range(3):
        manifest = copy.deepcopy(original)
        manifest["splits"] = fold_splits(original)[fold]
        write_json(tmp_path / "folds" / f"fold_{fold}" / "manifest.json", manifest)
        for seed in range(3):
            for method in METHODS:
                if fold == seed == 0:
                    relative = f"references/{method}"
                    fake_run(tmp_path / relative, manifest, method, seed, .8 if method == "baseline" else .9)
                    refs.append({"fold": fold, "seed": seed, "method": method, "frozen_record_directory": relative})
                else:
                    run_id = f"fold{fold}_seed{seed}_{method}"
                    tasks.append({"fold": fold, "seed": seed, "method": method, "run_id": run_id})
                    fake_run(tmp_path / "results" / run_id, manifest, method, seed, .8 if method == "baseline" else .9)
    write_json(tmp_path / "plan.json", {"references": refs, "new_runs": tasks})
    record = aggregate(tmp_path, complete=True)
    assert record["cells_completed"] == 54
    assert record["new_runs_completed"] == 48
    assert record["overall"]["bands"]["mean_paired_gain_pp"] == pytest.approx(10)
    assert record["per_fold_three_seeds"]["fold2_bands"]["seeds"] == 3
    (tmp_path / "results" / tasks[-1]["run_id"] / "completion.json").unlink()
    with pytest.raises(RuntimeError, match="all 54"):
        aggregate(tmp_path, complete=True)
