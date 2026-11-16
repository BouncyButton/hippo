"""Check full-size runs against the paired low-data execution's actual settings and LR log."""
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from comparisons import ROOT, DEST, true_voxel_counts


def low_files(row, reference):
    ds = row["dataset"]
    reports = ROOT / "reports"
    key = f"{ds}_{row['model']}_dice_f{row['partition']}s{row['seed']}"
    files = ("config.json", "metrics.csv", "initialization.json", "validation_latest.json")
    if ds == "HFH":
        path = ROOT / reference.split("!")[0]
        prefix = "results/"
    elif ds == "MSD" and row["seed"] == "20":
        folder = reports / "background_dice_archive_recovery_20261023/extracted/MSD" / key
        receipt = json.loads((folder / "recovery_receipt.json").read_bytes())
        path = ROOT / receipt["source"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        prefix = ""
    else:
        directory = {"MSD": reports / "msd_colab_seeds0220_recovery_20261018/cloud_backup/runs",
                     "MNI": reports / "mni_winning_recipes_matrix2030_53t_0519_20261109/frozen/reused_runs",
                     "ADNI": DEST / "audit_sources/ADNI"}[ds] / key
        data = {name: (directory / name).read_bytes() for name in files if (directory / name).exists()}
        if ds == "ADNI":
            source = reports / "background_dice_archive_recovery_20261023/remote/ADNI_conventional" / key
            data["validation_latest.json"] = (source / "validation_latest.json").read_bytes()
        return data
    with zipfile.ZipFile(path) as archive:
        return {n: archive.read(prefix+n) for n in files if prefix+n in archive.namelist()}


def audit(csv_path=DEST / "paired_comparisons.csv"):
    counts = Counter()
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["model"] == "medsam3":
            continue
        path = row["paper_matched_archive"] or row["native_pass_archive"]
        if not path:
            continue
        native_pass = bool(row["native_pass_archive"])
        with zipfile.ZipFile(ROOT / path) as z:
            key = row["cell"]
            receipt = json.loads(z.read("matched_receipt.json"))
            low = low_files(row, receipt["reference"])
            src, full = [json.loads(raw) for raw in (low["config.json"], z.read("config.json"))]
            if src.get("native_loss") != full.get("native_loss"):
                assert row["dataset"] == "MNI" and row["model"] == "nnunet"
                assert src["nnunet_ce_weight"] == 1 and full.get("nnunet_ce_weight", 1) == 1
                assert src["native_objective_sha256"] == full["native_objective_sha256"]
                assert (src["native_loss"], full["native_loss"]) == (
                    "Native foreground Dice + 1.0*CE, original deep supervision; SGD/Nesterov momentum0.95",
                    "Native foreground Dice+CE with deep supervision; SGD/Nesterov momentum0.95")
            for field in ("model", "model_identity", "fold", "seed", "precision", "objective",
                          "initial_lr", "lr_horizon_updates", "augmentation", "nnunet_sgd_momentum",
                          "nnunet_warmup_fraction", "constraint_warmup_updates"):
                assert src.get(field) == full.get(field), (key, field, src.get(field), full.get(field))
            for field in ("flops_per_update", "baseline_flops_per_update", "prefix_updates"):
                assert src["schedule"].get(field) == full["schedule"].get(field), (key, field)
            assert src["max_steps"] == (receipt["original_stop"] if native_pass else full["max_steps"]), key
            assert set(src["train_cases"]) <= set(full["train_cases"])
            assert set(src["validation_cases"]) == set(full["validation_cases"])
            assert not set(full["train_cases"]) & set(full["validation_cases"])
            lo = json.loads(low["validation_latest.json"])
            assert math.isclose(100*lo["summary"]["macro_hard_dice"], float(row["few_data_baseline_pct"]), abs_tol=1e-7), key
            assert true_voxel_counts(lo) == true_voxel_counts(json.loads(z.read("validation_latest.json"))), key
            trace = [json.loads(line)["lr"] for line in z.read("control_exposure.jsonl").splitlines()]
            observed = list(csv.DictReader(io.StringIO(low["metrics.csv"].decode())))
            assert observed and int(observed[-1]["step"]) == src["max_steps"], key
            for record in observed:
                assert math.isclose(float(record["learning_rate"]), trace[int(record["step"])-1], rel_tol=1e-10), (key, record["step"])
            if "initialization.json" in low:
                assert json.loads(low["initialization.json"])["model_state_sha256"] == json.loads(z.read("initialization.json"))["model_state_sha256"], key
                counts[row["dataset"], row["model"], "initial"] += 1
            counts[row["dataset"], row["model"], "recipe"] += 1
    return counts


if __name__ == "__main__":
    for key,value in sorted(audit().items()):
        print(*key,value)
