"""Independently reconstruct low, constrained and larger-pool Dice from case matrices."""
import csv
import json
import math
from pathlib import Path
from statistics import mean
import zipfile

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent


def read(path, name):
    if path.is_dir():
        return json.loads((path / (name + ".json")).read_bytes())
    with zipfile.ZipFile(path) as z:
        member = name + ".json"
        if member not in z.namelist():
            member = "results/" + member
        return json.loads(z.read(member))


def low_sources(ds, fold, seed, arm):
    arm = arm.removeprefix("dice_")
    pair = f"f{fold}s{seed}"
    if ds == "HFH":
        root = REPORTS / "hfh_colab_exploration_20261111"
        stem = f"__f{fold}_s{seed}.zip"
        if str(seed) == "0":
            loc = root / "medsam_adaptive_matrix_seed0"
            refs = json.loads((loc / "reused.json").read_bytes())
            baseline, = [root / v["source"] for v in refs.values() if str(v["fold"]) == str(fold) and v["arm"] == "dice"]
            return [baseline, loc / "archives" / ("matrix125__dice_"+arm+stem)]
        if str(seed) == "2":
            loc = root / "medsam_adaptive_matrix_replication/seed2"
            return [loc / ("matrix125__"+a+stem) for a in ("dice","dice_"+arm)]
        loc = HERE / "audit_sources/HFH"
        baseline = (root / "medsam_gain_replication125/compact_archives" / ("gain125__dice"+stem)
                    if str(fold) in ("0", "1") else loc / ("matrix125__dice"+stem))
        return [baseline, loc / ("matrix125__dice_"+arm+stem)]
    if ds == "MSD":
        root = REPORTS / "background_dice_archive_recovery_20261023/remote/MSD_MedSAM3_extension"
        return [root / f"MSD_MedSAM3_{a}_{pair}" for a in ("baseline", arm)]
    if ds == "ADNI":
        root = REPORTS / "adni_medsam3_2sample_3fold_3seed_14arms_20261018/runs"
        return [root / f"ADNI_MedSAM3_2sample_{a}_{pair}" for a in ("dice", "dice_" + arm)]
    root = REPORTS / "mni_medsam3_n5_third_seed_search_20261110/archive/candidates/candidates/seed2"
    native = REPORTS / "mni_medsam3_n5_adaptive60_2gpu_20261110/archive/runs" / f"MNI_MedSAM3_n5_{pair}_native400"
    if not (native / "evaluation.json").exists():
        native = root / "baseline/runs" / native.name
    parent = root / "constraints/runs" if str(seed) == "2" else REPORTS / "mni_medsam3_n5_f012_s01_constraints_20261110/archive/runs"
    return [native, parent / f"MNI_MedSAM3_n5_dice_{arm}_{pair}"]


def heldout(source, names=None):
    records = [r for r in read(source, "evaluation")["records"] if r["split"] == "validation"]
    truth, dice = {}, []
    for r in records:
        name = r.get("case_name") or r["case"]
        if names is not None and name not in names:
            continue
        matrix = r.get("confusion") or r["confusion_true_rows_predicted_columns"]
        assert name not in truth
        truth[name] = (r["truth_sha256"], tuple(map(sum, matrix)))
        classes = []
        for i in range(1, len(matrix)):
            denom = sum(matrix[i]) + sum(row[i] for row in matrix)
            assert denom > 0
            classes.append(2 * matrix[i][i] / denom)
        dice.append(mean(classes))
    return truth, 100 * mean(dice)


def audit(csv_path=HERE / "paired_comparisons.csv"):
    out = {}
    with csv_path.open() as stream:
        for r in csv.DictReader(stream):
            ds = r["dataset"]
            if r["model"] != "medsam3" or not r["paper_matched_archive"]:
                continue
            sources = low_sources(ds, r["partition"], r["seed"], r["selected_arm"])
            full = HERE.parents[1] / r["paper_matched_archive"]
            sources.append(full)
            results = [heldout(p) for p in sources[:2]]
            results.append(heldout(full, results[0][0] if ds == "MSD" else None))
            assert results[0][0] == results[1][0] == results[2][0], r["cell"]
            for (truth, value), column in zip(results, ("few_data_baseline_pct", "few_data_selected_pct", "paper_matched_full_pct")):
                assert math.isclose(value, float(r[column]), abs_tol=1e-8), (r["cell"], value, column)
            cfg = [read(p, "protocol") for p in sources]
            for name in ("base_sha256", "classes", "augmentation", "learning_rate", "warmup_updates", "accumulation", "final_learning_rate", "inference"):
                assert cfg[0][name] == cfg[1][name] == cfg[2][name], (r["cell"], name)
            assert all((p.get("prompts") or p["class_prompts"]) == (cfg[0].get("prompts") or cfg[0]["class_prompts"]) for p in cfg)
            assert cfg[0]["train_cases"] == cfg[1]["train_cases"]
            assert set(cfg[0]["train_cases"]) <= set(cfg[2]["train_cases"])
            assert cfg[0]["evaluation_cases"] == cfg[1]["evaluation_cases"]
            if ds == "MSD":
                assert cfg[2]["paired_evaluation_cases"] == cfg[0]["evaluation_cases"]
                assert len(cfg[2]["train_cases"]) == 208 and len(cfg[2]["evaluation_cases"]) == 52
            else:
                assert cfg[0]["evaluation_cases"] == cfg[2]["evaluation_cases"]
            completed = [read(p, "completion") for p in sources]
            assert len({done["updates"] for done in completed}) == 1
            if ds != "MSD":
                states = [read(p, "training_summary") for p in sources]
                assert all(t["frozen_before"] == t["frozen_after"] and t["adapter_before"] != t["adapter_after"] for t in states)
                for name in ("frozen_before", "adapter_before"):
                    assert len({t[name] for t in states}) == 1
            else:
                assert all(p["data_hashes"][k] == v for k,v in cfg[0]["data_hashes"].items() for p in cfg)
                assert completed[1]["budget"] == "200 updates; not FLOP capped"
            out[r["cell"]] = [round(result[1], 8) for result in results]
    return out


if __name__ == "__main__":
    rows = audit()
    for ds in ("MNI", "ADNI", "MSD", "HFH"):
        group = [scores for k, scores in rows.items() if k.startswith(ds + "_")]
        print(ds, len(group), "constrained wins", sum(c > f for _, c, f in group), "means", [round(mean(v), 4) for v in zip(*group)])
