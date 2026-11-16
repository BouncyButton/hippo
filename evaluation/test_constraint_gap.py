"""Reporting invariants using synthetic scores; no training archives required."""
import csv
import json

import pytest

from constraint_gap import ARMS, DATASETS, MODELS, cohort, index_source, write_constraint_gaps


def test_all_objectives_keep_losses_missing_pairs_and_execution_batches(tmp_path):
    controls, records = [], []
    for dataset in DATASETS:
        for model in MODELS:
            study = dataset + ("_MedSAM3" if model == "medsam3" else "_capped")
            for fold, seed in cohort(dataset, model):
                controls.append(dict(dataset=dataset, model=model, partition=fold, seed=seed,
                                     source="scores.json", few_data_baseline_pct=60,
                                     paper_matched_full_pct=65, native_pass_full_pct="",
                                     development_peak_lr_0_001_pct=""))
                for arm, score in (("baseline", 60), ("sdf", 0)):
                    if arm == "sdf" and (dataset, model, fold, seed) == ("HFH", "nnunet", 0, 0):
                        continue
                    records.append(dict(study=study, model=model, fold=fold, seed=seed,
                                        arm=arm, dice=score))
                if study == "MSD_MedSAM3":
                    for arm, score in (("baseline", 30), ("bands", 40)):
                        records.append(dict(study=study + "_original", model=model, fold=fold,
                                            seed=seed, arm=arm, dice=score))

    (tmp_path / "scores.json").write_text(json.dumps({"rows": records}))
    write_constraint_gaps(reversed(controls), tmp_path, tmp_path)
    with (tmp_path / "objective_pairs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(DATASETS) * len(MODELS) * len(ARMS) * 9
    old = next(r for r in rows if r["dataset"] == "MSD" and r["model"] == "medsam3"
               and r["objective"] == "bands")
    assert old["baseline_pct"] == "30.0" and old["gain_pp"] == "10.0"
    assert old["paper_matched_full_pct"] == ""
    report = (tmp_path / "constraint_gap.md").read_text()
    assert "| HFH | nnunet | sdf | 8/9 | — | — |" in report
    assert "| ADNI | nnunet | sdf | 9/9 | 0.00 | -60.00 |" in report
    assert "| MNI | nnunet | depth | 0/9 | — | — |" in report
    controls[0]["seed"] = 9876
    with pytest.raises(ValueError, match="cohort"):
        write_constraint_gaps(controls, tmp_path, tmp_path)


def test_csv_units_duplicates_and_nonfinite_scores():
    row = dict(objective="dice_sdf", fold="0", seed="2", final_dice="0.30")
    def read(rows):
        return index_source(rows, "HFH", "nnunet", {(0, 2)}, False)[0]
    assert read([row])["sdf", (0, 2)] == 30
    assert read([dict(row, status="incomplete")])["sdf", (0, 2)] is None
    with pytest.raises(ValueError, match="duplicate"):
        read([row, row])
    with pytest.raises(ValueError, match="Invalid Dice"):
        read([dict(row, final_dice="nan")])
