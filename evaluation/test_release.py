"""Cross-check published gains against their independently archived deltas."""
import csv
import json
from functools import lru_cache

import pytest

from constraint_gap import ROOT, comparison_rows, objective_name
from paper_tables import manuscript_rows


def load_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def test_all_paired_gains_match_the_archives():
    controls = load_csv("controls.csv")
    scores = list(comparison_rows(controls, ROOT))
    sources = {(r["dataset"], r["model"]): r["source"] for r in controls}

    @lru_cache(None)
    def archived(source, dataset, model, batch):
        if source == "published.json":
            study = dataset + ("_MedSAM3" if model == "medsam3" else "_capped")
            if batch == "original":
                study += "_original"
            rows = (r for r in json.loads((ROOT / source).read_text())["rows"]
                    if r["study"] == study and r["model"].lower() == model)
            field, scale = "gain_pp", 1
        else:
            rows = (r for r in load_csv(source) if r.get("model", model) == model)
            field, scale = {
                "mni_adapter.csv": ("delta_dice", 100),
                "hfh_adapter.csv": ("gain_over_matched_baseline_pp", 1),
                "hfh_scratch.csv": ("delta_pp", 1),
                "hfh_nnunet.csv": ("delta_pp", 1),
            }[source]
        return {(objective_name(r), int(r["fold"]), int(r["seed"])): scale * float(r[field])
                for r in rows}

    assert len(scores) == 2520
    for row in scores:
        ds, model = row["dataset"], row["model"]
        expected = archived(sources[ds, model], ds, model, row["batch"])
        assert row["gain_pp"] == pytest.approx(
            expected[row["objective"], row["fold"], row["seed"]], abs=1e-8)
    winners = list(manuscript_rows(scores))
    assert len(winners) == 20
    msd = next(r for r in winners if (r["dataset"], r["model"]) == ("MSD", "nnunet"))
    assert msd["objective"] == "bands_sdf_depth"
    assert msd["objective_pct_fold_sd"] == pytest.approx(2.549, abs=.001)
