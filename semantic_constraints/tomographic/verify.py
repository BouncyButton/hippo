"""Verify saved gradient-audit hashes, split membership and error accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probe import METHODS, sha256, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    protocol = json.loads((root / "protocol.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    rows = json.loads((root / "rows.json").read_text())
    summaries = json.loads((root / "summary.json").read_text())
    for name, expected in completion["output_sha256"].items():
        assert sha256(root / name) == expected, name
    assert len(rows) == completion["rows"]
    assert len(protocol["selected_names"]) == completion["cases"]
    assert protocol["validation_used"] is False
    assert {r["case"] for r in rows} == set(protocol["selected_names"])
    assert len({(r["case"], r["kind"], r["method"]) for r in rows}) == len(rows)
    pairs = {(r["case"], r["kind"]) for r in rows}
    assert len(pairs) * len(METHODS) == len(rows)
    steps = 0
    loss_increases = []
    baseline_by_case = {}
    for row in rows:
        before = row["baseline"]
        assert before["errors"] == before["fp"] + before["fn"] + before["swaps"]
        assert np.isfinite(row["loss"])
        assert row["gradient"]["rms"] >= 0
        if row["kind"] == "real":
            if row["case"] in baseline_by_case:
                assert before == baseline_by_case[row["case"]]
            baseline_by_case[row["case"]] = before
        if row["kind"] == "correct":
            assert before["errors"] == 0
        if row["loss_after_max_0p001"] > row["loss"] + max(1e-9, abs(row["loss"])*1e-5):
            loss_increases.append({k:row[k] for k in ("case", "kind", "method", "loss", "loss_after_max_0p001")})
        for update in row["steps"]:
            steps += 1
            after = update["after"]
            assert after["errors"] == after["fp"] + after["fn"] + after["swaps"]
            assert after["errors"] == before["errors"] - update["fixed"] + update["introduced"]
            assert after["fp"] + after["fn"] == before["fp"] + before["fn"] - update["external_fixed"] + update["external_introduced"]
            assert sum(update["fixed_by_true_class"]) == update["fixed"]
            assert sum(update["introduced_by_true_class"]) == update["introduced"]
            for name, value in update["delta"].items():
                assert np.isclose(value, after[name] - before[name], atol=1e-12)
            measured = update["update_rms"] if update["normalization"] != "max" else update["update_max"]
            # Zero-gradient arms are identities; all other arms meet their budget.
            assert measured == 0 or np.isclose(measured, update["step"], rtol=1e-5, atol=1e-8)
    assert summarize(rows) == summaries
    result = dict(verified=True, cases=completion["cases"], case_scenarios=len(pairs),
                  gradient_rows=len(rows), update_rows=steps,
                  verifier_sha256=sha256(Path(__file__)),
                  small_step_loss_increases=loss_increases,
                  checks=["output hashes", "case membership and unique records", "identical per-case baselines",
                          "semantic and foreground error accounting", "per-class edit accounting",
                          "update budgets", "all metric deltas", "summary recomputation"],
                  limitations="Numerical accounting verification; does not independently rederive all gradients or validate network training.")
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
