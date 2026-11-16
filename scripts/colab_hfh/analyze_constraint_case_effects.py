"""Describe paired foreground errors from independently verified HFH archives."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile

from scripts.colab_hfh.summarize_constraints import verify


def analyze(directory: Path, payload: Path):
    references = json.loads(payload.read_text())["references"]
    rows, pairs = [], []
    for path in sorted(directory.glob("*.zip")):
        pair = verify(path, references)
        pairs.append(pair)
        key = f'{pair["model"]}_f{pair["fold"]}_s{pair["seed"]}'
        baseline = {c["case_name"]: c for c in references[key]["validation"]["cases"]}
        with zipfile.ZipFile(path) as archive:
            final = json.loads(archive.read("results/validation_latest.json"))["cases"]
        assert len(final) == len(baseline)
        assert {c["case_name"] for c in final} == set(baseline)
        local = []
        for case in final:
            before = baseline[case["case_name"]]
            native = before["per_class"]["hippocampus"]
            constrained = case["per_class"]["hippocampus"]
            volume = native["tp"] + native["fn"]
            assert volume == constrained["tp"] + constrained["fn"] and volume > 0
            native_ratio = (native["tp"] + native["fp"]) / volume
            local.append(dict(model=pair["model"], fold=pair["fold"], seed=pair["seed"],
                case=case["case_name"], subject=case["case_name"].rsplit("_h", 1)[0],
                true_volume=volume, baseline_volume_ratio=native_ratio,
                constrained_volume_ratio=(constrained["tp"] + constrained["fp"]) / volume,
                baseline_dice=before["macro_hard_dice"], final_dice=case["macro_hard_dice"],
                delta_pp=100 * (case["macro_hard_dice"] - before["macro_hard_dice"]),
                baseline_fp=native["fp"], final_fp=constrained["fp"],
                baseline_fn=native["fn"], final_fn=constrained["fn"],
                baseline_volume_bias="under" if native_ratio < 1 else "over_or_equal"))
        assert abs(statistics.mean(r["delta_pp"] for r in local) - pair["delta_dice_pp"]) < 1e-9
        for label, column in (("baseline", "baseline"), ("constraint", "final")):
            for error in ("fp", "fn"):
                assert sum(r[f"{column}_{error}"] for r in local) == pair[f"{label}_foreground_{error}"]
        rows.extend(local)
    if not rows:
        raise ValueError("No verified archives")

    def summarize(group):
        return dict(case_seed_observations=len(group), unique_subjects=len({r["subject"] for r in group}),
            mean_delta_pp=statistics.mean(r["delta_pp"] for r in group),
            positive_observations=sum(r["delta_pp"] > 0 for r in group),
            delta_false_positives=sum(r["final_fp"] - r["baseline_fp"] for r in group),
            delta_false_negatives=sum(r["final_fn"] - r["baseline_fn"] for r in group),
            mean_baseline_volume_ratio=statistics.mean(r["baseline_volume_ratio"] for r in group))

    result = dict(recipe=directory.name, verified_pairs=len(pairs),
        complete_nine=len(pairs) == 9 and {(r["fold"], r["seed"]) for r in pairs} == {(f,s) for f in range(3) for s in range(3)},
        interpretation="Descriptive post-hoc validation analysis. Repeated seeds and hemispheres are not independent subjects. Volume bias does not locate boundary errors or establish a causal mechanism.",
        by_baseline_volume_bias={label:summarize([r for r in rows if r["baseline_volume_bias"] == label])
            for label in sorted({r["baseline_volume_bias"] for r in rows})},
        by_subject={subject:summarize([r for r in rows if r["subject"] == subject])
            for subject in sorted({r["subject"] for r in rows})},
        source_archives=[dict(fold=r["fold"],seed=r["seed"],sha256=r["archive_sha256"]) for r in pairs])
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--payload", type=Path, default=Path("reports/hfh_colab_exploration_20261111/bands_v1_payload.json"))
    args = parser.parse_args()
    result, rows = analyze(args.directory, args.payload)
    (args.directory / "case_effects_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.directory / "case_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ("source_archives", "by_subject")}, indent=2))


if __name__ == "__main__":
    main()
