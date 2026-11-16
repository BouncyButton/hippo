"""Run the actual archived HFH gain-only native trainer on all forty training crops."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ORIGINAL = PROJECT / "reports/hfh_colab_exploration_20261111"
BUNDLE = (Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())) / 'hfh_medsam3_seed2_20261114/bundle')
KEYS = [f"HFH_medsam3_f{fold}_s{seed}" for fold in range(3) for seed in range(3)]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(inputs, output):
    files = {}
    with zipfile.ZipFile(ORIGINAL / "medsam_adaptive_matrix_replication/replication_source.zip") as source:
        code_hashes = {n: sha(source.read(n)) for n in source.namelist() if n.endswith(".py")}
    with zipfile.ZipFile(inputs) as data:
        data_hashes = {n: sha(data.read(n)) for n in data.namelist() if n.startswith("data/") and n.endswith(".npz")}
    assert len(data_hashes) == 50
    files["original_inputs.json"] = json.dumps(dict(code=code_hashes, data=data_hashes), indent=2).encode()
    splits = json.loads((HERE / "splits.json").read_bytes())
    for key in KEYS:
        fold, seed = int(key[-4]), int(key[-1])
        evidence = ORIGINAL / f"medsam_adaptive_matrix_replication/seed2/matrix125__dice__f{fold}_s2.zip"
        with zipfile.ZipFile(evidence) as z:
            low = json.loads(z.read("results/protocol.json"))
        old_full = HERE / "archive/colab" / (key + "_coverage_control_results.zip")
        with zipfile.ZipFile(old_full) as z:
            previous = json.loads(z.read("protocol.json"))
            schedule = json.loads(z.read("slab_schedule.json"))
        split = splits[f"HFH_medsam3_f{fold}"]
        assert low["augmentation"]["component"] == "gain" and low["training_numerics"] == "strict_empty_grid_v1"
        assert split["validation"] == low["evaluation_cases"] == previous["evaluation_cases"]
        assert set(low["train_cases"]) <= set(split["train"]) == {case for case, _ in schedule}
        assert len(split["train"]) == 40 and len(schedule) == 250
        p = dict(low, name=key + "_gain_coverage_control", seed=seed, train_cases=split["train"],
                 support_subjects=previous["support_subjects"], train_subjects=20, train_crops=40,
                 frozen_sample_schedule=schedule, support_selection="All 20 training subjects, both crops.",
                 sampling="250 adjacent-slice draws from the frozen full-coverage schedule.",
                 comparison="All training crops, original native loss, gain-only augmentation, strict numerics and 125-update stop.")
        p.pop("arm")
        assert p["updates"] == 125 and len(p["frozen_learning_rates"]) == p["learning_rate_horizon_updates"] == 200
        files[f"protocols/{key}.json"] = json.dumps(p, indent=2).encode()
        files[f"references/{key}.json"] = json.dumps(dict(low_control=evidence.name, low_control_sha256=sha(evidence.read_bytes()),
                frozen_rate_sha256=sha(json.dumps(low["frozen_learning_rates"]).encode()), previous_full_sha256=sha(old_full.read_bytes())), indent=2).encode()
    for n in ("cluster_hfh_gain.py", "cluster_hfh_gain.sbatch"):
        files[n] = (HERE / n).read_bytes()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for n, raw in files.items():
            z.writestr(n, raw)
        z.writestr("SHA256SUMS", "".join(f"{sha(raw)}  {n}\n" for n,raw in sorted(files.items())))


def preflight():
    from scripts.cross_dataset_campaign.common import digest
    checks = json.loads((HERE / "original_inputs.json").read_bytes())
    for group in checks.values():
        for name, expected in group.items():
            assert digest(BUNDLE / name) == expected, name
    p = json.loads((HERE / "protocols" / (KEYS[0] + ".json")).read_bytes())
    assert digest(p["lora_weights"]) == p["lora_sha256"] and digest(os.environ["HFH_BASE_WEIGHTS"]) == p["base_sha256"]


def archive(key):
    from scripts.cross_dataset_campaign.common import digest, read, write
    from scripts.cross_dataset_campaign.storage import prepare
    out = HERE / "work" / key
    spec = HERE / "protocols" / (key + ".json")
    p = read(spec)
    assert read(out / "protocol.json") == dict(p, arm="dice")
    assert read(out / "slab_schedule.json") == p["frozen_sample_schedule"]
    tape = read(out / "augmentation_schedule.json")
    assert [(x["case"], x["anchor"]) for x in tape] == [tuple(s) for s in p["frozen_sample_schedule"]]
    assert len(tape) == 250 and all(x["component"] == "gain" for x in tape) and any(x["gain"] != 1 for x in tape)
    assert [x["lr"] for x in read(out / "training_history.json")] == p["frozen_learning_rates"][:125]
    assert read(out / "training_summary.json")["augmentation"] == p["augmentation"]
    write(out / "cluster_protocol.json", dict(key=key, input_manifest=digest(HERE / "SHA256SUMS"),
          protocol_sha256=digest(spec), allocation=os.environ["SLURM_JOB_ID"], comparison="gain-matched full-data native control"))
    prepare(out, dict(id=key, dataset="HFH", kind="foundation", fold=p["fold"], seed=p["seed"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("inputs", type=Path)
    build_parser.add_argument("output", type=Path)
    sub.add_parser("preflight")
    sub.add_parser("archive").add_argument("key", choices=KEYS)
    args = parser.parse_args()
    if args.command == "build":
        build(args.inputs, args.output)
    elif args.command == "archive":
        archive(args.key)
    else:
        preflight()
