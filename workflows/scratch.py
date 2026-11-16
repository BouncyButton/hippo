"""Run one final scratch experiment with its archived data and protocol."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def prepare(args):
    bundle = args.bundle.resolve()
    spec = next((r for r in read(HERE / "scratch_bundles.json")
                 if r["dataset"] == args.dataset and args.seed in r["seeds"]), None)
    if spec is None:
        raise ValueError("Seed is outside the reported cohort")
    with (HERE / "scratch_protocols.csv").open() as stream:
        matches = [r for r in csv.DictReader(stream)
                   if (r["dataset"], r["model"], r["arm"]) == (args.dataset, args.model, args.arm)
                   and str(args.fold) in r["folds"].split("/") and str(args.seed) in r["seeds"].split("/")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching final protocol")
    row, = matches
    folder = bundle / "protocols" / row["protocol"]
    verify(folder / "protocol.json", row["protocol_sha256"])
    protocol = read(folder / "protocol.json")
    for name, key in (("splits_10patients.json", "splits_sha256"), ("nnunet_architecture.json", "nnunet_architecture_sha256")):
        verify(folder / name, protocol[key])
    if "initialization_sha256" in spec:
        verify(bundle / "initialization_reference.json", spec["initialization_sha256"])
    filename = "MNI.pkl" if args.dataset == "MNI" else args.dataset.lower() + "_hippocampus_full.pkl"
    data = args.data.resolve() if args.data else bundle / "data" / filename
    verify(data, protocol["pkl_sha256"])
    output = args.output.resolve() / f"{args.dataset}_{args.model}_{args.arm}_f{args.fold}s{args.seed}"
    if output.is_relative_to(bundle):
        raise ValueError("Use an output directory outside the source bundle")
    if args.dataset == "MNI":
        entry = ["-m", "scripts.mni_constraints.campaign"]
    else:
        entry = [str(HERE.parent / "scripts" / ("colab_" + args.dataset.lower()) / "entry.py")]
    command = [sys.executable, *entry, "--model", protocol["trainer_alias"], "--objective", args.arm,
               "--fold", str(args.fold), "--seed", str(args.seed), "--protocol-dir", str(folder),
               "--pkl", str(data), "--output", str(output / "run"), "--threads",
               "4" if args.dataset == "MNI" else "2", "--resume"]
    source = HERE.parent
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source, source / "scripts",
                                                               source / "baselines/nnUNet"))),
               WANDB_MODE="disabled", HIPPO_COLAB_BUNDLE=str(bundle),
               HIPPO_COLAB_RECOVERY=str(output / "recovery"), MPLCONFIGDIR=str(output / "matplotlib"))
    return command, source, env, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MSD", "ADNI", "MNI"), required=True)
    parser.add_argument("--model", choices=("nnunet", "swinunetr", "basic_unet16", "unetr_vit_b16"), required=True)
    parser.add_argument("--arm", required=True, help="Native objective name, e.g. dice_bands_sdf_depth")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Prepared pickle matching the protocol's data hash")
    parser.add_argument("--output", type=Path, required=True, help="Parent directory for new runs")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs and print the command")
    args = parser.parse_args()
    command, cwd, env, output = prepare(args)
    print(shlex.join(command), flush=True)
    if not args.check_only:
        # Keep concurrent/repeated invocations from reusing a charged run directory.
        output.mkdir(parents=True, exist_ok=False)
        (output / "run").mkdir()
        (output / "recovery").mkdir()
        subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
