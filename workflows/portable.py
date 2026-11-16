"""Run one final HFH scratch or MedSAM3 experiment in a fresh directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOLS = HERE / "protocols"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hippo"
MODELS = ("basic_unet16", "nnunet", "swinunetr", "unetr_vit_b16", "medsam3")
LORA = "499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91"


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
        return checksum.hexdigest()


def verify(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed input: {path}")


def select(dataset, model, arm, fold, seed):
    stem = f"medsam3_{dataset.lower()}" if model == "medsam3" else f"hfh_{model}"
    if dataset != "HFH" and model != "medsam3":
        raise ValueError("Use workflows/scratch.py for scratch models outside HFH")
    catalog = read(PROTOCOLS / (stem + ".json"))
    key = f"{arm}/{fold}/{seed}"
    if key not in catalog["cells"]:
        raise ValueError(f"Unknown cell {dataset}/{model}/{key}; use --list")
    selection = catalog["profiles"][catalog["cells"][key]]
    protocol = dict(catalog["common"])
    for field, index in selection["fields"].items():
        protocol[field] = catalog["variants"][field][index]
    return protocol, selection


def validate_split(train, val, *, paired):
    if not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Empty partition or duplicate crop in frozen split")
    subject = (lambda name: name.rsplit("_", 1)[0]) if paired else (lambda name: name)
    if {subject(x) for x in train} & {subject(x) for x in val}:
        raise ValueError("Subject in both support and evaluation")


def check_arrays(folder, inventory, names):
    import numpy as np
    for name in sorted(set(names)):
        with np.load(folder / (name + ".npz"), allow_pickle=False) as item:
            for key, expected in inventory[name].items():
                array = item[key]
                actual = [str(array.dtype), list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]
                if actual != expected:
                    raise ValueError(f"Changed {key} array for {name}")


def bind(args, protocol):
    data = args.data
    dataset, model = args.dataset, args.model
    if model == "medsam3":
        if not (args.base_weights and args.lora_weights):
            raise ValueError("MedSAM3 requires --base-weights and --lora-weights")
        verify(args.base_weights, protocol["base_sha256"])
        verify(args.lora_weights, LORA)
        if protocol.get("lora_sha256", LORA) != LORA:
            raise ValueError("LoRA identity differs from catalog")
        for name, checksum in read(PROTOCOLS / "upstream.json")["files"].items():
            verify(args.upstream / name, checksum)
    if model != "medsam3":
        split = read(PROTOCOLS / "hfh/splits_10patients.json")[args.fold]
        train, val = split["train"], split["val"]
    else:
        train, val = protocol["train_cases"], protocol["evaluation_cases"]
    if dataset == "ADNI":
        # Validate bytes before unpickling. Only the verified research input is accepted.
        verify(data, protocol["dataset_sha256"])
        from scripts.train_compute_matched_baselines import _load_pkl_dataframe
        from scripts.adni_baselines.core import case_name
        records = [(case_name(r), str(r.subject_id)) for r in _load_pkl_dataframe(data).itertuples()]
        names = [name for name, _ in records]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ADNI crop identity")
        by_case = dict(records)
        ids = set(names) | set(by_case.values())
        lookup = {"sha256:" + hashlib.sha256(x.encode()).hexdigest(): x for x in ids}
        for field in ("train_cases", "evaluation_cases", "support_subjects"):
            protocol[field] = [lookup[token] for token in protocol[field]]
        train, val = protocol["train_cases"], protocol["evaluation_cases"]
        support = {by_case[name] for name in train}
        if support != set(protocol["support_subjects"]) or support & {by_case[name] for name in val}:
            raise ValueError("ADNI participant assignment differs from the frozen split")
    elif dataset == "MSD":
        for name, checksum in protocol["data_hashes"].items():
            verify(data / name, checksum)
        verify(PROTOCOLS / "msd/splits_final.json", protocol["split_sha256"])
    else:
        check_arrays(data, read(PROTOCOLS / (dataset.lower() + "_files.json")), train + val)
    validate_split(train, val, paired=dataset != "MSD")
    if "frozen_sample_schedule" in protocol:
        count = protocol["updates"] * protocol["accumulation"]
        order = protocol["frozen_sample_schedule"]
        if len(order) != count or any(c not in train or z < 0 for c, z in order):
            raise ValueError("Frozen samples outside training support or update budget")
    return train, val


def check_runtime(args):
    import torch
    if args.device == "cpu" and args.smoke_steps:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a CUDA GPU with native BF16; --check-only runs on CPU")
    if torch.__version__.split("+")[0] != "2.12.1":
        raise ValueError("Use the profiled PyTorch 2.12.1 runtime for full training")
    if args.model == "medsam3" and torch.cuda.get_device_properties(0).total_memory < 38 * 1024**3:
        raise ValueError("MedSAM3 requires a GPU with at least 40 GB of VRAM")


def run(args, p, selection):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    worker = selection["worker"]
    extra = selection.get("inputs", {})
    lock = read(PROTOCOLS / "release.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(x) for x in
               (ROOT, ROOT / "scripts", ROOT / "baselines/nnUNet", ROOT / "scripts/colab_medsam3")),
               MNI_MANIFEST_SHA=digest(PROTOCOLS / "release.json"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    def execute(*command):
        subprocess.run([sys.executable, *map(str, command)], env=env, cwd=ROOT, check=True)

    protocol = inputs / "protocol.json"
    output = out / "training"
    if args.model != "medsam3":
        for name in ("splits_10patients.json", "nnunet_architecture.json", "baseline_references.json"):
            (inputs / name).symlink_to(PROTOCOLS / "hfh" / name)
        index = inputs / "data_index.json"
        write(index, dict(data=str(args.data), cases=sorted(read(PROTOCOLS / "hfh_files.json"))))
        p.update(data=str(index), pkl_sha256=digest(index))
        if "anchor" in extra:
            path = inputs / "native_prefix.json"
            write(path, extra["anchor"])
            p["a100_native_anchor"] = str(path)
        if args.model == "nnunet":
            p["warm_ready_source_sha256"] = {name: lock["scripts/colab_hfh/" + name]
                for name in ("nnunet_warm_ready.py", "nnunet_warm_ready_worker.py")}
        write(protocol, p)
        flags = ["--device", args.device, "--smoke-steps", args.smoke_steps]
        execute("-m", "scripts.colab_hfh." + worker, "--protocol-dir", inputs, "--pkl", index,
                "--fold", args.fold, "--seed", args.seed, "--output", output, "--durable", out / "recovery", *flags)
        return
    if args.dataset == "MSD":
        data = inputs / "data"
        data.mkdir()
        for name in ("imagesTr", "labelsTr"):
            (data / name).symlink_to(args.data / name, target_is_directory=True)
        (data / "splits_final.json").symlink_to(PROTOCOLS / "msd/splits_final.json")
        if worker == "msd_original":
            write(protocol, p)
            flags = []
            if p["calibration_source_run"] != p["study_run"]:
                cal = PROTOCOLS / "msd" / f"calibration{args.fold}.json"
                flags = ["--calibration-input", cal, "--calibration-sha256", digest(cal)]
            execute("-m", "scripts.train_medsam3_multiclass", "--protocol", protocol, "--data", data,
                    "--base-weights", args.base_weights, "--lora-weights", args.lora_weights,
                    "--upstream", args.upstream, "--output", output, *flags)
        else:
            (inputs / "upstream").symlink_to(args.upstream, target_is_directory=True)
            (inputs / "source").symlink_to(ROOT, target_is_directory=True)
            engine = ROOT / "scripts/colab_medsam3/engine.py"
            (inputs / "engine.py").symlink_to(engine)
            for name, path in (("weights/sam3.pt", args.base_weights),
                               ("weights/best_lora_weights.pt", args.lora_weights),
                               (f"calibration/fold{args.fold}.json", PROTOCOLS / "msd" / f"calibration{args.fold}.json")):
                link = inputs / name
                link.parent.mkdir(exist_ok=True)
                link.symlink_to(path)
            for s in {17, args.seed}:
                q, _ = select("MSD", "medsam3", args.arm, args.fold, s)
                write(inputs / "protocols" / f"fold{args.fold}_seed{s}.json", q)
            objective = "baseline" if args.arm == "dice" else args.arm.removeprefix("dice_")
            task = dict(id=f"MSD_MedSAM3_{objective}_f{args.fold}s{args.seed}",
                        fold=args.fold, seed=args.seed, arm=objective)
            write(inputs / "plan.json", dict(tasks=[task]))
            write(inputs / "bundle_manifest.json", dict(release=digest(PROTOCOLS / "release.json"),
                  fold=args.fold, task=task, protocol=digest(inputs / "protocols" / f"fold{args.fold}_seed{args.seed}.json")))
            (inputs / "work").symlink_to(output, target_is_directory=True)
            output.mkdir()
            flags = ["--bundle", inputs, "--durable", out / "recovery"]
            execute(engine, "preflight", *flags, "--fold", args.fold)
            execute(engine, "train", *flags, "--task", task["id"])
        return
    p.update(data=str(args.data), upstream=str(args.upstream), lora_weights=str(args.lora_weights))
    if "paired_calibration" in extra:
        p["paired_calibration"] = str(PROTOCOLS / "hfh" / extra["paired_calibration"])
        p["paired_calibration_sha256"] = lock["workflows/protocols/hfh/" + extra["paired_calibration"]]
    if "matrix_worker" in worker:
        p["matrix_source_hashes"] = {name.rsplit("/", 1)[1]: checksum for name, checksum in lock.items()
                                     if name.startswith("scripts/colab_hfh/") and name.endswith(".py")}
    write(protocol, p)
    flags = ["--protocol", protocol, "--base-weights", args.base_weights, "--arm", args.arm, "--output", output]
    if args.dataset == "HFH":
        execute("-m", "scripts.colab_hfh." + worker, *flags, "--durable", out / "recovery")
    elif args.dataset == "MNI":
        execute(HERE / "mni_adapter.py", *flags)
    else:
        execute("-m", "scripts.cross_dataset_campaign.adni_foundation_full_runtime", *flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("HFH", "MNI", "MSD", "ADNI"))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--arm")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    for name in ("data", "base-weights", "lora-weights", "upstream", "output"):
        parser.add_argument("--" + name, type=lambda s: Path(s).expanduser().resolve())
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="HFH scratch worker device")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Limit HFH scratch training to a smoke test")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--show-protocol", action="store_true", help="Print the resolved final protocol without data, weights or training")
    mode.add_argument("--check-only", action="store_true", help="Verify release, data, weights and split without training")
    args = parser.parse_args()
    if args.smoke_steps < 0 or (args.model == "medsam3" and (args.smoke_steps or args.device != "cuda")):
        parser.error("Only scratch training supports a CPU device or a positive smoke-step limit")
    if args.dataset != "HFH" and args.model != "medsam3":
        parser.error("Use workflows/scratch.py for these models")
    if args.list:
        group = "medsam3_" + args.dataset.lower() if args.model == "medsam3" else "hfh_" + args.model
        print("\n".join(read(PROTOCOLS / (group + ".json"))["cells"]))
        return
    if None in (args.arm, args.fold, args.seed):
        parser.error("Supply --arm, --fold and --seed to select a run")
    p, selection = select(args.dataset, args.model, args.arm, args.fold, args.seed)
    if args.show_protocol:
        print(json.dumps(dict(worker=selection["worker"], protocol=p,
                              inputs=selection.get("inputs", {})), indent=2))
        return
    if args.data is None or (not args.check_only and args.output is None):
        parser.error("Supply --data and a new --output (unless --check-only)")
    for name, checksum in read(PROTOCOLS / "release.json").items():
        verify(ROOT / name, checksum)
    if args.upstream is None:
        args.upstream = CACHE / ("medsam3-" + read(PROTOCOLS / "upstream.json")["commit"])
    train, val = bind(args, p)
    print(f"{args.dataset}/{args.model}/{args.arm} fold={args.fold} seed={args.seed}: "
          f"verified inputs, {len(train)} training and {len(val)} evaluation crops", flush=True)
    if not args.check_only:
        if args.device == "cpu" and not args.smoke_steps:
            parser.error("Full HFH runs use CUDA BF16; use --smoke-steps for a CPU smoke run")
        check_runtime(args)
        run(args, p, selection)


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "baselines/nnUNet")]
    main()
