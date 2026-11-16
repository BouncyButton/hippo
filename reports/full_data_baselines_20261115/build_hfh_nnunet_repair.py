"""Create the isolated HFH nnU-Net 0.1 recipe beside a staged scratch bundle."""
import argparse
import json
from pathlib import Path
import shutil


def dump(path, value):
    contents = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text() != contents:
        raise ValueError(f"Existing recipe differs at {path}")
    path.write_text(contents)


def build(original, repair):
    if repair.resolve() == original.resolve():
        raise ValueError("Use a separate output directory for the repaired experiments")
    cells = json.loads((original / "cells.json").read_text())
    corrected = {}
    for key, row in cells.items():
        if row["dataset"] != "HFH" or row["model"] != "nnunet":
            continue
        parent = original / "protocols" / key
        protocol = json.loads((parent / "protocol.json").read_text())
        alias = protocol["trainer_alias"]
        assert row["lr"] == protocol["initial_lrs"][alias] == 0.01
        assert protocol["lr_horizon_updates"] == 174 and protocol["augmentation"] == "mild_v1"
        assert row["target_updates"] == 90
        corrected[key] = dict(row, lr=0.1)
        path = repair / "protocols" / key
        path.mkdir(exist_ok=True, parents=True)
        for name in ("splits_10patients.json", "nnunet_architecture.json"):
            source = parent / name
            dest = path / name
            if dest.exists() and dest.read_bytes() != source.read_bytes():
                raise ValueError(f"Existing input differs at {dest}")
            shutil.copy2(source, dest)
        protocol["initial_lrs"][alias] = 0.1
        protocol["archive_recipe_differences"]["full_data_lr"] = 0.1
        dump(path / "protocol.json", protocol)
        for name in ("data", protocol["data_file"]):
            dest = repair / name
            source = (original / name).resolve(strict=True)
            if dest.is_symlink() or dest.exists():
                if dest.resolve(strict=True) != source:
                    raise ValueError(f"Unexpected input at {dest}")
            else:
                dest.symlink_to(source)
    assert len(corrected) == 9
    dump(repair / "cells.json", corrected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("repair", type=Path)
    args = parser.parse_args()
    build(args.original, args.repair)
