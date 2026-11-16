"""Measure training-patch graphs against each archived native-volume control."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.cross_dataset_campaign import conventional, profile_adni_nnunet as adni, profile_msd_nnunet as msd
from profile_intermediate_baselines import audit, intermediate_supplement

D = Path(__file__).resolve().parent


def measure(dataset, model):
    shape = (48, 48, 32) if model == "nnunet" else (64, 64, 32)
    with zipfile.ZipFile(D / "matched_controls" / (dataset + "_matched.zip")) as z:
        parent = f"protocols/{dataset}_{model}_f0_s0/"
        protocol = json.loads(z.read(parent + "protocol.json"))
        architecture = json.loads(z.read(parent + "nnunet_architecture.json"))
    expected = protocol["models"][protocol["trainer_alias"]]["objectives"]["dice"]["flops_per_update"]
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = intermediate_supplement
    module = msd if dataset == "MSD" else adni
    module.dice_only_optimizer = conventional.ORIGINAL_LOSS_OPTIMIZER
    if dataset == "ADNI":
        adni.obj.bind()
    def row(size, geometry):
        if dataset == "MSD":
            return msd.profile(model, "dice", geometry, False, architecture, protocol, spatial_shape=size)
        return adni.measure(model, architecture, protocol, "dice", geometry, False, spatial_shape=size)
    retained = []
    for size, geometry in [((64,64,64),"ellipsoid"), (shape,"ellipsoid"), (shape,"box")]:
        result = row(size, geometry)
        retained.append(dict(spatial_shape=size, **result))
        print(dataset, model, size, geometry, result["flops"], flush=True)
        if size == (64,64,64):
            assert result["flops"] == expected, (result["flops"], expected)
    cost = retained[1]["flops"]
    assert cost == retained[2]["flops"] and cost < expected
    cap = protocol["budget_flops"]
    assert cap // cost >= protocol["train_count"]
    directory = D / "matched_controls" / "profiles"
    directory.mkdir(exist_ok=True)
    output = directory / f"{dataset}_{model}.json"
    data = dict(dataset=dataset, model=model, shape=shape, native_shape=[64]*3,
                native_flops=expected, flops_per_update=cost, max_updates=cap//cost, cap=cap,
                torch=audit.torch.__version__, monai=audit.monai.__version__,
                profiler_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                profiles=retained)
    pending = output.with_suffix(".partial")
    pending.write_text(json.dumps(data, indent=2))
    pending.replace(output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("MSD","ADNI"))
    parser.add_argument("model", choices=("nnunet","swinunetr"))
    args=parser.parse_args()
    measure(args.dataset, args.model)
