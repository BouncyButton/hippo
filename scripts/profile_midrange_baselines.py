"""Screen segmentation baseline configurations around 50--100 GFLOPs/update."""
import hashlib
import json
from pathlib import Path

from profile_intermediate_baselines import audit, intermediate_supplement
from monai.networks.nets import SegResNet, UNet


def main():
    assert audit.monai.__version__ == "1.6.0" and audit.torch.__version__.startswith("2.2.2")
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = intermediate_supplement
    root = Path(__file__).resolve().parents[1]
    output = root / "reports/midrange_baseline_profiles_20261004"
    output.mkdir(parents=True, exist_ok=True)
    budget = json.loads((root / "experiments/compute_budget_reference.json").read_text())["budget_flops"]
    candidates = {
        "segresnet_filters12_deconv": (SegResNet, dict(spatial_dims=3, in_channels=1, out_channels=3,
                                                      init_filters=12, norm=("GROUP", {"num_groups": 4}),
                                                      upsample_mode="deconv")),
        "unet_width40": (UNet, dict(spatial_dims=3, in_channels=1, out_channels=3,
                                   channels=(40, 80, 160, 320, 640), strides=(2, 2, 2, 2), num_res_units=2)),
        "unet_width48": (UNet, dict(spatial_dims=3, in_channels=1, out_channels=3,
                                   channels=(48, 96, 192, 384, 768), strides=(2, 2, 2, 2), num_res_units=2)),
    }
    summary = []
    for name, (constructor, kwargs) in candidates.items():
        audit.matched.build_network = lambda _name, recipe: (constructor(**kwargs), {})
        profiles = []
        for geometry in ("ellipsoid", "box"):
            row = audit.profile_arm("dice", geometry)
            profiles.append(row)
            print(json.dumps({"model": name, "geometry": geometry,
                              "flops_per_update": row["flops_per_update"], "parameters": row["parameters"]}), flush=True)
        assert len({p["flops_per_update"] for p in profiles}) == 1
        schedule = audit.schedule(profiles[0]["flops_per_update"], budget)
        result = {"model": name, "class": constructor.__name__, "kwargs": kwargs,
                  "parameters": profiles[0]["parameters"], "budget_flops": budget,
                  "schedule": schedule, "profiles": profiles,
                  "runtime": {"torch": audit.torch.__version__, "monai": audit.monai.__version__},
                  "accounting": "Same declared arithmetic convention as the official reference. GroupNorm uses analogous fused-normalization estimates. SegResNet uses deconvolution upsampling and four normalization groups, compatible with twelve initial channels.",
                  "purpose": "Synthetic compute profiling only; no patient-data training or validation.",
                  "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__), root / "scripts/profile_intermediate_baselines.py",
                                              Path(audit.__file__), audit.PROFILE_ROOT / "profile_reference.py")}}
        (output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        summary.append({k: result[k] for k in ("model", "class", "kwargs", "parameters", "schedule")})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
