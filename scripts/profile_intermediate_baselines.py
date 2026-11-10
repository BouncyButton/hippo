"""Screen intermediate models using the existing fixed-compute Dice update audit."""
import hashlib
import json
from pathlib import Path

import profile_unet_constraint_budget as audit
from monai.networks.nets import SegResNet, SwinUNETR, UNet


def intermediate_supplement(func, args, kwargs, result):
    value, category = audit.constraint_supplement(func, args, kwargs, result)
    if category != "no_supplement":
        return value, category
    # Apply the reference's declared fused-normalization estimates to GroupNorm.
    # Comparisons, indexing, and memory transfers remain excluded.
    name = str(func).split(".")[1] if str(func).startswith("aten.") else ""
    if name == "native_group_norm":
        affine = int(args[1] is not None) + int(args[2] is not None)
        return (5 + affine) * result[0].numel() + 2 * result[1].numel(), "group_normalization_estimate"
    if name == "native_group_norm_backward":
        mask = args[9]
        affine = args[4] is not None
        return ((8 + int(affine)) * int(mask[0]) + 2 * int(mask[1]) + int(mask[2])) * args[0].numel(), "group_normalization_estimate"
    return value, category


def main():
    assert audit.monai.__version__ == "1.6.0" and audit.torch.__version__.startswith("2.2.2")
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = intermediate_supplement
    root = Path(__file__).resolve().parents[1]
    output = root / "reports/intermediate_baseline_profiles_20261004"
    output.mkdir(parents=True, exist_ok=True)
    budget = json.loads((root / "experiments/compute_budget_reference.json").read_text())["budget_flops"]
    candidates = {
        "unet_width24": (UNet, dict(spatial_dims=3, in_channels=1, out_channels=3,
                                   channels=(24, 48, 96, 192, 384), strides=(2, 2, 2, 2), num_res_units=2)),
        "segresnet_filters8_deconv": (SegResNet, dict(spatial_dims=3, in_channels=1, out_channels=3,
                                                     init_filters=8, upsample_mode="deconv")),
        "segresnet_filters16_deconv": (SegResNet, dict(spatial_dims=3, in_channels=1, out_channels=3,
                                                      init_filters=16, upsample_mode="deconv")),
        "swinunetr_feature12": (SwinUNETR, dict(in_channels=1, out_channels=3, feature_size=12,
                                               use_checkpoint=False, drop_rate=0.0)),
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
        cost = profiles[0]["flops_per_update"]
        schedule = audit.schedule(cost, budget)
        result = {"model": name, "class": constructor.__name__, "kwargs": kwargs,
                  "parameters": profiles[0]["parameters"], "budget_flops": budget,
                  "schedule": schedule, "profiles": profiles,
                  "runtime": {"torch": audit.torch.__version__, "monai": audit.monai.__version__},
                  "accounting": "Same declared arithmetic convention as official reference; GroupNorm uses analogous fused-normalization estimates. SegResNet uses trainable deconvolution upsampling, not its default interpolation.",
                  "purpose": "Compute profiling only; no clinical data training or validation performed.",
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        summary.append({k: result[k] for k in ("model", "class", "kwargs", "parameters", "schedule")})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
