"""Profile the published UNETR preset at 64 cubed; do not train on clinical data."""
import hashlib
import json
from pathlib import Path

from monai.networks.nets import UNETR
from profile_intermediate_baselines import audit, intermediate_supplement


def main():
    if audit.monai.__version__ != "1.6.0" or not audit.torch.__version__.startswith("2.2.2"):
        raise ValueError("Use the established profiling runtime")
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = intermediate_supplement
    kwargs = dict(in_channels=1, out_channels=3, img_size=(64, 64, 64),
                  feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                  proj_type="perceptron", norm_name="instance", conv_block=True,
                  res_block=True, dropout_rate=0.0, spatial_dims=3, qkv_bias=False)
    audit.matched.build_network = lambda name, recipe: (UNETR(**kwargs), {})
    profiles = []
    for geometry in ("ellipsoid", "box"):
        row = audit.profile_arm("dice", geometry)
        profiles.append(row)
        print(json.dumps({"geometry": geometry, "flops_per_update": row["flops_per_update"],
                          "parameters": row["parameters"]}), flush=True)
    if len({row["flops_per_update"] for row in profiles}) != 1:
        raise ValueError("Target geometry changed the arithmetic count")
    root = Path(__file__).resolve().parents[1]
    reference = root / "experiments/compute_budget_reference.json"
    budget = json.loads(reference.read_text())["budget_flops"]
    sources = [Path(__file__), root / "scripts/profile_intermediate_baselines.py",
               Path(audit.__file__), audit.PROFILE_ROOT / "profile_reference.py",
               audit.PROFILE_ROOT / "profile_matched_models.py",
               root / "scripts/compute_constraint_objectives.py"]
    result = {"status": "profiled_only", "model": "unetr_vit_b16", "class": "monai.networks.nets.UNETR",
              "kwargs": kwargs, "parameters": profiles[0]["parameters"],
              "budget_flops": budget, "schedule": audit.schedule(profiles[0]["flops_per_update"], budget),
              "profiles": profiles, "runtime": {"torch": audit.torch.__version__, "monai": audit.monai.__version__},
              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "architecture_source": "https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV",
              "adaptations": "Published feature16/ViT-B16 preset in MONAI 1.6.0, with input64 cubed and three output labels. No pretrained weights.",
              "objective": "Same Dice-only plus connected-zero graph and AdamW arithmetic as the current reference; a Dice+CE recipe needs separate profiling.",
              "purpose": "Synthetic arithmetic audit only; does not predict validation Dice or authorize a training job."}
    output = root / "reports/unetr_candidate_20261005"
    output.mkdir(parents=True, exist_ok=True)
    (output / "unetr_vit_b16_profiles.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"parameters": result["parameters"], "schedule": result["schedule"]}), flush=True)


if __name__ == "__main__":
    main()
