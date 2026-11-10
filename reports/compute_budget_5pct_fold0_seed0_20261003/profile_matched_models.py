"""Profile explicit 64-cubed, batch-one baselines and derive capped schedules."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, "/private/tmp/hippo_bref_networkdeps")
import profile_reference as reference
import monai
import numpy as np
import torch
from monai.networks.nets import UNet, SwinUNETR
from monai.losses import DiceLoss
from torch.utils.flop_counter import FlopCounterMode

ROOT = reference.ROOT
OUT = reference.OUT
sys.path.insert(0, str(ROOT / "baselines/nnUNet"))

BASE_SUPPLEMENT = reference.supplementary_flops


def matched_supplement(func, args, kwargs, result):
    value, category = BASE_SUPPLEMENT(func, args, kwargs, result)
    if category != "no_supplement":
        return value, category
    name = str(func).split(".")[1] if str(func).startswith("aten.") else ""
    if name in {"_prelu_kernel", "prelu"}:
        return result.numel(), "prelu_estimate"
    if name == "_prelu_kernel_backward":
        return 2 * args[0].numel(), "prelu_estimate"
    if name == "_log_softmax":
        return 4 * args[0].numel(), "log_softmax"
    if name == "_log_softmax_backward_data":
        n = args[0].numel()
        return 4 * n - n // args[0].shape[args[2]], "log_softmax"
    if name in {"nll_loss_forward", "nll_loss2d_forward"}:
        return 2 * args[1].numel(), "cross_entropy_estimate"
    if name in {"nll_loss_backward", "nll_loss2d_backward"}:
        return 2 * args[2].numel(), "cross_entropy_estimate"
    if name == "linalg_vector_norm":
        return 2 * args[0].numel(), "gradient_norm_estimate"
    return value, category


reference.supplementary_flops = matched_supplement


def build_network(model_name, recipe):
    if model_name == "unet":
        path = ROOT / "baselines/3dunet/3dunet.py"
        assignment = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model" for t in n.targets))
        call = assignment.value.func.value  # UNet(...).to(device)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        kwargs["out_channels"] = 3
        return UNet(**kwargs), {"class": "monai.networks.nets.UNet", "kwargs": kwargs, "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "adaptations": ["Three classes rather than binary example", "Batch one and one input per patient rather than four random crops and batch two", "Softmax Dice and AdamW matched to reference"]}
    if model_name == "swinunetr":
        kwargs = dict(in_channels=1, out_channels=3, use_checkpoint=False, drop_rate=0.0)
        return SwinUNETR(**kwargs), {"class": "monai.networks.nets.SwinUNETR", "kwargs": kwargs}
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
    pools, strides, kernels, shape, divisor = get_pool_and_conv_props((1, 1, 1), (64, 64, 64), 4, 999999)
    stages = len(strides)
    kwargs = dict(input_channels=1, num_classes=3, n_stages=stages,
                  features_per_stage=[min(32 * 2 ** i, 320) for i in range(stages)],
                  conv_op=torch.nn.Conv3d, kernel_sizes=kernels, strides=strides,
                  n_conv_per_stage=[2] * stages, n_conv_per_stage_decoder=[2] * (stages - 1),
                  conv_bias=True, norm_op=torch.nn.InstanceNorm3d,
                  norm_op_kwargs={"eps": 1e-5, "affine": True},
                  dropout_op=None, dropout_op_kwargs=None,
                  nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True},
                  deep_supervision=recipe == "native")
    model = PlainConvUNet(**kwargs)
    config = {"class": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
              "kwargs": {k: (v.__module__ + "." + v.__name__ if isinstance(v, type) else v) for k, v in kwargs.items()},
              "topology_rule": "Repository default planner: isotropic spacing, fixed 64^3 patch, min bottleneck edge 4, base features 32, max features 320, two convs per stage",
              "plan_origin": "New explicit matched-input configuration; historical plan was not retained",
              "spacing_assumption": [1, 1, 1], "shape": [int(x) for x in shape]}
    (OUT / f"nnunet_matched64_architecture_{recipe}.json").write_text(json.dumps(config, indent=2) + "\n")
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "nnunet", "swinunetr"], required=True)
    parser.add_argument("--recipe", choices=["matched", "native"], default="matched")
    parser.add_argument("--device", choices=["meta", "cpu"], default="meta")
    args = parser.parse_args()
    assert monai.__version__ == "1.6.0"
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model, config = build_network(args.model, args.recipe)
    model.to(args.device).train()
    native = args.model == "nnunet" and args.recipe == "native"
    if native:
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        base_loss = DC_and_CE_loss({"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {}, dice_class=MemoryEfficientSoftDiceLoss)
        weights = np.array([1 / 2 ** i for i in range(config["kwargs"]["n_stages"] - 1)])
        weights[-1] = 0
        weights /= weights.sum()
        criterion = DeepSupervisionWrapper(base_loss, weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=3e-5, momentum=0.99, nesterov=True, foreach=False)
        # Profile the recurring momentum update, rather than its cheaper first
        # initialization. The recurring cost conservatively bounds every step.
        for parameter in model.parameters():
            optimizer.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)
    else:
        criterion = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, foreach=False)
    image = torch.randn((1, 1, 64, 64, 64), device=args.device)
    target = torch.zeros((1, 1, 64, 64, 64), dtype=torch.long, device=args.device)
    audit = reference.OperatorAudit()
    stage = {}
    print(f"Profiling {args.model} {args.recipe} on {args.device}", flush=True)
    with audit, FlopCounterMode(display=False, custom_mapping={torch.ops.aten.convolution_backward: reference.conv_backward_flops}) as counter:
        before = counter.get_total_flops()
        output = model(image)
        stage["forward"] = counter.get_total_flops() - before
        audit.stage = "loss"
        before = counter.get_total_flops()
        if native:
            targets = [torch.zeros((1, 1, *o.shape[2:]), dtype=torch.long, device=args.device) for o in output]
            loss = criterion(output, targets)
        else:
            loss = criterion(output, target) + 0.2 * (output.float().reshape(-1)[0] * 0.0)
        stage["loss"] = counter.get_total_flops() - before
        audit.stage = "backward"
        before = counter.get_total_flops()
        loss.backward()
        stage["backward"] = counter.get_total_flops() - before
        audit.stage = "optimizer"
        before = counter.get_total_flops()
        if native:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False)
        optimizer.step()
        stage["optimizer"] = counter.get_total_flops() - before
    f = sum(stage.values()) + sum(audit.supplement.values())
    # Keep the full reconstructed reference rather than a rounded display value.
    budget = json.loads((OUT / "profile_cpu.json").read_text())["estimated_total_B_ref"]
    max_steps = budget // f
    full_epochs, remainder = divmod(max_steps, 10)
    result = {"model": args.model, "recipe": args.recipe, "device": args.device,
              "torch": torch.__version__, "monai": monai.__version__,
              "parameters": sum(p.numel() for p in model.parameters()),
              "input_shape": list(image.shape), "training_patients": 10,
              "configuration": config, "loss": "deep-supervised foreground Dice+CE" if native else "MONAI softmax Dice, including background",
              "optimizer": "SGD Nesterov + gradient clipping" if native else "AdamW",
              "core_flops_per_stage": stage, "supplementary_estimate_per_stage": audit.supplement,
              "flops_per_update": f, "budget_flops": budget,
              "max_updates": max_steps, "complete_patient_pass_epochs": full_epochs,
              "additional_updates": remainder, "patient_pass_equivalents": max_steps / 10,
              "consumed_flops": max_steps * f, "unused_flops": budget - max_steps * f,
              "next_update_exceeds_budget": (max_steps + 1) * f > budget,
              "operator_audit": audit.ops,
              "limitations": ["Same declared arithmetic-estimate convention as reference; excludes CUDA GradScaler, data processing and validation.",
                              "Patient-pass epochs mean 10 explicitly scheduled patient visits, not nnU-Net's native 250-iteration epochs or a guarantee under random patch sampling.",
                              "Schedule applies to unconstrained baseline; constraints need their own step-cost accounting."]}
    assert result["consumed_flops"] <= budget and result["next_update_exceeds_budget"]
    (OUT / f"matched_{args.model}_{args.recipe}_{args.device}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:result[k] for k in ["model", "recipe", "parameters", "flops_per_update", "max_updates", "complete_patient_pass_epochs", "additional_updates", "unused_flops"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
