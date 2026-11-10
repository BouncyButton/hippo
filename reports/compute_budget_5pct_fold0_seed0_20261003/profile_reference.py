"""Reconstruct historical SwinUNETR arithmetic costs without accessing MRI data."""
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hippo_bref_mpl")
if os.environ.get("HIPPO_PROFILE_MONAI_PATH"):
    sys.path.insert(0, os.environ["HIPPO_PROFILE_MONAI_PATH"])

import monai
import torch
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG = ROOT / "reports/separated_edge_generalization_20260929/historical_baseline_configs/low_data_noaug_seed0_20260917_01/runs/baseline_5pct_seed0_noaug/config.json"
SOURCE = ROOT / "reports/separated_edge_generalization_20260929/historical_baseline_trainer.py"


def conv_backward_flops(grad_shape, input_shape, weight_shape, bias, stride,
                        padding, dilation, transposed, output_padding, groups,
                        output_mask, *, out_shape=None, **kwargs):
    """Dense MAC accounting for requested input/weight convolution gradients.

    Each requested gradient uses the forward multiply-add connectivity count.
    This avoids the incorrect weight-gradient formula in torch 2.2.2's counter.
    Padding follows the conventional dense-kernel convention.
    """
    assert groups == 1
    spatial = input_shape[2:] if transposed else grad_shape[2:]
    forward = 2 * input_shape[0] * math.prod(spatial) * math.prod(weight_shape)
    return forward * (int(output_mask[0]) + int(output_mask[1]))


def supplementary_flops(func, args, kwargs, result):
    """Declared scalar-operation estimates for kernels outside the dense counter.

    Basic arithmetic, sqrt, rsqrt, exp and erf each count as one operation.
    Fused normalization and GELU backward use explicit approximation formulas.
    """
    name = str(func).split(".")[1] if str(func).startswith("aten.") else ""
    outputs = [x for x in tree_flatten(result)[0] if isinstance(x, torch.Tensor) and x.is_floating_point()]
    n = outputs[0].numel() if outputs else 0
    simple = {"add", "add_", "sub", "sub_", "rsub", "mul", "mul_", "div", "div_", "neg", "sqrt", "rsqrt", "exp", "erf", "pow"}
    if name in simple:
        return n, "scalar_arithmetic"
    if name in {"lerp", "lerp_", "addcmul", "addcmul_", "addcdiv", "addcdiv_"}:
        return 3 * n, "scalar_arithmetic"
    if name in {"sum", "mean"} and args[0].is_floating_point():
        return args[0].numel() - n + (n if name == "mean" else 0), "reductions"
    if name == "_softmax":
        groups = n // args[0].shape[args[1]]
        return 4 * n - groups, "softmax"  # shift, exp, sum, divide; maximum comparison excluded
    if name == "_softmax_backward_data":
        groups = n // args[1].shape[args[2]]
        return 4 * n - groups, "softmax"
    if name == "gelu":
        return 5 * n, "gelu_estimate"
    if name == "gelu_backward":
        return 10 * n, "gelu_estimate"
    if name in {"leaky_relu", "leaky_relu_", "leaky_relu_backward"}:
        return n, "activation_estimate"  # one multiplication per element
    if name in {"native_layer_norm", "native_batch_norm"}:
        r = result[1].numel()
        affine_index = 2 if name == "native_layer_norm" else 1
        affine = int(args[affine_index] is not None) + int(args[affine_index + 1] is not None)
        return (5 + affine) * n + 2 * r, "normalization_estimate"
    if name in {"native_layer_norm_backward", "native_batch_norm_backward"}:
        # Common two-reduction input gradient plus optional affine gradients.
        if name == "native_layer_norm_backward":
            mask = args[7]
            affine = args[5] is not None
        else:
            mask = args[9]
            affine = args[2] is not None
        numel = args[0].numel()
        return ((8 + int(affine)) * int(mask[0]) + 2 * int(mask[1]) + int(mask[2])) * numel, "normalization_estimate"
    if name == "convolution":
        return n if args[2] is not None else 0, "bias"
    if name == "convolution_backward":
        return args[0].numel() - args[0].shape[1] if args[10][2] else 0, "bias"
    if name in {"addmm", "baddbmm"}:
        return n, "bias"
    if name == "index_put" and len(args) > 3 and args[3]:
        return args[2].numel(), "scatter_add"
    return 0, "no_supplement"


class OperatorAudit(TorchDispatchMode):
    """Record dispatched operations, tensor sizes, and stage for coverage review."""
    def __init__(self):
        super().__init__()
        self.stage = "forward"
        self.ops = {}
        self.supplement = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        tensors = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
        outputs = [x for x in tree_flatten(result)[0] if isinstance(x, torch.Tensor)]
        key = self.stage + ":" + str(func)
        row = self.ops.setdefault(key, {"calls": 0, "input_float_elements": 0, "output_float_elements": 0, "example_shapes": [list(x.shape) for x in tensors]})
        row["calls"] += 1
        row["input_float_elements"] += sum(x.numel() for x in tensors if x.is_floating_point())
        row["output_float_elements"] += sum(x.numel() for x in outputs if x.is_floating_point())
        extra, category = supplementary_flops(func, args, kwargs, result)
        row["supplementary_flops"] = row.get("supplementary_flops", 0) + extra
        row["supplementary_category"] = category
        self.supplement[self.stage] = self.supplement.get(self.stage, 0) + extra
        return result


def main():
    cfg = json.loads(CONFIG.read_text())
    run = cfg["run"]
    if monai.__version__ != cfg["runtime_provenance"]["monai"]:
        raise RuntimeError("Use the historical MONAI version; see this report's README.")
    assert run["constraint_set"] == "none"
    assert run["seed"] == run["fold"] == 0
    assert cfg["train_samples"] == 10 and run["epochs"] == 75
    torch.manual_seed(run["seed"])
    torch.set_num_threads(4)
    namespace = {"torch": torch, "math": math, "inspect": inspect, "SwinUNETR": SwinUNETR, "Any": Any}
    builder = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "build_swinunetr")
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(SOURCE), "exec"), namespace)
    model = namespace["build_swinunetr"](tuple(run["spatial_size"]), 3, torch.device("cpu"), drop_rate=run["drop_rate"], activation_checkpointing=run["activation_checkpointing"])
    device = os.environ.get("HIPPO_PROFILE_DEVICE", "meta")
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=run["learning_rate"], weight_decay=run["weight_decay"], foreach=False)
    image = torch.randn((run["batch_size"], 1, *run["spatial_size"]), device=device)
    label = torch.zeros((run["batch_size"], 1, *run["spatial_size"]), dtype=torch.long, device=device)
    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    audit = OperatorAudit()
    stages = {}
    print(f"Profiling MONAI {monai.__version__}, torch {torch.__version__}, device {device}", flush=True)
    with audit, FlopCounterMode(display=False, custom_mapping={torch.ops.aten.convolution_backward: conv_backward_flops}) as counter:
        start = counter.get_total_flops()
        logits = model(image)
        stages["forward"] = counter.get_total_flops() - start
        audit.stage = "loss"
        start = counter.get_total_flops()
        # The baseline objective returns a differentiably connected zero.
        loss = criterion(logits, label) + 0.2 * (logits.float().reshape(-1)[0] * 0.0)
        stages["loss"] = counter.get_total_flops() - start
        audit.stage = "backward"
        start = counter.get_total_flops()
        loss.backward()
        stages["backward"] = counter.get_total_flops() - start
        audit.stage = "optimizer"
        start = counter.get_total_flops()
        optimizer.step()
        stages["optimizer"] = counter.get_total_flops() - start
    counted = {str(op): int(n) for op, n in counter.flop_counts["Global"].items()}
    total = sum(stages.values())
    estimated_total = total + sum(audit.supplement.values())
    batches = math.ceil(cfg["train_samples"] / run["batch_size"])
    steps = batches * run["epochs"]
    evidence = json.loads((ROOT / "reports/hippocampus_research_review_20260924/evidence/normalized.json").read_text())
    historical = next(r for r in evidence["runs"] if r["path"].endswith("/baseline_5pct_seed0_noaug"))
    assert historical["optimizer_updates_at_stop"] == steps
    report = {
        "run": historical["path"], "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "historical_runtime": cfg["runtime_provenance"],
        "profiling_runtime": {"torch": torch.__version__, "monai": monai.__version__, "device": device},
        "model_constructor": ast.unparse(builder), "model_signature": str(inspect.signature(SwinUNETR)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": list(image.shape), "epochs": run["epochs"], "batches_per_epoch": batches,
        "training_steps": steps, "selected_checkpoint_epoch": historical["selected"],
        "stopped_epoch": historical["learning"]["stopped_epoch"],
        "core_flops_per_stage": stages, "core_flops_per_step": total,
        "core_B_ref": total * steps, "core_operator_flops": counted,
        "supplementary_estimate_per_stage": audit.supplement,
        "estimated_total_flops_per_step": estimated_total,
        "estimated_total_B_ref": estimated_total * steps,
        "operator_audit": audit.ops,
        "limitations": ["Reconstructed arithmetic counts, not historical GPU hardware measurements.",
                       "Dense convolution/matrix kernels are counted with MAC=2; convolution backward overrides the flawed torch 2.2.2 weight-gradient formula.",
                       "Supplementary scalar, loss and optimizer arithmetic is counted separately; fused normalization and activations use declared arithmetic estimates, not exact device instruction counts.",
                       "Validation, data processing, checkpoint I/O and final evaluation are outside the training allowance.",
                       "Shape-based profiling uses synthetic tensors; counts are independent of learned weights and MRI values for this fixed-shape architecture.",
                       "CUDA AMP and GradScaler overhead are not executed by this CPU/meta reconstruction."]
    }
    (OUT / f"profile_{device}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["parameters", "core_flops_per_stage", "core_flops_per_step", "core_B_ref", "supplementary_estimate_per_stage", "estimated_total_flops_per_step", "estimated_total_B_ref"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
