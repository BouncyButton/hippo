"""Screen discovered constraints by repairing frozen-model predictions.

This stage belongs between candidate induction and constraint selection::

    candidate_report.json -> counterfactual repair -> repair_report.json

For every candidate configuration, the script clones the frozen prediction as
editable logits and minimizes

    MSE(softmax(editable_logits), frozen_probs) + gamma * constraint_loss

Model parameters are never updated.  The resulting Dice change is a local
counterfactual signal, not an estimate of final fine-tuning performance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from tqdm import tqdm

from semantic_constraints.compiled_losses import SemanticConstraintLoss, class_mask
from semantic_constraints.probe_model import (
    DEFAULT_CHECKPOINT,
    build_dataset,
    build_fold_subsets,
    build_model,
    get_batch_paths,
    one_hot,
    resolve_device,
)
from semantic_constraints.select_constraints import parse_constraint


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_REPORT = SCRIPT_DIR / "probe_outputs" / "candidate_report.json"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "probe_outputs" / "repair_report.json"
DEFAULT_OUTPUT_MD = SCRIPT_DIR / "probe_outputs" / "repair_report.md"


@dataclass
class FrozenSample:
    sample_idx: int
    image: str
    probs: torch.Tensor
    gt: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counterfactually repair frozen predictions with discovered semantic constraints."
    )
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--repairs-dir",
        type=Path,
        default=SCRIPT_DIR / "probe_outputs" / "repairs",
        help="Directory for optional compressed original/repaired/GT mask examples.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Repair the first K ranked candidates.")
    parser.add_argument(
        "--candidate-ranks",
        type=int,
        nargs="+",
        default=None,
        help="One-based candidate ranks to repair instead of --top-k.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=("selected", "range"),
        default="selected",
        help="Use only selected alpha or also sweep its valid GT-purity range.",
    )
    parser.add_argument("--alpha-points", type=int, default=3, help="Grid points across the suggested alpha range.")
    parser.add_argument(
        "--soft-zero-floors",
        type=float,
        nargs="*",
        default=(1e-3, 5e-3, 1e-2),
        help=(
            "Positive alpha values substituted for exact-zero <= overlap/contains rules, which are unattainable "
            "for finite softmax logits. Pass the option with no values to keep alpha=0 for debugging."
        ),
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.1, 1.0), help="Constraint repair weights.")
    parser.add_argument("--steps", type=int, default=25, help="Adam steps per repair.")
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        nargs="+",
        default=(0, 1, 5, 10, 25, 50, 100),
        help="Completed repair steps at which aggregate diagnostics are recorded.",
    )
    parser.add_argument("--repair-lr", type=float, default=0.1, help="Editable-logit Adam learning rate.")
    parser.add_argument(
        "--fuzzy-mode",
        choices=("soft_exp", "lukasiewicz", "lukasiewicz_ste", "hinge"),
        default="soft_exp",
    )
    parser.add_argument("--fuzzy-margin", type=float, default=1.0)
    parser.add_argument("--fuzzy-beta", type=float, default=4.0)
    parser.add_argument(
        "--violation-tolerance",
        type=float,
        default=1e-6,
        help="Minimum normalized violation for a prediction to be repaired.",
    )
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=1e-2,
        help="Maximum normalized residual counted as successful repair.",
    )
    parser.add_argument("--harm-tolerance", type=float, default=1e-6, help="Dice decrease counted as harmful.")
    parser.add_argument(
        "--min-repair-success",
        type=float,
        default=0.5,
        help="Minimum success rate required for a configuration to receive a nonzero ranking score.",
    )
    parser.add_argument("--min-gt-volume", type=float, default=1e-6, help="Applicability threshold in voxels.")
    parser.add_argument(
        "--save-repairs",
        type=int,
        default=0,
        help="Save this many violated examples per configuration as compressed NPZ files.",
    )

    # These default to the probe report's config so repair reproduces discovery.
    parser.add_argument("--load-weights", type=Path, default=None)
    parser.add_argument("--root-dir", type=Path, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--split", choices=("val", "train"), default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pixdim", type=float, nargs=3, default=None)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--foreground-classes", type=int, nargs="+", default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="Limit inference batches for a smoke test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def resolve_source_path(raw_path: str | Path, relative_to: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (path, REPO_ROOT / path, relative_to / path):
        if candidate.exists():
            return candidate
    return path


def load_probe_config(candidate_payload: dict[str, Any], candidate_path: Path) -> tuple[Path | None, dict[str, Any]]:
    raw_probe = candidate_payload.get("probe_report")
    if not raw_probe:
        return None, {}
    probe_path = resolve_source_path(raw_probe, candidate_path.parent)
    if not probe_path.exists():
        return probe_path, {}
    probe_payload = load_json(probe_path)
    config = probe_payload.get("config", {})
    return probe_path, config if isinstance(config, dict) else {}


def inherit_probe_config(args: argparse.Namespace, probe_config: dict[str, Any]) -> None:
    defaults = {
        "load_weights": DEFAULT_CHECKPOINT,
        "root_dir": PROPOSAL_ROOT / "tmp",
        "task": "Task04_Hippocampus",
        "fold": 1,
        "folds": 5,
        "train_fraction": 0.1,
        "split": "val",
        "pixdim": (1.5, 0.5, 1.5),
        "spatial_size": (64, 64, 64),
        "num_classes": 3,
        "foreground_classes": (1, 2),
    }
    probe_aliases = {"foreground_classes": "classes"}
    path_fields = {"load_weights", "root_dir"}
    for field, fallback in defaults.items():
        if getattr(args, field) is not None:
            continue
        probe_field = probe_aliases.get(field, field)
        value = probe_config.get(probe_field, fallback)
        if field in path_fields:
            value = Path(value)
            if not value.is_absolute() and not value.exists() and (REPO_ROOT / value).exists():
                value = REPO_ROOT / value
        setattr(args, field, value)

    if args.split not in {"train", "val"}:
        raise ValueError("Repair requires a single source split; pass --split train or --split val.")


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    if args.alpha_points < 1:
        raise ValueError("--alpha-points must be positive.")
    if any(value <= 0.0 for value in args.soft_zero_floors):
        raise ValueError("Every --soft-zero-floors value must be positive.")
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    if args.repair_lr <= 0.0:
        raise ValueError("--repair-lr must be positive.")
    if not args.gammas or any(gamma <= 0.0 for gamma in args.gammas):
        raise ValueError("Every --gammas value must be positive.")
    if args.save_repairs < 0:
        raise ValueError("--save-repairs cannot be negative.")
    if not args.trajectory_steps or any(step < 0 for step in args.trajectory_steps):
        raise ValueError("--trajectory-steps must contain non-negative integers.")
    if not 0.0 <= args.min_repair_success <= 1.0:
        raise ValueError("--min-repair-success must be in [0, 1].")
    if not args.foreground_classes:
        raise ValueError("--foreground-classes must contain at least one class id.")
    if any(class_id < 0 or class_id >= args.num_classes for class_id in args.foreground_classes):
        raise ValueError("Every foreground class id must be in [0, num-classes).")


def select_candidates(payload: dict[str, Any], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    candidates = payload.get("top_candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{args.candidate_report} does not contain a top_candidates list.")
    ranks = args.candidate_ranks or list(range(1, min(args.top_k, len(candidates)) + 1))
    invalid = [rank for rank in ranks if rank < 1 or rank > len(candidates)]
    if invalid:
        raise ValueError(f"Candidate ranks outside 1..{len(candidates)}: {invalid}")
    return [(rank, candidates[rank - 1]) for rank in dict.fromkeys(ranks)]


def alpha_grid(
    candidate: dict[str, Any],
    mode: str,
    points: int,
    soft_zero_floors: list[float] | tuple[float, ...] = (),
) -> list[float]:
    selected = float(candidate["alpha"])
    parsed = parse_constraint(candidate["constraint"])
    is_unattainable_soft_zero = (
        parsed["primitive"] in {"overlap", "contains"}
        and parsed["direction"] == "<="
        and math.isclose(selected, 0.0, abs_tol=1e-12)
        and bool(soft_zero_floors)
    )
    values = list(soft_zero_floors) if is_unattainable_soft_zero else [selected]
    alpha_range = candidate.get("suggested_alpha_range")
    if mode == "range" and not is_unattainable_soft_zero and isinstance(alpha_range, list) and len(alpha_range) == 2:
        low, high = map(float, alpha_range)
        if math.isfinite(low) and math.isfinite(high):
            if low > high:
                low, high = high, low
            if points == 1:
                values.append((low + high) / 2.0)
            else:
                values.extend(low + index * (high - low) / (points - 1) for index in range(points))

    unique = []
    for value in values:
        if math.isfinite(value) and not any(math.isclose(value, old, rel_tol=1e-9, abs_tol=1e-12) for old in unique):
            unique.append(value)
    return unique


def make_constraint_spec(
    candidate: dict[str, Any],
    rank: int,
    alpha: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parsed = parse_constraint(candidate["constraint"])
    return {
        "name": f"candidate_{rank:02d}",
        "expression": candidate["constraint"],
        "primitive": parsed["primitive"],
        "args": parsed["args"],
        "direction": parsed["direction"],
        "alpha": alpha,
        "scale": max(float(candidate.get("mean_violation", 1.0)), 1e-8),
        "lambda": 1.0,
        "fuzzy": {
            "mode": args.fuzzy_mode,
            "margin": args.fuzzy_margin,
            "beta": args.fuzzy_beta,
        },
    }


def required_class_ids(spec: dict[str, Any]) -> list[int]:
    output = []
    for arg in spec.get("args", []):
        match = re.fullmatch(r"class_(\d+)", str(arg))
        if not match:
            raise ValueError(f"Unsupported class argument {arg!r}.")
        output.append(int(match.group(1)))
    return output


def is_applicable(gt: torch.Tensor, spec: dict[str, Any], min_gt_volume: float) -> bool:
    return all(float(gt[:, class_id].sum().item()) > min_gt_volume for class_id in required_class_ids(spec))


def mean_foreground_dice(
    probs: torch.Tensor,
    gt: torch.Tensor,
    class_ids: list[int] | tuple[int, ...],
    hard: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    if hard:
        labels = probs.argmax(dim=1)
        masks = F.one_hot(labels, num_classes=probs.shape[1]).movedim(-1, 1).to(probs.dtype)
    else:
        masks = probs
    per_class = []
    for class_id in class_ids:
        pred_mask = masks[:, class_id]
        gt_mask = gt[:, class_id]
        spatial_dims = tuple(range(1, pred_mask.ndim))
        intersection = (pred_mask * gt_mask).sum(dim=spatial_dims)
        denominator = pred_mask.sum(dim=spatial_dims) + gt_mask.sum(dim=spatial_dims)
        per_class.append((2.0 * intersection + eps) / (denominator + eps))
    return torch.stack(per_class, dim=1).mean(dim=1)


def constraint_state(
    probs: torch.Tensor,
    spec: dict[str, Any],
    module: SemanticConstraintLoss,
) -> dict[str, torch.Tensor]:
    masks = [class_mask(probs, arg) for arg in spec.get("args", [])]
    values = module.apply_primitive(spec["primitive"], masks, spec)
    alpha = torch.as_tensor(float(spec["alpha"]), device=values.device, dtype=values.dtype)
    if spec["direction"] == "<=":
        raw = F.relu(values - alpha)
    else:
        raw = F.relu(alpha - values)
    normalized = raw / max(float(spec["scale"]), 1e-8)
    return {"value": values, "raw_violation": raw, "normalized_violation": normalized}


def cache_frozen_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[FrozenSample]:
    samples = []
    sample_offset = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Frozen inference"):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            probs = torch.softmax(model(images), dim=1)
            gt = one_hot(labels, args.num_classes)
            paths = get_batch_paths(batch, images.shape[0])
            for index in range(images.shape[0]):
                samples.append(
                    FrozenSample(
                        sample_idx=sample_offset + index,
                        image=paths[index] if index < len(paths) else "",
                        probs=probs[index : index + 1].detach().cpu(),
                        gt=gt[index : index + 1].detach().cpu(),
                    )
                )
            sample_offset += images.shape[0]
    return samples


def repair_prediction(
    base_probs: torch.Tensor,
    gt: torch.Tensor,
    spec: dict[str, Any],
    module: SemanticConstraintLoss,
    gamma: float,
    steps: int,
    learning_rate: float,
    foreground_classes: list[int] | tuple[int, ...],
    trajectory_steps: list[int] | tuple[int, ...],
    success_tolerance: float,
) -> tuple[torch.Tensor, dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    editable_logits = base_probs.clamp_min(1e-8).log().detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([editable_logits], lr=learning_rate)
    checkpoints = {step for step in trajectory_steps if 0 <= step <= steps}
    checkpoints.update({0, steps})
    trajectory: dict[int, dict[str, float]] = {}
    gradient_norms: dict[int, dict[str, float]] = {}

    def record_snapshot(step: int) -> None:
        with torch.no_grad():
            probs = torch.softmax(editable_logits, dim=1)
            state = constraint_state(probs, spec, module)
            constraint_loss = module(editable_logits)["unweighted_loss"][spec["name"]]
            trajectory[step] = {
                "descriptor_value": float(state["value"].item()),
                "normalized_violation": float(state["normalized_violation"].item()),
                "constraint_loss": float(constraint_loss.item()),
                "repair_success": float(state["normalized_violation"].item() <= success_tolerance),
                "soft_dice": float(mean_foreground_dice(probs, gt, foreground_classes, hard=False).item()),
                "hard_dice": float(mean_foreground_dice(probs, gt, foreground_classes, hard=True).item()),
                "prediction_shift_mse": float(F.mse_loss(probs, base_probs).item()),
            }

    record_snapshot(0)
    for completed_steps in range(steps):
        optimizer.zero_grad(set_to_none=True)
        repaired_probs = torch.softmax(editable_logits, dim=1)
        preservation = F.mse_loss(repaired_probs, base_probs)
        constraint = module(editable_logits)["unweighted_loss"][spec["name"]]
        loss = preservation + gamma * constraint

        # At step 0 the preservation gradient is zero by construction. Step 1
        # is also logged so relative term strength becomes observable.
        if completed_steps in {0, 1}:
            preservation_grad = torch.autograd.grad(
                preservation, editable_logits, retain_graph=True
            )[0]
            constraint_grad = torch.autograd.grad(
                constraint, editable_logits, retain_graph=True
            )[0]
            weighted_constraint_grad = gamma * constraint_grad
            gradient_norms[completed_steps] = {
                "preservation_gradient_norm": float(torch.linalg.vector_norm(preservation_grad).item()),
                "constraint_gradient_norm": float(torch.linalg.vector_norm(constraint_grad).item()),
                "weighted_constraint_gradient_norm": float(
                    torch.linalg.vector_norm(weighted_constraint_grad).item()
                ),
                "total_gradient_norm": float(
                    torch.linalg.vector_norm(preservation_grad + weighted_constraint_grad).item()
                ),
            }
        loss.backward()
        optimizer.step()
        completed = completed_steps + 1
        if completed in checkpoints:
            record_snapshot(completed)

    return torch.softmax(editable_logits.detach(), dim=1), trajectory, gradient_norms


def confidence_interval_95(values: list[float]) -> list[float] | None:
    if not values:
        return None
    center = mean(values)
    if len(values) == 1:
        return [center, center]
    radius = 1.96 * stdev(values) / math.sqrt(len(values))
    return [center - radius, center + radius]


def optional_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def accumulate_step_metrics(
    accumulator: dict[int, dict[str, list[float]]],
    per_sample: dict[int, dict[str, float]],
) -> None:
    for step, metrics in per_sample.items():
        step_metrics = accumulator.setdefault(step, {})
        for name, value in metrics.items():
            step_metrics.setdefault(name, []).append(float(value))


def summarize_step_metrics(
    accumulator: dict[int, dict[str, list[float]]],
    include_dice_gain: bool = False,
) -> dict[str, dict[str, float]]:
    summary = {
        str(step): {name: mean(values) for name, values in metrics.items() if values}
        for step, metrics in sorted(accumulator.items())
    }
    if include_dice_gain and "0" in summary:
        baseline = summary["0"]
        for metrics in summary.values():
            metrics["soft_dice_gain"] = metrics["soft_dice"] - baseline["soft_dice"]
            metrics["hard_dice_gain"] = metrics["hard_dice"] - baseline["hard_dice"]
    return summary


def save_repair_example(
    path: Path,
    sample: FrozenSample,
    original: torch.Tensor,
    repaired: torch.Tensor,
    gt: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_idx=np.asarray(sample.sample_idx),
        image=np.asarray(sample.image),
        original_mask=original.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8),
        repaired_mask=repaired.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8),
        ground_truth=gt.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8),
        original_probs=original.squeeze(0).detach().cpu().numpy().astype(np.float16),
        repaired_probs=repaired.squeeze(0).detach().cpu().numpy().astype(np.float16),
    )


def evaluate_configuration(
    samples: list[FrozenSample],
    candidate: dict[str, Any],
    rank: int,
    alpha: float,
    gamma: float,
    config_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    spec = make_constraint_spec(candidate, rank=rank, alpha=alpha, args=args)
    parsed_candidate = parse_constraint(candidate["constraint"])
    source_alpha = float(candidate["alpha"])
    soft_zero_floor_applied = bool(
        parsed_candidate["primitive"] in {"overlap", "contains"}
        and parsed_candidate["direction"] == "<="
        and math.isclose(source_alpha, 0.0, abs_tol=1e-12)
        and alpha > 0.0
    )
    module = SemanticConstraintLoss([spec], spacing=tuple(args.pixdim), input_is_logits=True).to(device)

    soft_before_values: list[float] = []
    soft_after_values: list[float] = []
    hard_before_values: list[float] = []
    hard_after_values: list[float] = []
    descriptor_before_values: list[float] = []
    descriptor_after_values: list[float] = []
    raw_violations_before: list[float] = []
    normalized_violations_before: list[float] = []
    raw_residuals: list[float] = []
    normalized_residuals: list[float] = []
    shifts: list[float] = []
    applicable = 0
    violated = 0
    successful = 0
    saved_paths = []
    trajectory_accumulator: dict[int, dict[str, list[float]]] = {}
    gradient_accumulator: dict[int, dict[str, list[float]]] = {}

    description = f"repair rank={rank} alpha={alpha:.4g} gamma={gamma:g}"
    for sample in tqdm(samples, desc=description, leave=False):
        base_probs = sample.probs.to(device)
        gt = sample.gt.to(device)
        if not is_applicable(gt, spec, args.min_gt_volume):
            continue
        applicable += 1
        before_state = constraint_state(base_probs, spec, module)
        if float(before_state["normalized_violation"].item()) <= args.violation_tolerance:
            continue
        violated += 1

        repaired, sample_trajectory, sample_gradient_norms = repair_prediction(
            base_probs=base_probs,
            gt=gt,
            spec=spec,
            module=module,
            gamma=gamma,
            steps=args.steps,
            learning_rate=args.repair_lr,
            foreground_classes=args.foreground_classes,
            trajectory_steps=args.trajectory_steps,
            success_tolerance=args.success_tolerance,
        )
        accumulate_step_metrics(trajectory_accumulator, sample_trajectory)
        accumulate_step_metrics(gradient_accumulator, sample_gradient_norms)
        after_state = constraint_state(repaired, spec, module)
        descriptor_before_values.append(float(before_state["value"].item()))
        descriptor_after_values.append(float(after_state["value"].item()))
        raw_violations_before.append(float(before_state["raw_violation"].item()))
        normalized_violations_before.append(float(before_state["normalized_violation"].item()))
        raw_residual = float(after_state["raw_violation"].item())
        normalized_residual = float(after_state["normalized_violation"].item())
        successful += int(normalized_residual <= args.success_tolerance)

        soft_before_values.append(float(mean_foreground_dice(base_probs, gt, args.foreground_classes, hard=False).item()))
        soft_after_values.append(float(mean_foreground_dice(repaired, gt, args.foreground_classes, hard=False).item()))
        hard_before_values.append(float(mean_foreground_dice(base_probs, gt, args.foreground_classes, hard=True).item()))
        hard_after_values.append(float(mean_foreground_dice(repaired, gt, args.foreground_classes, hard=True).item()))
        raw_residuals.append(raw_residual)
        normalized_residuals.append(normalized_residual)
        shifts.append(float(F.mse_loss(repaired, base_probs).item()))

        if len(saved_paths) < args.save_repairs:
            filename = f"rank{rank:02d}_config{config_index:03d}_sample{sample.sample_idx:03d}.npz"
            path = args.repairs_dir / filename
            save_repair_example(path, sample, base_probs, repaired, gt)
            saved_paths.append(str(path))

    soft_gains = [after - before for before, after in zip(soft_before_values, soft_after_values)]
    hard_gains = [after - before for before, after in zip(hard_before_values, hard_after_values)]
    violation_rate = violated / applicable if applicable else 0.0
    success_rate = successful / violated if violated else None
    mean_soft_gain = optional_mean(soft_gains)
    mean_hard_gain = optional_mean(hard_gains)
    harmful_soft = sum(gain < -args.harm_tolerance for gain in soft_gains) / len(soft_gains) if soft_gains else None
    harmful_hard = sum(gain < -args.harm_tolerance for gain in hard_gains) / len(hard_gains) if hard_gains else None
    ranking_eligible = bool(
        violated
        and success_rate is not None
        and success_rate >= args.min_repair_success
        and mean_soft_gain is not None
        and mean_soft_gain > 0.0
        and mean_hard_gain is not None
        and mean_hard_gain >= -args.harm_tolerance
    )
    repair_score = 0.0
    if ranking_eligible:
        repair_score = (
            float(candidate.get("coverage", 0.0))
            * float(candidate.get("gt_purity", 0.0))
            * violation_rate
            * success_rate
            * mean_soft_gain
            * (1.0 - (harmful_soft or 0.0))
        )

    return {
        "candidate_rank": rank,
        "constraint": candidate["constraint"],
        "alpha": alpha,
        "source_alpha": source_alpha,
        "alpha_differs_from_source": not math.isclose(alpha, source_alpha, abs_tol=1e-12),
        "soft_zero_floor_applied": soft_zero_floor_applied,
        "gamma": gamma,
        "scale": spec["scale"],
        "fuzzy": spec["fuzzy"],
        "num_samples": len(samples),
        "num_applicable": applicable,
        "num_violated": violated,
        "violation_rate": violation_rate,
        "repair_success_rate": success_rate,
        "mean_soft_dice_before": optional_mean(soft_before_values),
        "mean_soft_dice_after": optional_mean(soft_after_values),
        "mean_soft_dice_gain": mean_soft_gain,
        "median_soft_dice_gain": median(soft_gains) if soft_gains else None,
        "soft_dice_gain_ci95": confidence_interval_95(soft_gains),
        "mean_hard_dice_before": optional_mean(hard_before_values),
        "mean_hard_dice_after": optional_mean(hard_after_values),
        "mean_hard_dice_gain": mean_hard_gain,
        "median_hard_dice_gain": median(hard_gains) if hard_gains else None,
        "hard_dice_gain_ci95": confidence_interval_95(hard_gains),
        "harmful_fraction_soft": harmful_soft,
        "harmful_fraction_hard": harmful_hard,
        "estimated_full_split_soft_dice_gain": (
            violated / len(samples) * mean_soft_gain if samples and mean_soft_gain is not None else None
        ),
        "estimated_full_split_hard_dice_gain": (
            violated / len(samples) * mean_hard_gain if samples and mean_hard_gain is not None else None
        ),
        "mean_prediction_shift_mse": optional_mean(shifts),
        "mean_descriptor_before": optional_mean(descriptor_before_values),
        "mean_descriptor_after": optional_mean(descriptor_after_values),
        "mean_constraint_violation_before": optional_mean(raw_violations_before),
        "mean_normalized_constraint_violation_before": optional_mean(normalized_violations_before),
        "mean_constraint_residual": optional_mean(raw_residuals),
        "mean_normalized_constraint_residual": optional_mean(normalized_residuals),
        "trajectory": summarize_step_metrics(trajectory_accumulator, include_dice_gain=True),
        "gradient_norms": summarize_step_metrics(gradient_accumulator),
        "ranking_eligible": ranking_eligible,
        "ranking_policy": {
            "min_repair_success": args.min_repair_success,
            "requires_positive_soft_dice_gain": True,
            "requires_nonnegative_hard_dice_gain": True,
            "score_uses_soft_dice_gain": True,
        },
        "repair_score": repair_score,
        "saved_repairs": saved_paths,
    }


def configuration_sort_key(config: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(bool(config.get("ranking_eligible", False))),
        float(config.get("repair_score") or 0.0),
        float(config.get("mean_soft_dice_gain") or 0.0),
        float(config.get("mean_hard_dice_gain") or 0.0),
    )


def summarize_candidates(
    selected: list[tuple[int, dict[str, Any]]], configurations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for rank, candidate in selected:
        matching = [item for item in configurations if item["candidate_rank"] == rank]
        best = max(matching, key=configuration_sort_key) if matching else None
        output.append(
            {
                "candidate_rank": rank,
                "constraint": candidate["constraint"],
                "discovery_score": candidate.get("score"),
                "coverage": candidate.get("coverage"),
                "gt_purity": candidate.get("gt_purity"),
                "best_configuration": best,
            }
        )
    return sorted(
        output,
        key=lambda item: configuration_sort_key(item["best_configuration"] or {}),
        reverse=True,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Counterfactual Constraint Repair",
        "",
        f"Candidate report: `{payload['candidate_report']}`",
        f"Probe report: `{payload.get('probe_report') or 'not available'}`",
        f"Frozen samples: `{payload['num_frozen_samples']}`",
        "",
        "> Repair gain is a local screening signal, not expected fine-tuning gain.",
        "",
        "## Best Configuration per Candidate",
        "",
        "| Repair rank | Candidate | Alpha | Gamma | Violated | Success | Soft Dice gain | Hard Dice gain | Harmful S/H | Eligible | Score |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for repair_rank, item in enumerate(payload["candidate_summary"], start=1):
        best = item.get("best_configuration")
        if best is None:
            continue
        lines.append(
            f"| {repair_rank} | `{item['constraint']}` | {fmt(best['alpha'], 6)} | {fmt(best['gamma'], 3)} "
            f"| {best['num_violated']}/{best['num_applicable']} | {fmt(best['repair_success_rate'])} "
            f"| {fmt(best['mean_soft_dice_gain'])} | {fmt(best['mean_hard_dice_gain'])} "
            f"| {fmt(best['harmful_fraction_soft'])}/{fmt(best['harmful_fraction_hard'])} "
            f"| {'yes' if best['ranking_eligible'] else 'no'} | {fmt(best['repair_score'], 8)} |"
        )
    lines.extend(["", "## Best-Configuration Trajectories", ""])
    for item in payload["candidate_summary"]:
        best = item.get("best_configuration")
        if best is None:
            continue
        lines.extend(
            [
                f"### `{item['constraint']}`",
                "",
                "| Step | Normalized violation | Success | Soft Dice gain | Hard Dice gain | Shift MSE |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for step, metrics in best.get("trajectory", {}).items():
            lines.append(
                f"| {step} | {fmt(metrics.get('normalized_violation'))} | {fmt(metrics.get('repair_success'))} "
                f"| {fmt(metrics.get('soft_dice_gain'))} | {fmt(metrics.get('hard_dice_gain'))} "
                f"| {fmt(metrics.get('prediction_shift_mse'), 6)} |"
            )
        lines.extend(
            [
                "",
                "Early gradient norms:",
                "",
                "| Step | Preservation | Constraint | Gamma × constraint | Total |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for step, metrics in best.get("gradient_norms", {}).items():
            lines.append(
                f"| {step} | {fmt(metrics.get('preservation_gradient_norm'), 6)} "
                f"| {fmt(metrics.get('constraint_gradient_norm'), 6)} "
                f"| {fmt(metrics.get('weighted_constraint_gradient_norm'), 6)} "
                f"| {fmt(metrics.get('total_gradient_norm'), 6)} |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Ranking requires the configured minimum repair success, positive soft Dice gain, and nonnegative hard "
            "Dice gain. The score uses soft Dice gain and penalizes harmful repairs. Exact-zero <= overlap/contains "
            "rules are evaluated at the configured positive floors unless that behavior is disabled. Results are "
            "computed only on applicable samples whose frozen prediction violated the tested rule.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_payload = load_json(args.candidate_report)
    probe_path, probe_config = load_probe_config(candidate_payload, args.candidate_report)
    inherit_probe_config(args, probe_config)
    validate_args(args)
    selected = select_candidates(candidate_payload, args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    dataset = build_dataset(args)
    train_subset, val_subset = build_fold_subsets(dataset, args.folds, args.fold, args.train_fraction)
    subset = val_subset if args.split == "val" else train_subset
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(args.load_weights, args.num_classes, device)
    samples = cache_frozen_predictions(model, loader, args, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not samples:
        raise RuntimeError("Frozen inference produced no samples.")

    configurations = []
    config_index = 0
    for rank, candidate in selected:
        for alpha in alpha_grid(
            candidate,
            args.alpha_mode,
            args.alpha_points,
            soft_zero_floors=args.soft_zero_floors,
        ):
            for gamma in args.gammas:
                config_index += 1
                result = evaluate_configuration(
                    samples=samples,
                    candidate=candidate,
                    rank=rank,
                    alpha=alpha,
                    gamma=gamma,
                    config_index=config_index,
                    args=args,
                    device=device,
                )
                configurations.append(result)

    summary = summarize_candidates(selected, configurations)
    payload = {
        "schema_version": "semantic_constraints.repair.v1",
        "candidate_report": str(args.candidate_report),
        "probe_report": str(probe_path) if probe_path is not None else None,
        "config": vars(args),
        "interpretation": (
            "Counterfactual repair measures locally available correction under editable logits; "
            "it is not expected final fine-tuning Dice gain."
        ),
        "num_frozen_samples": len(samples),
        "candidate_summary": summary,
        "configurations": configurations,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)

    print(f"device: {device}")
    print(f"loaded weights: {args.load_weights}")
    print(f"frozen samples: {len(samples)}")
    print(f"candidate configurations: {len(configurations)}")
    print(f"wrote JSON: {args.output_json}")
    print(f"wrote Markdown: {args.output_md}")
    for item in summary:
        best = item["best_configuration"]
        print(
            f"rank={item['candidate_rank']}",
            item["constraint"],
            f"soft_gain={fmt(best['mean_soft_dice_gain'])}",
            f"hard_gain={fmt(best['mean_hard_dice_gain'])}",
            f"harmful_soft={fmt(best['harmful_fraction_soft'])}",
            f"success={fmt(best['repair_success_rate'])}",
            f"eligible={best['ranking_eligible']}",
            f"alpha={best['alpha']:.6g}",
            f"gamma={best['gamma']:g}",
        )


if __name__ == "__main__":
    main()
