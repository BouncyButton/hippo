"""Discover constraints whose repairs improve held-out segmentations.

This is the end-to-end screening procedure described in the automatic
constraint-discovery proposal. Candidate thresholds are learned on one
cross-fitting partition and evaluated on another. A candidate is retained only
when its constraint gradient aligns with the supervised error and an explicit
output-space repair reduces both violation and segmentation loss.

The selected JSON uses the same contract as ``compiled_losses.py`` and can be
passed directly to ``train_with_constraints.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_constraints.compiled_losses import SemanticConstraintLoss
from semantic_constraints.counterfactual_repair import (
    FrozenSample,
    cache_frozen_predictions,
    constraint_state,
    is_applicable,
    make_constraint_spec,
    mean_foreground_dice,
    repair_prediction,
)
from semantic_constraints.evaluate_candidates import (
    build_candidates,
    candidate_to_dict,
    collect_observations,
)
from semantic_constraints.probe_model import (
    build_dataset,
    build_fold_subsets,
    build_model,
    probe_split,
    resolve_device,
)
from semantic_constraints.select_constraints import parse_constraint, slugify


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "probe_outputs" / "automatic_discovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit semantic constraints and retain only constraints with useful repairs."
    )
    parser.add_argument("--load-weights", type=Path, required=True, help="Frozen segmentation checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root-dir", type=Path, default=REPO_ROOT / "tmp")
    parser.add_argument("--task", default="Task04_Hippocampus")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--fold", type=int, default=1, help="Discovery fold, starting at 1.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pixdim", type=float, nargs=3, default=(1.5, 0.5, 1.5))
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--classes", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--pair", type=int, nargs=2, default=(1, 2))
    parser.add_argument("--connectedness-steps", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0, help="Limit discovery batches; 0 uses the full fold.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--cross-fit-folds", type=int, default=2)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--max-constraints", type=int, default=5)
    parser.add_argument("--min-gt-volume", type=float, default=1e-6)
    parser.add_argument("--min-fit-gt-purity", type=float, default=0.9)
    parser.add_argument("--min-fit-coverage", type=float, default=0.8)
    parser.add_argument("--min-fit-pred-violation-rate", type=float, default=0.05)
    parser.add_argument("--redundancy-penalty", type=float, default=0.5)

    parser.add_argument("--gammas", type=float, nargs="+", default=(0.1, 1.0))
    parser.add_argument("--repair-steps", type=int, default=25)
    parser.add_argument("--repair-lr", type=float, default=0.1)
    parser.add_argument("--soft-zero-floor", type=float, default=1e-3)
    parser.add_argument("--violation-tolerance", type=float, default=1e-6)
    parser.add_argument("--repair-success-tolerance", type=float, default=1e-2)
    parser.add_argument("--harm-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--fuzzy-mode",
        choices=("soft_exp", "lukasiewicz", "lukasiewicz_ste", "hinge"),
        default="hinge",
    )
    parser.add_argument("--fuzzy-margin", type=float, default=1.0)
    parser.add_argument("--fuzzy-beta", type=float, default=4.0)

    parser.add_argument("--max-heldout-gt-violation-rate", type=float, default=0.1)
    parser.add_argument("--min-heldout-pred-violation-rate", type=float, default=0.05)
    parser.add_argument("--min-mean-gradient-alignment", type=float, default=0.0)
    parser.add_argument("--min-positive-alignment-rate", type=float, default=0.5)
    parser.add_argument("--min-repair-success-rate", type=float, default=0.5)
    parser.add_argument("--max-harm-rate", type=float, default=0.25)
    parser.add_argument("--harm-penalty", type=float, default=2.0)
    parser.add_argument("--complexity-penalty", type=float, default=1e-5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--allow-nonpositive-ci",
        action="store_true",
        help="Do not require the 95%% bootstrap lower bound on repair gain to be positive.",
    )
    parser.add_argument("--lambda-default", type=float, default=0.1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.load_weights.exists():
        raise FileNotFoundError(args.load_weights)
    if args.cross_fit_folds < 2:
        raise ValueError("--cross-fit-folds must be at least 2.")
    if args.candidate_top_k < 1 or args.max_constraints < 1:
        raise ValueError("Candidate and selected-constraint limits must be positive.")
    if args.repair_steps < 1 or args.repair_lr <= 0.0:
        raise ValueError("Repair steps and learning rate must be positive.")
    if not args.gammas or any(value <= 0.0 for value in args.gammas):
        raise ValueError("Every repair gamma must be positive.")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")
    probability_fields = (
        "max_heldout_gt_violation_rate",
        "min_heldout_pred_violation_rate",
        "min_positive_alignment_rate",
        "min_repair_success_rate",
        "max_harm_rate",
    )
    for field in probability_fields:
        if not 0.0 <= getattr(args, field) <= 1.0:
            raise ValueError(f"--{field.replace('_', '-')} must be in [0, 1].")


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


def sample_ids_from_rows(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["sample_idx"]) for row in rows if row.get("sample_idx") is not None})


def make_cross_fit_assignments(sample_ids: list[int], folds: int, seed: int) -> dict[int, int]:
    if len(sample_ids) < folds:
        raise ValueError(f"Need at least {folds} discovery samples, got {len(sample_ids)}.")
    shuffled = np.asarray(sample_ids, dtype=int)
    np.random.RandomState(seed).shuffle(shuffled)
    return {int(sample_id): int(index % folds) for index, sample_id in enumerate(shuffled)}


def induce_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    groups, total_samples = collect_observations(rows, min_gt_volume=args.min_gt_volume)
    candidates = build_candidates(
        groups=groups,
        total_samples=total_samples,
        min_gt_purity=args.min_fit_gt_purity,
        min_coverage=args.min_fit_coverage,
        min_pred_violation_rate=args.min_fit_pred_violation_rate,
        failure_count=5,
        redundancy_penalty=args.redundancy_penalty,
    )
    return [candidate_to_dict(candidate) for candidate in candidates]


def effective_alpha(candidate: dict[str, Any], args: argparse.Namespace) -> float:
    alpha = float(candidate["alpha"])
    parsed = parse_constraint(candidate["constraint"])
    if (
        parsed["primitive"] in {"overlap", "contains"}
        and parsed["direction"] == "<="
        and math.isclose(alpha, 0.0, abs_tol=1e-12)
    ):
        return args.soft_zero_floor
    return alpha


def constraint_spec(candidate: dict[str, Any], rank: int, args: argparse.Namespace) -> dict[str, Any]:
    spec = make_constraint_spec(candidate, rank=rank, alpha=effective_alpha(candidate, args), args=args)
    spec["lambda"] = args.lambda_default
    return spec


def gradient_alignment(
    base_probs: torch.Tensor,
    gt: torch.Tensor,
    spec: dict[str, Any],
    module: SemanticConstraintLoss,
    foreground_classes: list[int] | tuple[int, ...],
) -> dict[str, float] | None:
    logits = base_probs.clamp_min(1e-8).log().detach().clone().requires_grad_(True)
    probs = torch.softmax(logits, dim=1)
    supervised_loss = 1.0 - mean_foreground_dice(probs, gt, foreground_classes, hard=False).mean()
    constraint_loss = module(logits)["unweighted_loss"][spec["name"]]
    supervised_grad = torch.autograd.grad(supervised_loss, logits, retain_graph=True)[0]
    constraint_grad = torch.autograd.grad(constraint_loss, logits)[0]
    supervised_norm = torch.linalg.vector_norm(supervised_grad)
    constraint_norm = torch.linalg.vector_norm(constraint_grad)
    denominator = supervised_norm * constraint_norm
    if not torch.isfinite(denominator) or float(denominator.item()) <= 1e-20:
        return None
    dot = torch.sum(supervised_grad * constraint_grad)
    return {
        "cosine": float((dot / denominator).item()),
        "dot_product": float(dot.item()),
        "supervised_gradient_norm": float(supervised_norm.item()),
        "constraint_gradient_norm": float(constraint_norm.item()),
    }


def evaluate_fold_candidate(
    samples: list[FrozenSample],
    candidate: dict[str, Any],
    rank: int,
    gamma: float,
    cross_fit_fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    spec = constraint_spec(candidate, rank=rank, args=args)
    module = SemanticConstraintLoss([spec], spacing=tuple(args.pixdim), input_is_logits=True).to(device)
    applicable = 0
    gt_violated = 0
    pred_violated = 0
    outcomes: list[dict[str, Any]] = []

    for sample in samples:
        base_probs = sample.probs.to(device)
        gt = sample.gt.to(device)
        if not is_applicable(gt, spec, args.min_gt_volume):
            continue
        applicable += 1
        gt_state = constraint_state(gt, spec, module)
        gt_violated += int(float(gt_state["normalized_violation"].item()) > args.violation_tolerance)
        before_state = constraint_state(base_probs, spec, module)
        before_violation = float(before_state["normalized_violation"].item())
        if before_violation <= args.violation_tolerance:
            continue
        pred_violated += 1

        alignment = gradient_alignment(base_probs, gt, spec, module, args.classes)
        repaired, _, _ = repair_prediction(
            base_probs=base_probs,
            gt=gt,
            spec=spec,
            module=module,
            gamma=gamma,
            steps=args.repair_steps,
            learning_rate=args.repair_lr,
            foreground_classes=args.classes,
            trajectory_steps=(0, args.repair_steps),
            success_tolerance=args.repair_success_tolerance,
        )
        after_state = constraint_state(repaired, spec, module)
        after_violation = float(after_state["normalized_violation"].item())
        soft_before = float(mean_foreground_dice(base_probs, gt, args.classes, hard=False).item())
        soft_after = float(mean_foreground_dice(repaired, gt, args.classes, hard=False).item())
        hard_before = float(mean_foreground_dice(base_probs, gt, args.classes, hard=True).item())
        hard_after = float(mean_foreground_dice(repaired, gt, args.classes, hard=True).item())
        outcomes.append(
            {
                "cross_fit_fold": cross_fit_fold,
                "sample_idx": sample.sample_idx,
                "image": sample.image,
                "gradient_alignment": alignment,
                "violation_before": before_violation,
                "violation_after": after_violation,
                "violation_reduction": before_violation - after_violation,
                "repair_success": after_violation <= args.repair_success_tolerance,
                "soft_dice_before": soft_before,
                "soft_dice_after": soft_after,
                "segmentation_loss_gain": soft_after - soft_before,
                "hard_dice_gain": hard_after - hard_before,
                "prediction_shift_mse": float(F.mse_loss(repaired, base_probs).item()),
            }
        )

    return {
        "cross_fit_fold": cross_fit_fold,
        "candidate": candidate,
        "spec": spec,
        "gamma": gamma,
        "num_samples": len(samples),
        "num_applicable": applicable,
        "num_gt_violated": gt_violated,
        "num_pred_violated": pred_violated,
        "outcomes": outcomes,
    }


def bootstrap_ci95(values: list[float], samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = np.random.RandomState(seed)
    data = np.asarray(values, dtype=float)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = float(rng.choice(data, size=len(data), replace=True).mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def optional_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_evidence(evidence: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    first = evidence[0]
    outcomes = [outcome for item in evidence for outcome in item["outcomes"]]
    applicable = sum(item["num_applicable"] for item in evidence)
    gt_violated = sum(item["num_gt_violated"] for item in evidence)
    pred_violated = sum(item["num_pred_violated"] for item in evidence)
    folds = sorted({int(item["cross_fit_fold"]) for item in evidence})

    alignments = [
        float(outcome["gradient_alignment"]["cosine"])
        for outcome in outcomes
        if outcome["gradient_alignment"] is not None
    ]
    gains = [float(outcome["segmentation_loss_gain"]) for outcome in outcomes]
    hard_gains = [float(outcome["hard_dice_gain"]) for outcome in outcomes]
    violation_reductions = [float(outcome["violation_reduction"]) for outcome in outcomes]
    harmful_magnitudes = [max(-gain, 0.0) for gain in gains]
    gain_ci = bootstrap_ci95(gains, args.bootstrap_samples, args.seed + len(outcomes))
    parsed = parse_constraint(first["candidate"]["constraint"])
    complexity = 1.0 + 0.5 * max(len(parsed["args"]) - 1, 0)
    mean_gain = optional_mean(gains)
    mean_harm = optional_mean(harmful_magnitudes) or 0.0
    utility = (mean_gain or 0.0) - args.harm_penalty * mean_harm - args.complexity_penalty * complexity

    gt_violation_rate = gt_violated / applicable if applicable else 1.0
    pred_violation_rate = pred_violated / applicable if applicable else 0.0
    positive_alignment_rate = sum(value > 0.0 for value in alignments) / len(alignments) if alignments else 0.0
    repair_success_rate = (
        sum(bool(outcome["repair_success"]) for outcome in outcomes) / len(outcomes) if outcomes else 0.0
    )
    harm_rate = sum(gain < -args.harm_tolerance for gain in gains) / len(gains) if gains else 1.0
    success_rate = sum(gain > 0.0 for gain in gains) / len(gains) if gains else 0.0

    checks = {
        "observed_in_every_cross_fit_fold": len(folds) == args.cross_fit_folds,
        "heldout_gt_feasible": gt_violation_rate <= args.max_heldout_gt_violation_rate,
        "prediction_violations_relevant": pred_violation_rate >= args.min_heldout_pred_violation_rate,
        "positive_mean_gradient_alignment": bool(alignments) and mean(alignments) > args.min_mean_gradient_alignment,
        "positive_alignment_rate": positive_alignment_rate >= args.min_positive_alignment_rate,
        "violation_reduced": bool(violation_reductions) and mean(violation_reductions) > 0.0,
        "repair_success_rate": repair_success_rate >= args.min_repair_success_rate,
        "positive_mean_segmentation_gain": mean_gain is not None and mean_gain > 0.0,
        "acceptable_harm_rate": harm_rate <= args.max_harm_rate,
        "positive_bootstrap_lower_bound": bool(
            args.allow_nonpositive_ci or (gain_ci is not None and gain_ci[0] > 0.0)
        ),
    }
    selected = all(checks.values())
    return {
        "constraint": first["candidate"]["constraint"],
        "gamma": first["gamma"],
        "cross_fit_folds_evaluated": folds,
        "fold_alphas": [float(item["spec"]["alpha"]) for item in evidence],
        "num_applicable": applicable,
        "num_gt_violated": gt_violated,
        "heldout_gt_violation_rate": gt_violation_rate,
        "num_pred_violated": pred_violated,
        "heldout_prediction_violation_rate": pred_violation_rate,
        "num_gradient_alignments": len(alignments),
        "mean_gradient_alignment": optional_mean(alignments),
        "positive_gradient_alignment_rate": positive_alignment_rate,
        "num_repairs": len(outcomes),
        "mean_violation_reduction": optional_mean(violation_reductions),
        "repair_success_rate": repair_success_rate,
        "mean_segmentation_loss_gain": mean_gain,
        "segmentation_gain_ci95_bootstrap": gain_ci,
        "segmentation_repair_success_rate": success_rate,
        "mean_hard_dice_gain": optional_mean(hard_gains),
        "harm_rate": harm_rate,
        "mean_harm_magnitude": mean_harm,
        "complexity": complexity,
        "utility": utility,
        "checks": checks,
        "selected": selected,
        "sample_outcomes": outcomes,
    }


def best_configuration_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    ci = item.get("segmentation_gain_ci95_bootstrap") or [-math.inf, -math.inf]
    return (
        float(bool(item.get("selected"))),
        float(item.get("utility") or 0.0),
        float(ci[0]),
        float(item.get("mean_segmentation_loss_gain") or 0.0),
    )


def final_constraint_spec(
    candidate: dict[str, Any],
    result: dict[str, Any],
    rank: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = constraint_spec(candidate, rank=rank, args=args)
    parsed = parse_constraint(candidate["constraint"])
    arg_slug = "_".join(parsed["args"])
    direction_slug = "le" if parsed["direction"] == "<=" else "ge"
    spec["name"] = f"{rank:02d}_{parsed['primitive']}_{arg_slug}_{direction_slug}"
    spec["source"] = {
        "procedure": "automatic_crossfit_repair_v1",
        "candidate_score": candidate.get("score"),
        "fit_gt_purity": candidate.get("gt_purity"),
        "fit_coverage": candidate.get("coverage"),
        "cross_fit": {key: value for key, value in result.items() if key != "sample_outcomes"},
    }
    spec["notes"] = {"slug": slugify(candidate["constraint"]), "refit_threshold_on_all_discovery_samples": True}
    return spec


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Automatic Constraint Discovery",
        "",
        f"Checkpoint: `{payload['config']['load_weights']}`",
        f"Discovery samples: `{payload['num_discovery_samples']}`",
        f"Cross-fit folds: `{payload['config']['cross_fit_folds']}`",
        f"Selected constraints: `{len(payload['selected_constraints'])}`",
        "",
        "| Rank | Candidate | Gamma | GT viol. | Pred viol. | Alignment | Repair gain | 95% CI | Harm | Selected |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for rank, result in enumerate(payload["candidate_results"], start=1):
        ci = result.get("segmentation_gain_ci95_bootstrap")
        ci_text = "n/a" if ci is None else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
        alignment = result.get("mean_gradient_alignment")
        gain = result.get("mean_segmentation_loss_gain")
        lines.append(
            f"| {rank} | `{result['constraint']}` | {result['gamma']:.4g} "
            f"| {result['heldout_gt_violation_rate']:.3f} | {result['heldout_prediction_violation_rate']:.3f} "
            f"| {'n/a' if alignment is None else f'{alignment:.4f}'} "
            f"| {'n/a' if gain is None else f'{gain:.6f}'} | {ci_text} | {result['harm_rate']:.3f} "
            f"| {'yes' if result['selected'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "A candidate is selected only if held-out labels remain feasible, predictions violate the rule often enough, "
            "the constraint gradient is aligned with supervised error, nonlinear repair reduces violation and soft-Dice "
            "loss, the harm rate stays bounded, and the bootstrap lower confidence bound is positive.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    dataset = build_dataset(args)
    _, discovery_subset = build_fold_subsets(dataset, args.folds, args.fold, args.train_fraction)
    loader = DataLoader(
        discovery_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(args.load_weights, args.num_classes, device)
    rows = probe_split(model, loader, "discovery", args, device)
    samples = cache_frozen_predictions(model, loader, args, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not rows or not samples:
        raise RuntimeError("Discovery inference produced no samples.")

    sample_ids = sample_ids_from_rows(rows)
    assignments = make_cross_fit_assignments(sample_ids, args.cross_fit_folds, args.seed)
    samples_by_id = {sample.sample_idx: sample for sample in samples}
    evidence_by_configuration: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    fold_summaries = []

    for heldout_fold in range(args.cross_fit_folds):
        fit_ids = {sample_id for sample_id, fold in assignments.items() if fold != heldout_fold}
        heldout_ids = {sample_id for sample_id, fold in assignments.items() if fold == heldout_fold}
        fit_rows = [row for row in rows if int(row["sample_idx"]) in fit_ids]
        heldout_samples = [samples_by_id[sample_id] for sample_id in sorted(heldout_ids)]
        candidates = induce_candidates(fit_rows, args)[: args.candidate_top_k]
        fold_summaries.append(
            {
                "heldout_fold": heldout_fold,
                "fit_samples": len(fit_ids),
                "heldout_samples": len(heldout_ids),
                "candidate_count": len(candidates),
                "candidate_constraints": [candidate["constraint"] for candidate in candidates],
            }
        )
        for rank, candidate in enumerate(candidates, start=1):
            for gamma in args.gammas:
                evidence = evaluate_fold_candidate(
                    samples=heldout_samples,
                    candidate=candidate,
                    rank=rank,
                    gamma=gamma,
                    cross_fit_fold=heldout_fold,
                    args=args,
                    device=device,
                )
                evidence_by_configuration[(candidate["constraint"], gamma)].append(evidence)

    all_candidates = induce_candidates(rows, args)
    all_candidates_by_constraint = {candidate["constraint"]: candidate for candidate in all_candidates}
    configurations = [summarize_evidence(items, args) for items in evidence_by_configuration.values()]
    best_by_constraint: dict[str, dict[str, Any]] = {}
    for result in configurations:
        current = best_by_constraint.get(result["constraint"])
        if current is None or best_configuration_key(result) > best_configuration_key(current):
            best_by_constraint[result["constraint"]] = result
    candidate_results = sorted(best_by_constraint.values(), key=best_configuration_key, reverse=True)

    selected_specs = []
    for result in candidate_results:
        if not result["selected"]:
            continue
        candidate = all_candidates_by_constraint.get(result["constraint"])
        if candidate is None:
            continue
        selected_specs.append(final_constraint_spec(candidate, result, len(selected_specs) + 1, args))
        if len(selected_specs) >= args.max_constraints:
            break

    report = {
        "schema_version": "semantic_constraints.automatic_discovery.v1",
        "config": vars(args),
        "device": str(device),
        "num_discovery_samples": len(sample_ids),
        "cross_fit_assignments": assignments,
        "fold_summaries": fold_summaries,
        "candidate_results": candidate_results,
        "selected_constraints": [spec["name"] for spec in selected_specs],
    }
    selected_payload = {
        "schema_version": "semantic_constraints.selected.v1",
        "discovery_report": str(args.output_dir / "automatic_discovery_report.json"),
        "selection_policy": {
            "procedure": "automatic_crossfit_repair_v1",
            "max_constraints": args.max_constraints,
            "requires_positive_bootstrap_ci": not args.allow_nonpositive_ci,
            "max_heldout_gt_violation_rate": args.max_heldout_gt_violation_rate,
            "max_harm_rate": args.max_harm_rate,
        },
        "constraints": selected_specs,
    }
    candidate_payload = {
        "schema_version": "semantic_constraints.candidates.refit.v1",
        "source": "all discovery samples after cross-fit family evaluation",
        "top_candidates": all_candidates[: args.candidate_top_k],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "probe_rows.json", {"config": vars(args), "rows": rows})
    write_json(args.output_dir / "candidate_report_refit.json", candidate_payload)
    write_json(args.output_dir / "automatic_discovery_report.json", report)
    write_json(args.output_dir / "selected_constraints.json", selected_payload)
    write_markdown(args.output_dir / "automatic_discovery_report.md", report)

    print(f"device: {device}")
    print(f"discovery samples: {len(sample_ids)}")
    print(f"cross-fit configurations: {len(configurations)}")
    print(f"candidate families: {len(candidate_results)}")
    print(f"selected constraints: {len(selected_specs)}")
    print(f"report: {args.output_dir / 'automatic_discovery_report.json'}")
    print(f"selected JSON: {args.output_dir / 'selected_constraints.json'}")
    for result in candidate_results[:5]:
        print(
            result["constraint"],
            f"alignment={result['mean_gradient_alignment']}",
            f"gain={result['mean_segmentation_loss_gain']}",
            f"harm={result['harm_rate']:.3f}",
            f"selected={result['selected']}",
        )


if __name__ == "__main__":
    main()
