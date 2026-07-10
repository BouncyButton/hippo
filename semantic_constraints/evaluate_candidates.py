"""Condense probe rows into ranked semantic-constraint candidates.

The probe report is deliberately detailed and can be too large for an LLM loop.
This script turns it into a compact candidate-level report:

    python semantic_constraints/evaluate_candidates.py

Each candidate has the shape ``descriptor <= alpha`` or ``descriptor >= alpha``.
The script chooses alpha by sweeping thresholds that are mostly satisfied by GT,
then scores whether predictions violate the constraint and whether violations
align with segmentation error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_PROBE_REPORT = Path(__file__).resolve().parent / "probe_outputs" / "probe_fold1_val.json"
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "probe_outputs" / "candidate_report.json"
DEFAULT_OUTPUT_MD = Path(__file__).resolve().parent / "probe_outputs" / "candidate_report.md"


@dataclass(frozen=True)
class Observation:
    split: str
    sample_idx: int
    image: str
    scope: str
    expression: str
    gt_value: float
    pred_value: float
    dice_loss: float | None


@dataclass
class Candidate:
    expression: str
    direction: str
    alpha: float
    alpha_range: tuple[float, float]
    support: int
    total_samples: int
    exceptions: int
    coverage: float
    gt_purity: float
    pred_violation_rate: float
    mean_violation: float
    normalized_violation_gap: float
    error_alignment: float | None
    raw_score: float
    redundancy: float
    score: float
    n: int
    failure_examples: list[dict[str, Any]]
    violation_by_sample: dict[tuple[str, int], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidate semantic constraints from probe diagnostics.")
    parser.add_argument("--probe-report", type=Path, default=DEFAULT_PROBE_REPORT, help="Probe JSON or CSV path.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Compact JSON report path.")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD, help="Markdown report path.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of candidates to emit.")
    parser.add_argument("--failure-examples", type=int, default=5, help="Representative failures per candidate.")
    parser.add_argument(
        "--min-gt-volume",
        type=float,
        default=1e-6,
        help="Minimum GT volume for a class to count as present/applicable.",
    )
    parser.add_argument("--min-gt-purity", type=float, default=0.9, help="Minimum GT satisfaction for alpha selection.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Minimum rule coverage: applicable GT samples divided by total samples.",
    )
    parser.add_argument(
        "--min-pred-violation-rate",
        type=float,
        default=0.05,
        help="Discard candidates that predictions almost never violate.",
    )
    parser.add_argument("--redundancy-penalty", type=float, default=0.5, help="Score penalty strength for redundant candidates.")
    return parser.parse_args()


def maybe_number(value: Any) -> Any:
    if value == "" or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def finite_number(value: Any) -> float | None:
    value = maybe_number(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of rows in {path}.")
        return rows

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [{key: maybe_number(value) for key, value in row.items()} for row in csv.DictReader(f)]

    raise ValueError(f"Unsupported report format: {path.suffix}. Use JSON or CSV.")


def row_sample_key(row: dict[str, Any]) -> tuple[str, int] | None:
    split = row.get("split")
    sample_idx = finite_number(row.get("sample_idx"))
    if split is None or sample_idx is None:
        return None
    return str(split), int(sample_idx)


def mean_dice_loss_by_sample(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("metric_scope") != "class":
            continue
        key = row_sample_key(row)
        dice = finite_number(row.get("dice"))
        if key is not None and dice is not None:
            values[key].append(1.0 - dice)
    return {key: mean(items) for key, items in values.items() if items}


def sample_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys = set()
    for row in rows:
        key = row_sample_key(row)
        if key is not None:
            keys.add(key)
    return keys


def class_presence_by_sample(rows: list[dict[str, Any]], min_gt_volume: float) -> dict[tuple[str, int], set[str]]:
    presence: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("metric_scope") != "class":
            continue
        key = row_sample_key(row)
        class_id = row.get("class_id")
        gt_volume = finite_number(row.get("gt_volume"))
        if key is None or class_id in ("", None) or gt_volume is None:
            continue
        if gt_volume > min_gt_volume:
            presence[key].add(f"class_{int(float(class_id))}")
    return presence


def collect_observations(
    rows: list[dict[str, Any]],
    min_gt_volume: float,
) -> tuple[dict[str, list[Observation]], int]:
    dice_loss = mean_dice_loss_by_sample(rows)
    all_sample_keys = sample_keys(rows)
    class_presence = class_presence_by_sample(rows, min_gt_volume=min_gt_volume)
    groups: dict[str, list[Observation]] = defaultdict(list)

    class_features = [
        ("volume", "gt_volume", "pred_volume"),
        ("boundary_length", "gt_boundary_length", "pred_boundary_length"),
        ("compactness", "gt_compactness", "pred_compactness"),
        ("connectedness", "gt_connectedness", "pred_connectedness"),
    ]
    pair_features = [
        ("distance", "gt_pair_distance", "pred_pair_distance", "distance({a}, {b})"),
        ("overlap", "gt_pair_overlap", "pred_pair_overlap", "overlap({a}, {b})"),
        ("adjacent", "gt_adjacent", "pred_adjacent", "adjacent({a}, {b})"),
        ("contains_a_b", "gt_contains_a_b", "pred_contains_a_b", "contains({a}, {b})"),
        ("contains_b_a", "gt_contains_b_a", "pred_contains_b_a", "contains({b}, {a})"),
    ]

    for row in rows:
        key = row_sample_key(row)
        if key is None:
            continue
        split, sample_idx = key
        image = str(row.get("image") or "")

        if row.get("metric_scope") == "class":
            class_id = row.get("class_id")
            if class_id in ("", None):
                continue
            class_name = f"class_{int(float(class_id))}"
            if class_name not in class_presence.get(key, set()):
                continue
            for name, gt_key, pred_key in class_features:
                gt_value = finite_number(row.get(gt_key))
                pred_value = finite_number(row.get(pred_key))
                if gt_value is None or pred_value is None:
                    continue
                expression = f"{name}({class_name})"
                groups[expression].append(
                    Observation(
                        split=split,
                        sample_idx=sample_idx,
                        image=image,
                        scope=class_name,
                        expression=expression,
                        gt_value=gt_value,
                        pred_value=pred_value,
                        dice_loss=dice_loss.get(key),
                    )
                )

        elif row.get("metric_scope") == "pair":
            raw_pair = str(row.get("pair") or "")
            if "-" not in raw_pair:
                continue
            a_raw, b_raw = raw_pair.split("-", maxsplit=1)
            a = f"class_{int(float(a_raw))}"
            b = f"class_{int(float(b_raw))}"
            present = class_presence.get(key, set())
            if a not in present or b not in present:
                continue
            for _name, gt_key, pred_key, template in pair_features:
                gt_value = finite_number(row.get(gt_key))
                pred_value = finite_number(row.get(pred_key))
                if gt_value is None or pred_value is None:
                    continue
                expression = template.format(a=a, b=b)
                groups[expression].append(
                    Observation(
                        split=split,
                        sample_idx=sample_idx,
                        image=image,
                        scope=f"{a}-{b}",
                        expression=expression,
                        gt_value=gt_value,
                        pred_value=pred_value,
                        dice_loss=dice_loss.get(key),
                    )
                )

    return groups, len(all_sample_keys)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    x = [value - mx for value in xs]
    y = [value - my for value in ys]
    denom = math.sqrt(sum(value * value for value in x)) * math.sqrt(sum(value * value for value in y))
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(x, y)) / denom


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of an empty list.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def value_scale(values: list[float]) -> float:
    if not values:
        return 1.0
    iqr = quantile(values, 0.75) - quantile(values, 0.25)
    full_range = max(values) - min(values)
    magnitude = abs(mean(values))
    return max(iqr, 0.1 * full_range, 0.1 * magnitude, 1e-8)


def violation(value: float, direction: str, alpha: float) -> float:
    if direction == "<=":
        return max(value - alpha, 0.0)
    if direction == ">=":
        return max(alpha - value, 0.0)
    raise ValueError(f"Unknown direction: {direction}")


def satisfied(value: float, direction: str, alpha: float) -> bool:
    return violation(value, direction, alpha) == 0.0


def alpha_grid(gt_values: list[float]) -> list[float]:
    if not gt_values:
        return []
    qs = [i / 100 for i in range(0, 101)]
    grid = {quantile(gt_values, q) for q in qs}
    grid.update(gt_values)
    return sorted(grid)


def evaluate_candidate(
    expression: str,
    observations: list[Observation],
    direction: str,
    total_samples: int,
    min_gt_purity: float,
    failure_count: int,
) -> Candidate | None:
    if not observations:
        return None

    gt_values = [obs.gt_value for obs in observations]
    pred_values = [obs.pred_value for obs in observations]
    scale = value_scale(gt_values + pred_values)

    viable: list[tuple[float, float, float, float, list[float]]] = []
    for alpha in alpha_grid(gt_values):
        gt_purity = sum(satisfied(value, direction, alpha) for value in gt_values) / len(gt_values)
        if gt_purity < min_gt_purity:
            continue
        violations = [violation(value, direction, alpha) for value in pred_values]
        pred_violation_rate = sum(value > 0.0 for value in violations) / len(violations)
        mean_violation = mean(violations)
        normalized_gap = mean_violation / scale
        viable.append((alpha, gt_purity, pred_violation_rate, normalized_gap, violations))

    if not viable:
        return None

    support = len(observations)
    coverage = support / max(total_samples, 1)
    best_tuple = None
    best_score = -1.0
    best_alignment = None

    for alpha, gt_purity, pred_violation_rate, normalized_gap, violations in viable:
        aligned_pairs = [
            (v, obs.dice_loss)
            for v, obs in zip(violations, observations)
            if obs.dice_loss is not None and math.isfinite(obs.dice_loss)
        ]
        alignment = pearson([item[0] for item in aligned_pairs], [item[1] for item in aligned_pairs]) if aligned_pairs else None
        positive_alignment = max(alignment or 0.0, 0.0)
        raw_score = coverage * gt_purity * (
            0.45 * pred_violation_rate + 0.35 * min(normalized_gap, 1.0) + 0.20 * positive_alignment
        )
        if raw_score > best_score:
            best_score = raw_score
            best_tuple = (alpha, gt_purity, pred_violation_rate, normalized_gap, violations)
            best_alignment = alignment

    if best_tuple is None:
        return None

    alpha, gt_purity, pred_violation_rate, normalized_gap, violations = best_tuple
    exceptions = sum(not satisfied(value, direction, alpha) for value in gt_values)
    viable_alphas = [item[0] for item in viable]
    failure_examples = []
    ranked_failures = sorted(
        zip(observations, violations),
        key=lambda item: (item[1], item[0].dice_loss or 0.0),
        reverse=True,
    )
    for obs, obs_violation in ranked_failures[:failure_count]:
        failure_examples.append(
            {
                "split": obs.split,
                "sample_idx": obs.sample_idx,
                "image": obs.image,
                "gt_value": obs.gt_value,
                "pred_value": obs.pred_value,
                "violation": obs_violation,
                "dice_loss": obs.dice_loss,
            }
        )

    return Candidate(
        expression=expression,
        direction=direction,
        alpha=alpha,
        alpha_range=(min(viable_alphas), max(viable_alphas)),
        support=support,
        total_samples=total_samples,
        exceptions=exceptions,
        coverage=coverage,
        gt_purity=gt_purity,
        pred_violation_rate=pred_violation_rate,
        mean_violation=mean(violations),
        normalized_violation_gap=normalized_gap,
        error_alignment=best_alignment,
        raw_score=best_score,
        redundancy=0.0,
        score=best_score,
        n=len(observations),
        failure_examples=failure_examples,
        violation_by_sample={(obs.split, obs.sample_idx): obs_violation for obs, obs_violation in zip(observations, violations)},
    )


def redundancy(candidate: Candidate, selected: list[Candidate]) -> float:
    max_corr = 0.0
    for previous in selected:
        common = sorted(set(candidate.violation_by_sample) & set(previous.violation_by_sample))
        if len(common) < 2:
            continue
        corr = pearson(
            [candidate.violation_by_sample[key] for key in common],
            [previous.violation_by_sample[key] for key in common],
        )
        if corr is not None:
            max_corr = max(max_corr, abs(corr))
    return max_corr


def build_candidates(
    groups: dict[str, list[Observation]],
    total_samples: int,
    min_gt_purity: float,
    min_coverage: float,
    min_pred_violation_rate: float,
    failure_count: int,
    redundancy_penalty: float,
) -> list[Candidate]:
    candidates = []
    for expression, observations in groups.items():
        for direction in ("<=", ">="):
            candidate = evaluate_candidate(
                expression=expression,
                observations=observations,
                direction=direction,
                total_samples=total_samples,
                min_gt_purity=min_gt_purity,
                failure_count=failure_count,
            )
            if (
                candidate is None
                or candidate.coverage < min_coverage
                or candidate.pred_violation_rate < min_pred_violation_rate
            ):
                continue
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.raw_score, reverse=True)
    selected_for_redundancy: list[Candidate] = []
    for candidate in candidates:
        candidate.redundancy = redundancy(candidate, selected_for_redundancy)
        candidate.score = candidate.raw_score * (1.0 - redundancy_penalty * candidate.redundancy)
        selected_for_redundancy.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    alpha_name = "alpha"
    return {
        "constraint": f"{candidate.expression} {candidate.direction} {alpha_name}",
        "alpha": candidate.alpha,
        "suggested_alpha_range": list(candidate.alpha_range),
        "support": candidate.support,
        "total_samples": candidate.total_samples,
        "exceptions": candidate.exceptions,
        "coverage": candidate.coverage,
        "gt_purity": candidate.gt_purity,
        "confidence": candidate.gt_purity,
        "prediction_violation_rate": candidate.pred_violation_rate,
        "mean_violation": candidate.mean_violation,
        "normalized_violation_gap": candidate.normalized_violation_gap,
        "error_alignment_pearson": candidate.error_alignment,
        "redundancy_with_higher_ranked": candidate.redundancy,
        "score": candidate.score,
        "raw_score": candidate.raw_score,
        "n": candidate.n,
        "representative_failure_examples": candidate.failure_examples,
    }


def summary_stats(candidates: list[Candidate]) -> dict[str, Any]:
    if not candidates:
        return {"num_candidates": 0}
    scores = [candidate.score for candidate in candidates]
    return {
        "num_candidates": len(candidates),
        "score_mean": mean(scores),
        "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "score_min": min(scores),
        "score_max": max(scores),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Candidate Constraint Report",
        "",
        f"Probe report: `{payload['probe_report']}`",
        f"Generated candidates: {payload['summary']['num_candidates']}",
        "",
        "## Top Candidates",
        "",
    ]
    for index, candidate in enumerate(payload["top_candidates"], start=1):
        lines.extend(
            [
                f"### {index}. `{candidate['constraint']}`",
                "",
                f"- alpha: `{candidate['alpha']:.6g}`",
                f"- suggested alpha range: `{candidate['suggested_alpha_range'][0]:.6g}` to `{candidate['suggested_alpha_range'][1]:.6g}`",
                f"- score: `{candidate['score']:.4f}`",
                f"- support: `{candidate['support']}/{candidate['total_samples']}`",
                f"- coverage: `{candidate['coverage']:.3f}`",
                f"- GT purity/confidence: `{candidate['gt_purity']:.3f}`",
                f"- GT exceptions: `{candidate['exceptions']}`",
                f"- prediction violation rate: `{candidate['prediction_violation_rate']:.3f}`",
                f"- normalized violation gap: `{candidate['normalized_violation_gap']:.3f}`",
                f"- error alignment Pearson: `{format_optional(candidate['error_alignment_pearson'])}`",
                f"- redundancy: `{candidate['redundancy_with_higher_ranked']:.3f}`",
                "",
                "Representative failures:",
            ]
        )
        for failure in candidate["representative_failure_examples"][:3]:
            lines.append(
                f"- split `{failure['split']}`, sample `{failure['sample_idx']}`: "
                f"gt `{failure['gt_value']:.6g}`, pred `{failure['pred_value']:.6g}`, "
                f"violation `{failure['violation']:.6g}`, dice_loss `{format_optional(failure['dice_loss'])}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.probe_report)
    groups, total_samples = collect_observations(rows, min_gt_volume=args.min_gt_volume)
    candidates = build_candidates(
        groups=groups,
        total_samples=total_samples,
        min_gt_purity=args.min_gt_purity,
        min_coverage=args.min_coverage,
        min_pred_violation_rate=args.min_pred_violation_rate,
        failure_count=args.failure_examples,
        redundancy_penalty=args.redundancy_penalty,
    )

    top = candidates[: args.top_k]
    payload = {
        "probe_report": str(args.probe_report),
        "selection_policy": {
            "min_gt_purity": args.min_gt_purity,
            "min_gt_volume": args.min_gt_volume,
            "min_coverage": args.min_coverage,
            "min_pred_violation_rate": args.min_pred_violation_rate,
            "score": "coverage * gt_purity * (0.45*prediction_violation_rate + 0.35*normalized_gap + 0.20*positive_error_alignment), with redundancy penalty",
        },
        "summary": summary_stats(candidates),
        "top_candidates": [candidate_to_dict(candidate) for candidate in top],
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)

    print(f"read rows: {len(rows)}")
    print(f"total samples: {total_samples}")
    print(f"candidate families: {len(groups)}")
    print(f"ranked candidates: {len(candidates)}")
    print(f"wrote JSON: {args.output_json}")
    print(f"wrote Markdown: {args.output_md}")
    for candidate in top[:5]:
        print(
            f"{candidate.expression} {candidate.direction} alpha",
            f"alpha={candidate.alpha:.6g}",
            f"score={candidate.score:.4f}",
            f"coverage={candidate.coverage:.3f}",
            f"gt={candidate.gt_purity:.3f}",
            f"pred_viol={candidate.pred_violation_rate:.3f}",
        )


if __name__ == "__main__":
    main()
