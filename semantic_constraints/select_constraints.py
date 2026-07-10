"""Select executable constraint specs from a ranked candidate report.

This is the bridge between rule induction diagnostics and training. It consumes
``candidate_report.json`` and emits ``selected_constraints.json`` with a stable,
machine-readable constraint spec. The selector is deliberately rule-based for
now; an LLM can later revise the same JSON contract without changing training.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_REPORT = Path(__file__).resolve().parent / "probe_outputs" / "candidate_report.json"
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "probe_outputs" / "selected_constraints.json"
DEFAULT_OUTPUT_MD = Path(__file__).resolve().parent / "probe_outputs" / "selected_constraints.md"

CONSTRAINT_RE = re.compile(
    r"^\s*(?P<primitive>[a-zA-Z_][a-zA-Z0-9_]*)\((?P<args>[^)]*)\)\s*"
    r"(?P<direction><=|>=)\s*alpha\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select executable constraints from candidate diagnostics.")
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-constraints", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-purity", type=float, default=0.9)
    parser.add_argument("--min-pred-violation-rate", type=float, default=0.1)
    parser.add_argument("--max-exceptions", type=int, default=3)
    parser.add_argument("--max-redundancy", type=float, default=0.9)
    parser.add_argument("--lambda-default", type=float, default=0.1)
    parser.add_argument(
        "--lambda-by-score",
        action="store_true",
        help="Scale lambda by candidate score instead of using lambda-default exactly.",
    )
    parser.add_argument(
        "--fuzzy-mode",
        choices=("soft_exp", "lukasiewicz", "lukasiewicz_ste", "hinge"),
        default="soft_exp",
    )
    parser.add_argument("--fuzzy-margin", type=float, default=1.0)
    parser.add_argument("--fuzzy-beta", type=float, default=4.0)
    return parser.parse_args()


def load_candidate_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "top_candidates" not in payload:
        raise ValueError(f"{path} does not look like a candidate report.")
    return payload


def parse_constraint(text: str) -> dict[str, Any]:
    match = CONSTRAINT_RE.match(text)
    if not match:
        raise ValueError(f"Cannot parse constraint expression: {text!r}")
    args = [item.strip() for item in match.group("args").split(",") if item.strip()]
    return {
        "primitive": match.group("primitive"),
        "args": args,
        "direction": match.group("direction"),
    }


def slugify(value: str) -> str:
    value = value.lower()
    value = value.replace("<=", "le").replace(">=", "ge")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def candidate_passes(candidate: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons = []
    checks = [
        (candidate.get("score", 0.0) >= args.min_score, "score"),
        (candidate.get("coverage", 0.0) >= args.min_coverage, "coverage"),
        (candidate.get("gt_purity", 0.0) >= args.min_purity, "gt_purity"),
        (candidate.get("prediction_violation_rate", 0.0) >= args.min_pred_violation_rate, "prediction_violation_rate"),
        (candidate.get("exceptions", 10**9) <= args.max_exceptions, "exceptions"),
        (candidate.get("redundancy_with_higher_ranked", 1.0) <= args.max_redundancy, "redundancy"),
    ]
    for passed, name in checks:
        if not passed:
            reasons.append(name)
    return not reasons, reasons


def lambda_for(candidate: dict[str, Any], args: argparse.Namespace) -> float:
    if not args.lambda_by_score:
        return args.lambda_default
    return args.lambda_default * max(float(candidate.get("score", 0.0)), 1e-8)


def make_constraint_spec(candidate: dict[str, Any], rank: int, args: argparse.Namespace) -> dict[str, Any]:
    parsed = parse_constraint(candidate["constraint"])
    expression_slug = slugify(candidate["constraint"])
    primitive = parsed["primitive"]
    arg_slug = "_".join(parsed["args"])
    direction_slug = "le" if parsed["direction"] == "<=" else "ge"
    name = f"{rank:02d}_{primitive}_{arg_slug}_{direction_slug}"

    return {
        "name": name,
        "expression": candidate["constraint"],
        "primitive": primitive,
        "args": parsed["args"],
        "direction": parsed["direction"],
        "alpha": candidate["alpha"],
        "suggested_alpha_range": candidate.get("suggested_alpha_range"),
        "scale": max(float(candidate.get("mean_violation", 0.0)), 1e-8),
        "lambda": lambda_for(candidate, args),
        "fuzzy": {
            "mode": args.fuzzy_mode,
            "margin": args.fuzzy_margin,
            "beta": args.fuzzy_beta,
        },
        "source": {
            "rank": rank,
            "score": candidate.get("score"),
            "raw_score": candidate.get("raw_score"),
            "support": candidate.get("support"),
            "total_samples": candidate.get("total_samples"),
            "coverage": candidate.get("coverage"),
            "gt_purity": candidate.get("gt_purity"),
            "confidence": candidate.get("confidence", candidate.get("gt_purity")),
            "exceptions": candidate.get("exceptions"),
            "prediction_violation_rate": candidate.get("prediction_violation_rate"),
            "normalized_violation_gap": candidate.get("normalized_violation_gap"),
            "error_alignment_pearson": candidate.get("error_alignment_pearson"),
            "redundancy_with_higher_ranked": candidate.get("redundancy_with_higher_ranked"),
            "representative_failure_examples": candidate.get("representative_failure_examples", []),
        },
        "notes": {
            "selector": "rule_based_v1",
            "slug": expression_slug,
        },
    }


def select_constraints(payload: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    rejected = []

    for rank, candidate in enumerate(payload["top_candidates"], start=1):
        passed, reasons = candidate_passes(candidate, args)
        if not passed:
            rejected.append(
                {
                    "rank": rank,
                    "constraint": candidate.get("constraint"),
                    "reasons": reasons,
                    "score": candidate.get("score"),
                }
            )
            continue

        selected.append(make_constraint_spec(candidate, rank=rank, args=args))
        if len(selected) >= args.max_constraints:
            break

    return selected, rejected


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Selected Constraint Specs",
        "",
        f"Candidate report: `{payload['candidate_report']}`",
        f"Selected constraints: {len(payload['constraints'])}",
        "",
    ]
    for index, constraint in enumerate(payload["constraints"], start=1):
        source = constraint["source"]
        fuzzy = constraint["fuzzy"]
        lines.extend(
            [
                f"## {index}. `{constraint['name']}`",
                "",
                f"- expression: `{constraint['expression']}`",
                f"- primitive: `{constraint['primitive']}`",
                f"- args: `{', '.join(constraint['args'])}`",
                f"- alpha: `{constraint['alpha']:.6g}`",
                f"- lambda: `{constraint['lambda']:.6g}`",
                f"- fuzzy: `{fuzzy['mode']}`",
                f"- support: `{source['support']}/{source['total_samples']}`",
                f"- coverage: `{source['coverage']:.3f}`",
                f"- GT purity/confidence: `{source['gt_purity']:.3f}`",
                f"- prediction violation rate: `{source['prediction_violation_rate']:.3f}`",
                f"- score: `{source['score']:.4f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_report = load_candidate_report(args.candidate_report)
    constraints, rejected = select_constraints(candidate_report, args)

    payload = {
        "schema_version": "semantic_constraints.selected.v1",
        "candidate_report": str(args.candidate_report),
        "selection_policy": {
            "max_constraints": args.max_constraints,
            "min_score": args.min_score,
            "min_coverage": args.min_coverage,
            "min_purity": args.min_purity,
            "min_pred_violation_rate": args.min_pred_violation_rate,
            "max_exceptions": args.max_exceptions,
            "max_redundancy": args.max_redundancy,
            "lambda_default": args.lambda_default,
            "lambda_by_score": args.lambda_by_score,
            "fuzzy_mode": args.fuzzy_mode,
            "fuzzy_margin": args.fuzzy_margin,
            "fuzzy_beta": args.fuzzy_beta,
        },
        "constraints": constraints,
        "rejected": rejected,
    }

    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)

    print(f"read candidates: {len(candidate_report['top_candidates'])}")
    print(f"selected constraints: {len(constraints)}")
    print(f"wrote JSON: {args.output_json}")
    print(f"wrote Markdown: {args.output_md}")
    for constraint in constraints:
        source = constraint["source"]
        print(
            constraint["name"],
            f"score={source['score']:.4f}",
            f"coverage={source['coverage']:.3f}",
            f"purity={source['gt_purity']:.3f}",
            f"pred_viol={source['prediction_violation_rate']:.3f}",
        )


if __name__ == "__main__":
    main()
