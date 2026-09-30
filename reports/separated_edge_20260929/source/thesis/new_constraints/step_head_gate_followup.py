"""Fold-1 replication and patient-grouped trust gate for the sagittal head."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe, _load_splits_json
from thesis.new_constraints.step_head_pilot import PilotModel, head_mask, plain, sha256, train_arm


FEATURE_NAMES = (
    "head_presence_probability", "seg_occupied", "head_occupied", "seg_fg_max",
    "seg_fg_mean", "seg_fg_on_added", "seg_fg_on_removed", "head_edge_peak",
    "head_edge_entropy", "head_length", "seg_length", "lower_displacement",
    "upper_displacement", "abs_lower_displacement", "abs_upper_displacement",
)
REGULARIZATION = (0.03, 0.3, 3.0)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass
class Case:
    name: str
    truth: np.ndarray
    seg: np.ndarray
    head: np.ndarray
    disagreement: np.ndarray
    features: np.ndarray
    head_gain: np.ndarray
    head_only: np.ndarray
    seg_only: np.ndarray


def union_dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_fg, truth_fg = pred > 0, truth > 0
    denominator = int(pred_fg.sum() + truth_fg.sum())
    return float(2 * (pred_fg & truth_fg).sum() / denominator) if denominator else 1.0


def class_dice(pred: np.ndarray, truth: np.ndarray, cls: int) -> float:
    p, t = pred == cls, truth == cls
    denominator = int(p.sum() + t.sum())
    return float(2 * (p & t).sum() / denominator) if denominator else 1.0


def column_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = mask.any(-1)
    z = np.arange(mask.shape[-1])[None, None, :]
    lower = np.where(mask, z, mask.shape[-1]).min(-1)
    upper = np.where(mask, z, -1).max(-1)
    return occupied, np.where(occupied, lower, 0), np.where(occupied, upper, 0)


def make_case(name: str, saved: dict, truth: np.ndarray) -> Case:
    seg = saved["hard_mask"].astype(np.uint8)
    head = saved["head_mask"].astype(np.uint8)
    assert seg.shape == head.shape == truth.shape == (64, 64, 64)
    seg_fg, head_fg, truth_fg = seg > 0, head > 0, truth > 0
    disagreement = (seg_fg ^ head_fg).any(-1)
    seg_occ, seg_lo, seg_hi = column_bounds(seg_fg)
    head_occ, head_lo, head_hi = column_bounds(head_fg)
    probability = saved["probability"].astype(np.float32)
    fg_probability = probability[1] + probability[2]
    presence_probability = saved["head_presence_probability"].astype(np.float32)
    edge_probability = saved["head_edge_probability"].astype(np.float32)
    edge_probability /= edge_probability.sum(-1, keepdims=True).clip(min=1e-8)
    edge_peak = edge_probability.max(-1).mean(0)
    edge_entropy = -(edge_probability * np.log(edge_probability.clip(min=1e-8))).sum(-1).mean(0) / math.log(64)
    added = head_fg & ~seg_fg
    removed = seg_fg & ~head_fg
    added_count = added.sum(-1)
    removed_count = removed.sum(-1)
    added_fg_probability = (fg_probability * added).sum(-1) / np.maximum(added_count, 1)
    removed_fg_probability = (fg_probability * removed).sum(-1) / np.maximum(removed_count, 1)
    features = np.stack(
        (
            presence_probability, seg_occ.astype(float), head_occ.astype(float),
            fg_probability.max(-1), fg_probability.mean(-1), added_fg_probability,
            removed_fg_probability, edge_peak, edge_entropy,
            np.where(head_occ, head_hi - head_lo + 1, 0) / 64,
            np.where(seg_occ, seg_hi - seg_lo + 1, 0) / 64,
            (head_lo - seg_lo) / 64, (head_hi - seg_hi) / 64,
            np.abs(head_lo - seg_lo) / 64, np.abs(head_hi - seg_hi) / 64,
        ),
        axis=-1,
    )[disagreement].astype(np.float32)
    assert np.isfinite(features).all()
    head_gain = ((head_fg == truth_fg).sum(-1) - (seg_fg == truth_fg).sum(-1))[disagreement]
    return Case(
        name, truth, seg, head, disagreement, features, head_gain,
        (head_occ & ~seg_occ)[disagreement], (seg_occ & ~head_occ)[disagreement],
    )


def gated_mask(case: Case, accept: np.ndarray) -> np.ndarray:
    assert len(accept) == len(case.features)
    selected = np.zeros(case.disagreement.shape, dtype=bool)
    selected[case.disagreement] = accept
    return np.where(selected[..., None], case.head, case.seg)


def fit_classifier(cases: list[Case], c: float):
    features = np.concatenate([case.features for case in cases])
    gains = np.concatenate([case.head_gain for case in cases])
    if len(features) == 0 or len(np.unique(gains > 0)) < 2:
        return None
    weight = np.concatenate([
        np.full(len(case.features), 1 / max(len(case.features), 1), dtype=np.float64)
        for case in cases
    ])
    weight *= len(features) / weight.sum()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=0))
    model.fit(features, gains > 0, logisticregression__sample_weight=weight)
    return model


def choose_gate(cases: list[Case]) -> tuple[dict, object | None]:
    assert len(cases) == 42 and len({case.name for case in cases}) == 42
    groups = np.arange(len(cases))
    folds = list(GroupKFold(n_splits=5).split(groups, groups=groups))
    candidate_rows = []
    best = {"enabled": False, "cv_gain": 0.0, "c": None, "threshold": None}
    for c in REGULARIZATION:
        oof = [np.zeros(len(case.features), dtype=np.float32) for case in cases]
        valid = True
        for train_idx, test_idx in folds:
            model = fit_classifier([cases[int(i)] for i in train_idx], c)
            if model is None:
                valid = False
                break
            for index in test_idx:
                case = cases[int(index)]
                if len(case.features):
                    oof[int(index)] = model.predict_proba(case.features)[:, 1].astype(np.float32)
        if not valid:
            continue
        for threshold in THRESHOLDS:
            gains = [union_dice(gated_mask(case, score >= threshold), case.truth)
                     - union_dice(case.seg, case.truth) for case, score in zip(cases, oof)]
            mean_gain = float(np.mean(gains))
            candidate_rows.append({"c": c, "threshold": threshold, "cv_gain": mean_gain,
                                   "cv_wins": int(np.count_nonzero(np.array(gains) > 0)),
                                   "cv_losses": int(np.count_nonzero(np.array(gains) < 0))})
            if mean_gain > best["cv_gain"] + 1e-12:
                best = {"enabled": True, "cv_gain": mean_gain, "c": c, "threshold": threshold}
    best["candidates"] = candidate_rows
    return best, fit_classifier(cases, best["c"]) if best["enabled"] else None


def paired_interval(values: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "n": len(values)}


def build_loaders(pkl: Path, splits: Path, output: Path):
    ds = _build_monai_dataset_from_pkl(_load_pkl_dataframe(pkl), "MSD", 3, spatial_size=(64, 64, 64))
    outer = _load_splits_json(splits)[1]
    train_names, dev_names = set(outer["train"]), set(outer["val"])
    assert len(train_names) == 208 and len(dev_names) == 52 and not train_names & dev_names
    rng = np.random.default_rng(0)
    shuffled = np.array(sorted(train_names))
    rng.shuffle(shuffled)
    inner_names = set(shuffled[:42].tolist())
    fit_names = train_names - inner_names
    split = {"seed": 0, "outer_fold": 1, "train": sorted(fit_names),
             "inner_val": sorted(inner_names), "outer_dev": sorted(dev_names)}
    path = output / "split.json"
    if path.exists():
        assert json.loads(path.read_text()) == split
    else:
        path.write_text(json.dumps(split, indent=2))
    def make(selected, shuffle=False, batch=2):
        items = [item for item in ds.data if item["case_name"] in selected]
        assert len(items) == len(selected)
        return DataLoader(Dataset(items, ds.transform), batch_size=batch, shuffle=shuffle, num_workers=0)
    return make(fit_names, True), make(inner_names), make(dev_names, batch=1), make(inner_names, batch=1)


def load_cases(loader, probability_dir: Path, model=None, device=None) -> list[Case]:
    probability_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for batch in loader:
        name = str(batch["case_name"][0])
        truth = batch["label"][0, 0].cpu().numpy().astype(np.uint8)
        path = probability_dir / f"{name}.npz"
        if not path.exists():
            assert model is not None and device is not None
            with torch.no_grad():
                logits, head = model(plain(batch["image"], device))
                probability = logits.float().softmax(1)[0].cpu().numpy().astype(np.float16)
                assert head is not None
                mask = head_mask(head, probability.astype(np.float32))
                np.savez_compressed(
                    path, probability=probability, hard_mask=probability.argmax(0).astype(np.uint8),
                    head_mask=mask,
                    head_presence_probability=head[0].float().sigmoid()[0].cpu().numpy().astype(np.float16),
                    head_edge_probability=head[1].float().softmax(-1)[0].cpu().numpy().astype(np.float16),
                )
        with np.load(path) as saved:
            cases.append(make_case(name, saved, truth))
    return sorted(cases, key=lambda case: case.name)


def load_baseline_masks(loader, probability_dir: Path, model, device) -> dict[str, np.ndarray]:
    probability_dir.mkdir(parents=True, exist_ok=True)
    masks = {}
    for batch in loader:
        name = str(batch["case_name"][0])
        path = probability_dir / f"{name}.npz"
        if not path.exists():
            with torch.no_grad():
                logits, _ = model(plain(batch["image"], device))
                probability = logits.float().softmax(1)[0].cpu().numpy().astype(np.float16)
            np.savez_compressed(path, probability=probability,
                                hard_mask=probability.argmax(0).astype(np.uint8))
        with np.load(path) as saved:
            masks[name] = saved["hard_mask"].astype(np.uint8)
    return masks


def evaluate(cases: list[Case], gate: dict, model, output: Path, a_masks: dict[str, np.ndarray]) -> dict:
    rows = []
    for case in cases:
        score = model.predict_proba(case.features)[:, 1] if model is not None and len(case.features) else np.zeros(len(case.features))
        accept = score >= gate["threshold"] if gate["enabled"] else np.zeros(len(case.features), dtype=bool)
        chosen = gated_mask(case, accept)
        baseline = a_masks[case.name]
        rows.append({
            "patient": case.name,
            "A_union_dice": union_dice(baseline, case.truth),
            "D_union_dice": union_dice(case.seg, case.truth),
            "head_union_dice": union_dice(case.head, case.truth),
            "gate_union_dice": union_dice(chosen, case.truth),
            "D_posterior_dice": class_dice(case.seg, case.truth, 1),
            "gate_posterior_dice": class_dice(chosen, case.truth, 1),
            "D_anterior_dice": class_dice(case.seg, case.truth, 2),
            "gate_anterior_dice": class_dice(chosen, case.truth, 2),
            "disagreement_columns": len(case.features),
            "accepted_columns": int(accept.sum()),
            "accepted_head_wins": int(np.count_nonzero(accept & (case.head_gain > 0))),
            "accepted_seg_wins": int(np.count_nonzero(accept & (case.head_gain < 0))),
            "accepted_ties": int(np.count_nonzero(accept & (case.head_gain == 0))),
            "head_only_columns": int(case.head_only.sum()),
            "accepted_head_only": int(np.count_nonzero(accept & case.head_only)),
            "seg_only_columns": int(case.seg_only.sum()),
            "accepted_seg_only": int(np.count_nonzero(accept & case.seg_only)),
        })
    with (output / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    def mean(key):
        return float(np.mean([row[key] for row in rows]))
    summary = {"outer_fold": 1, "dev_case_count": len(rows), "gate": gate,
               "A_union_dice": mean("A_union_dice"), "D_union_dice": mean("D_union_dice"),
               "head_union_dice": mean("head_union_dice"), "gated_union_dice": mean("gate_union_dice")}
    for key, left, right in (("D_minus_A", "D_union_dice", "A_union_dice"),
                             ("gate_minus_D", "gate_union_dice", "D_union_dice"),
                             ("gate_minus_A", "gate_union_dice", "A_union_dice")):
        summary[key] = paired_interval(np.array([row[left] - row[right] for row in rows]), seed=len(key))
    for key in ("disagreement_columns", "accepted_columns", "accepted_head_wins", "accepted_seg_wins",
                "accepted_ties", "head_only_columns", "accepted_head_only", "seg_only_columns", "accepted_seg_only"):
        summary[key] = int(sum(row[key] for row in rows))
    summary["gate_patient_wins"] = int(sum(row["gate_union_dice"] > row["D_union_dice"] for row in rows))
    summary["gate_patient_losses"] = int(sum(row["gate_union_dice"] < row["D_union_dice"] for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__).resolve(), Path(importlib.import_module("thesis.new_constraints.step_head_pilot").__file__).resolve())
    config = {"outer_fold": 1, "seed": 0, "arms": ["A", "D"], "train_count": 166,
              "inner_val_count": 42, "outer_dev_count": 52,
              "source_manifest": {str(path): sha256(path) for path in sources},
              "pkl_sha256": sha256(args.pkl), "splits_sha256": sha256(args.splits),
              "gate_features": FEATURE_NAMES, "gate_regularization": REGULARIZATION,
              "gate_thresholds": THRESHOLDS, "gate_selection": "five-fold patient-grouped inner-validation mean union Dice gain; no-op if no positive gain"}
    cfg = args.output / "config.json"
    if cfg.exists():
        assert json.loads(cfg.read_text()) == json.loads(json.dumps(config)), "source/data/config changed during resume"
    else:
        cfg.write_text(json.dumps(config, indent=2))
    train, val, dev, val_single = build_loaders(args.pkl, args.splits, args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arm in ("A", "D"):
        train_arm(args, arm, args.output, train, val, None, device)
    done = json.loads((args.output / "D" / "done.json").read_text())
    checkpoint = args.output / "D" / Path(done["checkpoint"]).name
    assert sha256(checkpoint) == done["checkpoint_sha256"]
    model = PilotModel("D").to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    inner_cases = load_cases(val_single, args.output / "D" / "inner_val_probabilities", model, device)
    gate, classifier = choose_gate(inner_cases)
    (args.output / "gate_selection.json").write_text(json.dumps(gate, indent=2))
    d_cases = load_cases(dev, args.output / "D" / "probabilities", model, device)
    a_done = json.loads((args.output / "A" / "done.json").read_text())
    a_checkpoint = args.output / "A" / Path(a_done["checkpoint"]).name
    assert sha256(a_checkpoint) == a_done["checkpoint_sha256"]
    a_model = PilotModel("A").to(device)
    a_model.load_state_dict(torch.load(a_checkpoint, map_location=device, weights_only=False)["model"])
    a_model.eval()
    a_masks = load_baseline_masks(dev, args.output / "A" / "probabilities", a_model, device)
    assert len(d_cases) == len(a_masks) == 52
    summary = evaluate(d_cases, gate, classifier, args.output, a_masks)
    lines = ["# Fold-1 sagittal contour trust-gate follow-up", "",
             "The gate used only the 42 fold-1 inner-validation patients. Model checkpoints were selected on that same inner-validation cohort; 52 fold-1 outer-DEV patients were reserved for final evaluation. Fold 1 is separate from the earlier fold-0 discovery cohort, although it has been studied in prior research.", "",
             f"Gate enabled: **{gate['enabled']}**; inner grouped-CV gain {gate['cv_gain']:+.6f}; C={gate['c']}; threshold={gate['threshold']}.", "",
             "| Mask | Outer-DEV mean union Dice |", "|---|---:|",
             f"| A Dice baseline | {summary['A_union_dice']:.6f} |",
             f"| D segmentation | {summary['D_union_dice']:.6f} |",
             f"| D full head mask | {summary['head_union_dice']:.6f} |",
             f"| D gated mask | {summary['gated_union_dice']:.6f} |", ""]
    for key in ("D_minus_A", "gate_minus_D", "gate_minus_A"):
        x = summary[key]
        lines.append(f"- {key}: {x['mean']:+.6f} (paired patient-bootstrap 95% interval {x['ci95'][0]:+.6f} to {x['ci95'][1]:+.6f}; n={x['n']}).")
    lines.extend(["", f"The gate accepted {summary['accepted_columns']} of {summary['disagreement_columns']} disagreement columns; among accepted columns the head beat segmentation on {summary['accepted_head_wins']}, segmentation beat head on {summary['accepted_seg_wins']}, and {summary['accepted_ties']} tied.",
                  f"It accepted {summary['accepted_head_only']} of {summary['head_only_columns']} head-only foreground columns and {summary['accepted_seg_only']} of {summary['seg_only_columns']} segmentation-only foreground columns. Patient outcomes: {summary['gate_patient_wins']} improved, {summary['gate_patient_losses']} worsened relative to D.",
                  "", "The gate features use saved model probabilities and masks only; no reference label is used at inference. The rule and threshold were fixed before opening outer DEV. Case results, gate candidates, split, source/data hashes, and selected checkpoints are saved with this run."])
    (args.output / "result.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "summary": summary}, default=float), flush=True)


if __name__ == "__main__":
    main()
