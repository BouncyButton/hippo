#!/usr/bin/env python3
"""Diagnose whether a trained candidate ranker uses MRI and mask evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..evaluate_predictions import prediction_map
from ..train_teacher import build_loaders
from .candidate_ranker import CSTCandidateRanker, CandidateSetDataset, build_candidate_batch, candidate_cuts_around
from .cut_model import VIEW_SPECS, best_cut_from_labels


def selected_cut(model, batch, cuts, specification, device, *, zero_mri=False, zero_masks=False):
    elements, metadata = build_candidate_batch(batch, cuts, specification)
    if zero_mri:
        elements[:, :, :3] = 0
    if zero_masks:
        elements[:, :, 3:] = 0
    with torch.no_grad():
        scores = model(elements.to(device), metadata.to(device)).reshape(cuts.shape)
    return int(cuts[0, int(scores.argmax(dim=1)[0])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    loader_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=32, slab_depth=3,
        inplane_size=32, cache_dataset=False, max_train_cases=0,
        max_val_cases=0, batch_size=16, num_workers=0,
    )
    _, loader = build_loaders(loader_args)
    cases = [loader.dataset.base_dataset[index] for index in range(len(loader.dataset.base_dataset))]
    paths = prediction_map(args.inference_dir)
    device = torch.device(args.device)
    report = {"schema": "semantic_constraints.cst_teacher.multiview_candidate_dependence.v1",
              "fold": args.fold, "views": {}}
    for name, specification in VIEW_SPECS.items():
        checkpoint = torch.load(args.checkpoint_dir / f"{name}_candidate.pt", map_location="cpu", weights_only=False)
        dataset = CandidateSetDataset(cases, specification)
        model = CSTCandidateRanker().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        rows = []
        for index, item in enumerate(dataset.items):
            case_name = item["case_name"]
            swin = np.load(paths[case_name], allow_pickle=False).argmax(axis=0).astype(np.int8)
            try:
                swin_cut = best_cut_from_labels(swin)[0]
            except ValueError:
                active = np.flatnonzero((swin > 0).any(axis=(0, 2)))
                swin_cut = int(np.median(active)) if len(active) else (swin.shape[1] - 1) // 2
            cuts = candidate_cuts_around([swin_cut], swin.shape[1])
            common = {"case_name": [case_name], "image_slabs": item["image_slabs"].unsqueeze(0),
                      "metadata": item["metadata"].unsqueeze(0)}
            swin_batch = common | {"union": torch.from_numpy(swin > 0).unsqueeze(0)}
            truth_batch = common | {"union": item["union"].unsqueeze(0)}
            original = selected_cut(model, swin_batch, cuts, specification, device)
            truth_union = selected_cut(model, truth_batch, cuts, specification, device)
            no_mri = selected_cut(model, swin_batch, cuts, specification, device, zero_mri=True)
            no_mask = selected_cut(model, swin_batch, cuts, specification, device, zero_masks=True)
            rows.append({"case_name": case_name, "true_cut": item["true_cut_y"], "swin_cut": swin_cut,
                         "original": original, "truth_union": truth_union, "zero_mri": no_mri,
                         "zero_masks": no_mask})
        truth = np.asarray([row["true_cut"] for row in rows])
        report["views"][name] = {
            "mae": {key: float(np.mean([abs(row[key] - row["true_cut"]) for row in rows]))
                    for key in ("swin_cut", "original", "truth_union", "zero_mri", "zero_masks")},
            "agreement_with_original": {key: float(np.mean([row[key] == row["original"] for row in rows]))
                                        for key in ("truth_union", "zero_mri", "zero_masks")},
            "original_prediction_std": float(np.std([row["original"] for row in rows])),
            "ground_truth_cut_std": float(np.std(truth)),
            "patients": rows,
        }
        print(json.dumps({"view": name, **{key: value for key, value in report["views"][name].items() if key != "patients"}}), flush=True)
    report["warning"] = "GT-union and zeroed-input probes are diagnostic interventions, not deployable inference or causal attribution."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
