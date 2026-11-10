#!/usr/bin/env python3
"""Audit saved nnU-Net baselines without updating or overwriting their weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F


def load_trainer(source_root):
    path = source_root / "scripts/train_compute_matched_baselines.py"
    spec = importlib.util.spec_from_file_location("frozen_compute_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counts(label):
    return [int((label == c).sum()) for c in range(3)]


def compare_case(logits, label):
    probabilities = logits.float().softmax(1)
    predicted = logits.argmax(1)
    truth = label[:, 0]
    confusion = [[int(((truth == t) & (predicted == p)).sum()) for p in range(3)] for t in range(3)]
    hard = []
    soft = []
    gt_probabilities = []
    margins = []
    for c in (1, 2):
        mask = truth == c
        proposal = predicted == c
        hard.append(float(2 * (mask & proposal).sum() / (mask.sum() + proposal.sum())))
        soft.append(float(2 * (probabilities[:, c] * mask).sum() / (mask.sum() + probabilities[:, c].sum())))
        gt_probabilities.append([float(probabilities[:, p][mask].mean()) for p in range(3)])
        alternatives = logits[:, [p for p in range(3) if p != c]].amax(1)
        margins.append(float((logits[:, c] - alternatives)[mask].mean()))
    return {"ground_truth_voxels": counts(truth), "predicted_voxels": counts(predicted),
            "confusion_true_rows_predicted_columns": confusion, "hard_dice": hard, "soft_dice": soft,
            "mean_probabilities_on_anterior_posterior": gt_probabilities,
            "mean_logit_margin_on_anterior_posterior": margins}


def audit_data(trainer, dataframe, splits):
    result = []
    for fold, split in enumerate(splits):
        dataset = trainer.PreparedPatients(dataframe, split["train"])
        for item in dataset:
            original = dataframe[dataframe["subject_id"].astype(str) == item["case_name"]].iloc[0]
            raw = np.asarray(original["label_data"])
            processed = item["label"]
            ds = [F.interpolate(processed[None].float(), size=(edge,) * 3, mode="nearest").long()
                  for edge in (64, 32, 16, 8)]
            raw_counts = [int((raw == c).sum()) for c in range(3)]
            processed_counts = counts(processed)
            result.append({"fold": fold, "case_name": item["case_name"],
                           "raw_shape": list(raw.shape), "raw_voxels": raw_counts,
                           "prepared_voxels": processed_counts, "deep_supervision_voxels": [counts(x) for x in ds],
                           "foreground_preserved_by_preprocessing": raw_counts[1:] == processed_counts[1:]})
            assert all(x > 0 for x in processed_counts[1:]), "Training foreground missing"
            assert all(all(x > 0 for x in counts(target)[1:]) for target in ds[:3]), "Active DS target lost a foreground class"
        # Exercise the actual epoch order and dedicated augmentation RNG without any model training.
        for seed in (0, 1, 2):
            generator = torch.Generator().manual_seed(seed + 2000)
            order = torch.randperm(10, generator=torch.Generator().manual_seed(seed + 1000)).tolist()
            augmented_counts = []
            for index in order:
                item = dataset[index]
                image, label, _ = trainer.augment_mild(item["image"][None], item["label"][None], generator=generator)
                assert torch.isfinite(image).all()
                assert set(torch.unique(label).tolist()) == {0, 1, 2}
                augmented_counts.append(counts(label))
            result.append({"fold": fold, "seed": seed, "first_epoch_augmented_class_counts": augmented_counts})
    return result


def audit_loss(trainer, model, protocol, example):
    criterion, _ = trainer.build_loss_and_optimizer("nnunet", model, protocol)
    target = example["label"][None].long()
    targets = [F.interpolate(target.float(), size=(edge,) * 3, mode="nearest").long()
               for edge in (64, 32, 16, 8)]
    outputs = [(F.one_hot(t[:, 0], 3).movedim(-1, 1).float() * 12 - 6).requires_grad_() for t in targets]
    perfect = criterion(outputs, targets)
    perfect_value = float(perfect.detach())
    swapped = criterion([o[:, [0, 2, 1]] for o in outputs], targets)
    swapped_value = float(swapped.detach())
    neutral = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    loss = criterion(neutral, targets)
    loss.backward()
    gradients = []
    for index, (output, gt) in enumerate(zip(neutral, targets)):
        if output.grad is None:
            gradients.append({"head": index, "active": False})
            continue
        values = [float(output.grad[:, c][gt[:, 0] == c].mean()) for c in (1, 2)]
        gradients.append({"head": index, "active": True,
                          "mean_correct_class_logit_gradient_anterior_posterior": values})
        assert all(x < 0 for x in values), "Loss does not encourage correct foreground logits"
    assert perfect_value < swapped_value
    assert not gradients[-1]["active"], "Lowest-resolution head should be disabled"

    class Oracle(torch.nn.Module):
        def forward(self, image):
            return outputs[0].detach()

    oracle = trainer.evaluate(Oracle(), [example], torch.device("cpu"), False)
    assert oracle["summary"]["macro_hard_dice"] == 1.0
    return {"perfect_loss": perfect_value, "swapped_foreground_loss": swapped_value,
            "neutral_loss": float(loss.detach()), "deep_supervision_gradients": gradients,
            "oracle_validation_summary": oracle["summary"]}


def curves(run):
    with (run / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return [{key: float(row[key]) for key in ("step", "learning_rate", "mean_train_loss", "anterior_hard_dice",
                                              "anterior_soft_dice", "posterior_hard_dice", "macro_hard_dice")} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--seeds", type=int, choices=(0, 1, 2), nargs="+", default=[0, 2])
    parser.add_argument("--existing-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source_root = args.run_root / "source"
    trainer = load_trainer(source_root)
    protocol_dir = args.protocol_dir or source_root / "experiments/compute_matched_baselines_20261003"
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    architecture = json.loads((protocol_dir / "nnunet_architecture.json").read_text())
    splits = json.loads((protocol_dir / "splits_10patients.json").read_text())
    assert trainer.digest(args.pkl) == protocol["pkl_sha256"]
    dataframe = trainer._load_pkl_dataframe(args.pkl)
    result = {"audit_scope": "CPU FP32, no optimizer steps or saved-weight changes",
              "dataset_sha256": trainer.digest(args.pkl), "training_source_sha256": trainer.digest(Path(trainer.__file__)),
              "data": audit_data(trainer, dataframe, splits), "loss": None, "checkpoints": [], "initializations": []}
    # Check the actual seed-specific starts on one common patient.
    examples = trainer.PreparedPatients(dataframe, splits[0]["train"])
    for seed in (0, 1, 2):
        trainer.seed_everything(seed)
        model = trainer.build_model("nnunet", architecture)
        model.eval()
        with torch.no_grad():
            outputs = model(examples[0]["image"][None])
        result["initializations"].append({"seed": seed, "output_shapes": [list(o.shape) for o in outputs],
                                          **compare_case(outputs[0], examples[0]["label"][None])})
        if result["loss"] is None:
            result["loss"] = audit_loss(trainer, model, protocol, examples[0])
        del model, outputs
    for fold in range(3):
        validation = trainer.PreparedPatients(dataframe, splits[fold]["val"][:args.cases])
        training = trainer.PreparedPatients(dataframe, splits[fold]["train"][:1])
        for seed in args.seeds:
            run = args.run_root / "nnunet" / f"fold{fold}" / f"seed{seed}" / "run"
            for name in ("best", "latest"):
                checkpoint_path = run / f"checkpoint_{name}.pt"
                if args.existing_only and not checkpoint_path.exists():
                    continue
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                expected = json.loads((run / "completion.json").read_text())[f"checkpoint_{name}_sha256"]
                assert trainer.digest(checkpoint_path) == expected
                assert checkpoint["configuration"] == json.loads((run / "config.json").read_text())
                model = trainer.build_model("nnunet", architecture)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                cases = []
                with torch.no_grad():
                    for example in validation:
                        output = model(example["image"][None])[0]
                        cases.append({"case_name": example["case_name"], **compare_case(output, example["label"][None])})
                # Backpropagate the unchanged loss once; do not take an optimizer step.
                model.train()
                example = training[0]
                output = model(example["image"][None])
                criterion, _ = trainer.build_loss_and_optimizer("nnunet", model, protocol)
                targets = [F.interpolate(example["label"][None].float(), size=o.shape[2:], mode="nearest").long() for o in output]
                loss = criterion(output, targets)
                with torch.no_grad():
                    ce_voxels = F.cross_entropy(output[0].float(), targets[0][:, 0], reduction="none")
                    ce_per_class = [float(ce_voxels[targets[0][:, 0] == c].mean()) for c in range(3)]
                    primary_dice_loss = float(criterion.loss.dc(output[0], targets[0]))
                    training_prediction = compare_case(output[0], example["label"][None])
                output[0].retain_grad()
                loss.backward()
                logit_gradient = output[0].grad
                head = model.decoder.seg_layers[-1]
                entry = {"fold": fold, "seed": seed, "checkpoint": name, "step": checkpoint["step"],
                         "cases_fp32": cases, "gradient_probe_loss": float(loss.detach()),
                         "last_segmentation_head_bias": head.bias.detach().tolist(),
                         "last_segmentation_head_weight_grad_norm_per_class": head.weight.grad.flatten(1).norm(dim=1).tolist(),
                         "mean_correct_class_logit_gradient_on_training_anterior_posterior": [
                             float(logit_gradient[:, c][example["label"] == c].mean()) for c in (1, 2)],
                         "curve": curves(run)}
                entry["primary_ce_per_true_class_background_anterior_posterior"] = ce_per_class
                entry["primary_dice_loss"] = primary_dice_loss
                entry["training_prediction_fp32"] = training_prediction
                entry["all_saved_model_tensors_finite"] = all(
                    bool(torch.isfinite(value).all()) for value in checkpoint["model"].values())
                entry["all_saved_momentum_buffers_finite"] = None if "optimizer" not in checkpoint else all(
                    bool(torch.isfinite(state["momentum_buffer"]).all())
                    for state in checkpoint.get("optimizer", {}).get("state", {}).values()
                    if "momentum_buffer" in state)
                result["checkpoints"].append(entry)
                print(json.dumps({key:entry[key] for key in ("fold", "seed", "checkpoint", "step", "gradient_probe_loss",
                                                            "last_segmentation_head_weight_grad_norm_per_class")}), flush=True)
                del model, checkpoint, output, criterion, logit_gradient
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output":str(args.output),"checkpoints_audited":len(result['checkpoints'])}), flush=True)


if __name__ == "__main__":
    main()
