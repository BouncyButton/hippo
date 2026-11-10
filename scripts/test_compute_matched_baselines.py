"""CPU smoke checks for arithmetic caps, training, interruption and aggregation.

Run with the profiled MONAI 1.6.0 and dynamic-network-architectures 0.4.x installed:
python -m unittest scripts.test_compute_matched_baselines -v
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_script("train_compute_matched_baselines")
reporter = load_script("summarize_compute_matched_baselines")
PROTOCOL_DIR = ROOT / "experiments/compute_matched_baselines_20261003"


class ScheduleTests(unittest.TestCase):
    def test_momentum_revision_isolated_and_used_by_optimizer(self):
        revised = ROOT / "experiments/nnunet_momentum095_20261004"
        protocol = json.loads((revised / "protocol.json").read_text())
        original = json.loads((PROTOCOL_DIR / "protocol.json").read_text())
        for name in ("models", "budget_flops", "initial_lrs", "pkl_sha256", "splits_sha256",
                     "nnunet_architecture_sha256", "precision", "augmentation", "validation_milestones"):
            self.assertEqual(protocol[name], original[name])
        self.assertEqual(protocol["folds"], [0])
        self.assertEqual(protocol["seeds"], [0, 1, 2])
        self.assertEqual(protocol["nnunet_warmup_fraction"], 0.0)
        for step in range(327):
            self.assertEqual(trainer.learning_rate("nnunet", step, 327, .01),
                             trainer.learning_rate("nnunet", step, 327, .01, protocol["nnunet_warmup_fraction"]))
        model = torch.nn.Conv3d(1, 3, 1)
        _, control = trainer.build_loss_and_optimizer("nnunet", model, original)
        _, revised_optimizer = trainer.build_loss_and_optimizer("nnunet", model, protocol)
        self.assertEqual(control.param_groups[0]["momentum"], .99)
        self.assertEqual(revised_optimizer.param_groups[0]["momentum"], .95)
        self.assertTrue(revised_optimizer.param_groups[0]["nesterov"])

    def test_nnunet_warmup_uses_17_of_the_same_327_updates(self):
        revised = ROOT / "experiments/nnunet_warmup_20261004"
        protocol = json.loads((revised / "protocol.json").read_text())
        original = json.loads((PROTOCOL_DIR / "protocol.json").read_text())
        self.assertEqual(protocol["models"], original["models"])
        for name in ("budget_flops", "initial_lrs", "pkl_sha256", "splits_sha256", "nnunet_architecture_sha256"):
            self.assertEqual(protocol[name], original[name])
        rates = [trainer.learning_rate("nnunet", i, 327, .01, protocol["nnunet_warmup_fraction"])
                 for i in range(327)]
        self.assertAlmostEqual(rates[0], .01 / 17)
        self.assertEqual(rates[16], .01)
        self.assertEqual(rates[17], .01)
        self.assertTrue(all(a < b for a, b in zip(rates[:16], rates[1:17])))
        self.assertTrue(all(a > b for a, b in zip(rates[17:-1], rates[18:])))
        self.assertGreater(rates[-1], 0)
        self.assertLess(rates[-1], .0001)
        self.assertEqual(trainer.learning_rate("nnunet", 0, 1, .01, .05), .01)
        for fraction in (-.1, 1):
            with self.assertRaises(ValueError):
                trainer.learning_rate("nnunet", 0, 327, .01, fraction)

    def test_literal_caps_and_array_coverage(self):
        protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text())
        self.assertEqual(protocol["budget_flops"], 116259000000000)
        expected = {"unet": 12134, "swinunetr": 749, "nnunet": 327}
        identities = {trainer.array_identity(i) for i in range(27)}
        self.assertEqual(identities, {(m, f, s) for m in expected for f in range(3) for s in range(3)})
        for model, count in expected.items():
            cost = protocol["models"][model]["flops_per_update"]
            self.assertEqual(protocol["budget_flops"] // cost, count)
            self.assertLessEqual(count * cost, protocol["budget_flops"])
            self.assertGreater((count + 1) * cost, protocol["budget_flops"])
            milestones = trainer.validation_steps(count)
            self.assertEqual(len(milestones), 20)
            self.assertEqual(milestones[-1], count)
            rates = [trainer.learning_rate(model, i, count, protocol["initial_lrs"][model]) for i in range(count)]
            self.assertTrue(all(0 < r <= protocol["initial_lrs"][model] for r in rates))
            self.assertLess(rates[-1], rates[count // 2])

    def test_patient_subsets_preserve_original_folds(self):
        splits = json.loads((PROTOCOL_DIR / "splits_10patients.json").read_text())
        original = json.loads((ROOT / "datasets/Dataset101_MSD/splits_final.json").read_text())
        for fold, split in enumerate(splits):
            self.assertEqual(len(set(split["train"])), 10)
            self.assertEqual(len(split["val"]), 52)
            self.assertFalse(set(split["train"]) & set(split["val"]))
            self.assertTrue(set(split["train"]).issubset(original[fold]["train"]))
            self.assertEqual(split["val"], original[fold]["val"])


class LossSemanticTests(unittest.TestCase):
    def test_nnunet_loss_supervises_both_foreground_classes_at_all_active_scales(self):
        protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text())
        criterion, _ = trainer.build_loss_and_optimizer("nnunet", torch.nn.Conv3d(1, 3, 1), protocol)
        targets = []
        outputs = []
        for edge in (16, 8, 4, 2):
            target = torch.zeros((1, 1, edge, edge, edge), dtype=torch.long)
            target[:, :, :edge // 2, :, :edge // 2] = 1
            target[:, :, edge // 2:, :, :edge // 2] = 2
            targets.append(target)
            outputs.append(torch.zeros((1, 3, edge, edge, edge), requires_grad=True))
        criterion(outputs, targets).backward()
        for index in range(3):
            for c in (1, 2):
                gradient = outputs[index].grad[:, c][targets[index][:, 0] == c]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient < 0).all(), "Gradient descent must raise the correct foreground logit")
        self.assertIsNone(outputs[-1].grad)
        correct = [torch.nn.functional.one_hot(t[:, 0], 3).movedim(-1, 1).float() * 12 - 6 for t in targets]
        self.assertLess(float(criterion(correct, targets)), float(criterion([o[:, [0, 2, 1]] for o in correct], targets)))

    def test_validation_class_mapping_with_perfect_and_swapped_oracles(self):
        label = torch.zeros((1, 16, 16, 16), dtype=torch.long)
        label[:, :8, :, :8] = 1
        label[:, 8:, :, :8] = 2
        example = {"image": torch.zeros_like(label, dtype=torch.float), "label": label, "case_name": "oracle"}
        logits = torch.nn.functional.one_hot(label, 3).movedim(-1, 1).float() * 12 - 6

        class Oracle(torch.nn.Module):
            def __init__(self, swap=False):
                super().__init__()
                self.swap = swap

            def forward(self, image):
                return logits[:, [0, 2, 1]] if self.swap else logits

        correct = trainer.evaluate(Oracle(), [example], torch.device("cpu"), False)["summary"]
        swapped = trainer.evaluate(Oracle(True), [example], torch.device("cpu"), False)["summary"]
        self.assertEqual(correct["anterior_hard_dice"], 1.0)
        self.assertEqual(correct["posterior_hard_dice"], 1.0)
        self.assertEqual(swapped["anterior_hard_dice"], 0.0)
        self.assertEqual(swapped["posterior_hard_dice"], 0.0)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hippo-compute-test-")
        self.root = Path(self.temporary.name)
        self.protocol_dir = self.root / "protocol"
        shutil.copytree(PROTOCOL_DIR, self.protocol_dir)
        names = [f"synthetic_{i:02}" for i in range(12)]
        splits = [{"train": names[:10], "val": names[10:]} for _ in range(3)]
        path = self.protocol_dir / "splits_10patients.json"
        path.write_text(json.dumps(splits))
        protocol = json.loads((self.protocol_dir / "protocol.json").read_text())
        protocol["splits_sha256"] = trainer.digest(path)
        protocol["validation_patients_per_fold"] = 2
        (self.protocol_dir / "protocol.json").write_text(json.dumps(protocol))
        rng = np.random.default_rng(37)
        label = np.zeros((24, 24, 24), dtype=np.int64)
        label[6:12, 6:18, 6:18] = 1
        label[12:18, 6:18, 6:18] = 2
        records = [{"subject_id": name, "image_data": rng.normal(size=label.shape).astype(np.float32),
                    "label_data": label.copy()} for name in names]
        self.pkl = self.root / "synthetic.pkl.gz"
        pd.DataFrame(records).to_pickle(self.pkl, compression="gzip")

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, model, steps=1):
        output = self.root / model
        output.mkdir(exist_ok=True)
        return argparse.Namespace(model=model, fold=0, seed=0, pkl=self.pkl, output=output,
                                  protocol_dir=self.protocol_dir, device="cpu", threads=2,
                                  resume=True, smoke_steps=steps)

    def test_three_actual_models_forward_backward_selection_and_idempotence(self):
        for model in ("unet", "swinunetr", "nnunet"):
            with self.subTest(model=model):
                args = self.args(model)
                if model == "nnunet":
                    path = self.protocol_dir / "protocol.json"
                    protocol = json.loads(path.read_text())
                    protocol["nnunet_warmup_fraction"] = .05
                    protocol["nnunet_sgd_momentum"] = .95
                    path.write_text(json.dumps(protocol))
                trainer.run(args)
                done_path = args.output / "completion.json"
                original = done_path.read_bytes()
                done = json.loads(original)
                self.assertEqual(done["status"], "smoke_complete")
                self.assertEqual(done["updates"], 1)
                self.assertEqual(done["training_attempts_charged"], 1)
                self.assertEqual(done["discarded_attempts"], 0)
                self.assertLessEqual(done["consumed_training_flops"], done["budget_flops"])
                if model == "nnunet":
                    self.assertEqual(json.loads((args.output / "config.json").read_text())["nnunet_warmup_fraction"], .05)
                    self.assertEqual(json.loads((args.output / "config.json").read_text())["nnunet_sgd_momentum"], .95)
                    checkpoint = torch.load(args.output / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
                    self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["momentum"], .95)
                trainer.run(args)
                self.assertEqual(done_path.read_bytes(), original)

    def test_interrupted_work_stays_charged_on_resume(self):
        args = self.args("unet", steps=3)
        original_evaluate = trainer.evaluate
        calls = 0

        def fail_second_validation(*values):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption before checkpoint commit")
            return original_evaluate(*values)

        with patch.object(trainer, "evaluate", fail_second_validation):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                trainer.run(args)
        self.assertEqual(json.loads((args.output / "budget_ledger.json").read_text())["charged_attempts"], 2)
        trainer.run(args)
        done = json.loads((args.output / "completion.json").read_text())
        self.assertEqual(done["training_attempts_charged"], 3)
        self.assertEqual(done["updates"], 2)
        self.assertEqual(done["discarded_attempts"], 1)
        self.assertEqual(done["consumed_training_flops"], 3 * json.loads((args.output / "config.json").read_text())["flops_per_update"])
        (args.output / "validation_best.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact binding changed"):
            trainer.run(args)

    def test_committed_checkpoint_recovers_unpublished_best_and_metrics(self):
        args = self.args("unet", steps=1)
        original_save = trainer.save_checkpoint

        def interrupt_best_publication(path, value):
            if path.name == "checkpoint_best.pt":
                raise RuntimeError("interrupted after committing latest checkpoint")
            original_save(path, value)

        with patch.object(trainer, "save_checkpoint", interrupt_best_publication):
            with self.assertRaisesRegex(RuntimeError, "after committing"):
                trainer.run(args)
        self.assertFalse((args.output / "metrics.csv").exists())
        trainer.run(args)
        done = json.loads((args.output / "completion.json").read_text())
        self.assertEqual(done["updates"], 1)
        self.assertEqual(done["training_attempts_charged"], 1)
        self.assertEqual(done["discarded_attempts"], 0)
        self.assertEqual(done["selected_step"], 1)
        self.assertTrue((args.output / "metrics.csv").exists())


class AggregationTests(unittest.TestCase):
    def test_complete_campaign_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(prefix="hippo-compute-report-") as directory:
            root = Path(directory)
            source = root / "source"
            protocol_dir = source / "experiments/compute_matched_baselines_20261003"
            shutil.copytree(PROTOCOL_DIR, protocol_dir)
            (source / "scripts").mkdir()
            shutil.copy(ROOT / "scripts/train_compute_matched_baselines.py", source / "scripts")
            self.assertIsNone(reporter.summarize(root, allow_incomplete=True))
            protocol = json.loads((protocol_dir / "protocol.json").read_text())
            splits = json.loads((protocol_dir / "splits_10patients.json").read_text())
            for index in range(27):
                model, fold, seed = trainer.array_identity(index)
                run = root / model / f"fold{fold}" / f"seed{seed}" / "run"
                run.mkdir(parents=True)
                cost = protocol["models"][model]["flops_per_update"]
                steps = protocol["budget_flops"] // cost
                score = 0.5 + seed * 0.01
                config = {"model": model, "fold": fold, "seed": seed, "smoke_test": False,
                          "budget_flops": protocol["budget_flops"], "flops_per_update": cost, "max_steps": steps,
                          "protocol_sha256": trainer.digest(protocol_dir / "protocol.json"),
                          "dataset_sha256": protocol["pkl_sha256"],
                          "script_sha256": trainer.digest(source / "scripts/train_compute_matched_baselines.py"),
                          "train_cases": splits[fold]["train"], "validation_cases": splits[fold]["val"]}
                trainer.atomic_json(run / "config.json", config)
                trainer.atomic_json(run / "budget_ledger.json", {"config_sha256": trainer.digest(run / "config.json"),
                                    "charged_attempts": steps, "charged_training_flops": steps * cost})
                trainer.atomic_json(run / "validation_best.json", {"summary": {"macro_hard_dice": score},
                                    "cases": [{"case_name": name} for name in splits[fold]["val"]]})
                for name in ("checkpoint_best.pt", "checkpoint_latest.pt"):
                    (run / name).write_bytes(b"test artifact binding")
                with (run / "metrics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["charged_attempts", "macro_hard_dice"])
                    writer.writeheader()
                    writer.writerows({"charged_attempts": k, "macro_hard_dice": score} for k in trainer.validation_steps(steps))
                done = {"status": "complete", "model": model, "fold": fold, "seed": seed,
                        "training_attempts_charged": steps, "updates": steps, "discarded_attempts": 0,
                        "selected_step": steps, "consumed_training_flops": steps * cost,
                        "unused_training_flops": protocol["budget_flops"] - steps * cost,
                        "best_validation_hard_dice": score,
                        **{key: trainer.digest(run / name) for key, name in reporter.ARTIFACTS.items()}}
                trainer.atomic_json(run / "completion.json", done)
            result = reporter.summarize(root)
            self.assertEqual(len(result["runs"]), 27)
            self.assertAlmostEqual(result["models"]["unet"]["selected_validation_macro_hard_dice"]["mean"], 0.51)
            single = reporter.summarize(root, protocol_dir=protocol_dir, models=["nnunet"])
            self.assertEqual(len(single["runs"]), 9)
            self.assertEqual(set(single["models"]), {"nnunet"})
            shutil.rmtree(root / "nnunet/fold0/seed2")
            with self.assertRaisesRegex(ValueError, "Missing 1 of 9"):
                reporter.summarize(root, protocol_dir=protocol_dir, models=["nnunet"])
            (root / "unet/fold0/seed0/run/checkpoint_best.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed artifact"):
                reporter.summarize(root)


if __name__ == "__main__":
    unittest.main()
