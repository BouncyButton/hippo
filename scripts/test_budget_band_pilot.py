"""Checks for target selection, cap accounting, and artifact verification."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("pilot", Path(__file__).with_name("run_budget_band_pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
REPO = Path(__file__).resolve().parents[1]


class PilotChecks(unittest.TestCase):
    def setUp(self):
        self.plan = {"base_budget_flops": 1000, "target_min": .70, "target_max": .77,
                     "max_pilot_runs": 8,
                     "escalation_multipliers": [[2, 1], [5, 2], [3, 1], [4, 1], [5, 1]]}

    def test_selection_uses_final_score_and_target_edges(self):
        for score in (.70, .735, .77):
            self.assertIsNone(pilot.next_budget(self.plan, [{"budget_flops": 1500, "final_dice": score}]))
        # A high best checkpoint does not select a run with an inadequate endpoint.
        self.assertEqual(pilot.next_budget(self.plan, [{"budget_flops": 1500, "final_dice": .65, "best_dice": .75}]), 2000)

    def test_overshoot_bisects_instead_of_adopting_high_score(self):
        history = [{"budget_flops": 1500, "final_dice": .79}]
        self.assertEqual(pilot.next_budget(self.plan, history), 1250)
        history.append({"budget_flops": 1250, "final_dice": .68})
        self.assertEqual(pilot.next_budget(self.plan, history), 1375)

    def test_search_is_bounded_and_rejects_invalid_curves(self):
        with self.assertRaises(RuntimeError):
            pilot.next_budget(self.plan, [{"budget_flops": 5000, "final_dice": .65}])
        with self.assertRaises(RuntimeError):
            pilot.next_budget(self.plan, [{"budget_flops": 1500, "final_dice": .79},
                                        {"budget_flops": 2000, "final_dice": .65}])
        with self.assertRaises(ValueError):
            pilot.next_budget(self.plan, [{"budget_flops": 1500, "final_dice": float("nan")}])

    def test_all_arm_schedules_charge_calibration_and_exhaust_cap(self):
        p = json.loads((REPO / "experiments/compute_budget_band_20261004/template_protocol.json").read_text())
        for numerator, denominator in ((3, 2), (2, 1), (5, 2)):
            budget = p["budget_flops"] * numerator // denominator
            for arm, s in pilot.schedules(p, budget).items():
                cost = s["prefix_updates"] * s["baseline_flops_per_update"] + s["constraint_updates"] * s["flops_per_update"] + s["calibration_flops"]
                self.assertEqual(cost, s["consumed_flops"])
                self.assertLessEqual(cost, budget)
                self.assertGreater(cost + s["flops_per_update"], budget)
                self.assertEqual(s["max_steps"], s["prefix_updates"] + s["constraint_updates"])

    def test_completed_score_is_bound_to_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, run, protocol_dir = root / "source", root / "run", root / "protocol"
            run.mkdir(); protocol_dir.mkdir()
            names = {"script_sha256": "scripts/train_compute_matched_constraints.py",
                     "objective_source_sha256": "scripts/compute_constraint_objectives.py",
                     "shared_source_sha256": "scripts/train_compute_matched_baselines.py",
                     "sdf_source_sha256": "thesis/new_constraints/signed_distance.py"}
            for name in names.values():
                path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
            schedule = {"max_steps": 10, "consumed_flops": 1000}
            protocol = {"budget_flops": 1005, "pkl_sha256": "data", "models": {"unet": {"objectives": {"dice": schedule}}}}
            pilot.write_json(protocol_dir / "protocol.json", protocol)
            config = {"smoke_test": False, "schedule": schedule, "budget_flops": 1005,
                      "protocol_sha256": pilot.digest(protocol_dir / "protocol.json"), "dataset_sha256": "data",
                      **{key: pilot.digest(source / name) for key, name in names.items()}}
            pilot.write_json(run / "config.json", config)
            pilot.write_json(run / "budget_ledger.json", {"charged_attempts": 10})
            (run / "metrics.csv").write_text("step,macro_hard_dice\n5,0.9\n10,0.72\n")
            for name in pilot.ARTIFACTS.values():
                if not (run / name).exists(): (run / name).write_text("fixture")
            done = {"status": "complete", "model": "unet", "objective": "dice", "fold": 0, "seed": 0,
                    "updates": 10, "discarded_attempts": 0, "consumed_total_flops": 1000,
                    **{key: pilot.digest(run / name) for key, name in pilot.ARTIFACTS.items()}}
            pilot.write_json(run / "completion.json", done)
            final, _ = pilot.verify_completed(run, protocol_dir, source, "dice")
            self.assertEqual(final, .72)
            (run / "metrics.csv").write_text("step,macro_hard_dice\n10,0.76\n")
            with self.assertRaises(ValueError): pilot.verify_completed(run, protocol_dir, source, "dice")

    def test_controller_freezes_budget_before_any_constraint_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"; source.mkdir()
            stage = root / "plan"; stage.mkdir()
            fixture = REPO / "experiments/compute_budget_band_20261004"
            for name in ("template_protocol.json", "splits_10patients.json", "split_provenance.json", "nnunet_architecture.json", "plan.json"):
                (stage / name).write_bytes((fixture / name).read_bytes())
            data = root / "data.pkl"; data.write_bytes(b"paired data fixture")
            template = json.loads((stage / "template_protocol.json").read_text())
            template["pkl_sha256"] = pilot.digest(data)
            pilot.write_json(stage / "template_protocol.json", template)
            plan = json.loads((stage / "plan.json").read_text())
            plan["template_protocol_sha256"] = pilot.digest(stage / "template_protocol.json")
            pilot.write_json(stage / "plan.json", plan)
            pilot.write_json(root / "runtime_origin_manifest.json", {})
            calls = []

            def fake_child(command, **kwargs):
                calls.append(command)
                if "--objective" not in command: return
                arm = command[command.index("--objective") + 1]
                output = Path(command[command.index("--output") + 1]); output.mkdir(parents=True)
                protocol_dir = Path(command[command.index("--protocol-dir") + 1])
                protocol = json.loads((protocol_dir / "protocol.json").read_text())
                schedule = protocol["models"]["unet"]["objectives"][arm]
                pilot.write_json(output / "completion.json", {"updates": schedule["max_steps"],
                                 "best_validation_hard_dice": .79,
                                 "consumed_total_flops": schedule["consumed_flops"]})
                if arm != "dice":
                    selection = json.loads((root / "budget_selection.json").read_text())
                    self.assertEqual(protocol["budget_flops"], selection["budget_flops"])
                    self.assertTrue((root / "source_manifest.json").exists())

            scores = iter((.79, .72))
            def fake_verify(output, *args):
                return next(scores), json.loads((output / "completion.json").read_text())
            with patch.object(pilot.subprocess, "run", fake_child), patch.object(pilot, "verify_completed", fake_verify), contextlib.redirect_stdout(io.StringIO()):
                pilot.run(SimpleNamespace(root=root, plan=stage / "plan.json", pkl=data, threads=1))
            selection = json.loads((root / "budget_selection.json").read_text())
            self.assertEqual(selection["final_dice"], .72)
            self.assertEqual(selection["budget_flops"], plan["base_budget_flops"] * 5 // 4)
            arms = [c[c.index("--objective") + 1] for c in calls if "--objective" in c]
            self.assertEqual(arms[:2], ["dice", "dice"])
            self.assertEqual(set(arms[2:]), set(template["models"]["unet"]["objectives"]) - {"dice"})
            self.assertEqual(len(arms[2:]), 7)
            self.assertTrue((root / "unet/dice/fold0/seed0/run").is_symlink())
            self.assertIn("report.DEFAULT_PROTOCOL=Path(sys.argv[2])", calls[-1][2])


if __name__ == "__main__":
    unittest.main()
