"""Release invariants for final recipes, independently checked against score records."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import portable as flow


class PortableRecipes(unittest.TestCase):
    def test_all_reported_cells(self):
        groups = [("HFH", model) for model in flow.MODELS[:-1]]
        groups += [(d, "medsam3") for d in ("HFH", "MSD", "ADNI", "MNI")]
        for d, m in groups:
            stem = f"medsam3_{d.lower()}" if m == "medsam3" else f"hfh_{m}"
            keys = flow.read(flow.PROTOCOLS / (stem + ".json"))["cells"]
            self.assertEqual(len(keys), 135 if d == "MSD" else 126)
            for key in keys:
                a, f, s = key.split("/")
                p, profile = flow.select(d, m, a, int(f), int(s))
                if m == "medsam3":
                    self.assertEqual((p.get("fold_index", p.get("fold")), p["seed"]), (int(f), int(s)))
                    if d != "ADNI":  # ADNI tokens are resolved after checking the data file.
                        flow.validate_split(p["train_cases"], p["evaluation_cases"], paired=d != "MSD")
                    if "frozen_sample_schedule" in p:
                        self.assertEqual(len(p["frozen_sample_schedule"]), p["updates"] * p["accumulation"])
                        self.assertTrue(all(c in p["train_cases"] for c, z in p["frozen_sample_schedule"]))
                else:
                    for field, name in (("splits_sha256", "splits_10patients.json"),
                                        ("nnunet_architecture_sha256", "nnunet_architecture.json")):
                        flow.verify(flow.PROTOCOLS / "hfh" / name, p[field])
                    self.assertIn(a, p["models"][p["trainer_alias"]]["objectives"])

    def test_hfh_budget_and_activation_match_score_records(self):
        for name in ("hfh_scratch.csv", "hfh_nnunet.csv"):
            with (flow.ROOT / "evaluation/data" / name).open() as stream:
                for row in csv.DictReader(stream):
                    p, profile = flow.select("HFH", row["model"], row["objective"], int(row["fold"]), int(row["seed"]))
                    schedule = p["models"][p["trainer_alias"]]["objectives"][row["objective"]]
                    self.assertEqual(schedule["max_steps"], int(row["updates"]))
                    if name == "hfh_nnunet.csv" and row["objective"] != "dice":
                        self.assertEqual(profile["inputs"]["anchor"]["after_update"], int(row["activation_after_update"]))
                        self.assertEqual(p.get("weight_multiplier", 1), float(row["weight_multiplier"]))

    def test_msd_keeps_the_two_controls_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            for a, worker in (("dice", "msd_primary"), ("dice_bands", "msd_original"),
                              ("original_dice", "msd_original")):
                p, selected = flow.select("MSD", "medsam3", a, 0, 83)
                self.assertEqual(selected["worker"], worker)
                args = SimpleNamespace(dataset="MSD", model="medsam3", output=Path(tmp) / a, arm=a,
                        fold=0, seed=83, data=Path(tmp), upstream=Path(tmp), base_weights=Path(tmp), lora_weights=Path(tmp))
                with patch.object(flow.subprocess, "run") as run:
                    flow.run(args, p, selected)
                commands = [c.args[0] for c in run.call_args_list]
                if worker == "msd_primary":
                    self.assertEqual([c[2] for c in commands], ["preflight", "train"])
                    self.assertTrue((args.output / "inputs/protocols/fold0_seed17.json").is_file())
                else:
                    self.assertIn("--calibration-input", commands[0])
                with self.assertRaises(FileExistsError):
                    flow.run(args, p, selected)

    def test_reject_overlap_and_changed_array(self):
        import numpy as np
        with self.assertRaises(ValueError):
            flow.validate_split(["s1_L"], ["s1_R"], paired=True)
        with tempfile.TemporaryDirectory() as tmp:
            import hashlib
            a = np.zeros((2, 3), dtype=np.uint8)
            expected = {"s": {"label": ["uint8", [2, 3], hashlib.sha256(a.tobytes()).hexdigest()]}}
            np.savez(Path(tmp) / "s.npz", label=a)
            flow.check_arrays(Path(tmp), expected, ["s"])
            a[0, 0] = 1
            np.savez(Path(tmp) / "s.npz", label=a)
            with self.assertRaises(ValueError):
                flow.check_arrays(Path(tmp), expected, ["s"])


if __name__ == "__main__":
    unittest.main()
