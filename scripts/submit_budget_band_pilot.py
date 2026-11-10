"""Submit the authorized single budget-selection and U-Net ablation allocation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.getuid() == 1395, "Use the authorized cluster account"
    stage = Path(__file__).resolve().parents[1]
    receipt = stage / "submission.json"
    with (stage / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            print(receipt.read_text()); return
        for name, expected in json.loads((stage / "payload_manifest.json").read_text()).items():
            assert digest(stage / name) == expected, name
        for name, expected in json.loads((stage / "runtime_origin_manifest.json").read_text()).items():
            assert digest(stage / "frozen_runtime" / name) == expected, name
        plan_path = stage / "experiments/compute_budget_band_20261004/plan.json"
        plan = json.loads(plan_path.read_text())
        check = json.loads((stage / "verification.json").read_text())
        assert check["cluster_tests_passed"] == 6 and check["plan_sha256"] == digest(plan_path)
        assert plan["target_min"] == .70 and plan["target_max"] == .77 and plan["max_pilot_runs"] == 8
        template = json.loads((plan_path.parent / "template_protocol.json").read_text())
        assert template["folds"] == template["seeds"] == [0] and list(template["models"]) == ["unet"]
        assert len(template["models"]["unet"]["objectives"]) == 8
        data = Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
        assert digest(data) == template["pkl_sha256"]
        old = Path('/mnt/beegfsstudents/home/3160552/compute_matched_constraints20_682721/unet/dice/fold0/seed0/run')
        completion = json.loads((old / "completion.json").read_text())
        assert completion["status"] == "complete" and completion["budget_flops"] == plan["base_budget_flops"]
        with (old / "metrics.csv").open() as stream:
            import csv
            history = list(csv.DictReader(stream))
        assert float(history[-1]["macro_hard_dice"]) == plan["base_final_dice"]
        quota = subprocess.run(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'],
                               capture_output=True, text=True, check=True).stdout
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2]) - float(limits[1]) < 5:
            raise RuntimeError("Need at least 5 GiB free in the actual user quota: " + quota)
        queue = subprocess.run(['squeue', '-u', '3160552', '-h', '-o', '%i|%T|%j'],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        states = [line.split('|')[1] for line in queue if line.strip()]
        if states.count('PENDING') >= 1 or len(states) > 1:
            raise RuntimeError("Submission would exceed one-running/one-queued limit: " + repr(queue))
        sbatch = stage / "scripts/run_budget_band_pilot.sbatch"
        response = subprocess.run(['sbatch', '--parsable', str(sbatch)], cwd=stage,
                                  capture_output=True, text=True, check=True)
        job = int(response.stdout.strip().split(';')[0])
        record = {"job_id": job, "submitted_at_unix": time.time(), "stage": str(stage),
                  "run_root": f"/mnt/beegfsstudents/home/3160552/compute_budget_band_{job}",
                  "plan_sha256": digest(plan_path), "sbatch_sha256": digest(sbatch),
                  "queue_before_submission": queue, "quota_before_submission": quota,
                  "models_trained": ["unet"], "folds": [0], "seeds": [0], "target_baseline_dice": [.70, .77],
                  "final_objectives": 8, "maximum_budget_selection_runs": 8,
                  "first_candidate_budget_flops": plan["base_budget_flops"] * 3 // 2}
        temporary = receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n'); temporary.replace(receipt)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
