"""Record passed checks, freeze payload and estimate bounded cluster storage."""
from pathlib import Path
import json
import shutil
import subprocess
import tarfile
from scripts.geofwi_cluster.campaign import read, write, sha, MODELS
from scripts.geofwi_constraints.prepare import ROOT, NAME, REPORT


def main():
    stage=ROOT/"experiments"/NAME
    assert not (stage/"submission.json").exists()
    shutil.copyfile(Path(__file__).with_name("campaign.py"),stage/"campaign.py")
    checks={}
    for model in MODELS:
        p=stage/"checks"/model/"objective_checks.json"
        doc=read(p)
        assert doc["status"]=="passed" and doc["optimizer_updates"]==0
        checks[str(p.relative_to(stage))]=sha(p)
    geometry=read(stage/"checks/geometry.json")
    assert geometry["status"]=="passed"
    checks["checks/geometry.json"]=sha(stage/"checks/geometry.json")
    subprocess.run([str(ROOT/".venv/bin/python"),str(Path(__file__).with_name("test_controller.py"))],check=True)
    write(stage/"cpu_contract_checks.json",dict(status="passed",optimizer_updates=0,checks=checks,
          constraint_loss_checks=52,controller_tests=8,xy_transforms=8,
          source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
          initialization="Twelve exact seeded CPU initialization audits inherited from the unchanged frozen builders; each GPU run asserts the corresponding hash"))
    # Save the preparation/test source alongside the executable frozen bundle.
    for p in Path(__file__).parent.glob("*.py"):
        target=stage/"preparation_source"/p.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,target)
    plan=read(stage/"plan.json")
    base=ROOT/"reports/geofwi3d_frozen_3fold_3seed_20261020/runs"
    archive_sizes=[sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) for d in base.iterdir() if d.is_dir()]
    estimate=dict(checked_quota_gib=93.13,observed_used_gib=27.88,observed_free_gib=65.25,
                  two_active_runs_logical_gib=14,mirrored_active_allowance_gib=28,
                  full_campaign_metadata_logical_allowance_gib=2,
                  mirrored_metadata_allowance_gib=4,
                  estimated_peak_additional_quota_gib=32,
                  estimated_remaining_quota_at_peak_gib=33.25,
                  observed_baseline_max_archive_bytes=max(archive_sizes),
                  projected_two_copies_468_results_bytes=max(archive_sizes)*468*2,
                  data_shared_with_verified_baselines=True,
                  delete_completed_checkpoints_only_after_verified_cluster_archive=True,
                  retain_failed_and_active_recovery=True,laptop_required=False)
    write(stage/"storage_estimate.json",estimate)
    inventory={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob("*"))
               if p.is_file() and "data" not in p.relative_to(stage).parts and "__pycache__" not in p.parts
               and p.name!="manifest.json"}
    write(stage/"manifest.json",inventory)
    subprocess.run([str(ROOT/".venv/bin/python"),str(stage/"campaign.py"),"verify","--stage",str(stage)],check=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    payload=REPORT/"frozen_payload.tar.gz"
    with tarfile.open(payload,"w:gz",compresslevel=1) as tar:
        for name in sorted(inventory):
            tar.add(stage/name,arcname=name,recursive=False)
        tar.add(stage/"manifest.json",arcname="manifest.json",recursive=False)
    write(REPORT/"preparation.json",dict(status="verified_ready_to_submit",manifest_sha256=sha(stage/"manifest.json"),
          payload_sha256=sha(payload),payload_bytes=payload.stat().st_size,new_runs=len(plan["tasks"]),reused_constraints=1,
          baseline_runs=0,baselines_reused=36,storage=estimate))
    for name in ("plan.json","manifest.json","cpu_contract_checks.json","storage_estimate.json"):
        shutil.copyfile(stage/name,REPORT/name)
    print(json.dumps(read(REPORT/"preparation.json")))


if __name__=="__main__":
    main()
