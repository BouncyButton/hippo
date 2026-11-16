"""Verify every charged scratch update and each archived full-data evaluation."""
from pathlib import Path
from scripts.cross_dataset_campaign.common import digest, read
from scripts.cross_dataset_campaign import storage


def verify(output, task):
    done=read(output/"completion.json")
    cfg=read(output/"config.json")
    assert done["status"]=="complete" and done["discarded_attempts"]==0 and not cfg["smoke_test"]
    assert cfg["model_identity"]==task["model"] and cfg["fold"]==task["fold"] and cfg["seed"]==task["seed"]
    assert cfg["augmentation"]=="mild_v1" and cfg["objective"]=="dice"
    cap=cfg["budget_flops"]
    cost=cfg["schedule"]["flops_per_update"]
    assert cap==53_305_109_592_450 and done["updates"]==cap//cost
    assert done["consumed_training_flops"]==cost*done["updates"] and done["calibration_flops"]==0
    for filename, field in [("checkpoint_latest.pt","checkpoint_latest_sha256"),("checkpoint_best.pt","checkpoint_best_sha256"),
                            ("metrics.csv","metrics_sha256"),("budget_ledger.json","budget_ledger_sha256")]:
        assert digest(output/filename)==done[field]
    for name, cohort in [("training_final","train_cases"),("validation_latest","validation_cases"),("validation_best","validation_cases")]:
        rows=read(output/(name+".json"))["cases"]
        assert len(rows)==len(cfg[cohort]) and {r["case_name"] for r in rows}==set(cfg[cohort])
        assert all("foreground_fp" in r and "foreground_fn" in r and "confusion" in r for r in rows)
    covered=read(output/"coverage.json")
    assert covered["successful_updates"]==done["updates"] and set(covered["sampled_crops"])<=set(cfg["train_cases"])


def prepare(folder, key):
    row=read(folder/"cells.json")[key]
    task=dict(id=key,kind="conventional",dataset="MNI",model=row["model"],fold=row["partition"],seed=row["seed"])
    storage.verify_results=verify
    return storage.prepare(folder.parent/"runs"/key,task)


if __name__=="__main__":
    import sys
    prepare(Path(sys.argv[1]),sys.argv[2])
