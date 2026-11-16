"""Extend the archived native scratch schedule just far enough for one full-resolution pass."""
import json
import zipfile
from build_matched_controls import D, dump, sha


def build(ds):
    root = D / "matched_controls"
    target = root / (ds + "_native_pass.zip")
    with zipfile.ZipFile(root / (ds + "_matched.zip")) as source, zipfile.ZipFile(target.with_suffix(".partial"), "w", zipfile.ZIP_DEFLATED) as z:
        jobs = {}
        for key, row in json.loads(source.read("jobs.json")).items():
            if row["model"] not in ("nnunet", "swinunetr"):
                continue
            prefix = "protocols/" + key + "/"
            p = json.loads(source.read(prefix + "protocol.json"))
            sched = p["models"][p["trainer_alias"]]["objectives"]["dice"]
            old, steps, cost = sched["max_steps"], row["eligible"], sched["flops_per_update"]
            assert not row["feasible"] and old < steps and p["batch_size"] == 1
            original_cap = p["budget_flops"]
            p.update(original_budget_updates=old, original_budget_flops=original_cap, budget_flops=steps*cost,
                     budget_exception="One complete full-size pass. Reproduce every original learning rate through the archived stop, then hold that final rate for the extra updates.")
            sched.update(max_steps=steps, patient_pass_epochs=1., unused_flops=0)
            for n in ("consumed_flops","consumed_training_flops","consumed_total_flops"):
                sched[n] = steps*cost
            raw = dump(p)
            z.writestr(prefix+"protocol.json", raw)
            for n in ("archived_protocol.json", "splits_10patients.json", "nnunet_architecture.json"):
                z.writestr(prefix+n,source.read(prefix+n))
            jobs[key] = dict(row, feasible=True, original_stop=old, original_cap=original_cap, updates=steps, protocol_sha256=sha(raw))
        assert len(jobs) == 18
        z.writestr("jobs.json",dump(jobs))
        name = "source/scripts/cross_dataset_campaign/capped_core.py"
        core = source.read(name).decode()
        edits = [
            ("        effective_max = max_steps", "        effective_max = protocol['original_budget_updates']"),
            ('case=batch["case_name"], lr=lr)', 'case=batch["case_name"], lr=lr, shape=list(image.shape[-3:]))'),
            ('milestones = base.validation_steps(max_steps, protocol["validation_milestones"])',
             'milestones = set(base.validation_steps(protocol["original_budget_updates"], protocol["validation_milestones"])) | {max_steps, protocol["original_budget_updates"]}'),
            ('        score = result["summary"]["macro_hard_dice"]','        if step == protocol["original_budget_updates"]:\n            base.atomic_json(args.output / "validation_original_stop.json", result)\n        score = result["summary"]["macro_hard_dice"]')
        ]
        for old, new in edits:
            assert core.count(old) == 1, old
            core = core.replace(old,new)
        compile(core,name,"exec")
        z.writestr(name,core)
        for n in ("run_matched_controls.py","fast_msd_storage.py","link_patch_inputs.py"):
            z.write(D/n,n)
    target.with_suffix(".partial").replace(target)
    print(target.name,sha(target.read_bytes()),target.stat().st_size)


if __name__ == "__main__":
    for ds in ("ADNI","MSD"):
        build(ds)
