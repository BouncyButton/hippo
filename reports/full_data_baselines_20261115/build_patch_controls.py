"""Build a small Colab overlay with profiled, separately named training-patch runs."""
import json
import sys
import zipfile
from pathlib import Path
from build_matched_controls import D, dump, sha


def build(dataset):
    folder = D / "matched_controls"
    output = folder / (dataset + "_training_patches.zip")
    temp = output.with_suffix(".partial")
    with zipfile.ZipFile(folder/(dataset+"_matched.zip")) as source, zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        jobs = json.loads(source.read("jobs.json"))
        patched = {}
        for model in ("nnunet", "swinunetr"):
            f = folder/"profiles"/f"{dataset}_{model}.json"
            profile = json.loads(f.read_text())
            z.write(f, "profiles/" + model + ".json")
            for key,r in jobs.items():
                if r["model"] != model:
                    continue
                assert not r["feasible"] and r["dataset"] == dataset
                prefix = f"protocols/{key}/"
                p = json.loads(source.read(prefix+"protocol.json"))
                alias = p["trainer_alias"]
                sched = p["models"][alias]["objectives"]["dice"]
                cost, steps = profile["flops_per_update"], profile["max_updates"]
                assert profile["native_flops"] == sched["flops_per_update"] and steps == p["budget_flops"] // cost
                assert steps >= r["eligible"]
                native_steps = sched["max_steps"]
                p["training_patch"] = profile["shape"]
                p["native_lr_horizon_updates"] = p["lr_horizon_updates"]
                p["lr_horizon_updates"] = round(p["lr_horizon_updates"] * profile["native_flops"] / cost)
                p["patch_sampling"] = "Training foreground bounding box with seeded position when it fits; centered on its bounds otherwise."
                p["patch_profile_sha256"] = sha(f.read_bytes())
                for name in ("baseline_flops_per_update","early_flops_per_update","flops_per_update"):
                    sched[name] = cost
                sched.update(max_steps=steps, patient_pass_epochs=steps/r["eligible"], unused_flops=p["budget_flops"]-steps*cost)
                for name in ("consumed_training_flops","consumed_flops","consumed_total_flops"):
                    sched[name] = steps*cost
                for name in ("archived_protocol.json","splits_10patients.json","nnunet_architecture.json"):
                    z.writestr(prefix+name,source.read(prefix+name))
                z.writestr(prefix+"protocol.json",dump(p))
                patched[key] = dict(r, feasible=True, training_patch=profile["shape"], native_updates=native_steps,
                                    updates=steps, protocol_sha256=sha(dump(p)))
        assert len(patched) == 18
        z.writestr("jobs.json",dump(patched))
        core_name = "source/scripts/cross_dataset_campaign/capped_core.py"
        core = source.read(core_name).decode()
        needle = '        image, label = batch["image"][None], batch["label"][None]'
        assert core.count(needle) == 1 and core.count('case=batch["case_name"], lr=lr)') == 1
        core = core.replace(needle,needle+'\n        from scripts.cross_dataset_campaign.training_patch import sample_patch\n        image, label, patch_stats = sample_patch(image, label, protocol["training_patch"], args.seed, charged)')
        core = core.replace('case=batch["case_name"], lr=lr)','case=batch["case_name"], lr=lr, **patch_stats)')
        compile(core,core_name,"exec")
        z.writestr(core_name,core)
        z.write(D/"training_patch.py","source/scripts/cross_dataset_campaign/training_patch.py")
        z.write(D/"run_matched_controls.py","run_matched_controls.py")
        if dataset == "MSD":
            z.write(D/"fast_msd_storage.py","fast_msd_storage.py")
        z.write(D/"link_patch_inputs.py","link_patch_inputs.py")
    temp.replace(output)
    print(output, output.stat().st_size)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        build(ds)
