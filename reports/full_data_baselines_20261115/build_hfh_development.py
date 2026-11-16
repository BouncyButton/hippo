"""Prepare one fixed learning-rate experiment for Basic U-Net and UNETR."""
import json
import sys
import zipfile
from build_matched_controls import D, dump, sha

directory = D / "matched_controls"
ds = sys.argv[1] if len(sys.argv)>1 else "HFH"
assert ds in ("HFH","ADNI","MSD")
with zipfile.ZipFile(directory/(ds+"_matched.zip")) as base, zipfile.ZipFile(
        directory/(ds+"_lr_development.zip"), "w", zipfile.ZIP_DEFLATED) as out:
    jobs = {}
    for key, row in json.loads(base.read("jobs.json")).items():
        if row["model"] not in ("basic_unet16", "unetr_vit_b16"):
            continue
        folder = "protocols/" + key + "/"
        p = json.loads(base.read(folder+"protocol.json"))
        alias = p["trainer_alias"]
        before, after = p["initial_lrs"][alias], .001
        assert before in (.0001,.0003)
        p["initial_lrs"][alias] = after
        p["development_comparison"] = "One fixed peak rate across both networks, all three folds and seeds, with final-budget evaluation."
        out.writestr(folder+"protocol.json", dump(p))
        for name in ("nnunet_architecture.json", "splits_10patients.json", "archived_protocol.json"):
            out.writestr(folder+name, base.read(folder+name))
        jobs[key] = dict(row, development_peak_lr=after, original_peak_lr=before, protocol_sha256=sha(dump(p)),
                         prior_lr={k: v*after/before for k,v in row["prior_lr"].items()})
    assert len(jobs) == 18
    out.writestr("jobs.json",dump(jobs))
    for name in ("paper_data_index.json", "hfh_data_sha256.json", "archived_hfh_schedule.py"):
        if name in base.namelist():
            out.writestr(name,base.read(name))
    out.write(D/"run_matched_controls.py","run_matched_controls.py")
    if ds == "MSD":
        out.write(D/"fast_msd_storage.py","fast_msd_storage.py")
    out.write(D/"link_patch_inputs.py","link_patch_inputs.py")
