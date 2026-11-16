"""Freeze the Colab adapter protocols and stage only their required VM inputs."""
import json
from pathlib import Path
from hashlib import sha256
import sys
import zipfile

D=Path(__file__).resolve().parent
ROOT=D.parents[1]
HFH=ROOT/"reports/hfh_colab_exploration_20261111/bundle"
DATA=ROOT/"datasets/Dataset101_MSD"


def build(ds,target):
    m=json.loads((D/"manifest.json").read_text())
    splits=json.loads((D/"splits.json").read_text())
    target.mkdir(exist_ok=True,parents=True)
    location=f"/content/full_data_{ds}_medical"
    data=dict(MSD=location+"/data",ADNI="/content/full_data_ADNI/data.pkl",HFH="/content/full_data_HFH/data")[ds]
    raw_names=set()
    all_hashes={}
    with zipfile.ZipFile(D/f"{ds}_medical_protocols.zip","w",zipfile.ZIP_DEFLATED) as protocols:
        for key,r in m.items():
            if r["dataset"]!=ds or r["model"]!="medsam3":continue
            old=json.loads((ROOT/r["archived_protocol"]["file"]).read_text())
            s=splits[r["split"]]
            assert old["evaluation_cases"]==s["validation"] and set(old["train_cases"])<=set(s["train"])
            p=dict(old)
            p.update(name=key,fold=r["partition"],seed=r["seed"],dataset=ds,updates=r["target_updates"],
                     train_cases=s["train"],evaluation_cases=s["validation"],data=data,
                     train_crops=len(s["train"]),train_subjects=r["eligible_subjects"],
                     support_selection="All eligible original training crops after applying archived holdouts.",
                     sampling="Uniform training crop, then uniform adjacent-slice anchor; two pairs per update.",
                     learning_rate_horizon_updates=r["target_updates"],upstream=location+"/upstream",
                     lora_weights=location+"/weights/best_lora_weights.pt",requested_arms=["dice"],
                     lora_sha256="499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91",
                     primary_checkpoint="final_update",augmentation="none")
            p.pop("subset_seed",None)
            if ds!="MSD":
                p["support_subjects"]=sorted({c.rsplit("_",1)[0] for c in s["train"]})
            else:
                p["prompts"]=p["class_prompts"]
                names=sorted(f"{part}/{c}{ext}.nii.gz" for part,ext in [("imagesTr","_0000"),("labelsTr","")]
                             for c in s["train"]+s["validation"])
                raw_names.update(names)
                for n in names:
                    if n not in all_hashes:all_hashes[n]=sha256((DATA/n).read_bytes()).hexdigest()
                assert all(all_hashes[n]==h for n,h in old["data_hashes"].items())
                p["data_hashes"]={n:all_hashes[n] for n in names}
            p["archive_recipe_differences"]=dict(few_data_train=len(old["train_cases"]),full_data_train=len(s["train"]),
                paper_lr_horizon_updates=200 if ds=="HFH" else old["updates"],full_data_lr_horizon_updates=p["learning_rate_horizon_updates"],
                few_data_augmentation=old["augmentation"],full_data_augmentation="none")
            protocols.writestr(f"protocols/{key}.json",json.dumps(p,indent=2)+"\n")
        protocols.write(D/"medical_worker.py","medical_worker.py")
    with zipfile.ZipFile(target/f"{ds}_medical_inputs.zip","w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        with zipfile.ZipFile(D/f"{ds}_medical_protocols.zip") as p:
            for n in p.namelist():z.writestr(n,p.read(n))
        z.write(HFH/"requirements-colab.txt","requirements.txt")
        if ds!="HFH":
            for path in sorted((HFH/"upstream").rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:z.write(path,path.relative_to(HFH))
            z.write(HFH/"weights/best_lora_weights.pt","weights/best_lora_weights.pt")
        for n in sorted(raw_names):z.write(DATA/n,"data/"+n)
    f=target/f"{ds}_medical_inputs.zip"
    print(ds,f.stat().st_size,sha256(f.read_bytes()).hexdigest())


if __name__=="__main__":
    build(sys.argv[1],Path(sys.argv[2]))
