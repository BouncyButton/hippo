"""Recover the four eligible pools and compare each run's archived validation IDs."""
import hashlib
import json
from pathlib import Path

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
REPORTS = ROOT / "reports"
MODELS = ("nnunet", "basic_unet16", "unetr_vit_b16", "swinunetr", "medsam3")
SEEDS = {"MSD": (0, 2, 20), "ADNI": (0, 1, 4), "MNI": (0, 5, 19), "HFH": (0, 1, 2)}
UPDATES = {"MSD": 200, "ADNI": 200, "MNI": 400, "HFH": 125}
MSD = REPORTS / "msd_colab_seeds0220_recovery_20261018/package/seed20_bundle/protocols"
ADNI = REPORTS / "adni_colab_seeds014_drive_optimized_20261018/package/seed4_bundle/protocols"
MNI = REPORTS / "mni_winning_recipes_matrix2030_53t_0519_20261109/frozen/protocols"
HFH = REPORTS / "hfh_colab_exploration_20261111/bundle"


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return {"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def archived(ds, model, fold, seed):
    if model == "medsam3":
        paths = {
            "MSD": REPORTS / f"medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle/protocols/fold{fold}_seed{seed}.json",
            "ADNI": REPORTS / f"adni_medsam3_2sample_3fold_3seed_14arms_20261018/frozen/protocols/ADNI_f{fold}_s{seed}/protocol.json",
            "MNI": REPORTS / f"mni_medsam3_n5_adaptive60_2gpu_20261110/stage/protocols/MNI_MedSAM3_n5_f{fold}s{seed}_native400/protocol.json",
            "HFH": HFH / f"medsam_protocols/f{fold}_s{seed}.json",
        }
        path = paths[ds]
        recipe = read(path)
        assert recipe["seed"] == seed and recipe.get("fold", recipe.get("fold_index")) == fold
        return path, recipe["train_cases"], recipe["evaluation_cases"], path, recipe
    paths = {"MSD": MSD / f"{model}_s{seed}", "ADNI": ADNI / f"{model}_f{fold}_s{seed}",
             "MNI": MNI / f"{model}_dice", "HFH": HFH / "protocols" / model}
    path = paths[ds] / "splits_10patients.json"
    split = read(path)[fold]
    protocol = path.with_name("protocol.json")
    recipe = read(protocol)
    assert fold in recipe['folds'] and seed in recipe['seeds'], (ds,model,fold,seed,recipe['seeds'])
    return path, split["train"], split["val"], protocol, recipe


def main():
    study_path = ROOT / "docs/experiments/medsam3_fold3_extension_20260929/protocols/study.json"
    study = read(study_path)
    rows, splits = {}, {}
    for ds, seeds in SEEDS.items():
        source = next((ROOT / "datasets").glob(f"*_{ds}/splits_final.json"))
        original = read(source)
        for model in MODELS:
            adapter = model == "medsam3"
            partitions = (0, 1, 3) if adapter and ds in ("MSD", "ADNI") else (0, 1, 2)
            for fold in partitions:
                for seed in ((17,83,191) if ds == "MSD" else (0,1,2)) if adapter else seeds:
                    evidence, old_train, val, protocol, recipe = archived(ds, model, fold, seed)
                    train = list(original[fold]["train"])
                    full_val = list(original[fold]["val"])
                    if ds == "HFH":
                        train = [f"{sub}_h{i}" for sub in train for i in (1,2)]
                        full_val = [f"{sub}_h{i}" for sub in full_val for i in (1,2)]
                    excluded = []
                    if ds == "MSD" and adapter:
                        assert val == study["protocols"][f"fold{fold}_seed{seed}"]["evaluation_cases"]
                        blocked = set(study["evaluation_union"] + study["excluded_pilot_cases"])
                        excluded = sorted(set(train) & blocked)
                        train = [c for c in train if c not in blocked]
                        assert set(val) <= set(full_val)
                    else:
                        assert set(val) == set(full_val), (ds, model, fold, len(val), len(full_val))
                    assert len(val) == len(set(val)) and len(train) == len(set(train))
                    assert set(old_train) <= set(train) and not (set(train) & set(full_val))
                    if ds != "MSD":
                        subjects = lambda cases: {c.rsplit('_',1)[0] for c in cases}
                        assert len(subjects(train))*2 == len(train) and not subjects(train) & subjects(val)
                    key = f"{ds}_{model}_f{fold}_s{seed}"
                    split_key = f"{ds}_{model}_f{fold}"
                    split = dict(train=train, validation=val, original_validation=full_val,
                                 additional_excluded_training_ids=excluded, original_split=ref(source))
                    if split_key in splits:
                        assert splits[split_key] == split
                    splits[split_key] = split
                    alias = recipe.get('trainer_alias')
                    target = None if adapter else recipe['models'][alias]['objectives']['dice']
                    rows[key] = dict(dataset=ds, model=model, partition=fold, seed=seed, status="planned",
                        split=split_key, archived_validation_source=ref(evidence), archived_protocol=ref(protocol),
                        archived_train_count=len(old_train), eligible_crops=len(train), validation_crops=len(val),
                        eligible_subjects=None if ds=="MSD" else len(train)//2, unique_crops_sampled=None,
                        budget_flops=None if adapter else recipe['budget_flops'],
                        target_updates=UPDATES[ds] if adapter else target['max_steps'],
                        archived_flops_per_update=None if adapter else target['flops_per_update'],
                        lr=5e-5 if adapter else (0.01 if model=='nnunet' else 0.0001),
                        augmentation='none' if adapter else 'mild_v1', job_or_runtime=None,
                        checkpoint='final_budget', final_dice=None)
    assert len(rows) == 180
    for file, obj in [('splits.json',splits),('manifest.json',rows)]:
        path = DEST / file
        if path.exists():
            raise FileExistsError(f"Preserve existing campaign state: {path}")
        path.write_text(json.dumps(obj,indent=2)+'\n')
    print('Verified',len(rows),'cell keys and',len(splits),'model/partition split records.')


if __name__ == '__main__':
    main()
