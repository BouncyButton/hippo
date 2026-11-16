"""Join the paper's selected arm to its execution-batch control before rounding."""
import csv
import hashlib
import json
import math
import zipfile
from contextlib import nullcontext
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev
from tempfile import NamedTemporaryFile
from constraint_gap import write_constraint_gaps

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def true_voxel_counts(evaluation):
    return {r['case_name']: tuple(map(sum, r['confusion'])) for r in evaluation['cases']}


@lru_cache(maxsize=None)
def reference_counts(dataset, fold):
    # Every scratch model on one dataset/fold uses the same held-out images.
    reports = ROOT / 'reports'
    if dataset == 'HFH':
        path = reports / 'hfh_colab_exploration_20261111/recipe_analysis' / f'basic_unet16_f{fold}_s0.zip'
        with zipfile.ZipFile(path) as archive:
            return true_voxel_counts(json.loads(archive.read('results/validation_latest.json')))
    directories = {
        'ADNI': 'background_dice_archive_recovery_20261023/remote/ADNI_conventional',
        'MSD': 'msd_colab_seeds0220_recovery_20261018/cloud_backup/runs',
        'MNI': 'mni_winning_recipes_matrix2030_53t_0519_20261109/frozen/reused_runs',
    }
    path = reports / directories[dataset] / f'{dataset}_basic_unet16_dice_f{fold}s0/validation_latest.json'
    return true_voxel_counts(json.loads(path.read_text()))


def verified_comparisons(manifest):
    splits = json.loads((DEST / "splits.json").read_text())
    matched, patches, development, recoveries, native = {}, {}, {}, {}, {}
    for directory in (DEST / "archive/colab", DEST / "matched_controls/archive"):
        for suffix, records in (("_paper_matched_results.zip", matched),
                                ("_training_patch_results.zip", patches),
                                ("_native_pass_results.zip", native),
                                ("_lr_development_results.zip", development),
                                ("_recovery_replicate_results.zip", recoveries)):
            for path in directory.glob("*" + suffix):
                key = path.name.removesuffix(suffix)
                cell = manifest[key]
                split = splits[cell["split"]]
                with zipfile.ZipFile(path) as archive:
                    assert archive.testzip() is None
                    def read(name):
                        return json.loads(archive.read(name + ".json"))
                    cfg, done = read("config"), read("completion")
                    assert sorted(cfg["train_cases"]) == sorted(split["train"])
                    assert sorted(cfg["validation_cases"]) == sorted(split["validation"])
                    assert (cfg["model_identity"], cfg["fold"], cfg["seed"]) == (cell["model"], cell["partition"], cell["seed"])
                    validation = read("validation_latest")
                    assert sorted(r["case_name"] for r in validation["cases"]) == sorted(split["validation"])
                    assert true_voxel_counts(validation) == reference_counts(cell['dataset'], cell['partition']), (key, 'held-out class counts differ')
                    dice = validation["summary"]["macro_hard_dice"]
                    assert done["status"] == "complete" and 0 <= dice <= 1
                    charged = done["consumed_total_flops"]
                    if records is not native:
                        assert charged <= cell["budget_flops"]
                    if records is not patches and records is not native:
                        assert done["updates"] == cell["target_updates"]
                    if records is not recoveries:
                        receipt = read("matched_receipt")
                        trace = [json.loads(line) for line in archive.read("control_exposure.jsonl").splitlines()]
                        assert [item["step"] for item in trace] == list(range(1, done["updates"] + 1))
                        assert {item["case"] for item in trace} == set(split["train"])
                        arm = ("native_full_pass_budget_exception" if records is native else
                               "training_patch_full_coverage" if records is patches else
                               "development_peak_lr_full_coverage" if records is development else "paper_matched_full_coverage")
                        assert receipt["arm"] == arm and receipt["key"] == key
                        if records is patches:
                            profile = read("patch_profile")
                            shape = receipt["training_patch"]
                            assert receipt["validation_shape"] == [64]*3 and shape == profile["shape"]
                            assert shape != [64]*3 and receipt["eligible"] <= done["updates"] == profile["max_updates"]
                            assert done["updates"] * profile["flops_per_update"] == charged
                            assert profile["native_flops"] == cell["archived_flops_per_update"]
                            assert all(r["source_shape"] == [64]*3 and r["patch_shape"] == shape for r in trace)
                            assert all(sum(map(sum, r["confusion"])) == 64**3 for r in validation["cases"])
                        assert receipt["observed"] == receipt["eligible"] == len(split["train"])
                        location, separator, member = receipt["reference"].partition("!")
                        if separator:
                            with zipfile.ZipFile(ROOT / location) as reference:
                                original = reference.read(member)
                                initial = json.loads(reference.read("results/initialization.json"))["model_state_sha256"]
                        else:
                            original = (ROOT / location).read_bytes()
                            initial = json.loads(original).get("baseline_reference", {}).get("initialization_sha256")
                        if initial:
                            assert read("initialization")["model_state_sha256"] == initial, (key, "initial weights")
                        assert receipt["dice"] == dice and receipt["reference_sha256"] == hashlib.sha256(original).hexdigest()
                        assert cfg["protocol_sha256"] == receipt["protocol_sha256"] == hashlib.sha256(archive.read("protocols/protocol.json")).hexdigest()
                        if records is native:
                            assert cell["dataset"] in ("ADNI", "MSD") and cell["model"] in ("nnunet", "swinunetr")
                            cut = receipt["original_stop"]
                            assert cut == cell["target_updates"] == (90 if cell["model"] == "nnunet" else 206)
                            assert done["updates"] == len(split["train"]) == (216 if cell["dataset"] == "ADNI" else 208)
                            assert receipt["original_cap"] == cell["budget_flops"] < charged
                            assert charged == done["updates"] * cell["archived_flops_per_update"]
                            assert receipt["native_shape"] == [64]*3 and all(r["shape"] == [64]*3 for r in trace)
                            validation_at_stop = read("validation_original_stop")
                            assert true_voxel_counts(validation_at_stop) == true_voxel_counts(validation)
                            at_stop = validation_at_stop["summary"]["macro_hard_dice"]
                            observed_at_stop = len({r["case"] for r in trace[:cut]})
                            assert 0 <= at_stop <= 1 and observed_at_stop == cut
                            prior = json.loads(original)
                            protocol = read("protocols/protocol")
                            for name in ("augmentation", "batch_size", "trainer_alias", "initial_lrs", "lr_horizon_updates", "validation_target_flops"):
                                assert protocol[name] == prior[name], (key, name)
                            horizon = prior["lr_horizon_updates"]
                            lr = prior["initial_lrs"][prior["trainer_alias"]]
                            warm = (math.ceil(horizon * prior.get("nnunet_warmup_fraction", 0.0))
                                    if cell["model"] == "nnunet" else max(1, math.ceil(.05 * horizon)))
                            for i, item in enumerate(trace):
                                step = min(i, cut - 1)
                                if step < warm:
                                    expected = lr * (step + 1) / warm
                                elif cell["model"] == "nnunet":
                                    expected = lr * (1 - (step - warm) / (horizon - warm)) ** .9
                                else:
                                    angle = math.pi * ((step - warm) / max(1, horizon - warm - 1))
                                    expected = lr * (.01 + .99 * (1 + math.cos(angle)) / 2)
                                assert math.isclose(item["lr"], expected, rel_tol=1e-12), (key, i)
                        if records is development:
                            prior = json.loads(original)
                            assert (cfg["initial_lr"], receipt["development_peak_lr"], receipt["original_peak_lr"]) == (
                                .001,.001,prior["initial_lrs"][prior["trainer_alias"]])
                            assert cell["dataset"] in ("HFH","ADNI","MSD") and cell["model"] in ("basic_unet16","unetr_vit_b16")
                    else:
                        proof = read("replicate")
                        assert proof["key"] == key and proof["original_zip_sha256"] == cell["runtime_archive_sha256"]
                        assert proof["original_validation_dice"] == cell["final_dice"]
                value = dict(dice=dice, updates=done["updates"], charged_flops=charged, archive=str(path.relative_to(ROOT)),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if records is native:
                    value.update(original_stop=cut, observed_at_stop=observed_at_stop, original_stop_dice=at_stop)
                if records is patches:
                    value.update(shape=shape, observed=receipt["observed"])
                assert key not in records or value["sha256"] == records[key]["sha256"]
                records[key] = value
    return matched, patches, development, recoveries, native


def medical_comparison(key, cell, split, *, historical_hfh=False, historical_msd=False):
    if cell["model"] != "medsam3":
        return None
    colab = DEST / "archive/colab"
    paths = (colab / (key + "_coverage_control_results.zip"),
             DEST / 'archive/cluster_medical' / key,
             colab / (key + "_results.zip"), DEST / "archive" / key)
    if cell["dataset"] == "HFH" and not historical_hfh:
        paths = (DEST / "archive/cluster_hfh_gain" / key, colab / (key + "_gain_coverage_control_results.zip"))
    standard_msd = cell["dataset"] == "MSD" and not historical_msd
    if standard_msd:
        paths = (colab / (key + "_standard_fold_results.zip"),)
        official_path = ROOT / "datasets/Dataset101_MSD/splits_final.json"
        official = json.loads(official_path.read_bytes())[cell["partition"]]
        split = dict(train=official["train"], validation=official["val"])
        assert len(set(split["train"])) == 208 and len(set(split["validation"])) == 52
    path = next((p for p in paths if p.is_file() or (p / 'archive_manifest.json').is_file()), None)
    if path is None:
        return None
    reference_path = ROOT / cell["archived_protocol"]["file"]
    raw_reference = reference_path.read_bytes()
    assert hashlib.sha256(raw_reference).hexdigest() == cell["archived_protocol"]["sha256"]
    reference = json.loads(raw_reference)
    if cell["dataset"] == "MSD" and "lora_sha256" not in reference:
        # The original MSD engine loads this weight from its checksum-indexed bundle.
        bundle_index = json.loads((reference_path.parents[1] / "bundle_manifest.json").read_text())
        reference["lora_sha256"] = bundle_index["weights/best_lora_weights.pt"]
    with (zipfile.ZipFile(path) if path.is_file() else nullcontext()) as archive:
        if archive:
            assert archive.testzip() is None
        def content(name):
            return archive.read(name) if archive else (path / name).read_bytes()
        protocol = json.loads(content("protocol.json"))
        done = json.loads(content("completion.json"))
        schedule = json.loads(content("slab_schedule.json"))
        history = json.loads(content("training_history.json"))
        assert done["status"] == "complete" and done["primary_checkpoint"] == "final_update"
        assert done["evaluation_sha256"] == hashlib.sha256(content("evaluation.json")).hexdigest()
        assert done["protocol_sha256"] == hashlib.sha256(content("protocol.json")).hexdigest()
        assert (done["fold"], done["seed"], done["arm"]) == (cell["partition"], cell["seed"], "dice")
        assert len(history) == done["updates"] == cell["target_updates"] == protocol["updates"]
        assert protocol["train_cases"] == split["train"] and protocol["evaluation_cases"] == split["validation"]
        if standard_msd:
            assert protocol["paired_evaluation_cases"] == reference["evaluation_cases"]
            assert set(reference["evaluation_cases"]) <= set(split["validation"])
            assert protocol["official_fold_sha256"] == hashlib.sha256(official_path.read_bytes()).hexdigest()
            assert content("splits_final.json") == official_path.read_bytes()
        else:
            assert reference["evaluation_cases"] == protocol["evaluation_cases"]
        assert set(reference["train_cases"]).issubset(protocol["train_cases"])
        assert protocol["classes"] == reference["classes"] and protocol["inference"] == reference["inference"]
        assert protocol["prompts"] == reference.get("prompts", reference.get("class_prompts"))
        if "data_hashes" in reference:
            assert all(protocol["data_hashes"][n] == h for n, h in reference["data_hashes"].items())
        else:
            assert protocol["dataset_sha256"] == reference["dataset_sha256"]
        assert len(schedule) == len(history) * protocol["accumulation"]
        if {c for c, z in schedule} != set(split["train"]):
            return None
        summary = json.loads(content("training_summary.json"))
        inventory = json.loads(content("training_inventory.json"))
        assert summary["updates"] == len(history)
        assert summary["schedule_sha256"] == hashlib.sha256(content("slab_schedule.json")).hexdigest()
        assert summary["frozen_before"] == summary["frozen_after"] and summary["adapter_before"] != summary["adapter_after"]
        assert sorted(item.get("case") or item["case_name"] for item in inventory) == sorted(split["train"])
        assert set(split["train"]).isdisjoint(split["validation"])
        assert [e["update"] for e in history] == list(range(1, len(history) + 1))
        assert all(math.isfinite(e["gradient_norm"]) and e["gradient_norm"] >= 0 for e in history)
        assert any(e["gradient_norm"] > 0 for e in history)
        assert all(math.isclose(e["native"], e["total"], abs_tol=1e-9) for e in history)
        assert json.loads(content("runtime_validation.json"))["status"] == "passed"
        gain_control = cell["dataset"] == "HFH" and not historical_hfh
        for field in ("learning_rate", "final_learning_rate", "warmup_updates", "accumulation", "lora_sha256", "base_sha256"):
            assert protocol[field] == reference[field], (key, field)
        if gain_control:
            low_archive = ROOT / "reports/hfh_colab_exploration_20261111/medsam_adaptive_matrix_seed0/archives/matrix125__dice_sdf_depth__f0_s0.zip"
            with zipfile.ZipFile(low_archive) as low:
                augmentation = json.loads(low.read("results/protocol.json"))["augmentation"]
            assert protocol["augmentation"] == summary["augmentation"] == augmentation
        else:
            assert protocol["augmentation"] == reference["augmentation"]
        horizon = 200 if cell["dataset"] == "HFH" else reference.get("learning_rate_horizon_updates", reference["updates"])
        if protocol.get("learning_rate_horizon_updates", protocol["updates"]) != horizon:
            return None
        peak, floor, warmup = (protocol[f] for f in ("learning_rate", "final_learning_rate", "warmup_updates"))
        for i, event in enumerate(history):
            expected = (peak * (i + 1) / warmup if i < warmup else
                        floor + .5 * (peak - floor) * (1 + math.cos(math.pi * (i - warmup) / max(1, horizon - warmup - 1))))
            assert math.isclose(event["lr"], expected, rel_tol=1e-10, abs_tol=1e-14), (key, i)
        records = json.loads(content("evaluation.json"))["records"]
        assert len(records) == len(split["train"]) + len(split["validation"])
        for label, cases in (("training_support", split["train"]), ("validation", split["validation"])):
            group = [r for r in records if r["split"] == label]
            assert sorted(r["case_name"] for r in group) == sorted(cases)
            assert all(0 <= r["macro_hard_dice"] <= 1 for r in group)
            score = done["training_dice" if label == "training_support" else "validation_dice"]
            assert math.isclose(mean(r["macro_hard_dice"] for r in group), score, abs_tol=1e-12)
        assert all("foreground_fn" in r and "foreground_fp" in r for r in records)
        digest = hashlib.sha256(path.read_bytes() if archive else content("archive_manifest.json")).hexdigest()
        paired_score = done["validation_dice"]
        if standard_msd:
            paired_names = set(reference["evaluation_cases"])
            paired_score = mean(r["macro_hard_dice"] for r in group if r["case_name"] in paired_names)
            receipt = json.loads(content("standard_fold_receipt.json"))
            assert receipt["train"] == 208 and receipt["validation"] == 52 and receipt["paired_cases"] == len(paired_names)
            assert math.isclose(paired_score, receipt["dice_paired"], abs_tol=1e-12)
        return dict(dice=paired_score, all_validation_dice=done["validation_dice"], archive=str(path.relative_to(ROOT)), sha256=digest)


def write_progress(scores):
    groups = defaultdict(list)
    for row in scores.values():
        groups[row['dataset'], row['model']].append(row)
    header = "| Dataset | Model | Verified runs | Low-data Dice | Full-data Dice | Difference |"
    separator = "|---|---|---:|---:|---:|---:|"
    primary = sum(r['paper_matched_full_pct'] is not None or r['native_pass_full_pct'] is not None for r in scores.values())
    secondary = sum(r['development_peak_lr_0_001_pct'] is not None for r in scores.values())
    lines = ["# Full-data baseline results", "",
             f"**{primary} of {len(scores)} primary runs verified.** The separate learning-rate experiment has {secondary} of 54 runs verified.", "",
             "[Full-data baselines and the gap closed by low-data constraints](constraint_gap.md). The primary table uses the baseline training settings and labels the one-pass compute exceptions. The separate higher-learning-rate experiment covers Basic U-Net and UNETR on MSD, ADNI and HFH.", "",
             "Dice is reported as a percentage at the final budget. Each row uses the same partitions, seeds and held-out cases on both sides. Means based on fewer than nine pairs are interim. The difference is in percentage points.", "",
             "Full data includes all eligible training crops for the paired protocol. HFH and MNI use 40 crops per partition, ADNI uses 216, and all MSD models use 208. MSD MedSAM3 evaluates all 52 official validation crops. Paired low-data comparisons use the original 46, 52 or 44 held-out cases within that fold. [Original study splits](splits.json) · [Official 208/52 MSD folds](../../datasets/Dataset101_MSD/splits_final.json)", ""]
    for title, column, description in (
        ("Full-resolution training", "paper_matched_full_pct", "The training trace covers every eligible crop and each archived paper recipe is checked. Scratch runs also match the low-data validation reference on the number of true voxels in every class for every case."),
        ("Full-size one-pass compute exceptions", "native_pass_full_pct", "Each training crop is used exactly once at 64 × 64 × 64. Training follows the original learning rates through its stopping update and holds the last rate through the extra steps. Validation is repeated at both stops. The per-run CSV records total FLOPs and FLOPs above the original cap."),
        ("Smaller training patches", "patch_training_full_pct", "These models train at native voxel spacing on a patch from every eligible crop. Validation uses each complete 64 × 64 × 64 model input. The low-data controls used full-size training inputs."),
        ("Learning-rate experiment", "development_peak_lr_0_001_pct", "Basic U-Net and UNETR use a fixed peak rate of 0.001. The low-data references retain their original rates, so these differences combine the changes in training pool and learning rate. Each model uses three folds and three seeds.")):
        lines.extend(["## "+title, "", description, "", header, separator])
        for (ds, model), group in sorted(groups.items()):
            if column in ("patch_training_full_pct", "native_pass_full_pct") and not (ds in ("ADNI","MSD") and model in ("nnunet","swinunetr")):
                continue
            if column == "development_peak_lr_0_001_pct" and not (ds in ("ADNI","MSD","HFH") and model in ("basic_unet16","unetr_vit_b16")):
                continue
            if column == "paper_matched_full_pct" and group[0]["maximum_native_training_passes"] is not None and group[0]["maximum_native_training_passes"] < 1:
                continue
            completed = [r for r in group if r[column] is not None]
            if completed:
                few, full = (mean(r[c] for r in completed) for c in ("few_data_baseline_pct",column))
                values = f"{few:.2f}% | {full:.2f}% | {full-few:+.2f}"
            else:
                values = "— | — | —"
            lines.append(f"| {ds} | {model} | {len(completed)}/{len(group)} | {values} |")
        lines.append("")
    lines.extend(["## MedSAM3 fine-tuning audit", "",
                  "Archived parameter hashes confirm that fine-tuning changed the LoRA weights and preserved the frozen backbone. The validator checks the schedule against the training inventory, all optimizer steps against the archived learning-rate schedule, positive gradient signal, native baseline loss and disjoint held-out cases. Each optimizer update draws two adjacent-slice pairs. These runs establish coverage of training crops at a fixed update count. Constraint calibration can add compute beyond the 200-update MSD optimizer schedule.", "",
                  "| Dataset | Low-data crops | Training crops used per fold | Updates per run | Verified runs |",
                  "|---|---:|---|---:|---:|"])
    for ds, low, pool, steps in (("HFH", 10, "40 of 40", 125), ("MNI", 10, "40 of 40", 400),
                                 ("ADNI", 2, "216 of 216", 200), ("MSD", 5, "208 of 208 on folds 0, 1 and 3", 200)):
        n = sum(r['paper_matched_full_pct'] is not None for r in groups[ds, 'medsam3'])
        lines.append(f"| {ds} | {low} | {pool} | {steps} | {n}/9 |")
    lines.extend(["", "### Paired MedSAM3 validation Dice", "",
                  "Only completed full-fold fine-tunes enter these means. Each row uses the same fold, seed and validation cases in the three paired columns. The final column additionally includes every official validation crop.", "",
                  "| Dataset | Pairs | Low data | Best constrained low data | Full data, paired | Full minus constrained (pp) | Full data, all validation |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for ds in ("MSD","ADNI","HFH","MNI"):
        paired = [r for r in groups[ds,"medsam3"] if r["paper_matched_full_pct"] is not None]
        if not paired:
            lines.append(f"| {ds} | 0/9 | — | — | — | — | — |")
            continue
        low, constrained, full = (mean(r[column] for r in paired) for column in
                                  ("few_data_baseline_pct","few_data_selected_pct","paper_matched_full_pct"))
        all_validation = mean(r["msd_standard_all_52_dice_pct"] for r in paired) if ds == "MSD" else full
        lines.append(f"| {ds} | {len(paired)}/9 | {low:.2f}% | {constrained:.2f}% | {full:.2f}% | {full-constrained:+.2f} | {all_validation:.2f}% |")
    lines.extend(["", "HFH's first nine larger-data fine-tunes omitted the gain augmentation used in the measured low-data arms. Their mean Dice was 69.85%. They are archived as a separate no-augmentation experiment. Only larger-data runs with matching gain augmentation count in the primary HFH MedSAM3 row.", "",
                  "Historical MSD MedSAM3 results on 60, 61 and 56 training crops are retained in the per-run CSV. The primary comparison requires all 208. The CSV also records each standard-fold run’s Dice over all 52 held-out crops. The historical 200-update adapter schedule is fixed before these runs. Training can use cases from other folds, as in standard cross-validation. Each run excludes its own 52 validation crops. MNI MedSAM3 averaged 27.28 distinct planes per 48-slice training crop, versus 46.40 on the smaller pool under the same 400-update budget. [Training exposure analysis](hfh_pilot_diagnosis.md) · [Frozen splits](splits.json)", "",
                  "[Low-data constraints and full-data gap closed](constraint_gap.md) · [Per-run scores, coverage and archive links](paired_comparisons.csv)"])
    with NamedTemporaryFile(dir=DEST, mode="w", delete=False) as stream:
        stream.write("\n".join(lines)+"\n")
        tmp = Path(stream.name)
    tmp.replace(DEST/"full_data_results.md")


def main():
    manifest = json.loads((DEST / "manifest.json").read_text())
    matched, patches, development, recoveries, native = verified_comparisons(manifest)
    splits = json.loads((DEST / "splits.json").read_text())
    scores = {}
    def record(dataset, model, fold, seed, arm, baseline, selected, source):
        key = f"{dataset}_{model}_f{fold}_s{seed}"
        assert key in manifest and key not in scores and 0 <= baseline <= 100 and 0 <= selected <= 100
        scores[key] = dict(dataset=dataset, model=model, partition=int(fold), seed=int(seed),
                           selected_arm=arm, few_data_baseline_pct=baseline, few_data_selected_pct=selected,
                           few_data_selected_minus_baseline_pp=selected-baseline, source=str(source.relative_to(ROOT)))

    paper = ROOT / "papers/review_20261114/source/data"
    path = paper / "results.json"
    archive = json.loads(path.read_text())["rows"]
    by_key = {(r['study'], r['model'], r['arm'], r['fold'], r['seed']):r for r in archive}
    for selected in json.loads((paper / "selected.json").read_text()):
        study, model, arm = (selected[n] for n in ('study','model','arm'))
        for r in archive:
            if (r['study'], r['model'], r['arm']) != (study, model, arm):
                continue
            control = by_key[study,model,'baseline',r['fold'],r['seed']]
            assert abs(r['dice'] - control['dice'] - r['gain_pp']) < 1e-8
            record(r['dataset'], model.lower() if model == 'MedSAM3' else model,r['fold'],r['seed'],
                   arm,control['dice'],r['dice'],path)

    hfh = ROOT / 'reports/hfh_colab_exploration_20261111'
    matrix = hfh/'unetr_completion_20261113/full_constraint_matrix.csv'
    nnunet = ROOT/'reports/hfh_nnunet_matrix_reconciliation_20261114/final_a100_matrix_126.csv'
    selected = dict(basic_unet16='dice_bands_edge_depth',unetr_vit_b16='dice_bands_depth',
                    swinunetr='dice_bands_sdf_depth',nnunet='dice_edge_depth')
    for source in (matrix, nnunet):
        for r in csv_rows(source):
            model = r['model']
            if (model=='nnunet') != (source==nnunet) or r['objective']!=selected[model]:
                continue
            assert r.get('status','verified')=='verified'
            record('HFH',model,r['fold'],r['seed'],r['objective'],100*float(r['baseline_dice']),100*float(r['final_dice']),source)
    mni = ROOT/'reports/mni_medsam3_n5_third_seed_search_20261110/archive/final_cells.csv'
    for r in csv_rows(mni):
        if r['arm']=='dice_bands_edge_sdf_depth':
            assert int(r['updates']) == 400
            record('MNI','medsam3',r['fold'],r['seed'],r['arm'],100*float(r['baseline_dice']),100*float(r['validation_dice']),mni)
    source = hfh/'medsam_adaptive_matrix_replication/validation_dice_all_126.csv'
    rows = csv_rows(source)
    controls = {(r['fold'],r['seed']):float(r['validation_dice_pct']) for r in rows if r['objective']=='dice'}
    for r in rows:
        if r['objective']=='dice_sdf_depth':
            b = controls[r['fold'],r['seed']]
            v = float(r['validation_dice_pct'])
            assert abs(v-b-float(r['gain_over_matched_baseline_pp']))<1e-8
            record('HFH','medsam3',r['fold'],r['seed'],r['objective'],b,v,source)
    assert scores.keys()==manifest.keys(), set(manifest)-set(scores)
    repair_path = DEST / 'hfh_nnunet_repair.json'
    repairs = {}
    if repair_path.is_file():
        for run in json.loads(repair_path.read_text())['runs']:
            key = run['key']
            assert key in scores and key not in repairs
            assert manifest[key]['dataset'] == 'HFH' and manifest[key]['model'] == 'nnunet'
            assert run['updates'] == manifest[key]['target_updates'] and 0 <= run['dice'] <= 1
            repairs[key] = 100 * run['dice']
    for key,r in scores.items():
        cell = manifest[key]
        r['full_data_status'] = cell['status']
        observed = cell.get('unique_crops_sampled')
        eligible = cell['eligible_crops']
        r['eligible_training_crops'] = eligible
        r['optimizer_update_budget'] = cell['target_updates']
        r['maximum_native_training_passes'] = (cell['target_updates'] / eligible if cell['model'] != 'medsam3' else None)
        r['observed_training_crops'] = observed
        r['observed_full_coverage'] = None if observed is None else observed == eligible
        r['comparison_protocol'] = ('standard scratch defaults; training recipe differs from archived control'
                                    if cell['model'] != 'medsam3' else 'medical adapter; check schedule and observed exposure')
        corrected = matched.get(key) or medical_comparison(key, cell, splits[cell["split"]])
        legacy_hfh = (medical_comparison(key, cell, splits[cell["split"]], historical_hfh=True)
                      if cell["dataset"] == "HFH" and cell["model"] == "medsam3" else None)
        r["hfh_medsam3_noaug_full_pct"] = 100 * legacy_hfh["dice"] if legacy_hfh else None
        r["hfh_medsam3_noaug_archive"] = legacy_hfh["archive"] if legacy_hfh else None
        standard_msd = cell["dataset"] == "MSD" and cell["model"] == "medsam3"
        if standard_msd:
            eligible = 208
            r["eligible_training_crops"] = eligible
            r["observed_training_crops"] = eligible if corrected else None
            r["observed_full_coverage"] = True if corrected else None
            r["full_data_status"] = "verified_complete" if corrected else "awaiting_standard_fold_archive"
            r["comparison_protocol"] = "Official MSD fold, 208 training crops and 52 held out, native adapter loss and 200 updates"
        r["msd_standard_all_52_dice_pct"] = 100 * corrected["all_validation_dice"] if standard_msd and corrected else None
        restricted = medical_comparison(key, cell, splits[cell["split"]], historical_msd=True) if standard_msd else None
        r["msd_prior_restricted_pool_pct"] = 100 * restricted["dice"] if restricted else None
        r["msd_prior_restricted_archive"] = restricted["archive"] if restricted else None
        r['paper_matched_full_pct'] = 100 * corrected['dice'] if corrected else None
        lr_dev = development.get(key, {})
        r['development_peak_lr_0_001_pct'] = 100 * lr_dev['dice'] if lr_dev else None
        r['development_minus_few_pp'] = 100 * lr_dev['dice'] - r['few_data_baseline_pct'] if lr_dev else None
        r['development_archive'] = lr_dev.get('archive')
        full_pass = native.get(key, {})
        r["native_pass_full_pct"] = 100 * full_pass["dice"] if full_pass else None
        r["native_pass_original_stop_pct"] = 100 * full_pass["original_stop_dice"] if full_pass else None
        for field in ("updates", "original_stop", "observed_at_stop", "charged_flops", "archive", "sha256"):
            r["native_pass_" + field] = full_pass.get(field)
        r["native_pass_flops_over_cap"] = full_pass["charged_flops"] - cell["budget_flops"] if full_pass else None
        patch = patches.get(key, {})
        r['patch_training_full_pct'] = 100 * patch['dice'] if patch else None
        r['patch_training_shape'] = 'x'.join(map(str, patch['shape'])) if patch else None
        r['patch_observed_training_crops'] = patch.get('observed')
        r['patch_updates'] = patch.get('updates')
        r['patch_charged_flops'] = patch.get('charged_flops')
        r['patch_archive'] = patch.get('archive')
        r['patch_minus_native_few_pp'] = 100 * patch['dice'] - r['few_data_baseline_pct'] if patch else None
        r['corrected_result_status'] = ('verified_full_coverage' if corrected else
            'verified_native_pass_exception' if full_pass else
            'verified_patch_awaiting_native_pass' if patch else
            'awaiting_patch_full_coverage' if cell['model'] != 'medsam3' and cell['target_updates'] < eligible else
            'awaiting_verified_full_coverage_result')
        r['paper_matched_observed_crops'] = eligible if corrected else None
        r['paper_matched_minus_few_baseline_pp'] = (100 * corrected['dice'] - r['few_data_baseline_pct'] if corrected else None)
        r['paper_matched_archive'] = corrected['archive'] if corrected else None
        r['paper_matched_sha256'] = corrected['sha256'] if corrected else None
        replica = recoveries.get(key, {})
        r['separate_recovery_dice_pct'] = 100 * replica['dice'] if replica else None
        r['separate_recovery_archive'] = replica.get('archive')
        r['separate_recovery_charged_flops'] = replica.get('charged_flops')
        dice = cell['final_dice'] if cell['status'] in ('verified_complete','evaluated_pending_durable_archive') else None
        if standard_msd:
            dice = corrected["dice"] if corrected else None
        if dice is not None:
            assert 0 <= dice <= 1
        r['full_data_baseline_pct'] = None if dice is None else 100*dice
        for name in ('baseline','selected'):
            r[f'full_minus_few_{name}_pp'] = None if dice is None else 100*dice-r[f'few_data_{name}_pct']
        r['hfh_lr_0_1_repair_pct'] = repairs.get(key)
        for name in ('baseline', 'selected'):
            r[f'hfh_repair_minus_few_{name}_pp'] = (
                repairs[key] - r[f'few_data_{name}_pct'] if key in repairs else None
            )
    out = DEST/'paired_comparisons.csv'
    with NamedTemporaryFile(dir=out.parent, mode='w', delete=False) as stream:
        pending = Path(stream.name)
        writer = csv.DictWriter(stream,fieldnames=['cell',*next(iter(scores.values()))])
        writer.writeheader()
        writer.writerows(dict(cell=k,**scores[k]) for k in sorted(scores))
    from scratch_schedule_audit import audit as audit_scratch
    from medsam3_paired_audit import audit as audit_adapters
    try:
        audit_scratch(pending)
        audit_adapters(pending)
        pending.replace(out)
    finally:
        pending.unlink(missing_ok=True)
    write_progress(scores)
    write_constraint_gaps(scores.values())
    from matched_conditions import write_matched_conditions
    write_matched_conditions(scores)
    print('Matched',len(scores),'pairs')

if __name__=='__main__':
    main()
