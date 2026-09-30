"""Assemble a reproducible atlas pilot report and verification manifest."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.atlas_registration import CACHE, REPORT, native, normalize, save_json, sha
from evaluation.atlas_pilot import paired_summary


def audit_all_family_gates(cases, assessment, settings):
    """Apply the written criteria to every locked atlas family, including affine."""
    gates = {}
    for arm, chosen in settings["whole"].items():
        if arm == "graph_only":
            continue
        r = assessment["comparisons"][arm]
        d = r["foreground_dice"]
        graph = paired_summary(cases, arm, "graph_only")
        checks = dict(nonzero_atlas_weight=chosen["config"][1] > 0,
            gain_vs_raw=d["delta"] >= .001, interval_excludes_zero=d["delta_ci95"][0] > 0,
            gain_vs_graph=graph["foreground_dice"]["delta"] >= .001,
            surface_not_worse=r["assd_mm"]["delta"] <= 0,
            more_helped_than_harmed=d["improved"] >= d["worsened"])
        gates[arm] = dict(passed=all(checks.values()), checks=checks)
    for arm in settings["ap"]:
        r = assessment["ap_vs_model_plane"][arm]
        d = r["cut_error_mm"]
        checks = dict(gain_vs_model_plane=d["delta"] <= -.1,
            interval_excludes_zero=d["delta_ci95"][1] < 0,
            ap_dice_not_worse=r["ap_dice"]["delta"] >= 0,
            at_most_one_spoiled_raw_correct_cut=assessment["comparisons"][arm]["spoiled_correct_cuts"] <= 1)
        gates[arm] = dict(passed=all(checks.values()), checks=checks)
    return gates


def main():
    cohort = json.loads((REPORT / "cohort.json").read_text())
    settings = json.loads((REPORT / "locked_settings.json").read_text())
    assessment = json.loads((REPORT / "assessment_summary.json").read_text())
    cases = json.loads((REPORT / "assessment_cases.json").read_text())
    gates = audit_all_family_gates(cases, assessment, settings)
    save_json(REPORT / "all_family_gate_audit.json", gates)
    names = cohort["calibration"] + cohort["assessment"]
    records = [json.loads((REPORT / "registrations" / f"{name}.json").read_text()) for name in names]
    sources = [s for record in records for s in record["sources"]]
    expected_protocol_hash = sha(REPORT / "PROTOCOL.md")
    outputs = {}
    for record in records:
        assert record["target"] not in cohort["atlas_bank"]
        assert len(set(s["source"] for s in record["sources"])) == 3
        assert all(s["source"] in cohort["atlas_bank"] for s in record["sources"])
        assert record["provenance"]["files"]["docs/experiments/atlas_pilot_20260924/PROTOCOL.md"] == expected_protocol_hash
        assert record["provenance"]["files"]["evaluation/atlas_registration.py"] == sha(ROOT / "evaluation/atlas_registration.py")
        for relative, digest in record["input_hashes"].items():
            assert sha(ROOT / relative) == digest
        path = CACHE / "priors" / f"{record['target']}.npz"
        assert sha(path) == record["prior_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            for prior in archive.values():
                assert np.isfinite(prior).all() and prior.min() >= 0 and prior.max() <= 1
                np.testing.assert_allclose(prior.sum(0), 1, atol=2e-6)
        outputs[str(path.relative_to(ROOT))] = sha(path)
        logit_path = CACHE / "network" / f"{record['target']}.npz"
        outputs[str(logit_path.relative_to(ROOT))] = sha(logit_path)
    network = json.loads((REPORT / "network_verification_train.json").read_text())
    assert set(network) == set(names)
    assert all(r["old_hard_prediction_match"] and r["native_truth_match"] for r in network.values())
    assert settings["runner_sha256"] == sha(ROOT / "evaluation/atlas_pilot.py")
    registration = dict(targets=len(records), registrations=len(sources), atlas_bank=16, selected_atlases_per_target=3,
        affine_accepted=sum(s["affine_accepted"] for s in sources),
        deformable_accepted=sum(s["deformable_accepted"] for s in sources),
        mirrored_sources=sum(s["mirror"] for s in sources),
        summed_registration_seconds=sum(s["seconds"] for s in sources),
        median_registration_seconds=float(np.median([s["seconds"] for s in sources])),
        mean_nmi={stage: float(np.mean([s[f"{stage}_nmi"] for s in sources])) for stage in ("centered", "affine", "deformable")})
    save_json(REPORT / "registration_summary.json", registration)
    save_json(REPORT / "manifest.json", dict(protocol_sha256=expected_protocol_hash,
        cohort_sha256=sha(REPORT / "cohort.json"),
        source_hashes={p.name: sha(p) for p in (ROOT / "evaluation").glob("*atlas*.py")},
        network_manifest=json.loads((CACHE / "network/manifest.json").read_text()),
        library=records[0]["provenance"]["sitk"], artifacts=outputs,
        checked_target_atlas_disjointness=True, checked_all_input_hashes=True,
        checked_all_prior_probabilities=True, checked_48_previous_predictions=True))
    labels = {"raw": "Frozen network", "graph_only": "Graph only", "centered_fusion": "Centered atlas + network",
              "affine_fusion": "Affine atlas + network", "deformable_fusion": "Deformable atlas + network",
              "deformable_graph": "Deformable atlas + network + graph", "model_plane": "Model-only plane",
              "atlas_plane": "Atlas-only plane", "centered_ap": "Centered atlas + model plane",
              "affine_ap": "Affine atlas + model plane", "deformable_ap": "Deformable atlas + model plane"}
    cmp = assessment["comparisons"]
    whole_arms = ["raw", "graph_only", "centered_fusion", "affine_fusion", "deformable_fusion", "deformable_graph"]
    lines = ["# Atlas-guided refinement: completed pilot", "", "24 September 2026.", "",
        "This pilot tested image-registered label transfer from a 16-case atlas bank, "
        "with 24 separate cases for calibration and 24 for locked assessment. All are "
        "MSD fold-0 training cases; they were excluded from the atlas bank but were "
        "seen by the frozen backbone. This is an exploratory mechanism screen, not "
        "out-of-fold-backbone or independent-participant validation.", "",
        "## Decision", "", f"Written advancement criteria, applied to every locked family: `{json.dumps({k: v['passed'] for k, v in gates.items()})}`.", ""]
    if not any(v["passed"] for v in gates.values()):
        lines += ["**The atlas pilot did not pass its advancement gate.** No development "
                  "evaluation or model retraining was launched. The following results "
                  "characterize this crop-registration and fusion implementation; they "
                  "do not establish that all atlas methods are ineffective.", ""]
    lines += ["## Interpretation and next experiment", "",
        "The calibration A/P signal did not transfer: affine fusion reduced "
        "calibration cut MAE from 0.4583 to 0.3333 mm, but increased assessment "
        "MAE from 0.5000 to 0.7917 mm. It improved one cut and worsened eight. "
        "Deformable fusion improved one cut (hippocampus_068) and worsened one "
        "previously correct cut (hippocampus_353); the other 22 were unchanged. "
        "Its A/P Dice gain versus the model-only plane was 0.0602 percentage "
        "points, with a paired 95% interval spanning -0.1578 to +0.3384. "
        "This does not support an A/P improvement claim.", "",
        "Registration improved atlas-alone assessment foreground Dice from "
        "70.927% to 79.348%, but the frozen network achieved 91.223%. The "
        "atlas-alone foreground corrected 5,282 network errors while introducing "
        "22,725 new errors across the 24 cases. This shows some complementary "
        "voxel information, with a much larger cost when applied indiscriminately. "
        "These counts describe direct atlas replacement, not the calibrated "
        "fusion, which selected zero foreground atlas weight.", "",
        "**Recommendation:** do not add this atlas prior as a training loss yet. "
        "If pursuing a second pilot, first improve registration on training-only "
        "cases with stronger deformation regularization and explicit local "
        "alignment checks; 65/144 nonlinear fits exceeded the displacement guard. "
        "Then test whether model uncertainty and agreement among independently "
        "registered atlases can identify useful boundary corrections. Keep "
        "individual atlas outputs to measure disagreement before averaging. "
        "Freeze any acceptance rule using out-of-fold backbone predictions, "
        "and evaluate on a fresh case/participant-separated cohort. These are "
        "proposed experiments, not improvements demonstrated here. The current "
        "locked A/P candidates offer only 0.0417 mm of gain even with perfect "
        "case selection, so a more elaborate case selector alone has little "
        "observed headroom.", "",
        "![Assessment effect sizes](../../../experiments/atlas_pilot_20260924/assessment_results.png)", "",
        "## Whole-hippocampus refinement", "",
        "Settings were selected by foreground Dice on calibration cases only. "
        "Foreground can expand and contract across the entire native image grid. "
        "Conditional A/P labels remain the network's choice in these arms.", "",
        "| Method | Foreground Dice % | Delta vs network, pp [95% CI] | ASSD, mm | HD95, mm | Better / worse / tied |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in whole_arms:
        r = cmp[arm]; d = r["foreground_dice"]; ci = d["delta_ci95"]
        lines.append(f"| {labels[arm]} | {100*d['mean']:.4f} | {100*d['delta']:+.4f} [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] | "
            f"{r['assd_mm']['mean']:.4f} | {r['hd95_mm']['mean']:.3f} | {d['improved']} / {d['worsened']} / {d['tied']} |")
    lines += ["", "| Arm | Selected alpha | Selected lambda | Foreground voxels corrected / introduced |", "|---|---:|---:|---:|"]
    for arm, chosen in settings["whole"].items():
        _, a, l = chosen["config"]; r = cmp[arm]
        lines.append(f"| {labels[arm]} | {a:g} | {l:g} | {r['fg_corrected_voxels']} / {r['fg_introduced_voxels']} |")
    lines += ["", "## A/P cut localization", "",
        "All A/P arms preserve original predicted foreground. The model-only plane "
        "controls for enforcing a plane, so changes versus the raw network cannot "
        "automatically be attributed to atlas information. A/P targets are reference "
        "best-fit planes, while Dice retains the original voxel labels.", "",
        "| Method | Cut MAE, mm | A/P Dice % | Swaps | Raw-correct cuts spoiled |", "|---|---:|---:|---:|---:|"]
    for arm in ["raw", "model_plane", "atlas_plane", "centered_ap", "affine_ap", "deformable_ap"]:
        r = cmp[arm]
        lines.append(f"| {labels[arm]} | {r['cut_error_mm']['mean']:.4f} | {100*r['ap_dice']['mean']:.4f} | {r['ap_swaps']} | {r['spoiled_correct_cuts']}/{r['initially_correct_cuts']} |")
    lines += ["", "| Atlas plane fusion | Selected alpha | MAE delta vs model plane, mm [95% CI] |", "|---|---:|---:|"]
    for arm, chosen in settings["ap"].items():
        d = assessment["ap_vs_model_plane"][arm]["cut_error_mm"]; ci=d["delta_ci95"]
        lines.append(f"| {labels[arm]} | {chosen['config'][1]:g} | {d['delta']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ["", "## Does registration add anatomical accuracy?", "",
        "These atlas-alone results use the same three sources per target at every "
        "stage. Source selection is based on image similarity, never target labels.", "",
        "| Atlas stage | Foreground Dice % | A/P Dice % | Mean relative volume error % |", "|---|---:|---:|---:|"]
    for stage in ("centered", "affine", "deformable"):
        r = cmp[f"{stage}_atlas_alone"]
        lines.append(f"| {stage} | {100*r['foreground_dice']['mean']:.3f} | {100*r['ap_dice']['mean']:.3f} | {100*r['relative_volume_error']['mean']:+.2f} |")
    lines += ["", f"Across all 48 targets, {registration['affine_accepted']}/144 affine transforms and "
        f"{registration['deformable_accepted']}/144 residual deformations passed the image-only engineering guards. "
        "Rejected residuals use the affine result; rejected affines use centered alignment. "
        "Thus the deformable prior is a guarded mixture of registration stages. "
        "These guards are not proof of anatomical registration correctness.", "",
        f"Mean image NMI (centered / affine / final): {registration['mean_nmi']['centered']:.4f} / "
        f"{registration['mean_nmi']['affine']:.4f} / {registration['mean_nmi']['deformable']:.4f}. "
        "Image similarity and label accuracy must be assessed separately: matching "
        "surrounding tissue does not guarantee the correct hippocampal boundary.", "",
        "![Predetermined calibration examples](../../../experiments/atlas_pilot_20260924/registration_examples.png)", "",
        "## Room for improvement: oracle diagnostic", "",
        "The following hypothetical selector uses the reference to choose, case by "
        "case, between the frozen network and the already locked correction. It is "
        "not deployable, not a fitted gate, and not a strict bound on all possible "
        "methods. It measures error complementarity for these candidates.", "",
        "| Candidate | Oracle foreground Dice gain, pp | Oracle cut MAE reduction, mm |", "|---|---:|---:|"]
    for arm in ("deformable_fusion", "deformable_graph", "centered_ap", "affine_ap", "deformable_ap"):
        oracle = cmp[arm]["oracle_case_selection"]
        lines.append(f"| {labels[arm]} | {100*oracle['foreground_dice_gain']:.4f} | {oracle['cut_mae_gain']:.4f} |")
    lines += ["", "No reliability gate was fitted to assessment outcomes. Before escalating "
        "to a new model, establish that a label-independent registration-quality or "
        "disagreement signal can distinguish useful corrections from damage, using "
        "out-of-fold backbone outputs and a fresh evaluation set. If registration "
        "itself reduces label accuracy, first improve registration on training-only "
        "data rather than assigning the prior more weight.", "",
        "## Verification and limitations", "",
        "All 48 new CPU hard predictions exactly match the prior caches and native "
        "reference masks. All atlas probabilities, source/input hashes, cohort "
        "separation, and locked-settings provenance were checked. Eleven new tests "
        "passed (36 passed together with related graph/partition tests), including "
        "exact small-graph energy enumeration, transform direction "
        "and composition, fractional label interpolation, foreground expansion, "
        "A/P support preservation, and plane optimization against direct enumeration.", "",
        "Binary graph cuts optimize the quantized energy at scale 10,000. A conservative "
        "floating-energy suboptimality bound per solve is (number of voxels + number "
        "of undirected edges)/10,000. This is a numerical solver guarantee, not an "
        "anatomical optimality guarantee. Surface distances use native 1-mm voxels; "
        "ASSD averages the two directed surface means and HD95 takes the 95th "
        "percentile of their concatenated distances.", "",
        "Bootstrap intervals use 10,000 paired case resamples, conditional on "
        "calibration-selected settings; they omit model-refitting uncertainty and "
        "are not corrected for multiple comparisons. Shared backbone training and "
        "unverified participant linkage limit generalization. There is no comparison "
        "to the 2008 paper's Dice numbers across different datasets/protocols.", "",
        "## Reproduction and evidence", "",
        "```bash", "rtk proxy .venv/bin/python evaluation/atlas_registration.py --subset train --workers 2",
        "rtk proxy .venv/bin/python evaluation/atlas_model_cache.py --subset train",
        "rtk proxy .venv/bin/python evaluation/atlas_pilot.py --stage calibrate",
        "rtk proxy .venv/bin/python evaluation/atlas_pilot.py --stage assessment",
        "rtk proxy env MPLCONFIGDIR=/tmp/hippo-atlas-mpl .venv/bin/python evaluation/report_atlas_pilot.py",
        "rtk proxy .venv/bin/python -m pytest evaluation/test_atlas_pilot.py -q", "```", "",
        "Requires SimpleITK 2.5.2 in the experiment dependencies folder, the local "
        "dataset, canonical checkpoint, and documented existing CPU caches. "
        "Code or protocol changes invalidate cached provenance and require a new "
        "versioned experiment.", "",
        "Evidence: [protocol](PROTOCOL.md), [cohort](cohort.json), "
        "[implementation notes](IMPLEMENTATION_NOTES.md), [locked settings](locked_settings.json), "
        "[calibration cases](calibration_cases.json), [assessment summary](assessment_summary.json), "
        "[assessment cases](assessment_cases.json), [registration summary](registration_summary.json), "
        "[all-family gate audit](all_family_gate_audit.json), [reporting audit](REPORTING_AUDIT.md), "
        "[manifest](manifest.json). Derived arrays and transforms remain in ignored "
        "`experiments/atlas_pilot_20260924/`.", ""]
    (REPORT / "README.md").write_text("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Atlas pilot · locked assessment on 24 cases", fontsize=17, weight="bold")
    for ax, arms, metric, reference, factor, xlabel in [
        (axes[0], whole_arms[1:], "foreground_dice", "raw", 100, "Foreground Dice change (percentage points)"),
        (axes[1], list(settings["ap"]), "cut_error_mm", "model_plane", 1, "Cut MAE change (mm; negative is better)")]:
        for y, arm in enumerate(arms):
            stat = assessment["comparisons"][arm] if reference == "raw" else assessment["ap_vs_model_plane"][arm]
            d = stat[metric]
            mean = d["delta"] * factor
            lo, hi = np.array(d["delta_ci95"]) * factor
            ax.errorbar(mean, y, xerr=[[max(0, mean-lo)], [max(0, hi-mean)]], fmt="o", capsize=4, color="#245c78")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_yticks(range(len(arms)), [labels[a].replace(" + network", "").replace(" + model plane", "") for a in arms])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    fig.savefig(CACHE / "assessment_results.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for row, name in enumerate(cohort["calibration"][:2]):
        image = normalize(native(name)); gt = native(name, labels=True); x = image.shape[0] // 2
        with np.load(CACHE / "priors" / f"{name}.npz") as priors:
            for col, kind in enumerate(("reference", "centered", "affine", "deformable")):
                ax=axes[row, col]
                ax.imshow(image[x].T, cmap="gray", origin="lower", vmin=0, vmax=1)
                mask = gt[x] > 0 if kind == "reference" else priors[kind][1:, x].sum(0) > .5
                ax.contour(mask.T, levels=[.5], colors=["#49c2c9" if kind == "reference" else "#ffbe55"], linewidths=1)
                if col:
                    ax.contour((gt[x] > 0).T, levels=[.5], colors=["#49c2c9"], linewidths=.8)
                ax.set_title(kind.title())
                ax.set_xlabel("A voxel")
                ax.set_ylabel(f"{name}\nS voxel" if not col else "S voxel")
    fig.suptitle("Predetermined calibration examples · atlas (gold), reference (cyan)", fontsize=14)
    fig.savefig(CACHE / "registration_examples.png", dpi=160)
    plt.close(fig)
    print(f"Report and verified manifest saved to {REPORT}")


if __name__ == "__main__":
    main()
