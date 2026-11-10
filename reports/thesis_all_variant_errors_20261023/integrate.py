"""Place the verified all-variant error tables in thesis Section 5.7."""
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("error_build", HERE / "build.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ROOT = mod.ROOT
PATH = ROOT / "Template_tesi/thesis-revision.tex"
conditions = json.loads((HERE / "condition_errors.json").read_text())
index = {(r["campaign"], r["model"], r["arm"]): r for r in conditions}
rendered = []
SCALES = {
    "GeoFWI3D_conventional": 6,
    "MSD_conventional": 3,
    "ADNI_conventional": 3,
    "GeoFWI3D_SAM3": 3,
    "MSD_MedSAM3_original": 2,
    "MSD_MedSAM3_extension": 2,
    "ADNI_MedSAM3": 2,
}


def error_sort_key(row):
    metrics = ["fn", "fp", "swaps"] if row["campaign"].startswith("MSD") else ["fn", "fp"]
    total = sum(row[m] for m in metrics) if row["complete"] else float("inf")
    return (not row["complete"], total, mod.ARMS.index(row["arm"]), row["campaign"])


def ordered_arms(campaign, model):
    return sorted(mod.ARMS, key=lambda arm: error_sort_key(index[campaign, model, arm]))


def cell(row, metric):
    if metric == "swaps" and not row["campaign"].startswith("MSD"):
        return "N/A"
    if not row["complete"]:
        return r"\samPending"
    scale = 10 ** SCALES[row["campaign"]]
    return f'{row[metric] / scale:.2f}' + r" $\pm$ " + f'{row[metric + "_fold_sd"] / scale:.2f}'


def data_line(campaign, model, arm, name=None, coverage=False):
    row = index[campaign, model, arm]
    metrics = ["fn", "fp", "swaps"] if campaign.startswith("MSD") else ["fn", "fp"]
    values = [name or mod.label(arm)] + [cell(row, m) for m in metrics]
    if coverage:
        values.append(f'{row["runs"]}/9')
    rendered.append({"campaign": campaign, "model": model, "arm": arm,
                     "cells": values, "complete": row["complete"]})
    return " & ".join(values) + r" \\" + "\n"


def longtable(caption, labels, body, coverage=False, unit="run", swaps=True, exponent=3):
    n = 3 + int(swaps) + int(coverage)
    total = r"FN $+$ FP $+$ A/P swaps" if swaps else r"FN $+$ FP"
    caption += " Within each model, rows ascend by mean total error (" + total + ")."
    if coverage:
        caption += " Pending conditions follow the complete rows."
    def heading(metric):
        return r"\shortstack{" + f"{metric} / {unit}" + r"\\($\times 10^{" + str(exponent) + r"}$)}"
    header = "Variant & " + heading("FN") + " & " + heading("FP")
    if swaps:
        header += " & " + heading("A/P swaps")
    if coverage:
        header += " & Runs"
    header += r" \\ \hline" + "\n"
    return (
        r"\begingroup\footnotesize\setstretch{1.0}" + "\n"
        r"\setlength{\tabcolsep}{4pt}" + "\n"
        r"\renewcommand{\arraystretch}{1.12}" + "\n"
        r"\begin{longtable}{@{}l" + "r" * (n - 1) + r"@{}}" + "\n"
        r"\caption{" + caption + "}\n" +
        "".join(r"\label{" + x + "}" for x in labels) + r"\\" + "\n"
        r"\hline" + "\n" + header + r"\endfirsthead" + "\n"
        r"\hline" + "\n" + header + r"\endhead" + "\n"
        r"\hline" + "\n" + r"\multicolumn{" + str(n) + r"}{r}{Continued on next page}\\" + "\n"
        r"\endfoot" + "\n" + r"\hline\endlastfoot" + "\n"
        + body + r"\end{longtable}" + "\n" + r"\endgroup" + "\n\n"
    )


def conventional(dataset, labels):
    body = ""
    n = 4 if dataset == "MSD" else 3
    for model in mod.MODELS:
        body += r"\multicolumn{" + str(n) + r"}{@{}l}{\textbf{" + mod.MODEL_NAMES[model] + r"}}\\*" + "\n"
        body += "".join(data_line(dataset + "_conventional", model, arm) for arm in ordered_arms(dataset + "_conventional", model))
        body += r"\hline" + "\n"
    caption = (dataset + r" conventional models: all fourteen objectives for each architecture. "
               r"Entries are mean $\pm$ sample SD across three seed-averaged fold means, "
               r"using all nine runs per row. Counts pool the full validation set within each run.")
    return longtable(caption, labels, body, swaps=dataset == "MSD", exponent=SCALES[dataset + "_conventional"])


section = r"""\subsection{Error Analysis}
\label{sec:error-analysis}
\label{sec:capped-errors-resources}

The following tables report errors for every training variant in the
six studies, in the same order as the Dice comparisons. A false negative
(FN) is reference foreground predicted as background; a false positive
(FP) is reference background predicted as foreground. On MSD, A/P swaps
count anterior voxels predicted as posterior plus posterior voxels
predicted as anterior. These swaps preserve foreground membership and
are counted separately from FN and FP. They are not applicable
to binary ADNI or GeoFWI3D segmentation. All counts cover the complete
validation prediction, including locations outside the constraint bands.

For the conventional models, counts are pooled over the validation
volumes in each run. For SAM3 and MedSAM3, counts are averaged per volume
or crop within a run. Each complete condition then averages the three
seeds within each fold and weights the three folds equally. Tables give
the resulting mean and sample standard deviation across the three fold
means. Both the mean and its standard deviation use the power of ten
shown in the column heading, with displayed values rounded to two decimal
places. For example, $2.12\pm0.70$ under $\times10^6$ denotes
approximately $2.12\times10^6\pm0.70\times10^6$ voxels. Full-precision
counts are retained in the numerical archive.
These dispersions describe variation between the evaluated folds;
they are not confidence intervals. Counts retain each study's evaluation
grid and denominator, so their absolute magnitudes should not be compared
across studies as a measure of relative performance.

Within each model and dataset, variants are ordered from the lowest to
the highest mean total error: FN $+$ FP for binary segmentation, and
FN $+$ FP $+$ A/P swaps for MSD. These categories are disjoint, so their
sum counts all incorrectly labelled voxels. Ordering uses unrounded
means; pending conditions appear last.

\subsubsection{GeoFWI3D: Conventional Models}

Table~\ref{tab:geo-all-errors} includes all 504 evaluations, including
the nine recovered baseline records. For each architecture's highest-mean
Dice variant, nnU-Net reduces both FN (34.4\%) and FP (11.4\%) relative
to baseline. SwinUNETR and 3D U-Net reduce FP by 12.4\% and 26.9\%,
respectively, with FN increases of 0.7\% and 3.9\%. UNETR's $B+E$
configuration instead reduces FN by 4.3\% while increasing FP by 2.9\%.
Thus, the overlap gains arise from different changes in foreground
retention and extension. The full table also retains variants with
negative Dice changes; no error category is selected for reporting
according to whether it improves.

"""
section += conventional("GeoFWI3D", ["tab:geo-all-errors"])
section += r"""\subsubsection{MSD: Conventional Models}

Table~\ref{tab:msd-all-errors} separates whole-foreground errors from
internal A/P confusion for all four models and fourteen objectives.
At the highest-mean Dice configuration of each architecture, nnU-Net
reduces missed foreground while increasing foreground extension
slightly; the other models reduce extension while increasing omissions.
All four selected configurations reduce mean A/P swaps. This does not
imply that every constrained variant reduces every error category.

"""
section += conventional("MSD", ["tab:msd-all-errors", "tab:capped-errors"])
section += r"""\subsubsection{ADNI: Conventional Models}

Table~\ref{tab:adni-all-errors} reports the corresponding binary-task
errors. At each architecture's highest-mean Dice configuration, nnU-Net
reduces both error types, while the other models exchange fewer false
positives for more false negatives. A Dice improvement therefore does
not by itself establish a reduction in both omissions and extensions.

"""
section += conventional("ADNI", ["tab:adni-all-errors"])
section += r"""\subsubsection{SAM3 on GeoFWI3D}

Table~\ref{tab:sam3-geo-errors} uses the verified archive snapshot of
23 October 2026 at 13:37 UTC, also used for the complementary Dice
analysis in Section~\ref{sec:background-dice-results}. It contains
116 of the 126 planned evaluations. Six conditions have all nine
fold/seed cells and can be summarized; the eight remaining rows show
their available run counts and retain pending aggregates. These are
archive-coverage counts, not a live statement of training completion.
No mean is calculated from an incomplete condition.

"""
section += longtable(
    r"SAM3 on GeoFWI3D: full-volume FN and FP for all fourteen variants. "
    r"Each run averages errors over 96 held-out volumes; complete rows report "
    r"mean $\pm$ sample SD across three seed-averaged fold means. Runs records "
    r"coverage in the stated archive snapshot. Pending rows await complete "
    r"nine-run evidence.",
    ["tab:sam3-geo-errors"],
    "".join(data_line("GeoFWI3D_SAM3", "SAM3", arm, coverage=True) for arm in ordered_arms("GeoFWI3D_SAM3", "SAM3")),
    coverage=True, unit="volume", swaps=False)
section += r"""\noindent\samResult{interpret the FN/FP changes of the final selected
SAM3 objective against its matched baseline after the complete
nine-run conditions have been verified; distinguish fewer misses,
less foreground extension and any tradeoff}.

\subsubsection{MedSAM3 on MSD}

Table~\ref{tab:medsam-msd-all-errors} covers all fourteen objectives.
The two baseline rows identify the controls matched to $B$ and $B+E$,
and to the other constrained variants, within the same campaign.
The preserved aggregate records supply FN, FP and swaps for $B$, $B+E$
and their matched baseline even though their per-volume confusion
matrices were not recovered. Consequently, those error results are
complete; the missing per-volume records affect the complementary
background-Dice analysis, not the error counts reported here.

The highest-mean MSD adapter, $B+E+S+D$, reduces mean FN by 36.0\%
and A/P swaps by 20.7\%, while FP increases by 40.7\% relative to its
matched baseline. Improved overlap therefore includes a shift toward
retaining more foreground, with fewer internal class confusions.

"""
msd_body = ""
msd_rows = [("MSD_MedSAM3_original", arm) for arm in ["baseline", "bands", "bands_edge"]]
msd_rows += [("MSD_MedSAM3_extension", arm) for arm in mod.ARMS if arm not in {"bands", "bands_edge"}]
for campaign, arm in sorted(msd_rows, key=lambda key: error_sort_key(index[key[0], "MedSAM3", key[1]])):
    name = None
    if arm == "baseline":
        name = r"Baseline ($B$, $B+E$)" if campaign == "MSD_MedSAM3_original" else "Baseline (other variants)"
    msd_body += data_line(campaign, "MedSAM3", arm, name)
section += longtable(
    r"MSD MedSAM3: all variants and their matched controls. Counts are mean "
    r"errors per validation crop within a run, then averaged over seeds and folds; "
    r"$\pm$ gives sample SD across the three fold means. Each row includes nine "
    r"runs. The two baseline labels identify their matched constrained variants.",
    ["tab:medsam-msd-all-errors", "tab:medsam-noaug-errors", "tab:medsam-noaug-error-ledger"],
    msd_body, unit="crop", exponent=2)
section += r"""\subsubsection{MedSAM3 on ADNI}

Table~\ref{tab:medsam-adni-all-errors} reports all fourteen binary
segmentation variants. For the highest-mean Dice configuration,
$E+S+D$, FN decreases by 24.0\% and FP increases by 26.2\% relative
to baseline. As on MSD, the overlap gain includes greater foreground
retention rather than a simultaneous reduction of both error types.
Surface measures and attributable prediction examples would be needed
to locate these changes. The dataset examples in
Chapter~\ref{sec:methodology} show reference annotations and should
not be read as model-performance examples.

"""
section += longtable(
    r"ADNI MedSAM3: all fourteen variants, each with nine runs. Counts are "
    r"mean errors per validation crop within each run; entries give mean $\pm$ "
    r"sample SD across three seed-averaged fold means.",
    ["tab:medsam-adni-all-errors"],
    "".join(data_line("ADNI_MedSAM3", "MedSAM3", arm) for arm in ordered_arms("ADNI_MedSAM3", "MedSAM3")), unit="crop", swaps=False, exponent=2)
section += r"""\noindent\textit{[Presentation evidence to insert: paired baseline and
constrained predictions selected by a stated rule, with classwise and
surface error summaries where retained outputs and valid geometry permit them.]}

"""
assert len(rendered) == 211
(HERE / "error_section.tex").write_text(section)
(HERE / "table_rows.json").write_text(json.dumps(rendered, indent=2) + "\n")

original = PATH.read_text()
s = original
old_start = s.index(r"\subsection{Error Patterns and Resource Use}")
old_end = s.index("% BEGIN BACKGROUND DICE ANALYSIS", old_start)
s = s[:old_start] + section + "\\FloatBarrier\n" + s[old_end:]

# Keep the SAM3 result section focused on its Dice comparison; errors now live in 5.7.
a = s.index(r"\subsubsection{Error Balance and Adaptation Cost}")
b = s.index("The comparison with MedSAM3 concerns", a)
s = s[:a] + r"""Full-volume errors for all SAM3 variants are reported in
Section~\ref{sec:error-analysis}, Table~\ref{tab:sam3-geo-errors}.

\noindent\samResult{paired baseline/constrained prediction panels,
selected by a stated rule, illustrating the measured FN/FP pattern}.

""" + s[b:]

# Remove duplicate error columns from appendix ledgers, preserving every other cell.
preserved = []
for label, drop, old_cols, new_cols, caption in [
    ("tab:geo-ledger", {4, 5}, 7, 5,
     r"GeoFWI3D complete condition ledger. Dice is in percent; gains are paired percentage points. Full-validation error counts are reported in Table~\ref{tab:geo-all-errors}."),
    ("tab:msd-noaug-ledger", {4, 5, 6}, 8, 5,
     r"MSD full condition ledger. Support Dice is in percent and gains are paired validation-Dice percentage points. Error counts are reported in Table~\ref{tab:msd-all-errors}."),
    ("tab:adni-noaug-ledger", {4, 5}, 7, 5,
     r"ADNI full condition ledger. Support Dice is in percent and gains are paired validation-Dice percentage points. Error counts are reported in Table~\ref{tab:adni-all-errors}."),
]:
    pos = s.index(r"\label{" + label + "}")
    start = s.rfind(r"\begin{longtable}", 0, pos)
    end = s.index(r"\end{longtable}", pos) + len(r"\end{longtable}")
    block = s[start:end]
    block = re.sub(r"\\begin\{longtable\}\{[^\n]*\}", lambda _: r"\begin{longtable}{@{}lrrrr@{}}", block, count=1)
    cap_start = block.index(r"\caption{")
    label_start = block.index(r"\label{")
    block = block[:cap_start] + r"\caption{" + caption + "}" + block[label_start:]
    block = block.replace(r"\multicolumn{" + str(old_cols) + "}", r"\multicolumn{" + str(new_cols) + "}")
    updated = []
    for line in block.splitlines():
        if line.startswith("Variant &") or line.startswith("Baseline &") or line.startswith("Baseline $"):
            parts = line.split(" & ")
            assert len(parts) == old_cols, (label, line)
            kept = [p for i, p in enumerate(parts) if i not in drop]
            if line.startswith("Baseline"):
                preserved.append({"table": label, "cells": kept})
            line = " & ".join(kept)
        updated.append(line)
    new_block = "\n".join(updated)
    s = s[:start] + new_block + s[end:]
assert len(preserved) == 168

# There is one copy of each full error table, in Section 5.7.
pos = s.index("\\caption{All retained MedSAM3 configurations.")
start = s.rfind(r"\begin{table}", 0, pos)
end = s.index(r"\end{table}", pos) + len(r"\end{table}")
s = s[:start] + r"""The complete MedSAM3 error tables are presented in
Section~\ref{sec:error-analysis}, Tables~\ref{tab:medsam-msd-all-errors}
and~\ref{tab:medsam-adni-all-errors}.
""" + s[end:]

def replace_once(old, new):
    global s
    assert s.count(old) == 1, old[:100]
    s = s.replace(old, new, 1)


replace_once("Error patterns, resource use, metric", "Error patterns, metric")
replace_once(
    "Table~\\ref{tab:geo-ledger} in Appendix~\\ref{sec:extended-results}\n"
    "gives all 56 condition summaries with\n"
    "paired gains, positive-cell counts, updates and available full-volume\n"
    "FN/FP counts. The appendix retains the error-count coverage of the\n"
    "22 October extraction. The nine reused fold-0 baseline evaluation\n"
    "records have since been recovered, completing the confusion-matrix\n"
    "evidence used in Section~\\ref{sec:background-dice-results}. Full-matrix\n"
    "baseline-relative FN/FP summaries remain to be incorporated into\n"
    "the appendix; all 504 final Dice results are already included.",
    "Table~\\ref{tab:geo-ledger} in Appendix~\\ref{sec:extended-results}\n"
    "gives all 56 condition summaries with paired gains, positive-cell\n"
    "counts and updates. The complete FN/FP comparison is presented in\n"
    "Section~\\ref{sec:error-analysis}, Table~\\ref{tab:geo-all-errors}.\n"
    "It includes the nine recovered baseline evaluations and covers all\n"
    "504 final predictions.")
replace_once(
    "errors that remain. Completing the geological FN/FP comparisons and\nend-to-end runtime and peak-memory reporting would strengthen the",
    "errors that remain. End-to-end runtime and peak-memory reporting would strengthen the")
replace_once(
    "appendix retains the earlier FN/FP coverage pending integration\nof the recovered baseline counts.",
    "error analysis in Section~\\ref{sec:error-analysis} incorporates\nall 504 evaluations, including the recovered baseline counts.")
replace_once(
    "The geological four-model matrix is complete; retrieving the nine\nbaseline raw archives would complete its baseline error aggregates. Presentation",
    "The geological four-model matrix and its FN/FP aggregates are complete. Presentation")
replace_once(
    "Conventional FN, FP and swaps are mean pooled validation counts per run;\n"
    "MedSAM3 errors are means per validation crop, and SAM3 errors are means\n"
    "per validation volume. Verified numerical reductions and full-precision\n"
    "values are preserved in the accompanying CSV/JSON extractions. SAM3\n"
    "placeholders retain the same reporting conventions for final insertion.",
    "The full error tables and their reductions are presented together in\n"
    "Section~\\ref{sec:error-analysis}. Verified numerical reductions and\n"
    "full-precision values are preserved in the accompanying CSV/JSON\n"
    "extractions. SAM3 placeholders retain the stated reporting conventions\n"
    "for final insertion.")
replace_once(
    "analysis identifies its separate SAM3 snapshot and uses only fully\npopulated conditions within it.",
    "and error analyses identify their separate SAM3 archive snapshot and\nuse only fully populated conditions within it.")

evidence = r"""\subsection{Full-Validation Error Tables}
\label{sec:all-variant-error-evidence}

The error tables in Section~\ref{sec:error-analysis} are generated from
\path{reports/thesis_all_variant_errors_20261023/}. Its \path{build.py}
reads the recovered per-volume confusion matrices, checks their dimensions,
nonnegative integer entries, voxel totals and unique case/run identities,
and verifies all 1,862 evaluation-file hashes. It reconstructs errors
from 124,860 validation-volume records and checks the results against
1,242 earlier medical run summaries and 990 available geological FN/FP
values. The remaining 27 MSD MedSAM3 records use the preserved aggregate
errors in \path{reports/thesis_noaug_update_20261021/results.json}; they
do not require reconstruction of unavailable per-volume matrices.

The extraction retains all 1,512 conventional evaluations, 135 MSD
MedSAM3 records including the repeated matched controls, 126 ADNI
MedSAM3 evaluations and the 116-record SAM3 snapshot. It produces
211 condition rows, including the two applicable MSD baseline rows.
Of these, 203 have all nine fold/seed cells; the other eight are SAM3
conditions whose aggregate errors remain pending. Every complete row
uses the declared fold and seed inventory. The directory preserves
\path{run_errors.json}, \path{condition_errors.json},
\path{verification.json} and the generated table rows, including source
hashes, full-precision values and fold dispersions. No training or
inference is rerun for this extraction.

"""
replace_once(r"\subsection{Annotation Evidence and Outstanding Items}", evidence + r"\subsection{Annotation Evidence and Outstanding Items}")

# Check section order, all labels/references and preservation of non-error results.
result_section = s[s.index(r"\section{Results and Analysis}"):s.index(r"\section{Conclusions}")]
headings = re.findall(r"\\subsection\{([^}]+)\}", result_section)
assert headings[6] == "Error Analysis", headings
labels = re.findall(r"\\label\{([^}]+)\}", s)
assert len(labels) == len(set(labels)), [x for x in labels if labels.count(x) > 1]
missing = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", s)) - set(labels)
assert not missing, missing
assert r"\label{tab:noaug-resources}" not in s
assert len(re.findall(r"\\subsubsection\{", section)) == 6
assert "training-loop timers" not in section and "PFLOPs" not in section
for label in ["tab:geo-ledger", "tab:msd-noaug-ledger", "tab:adni-noaug-ledger"]:
    pos = s.index(r"\label{" + label + "}")
    block = s[pos:s.index(r"\end{longtable}", pos)]
    current = [line.split(" & ") for line in block.splitlines() if line.startswith("Baseline")]
    expected = [row["cells"] for row in preserved if row["table"] == label]
    assert current == expected, label
snapshot = Path("/private/tmp/thesis-before-all-variant-errors.tex")
if not snapshot.exists():
    snapshot.write_text(original)
assert PATH.read_text() == original, "Thesis changed concurrently; rerun against the current file."
PATH.write_text(s)
(HERE / "integration_check.json").write_text(json.dumps({
    "tables": 6, "variant_rows": len(rendered), "complete_rows": sum(x["complete"] for x in rendered),
    "appendix_non_error_rows_preserved": len(preserved), "results_headings": headings,
    "unresolved_references": sorted(missing), "resource_use_removed_from_section_5_7": True,
}, indent=2) + "\n")
print(json.dumps({"saved": str(PATH), "tables": 6, "rows": 211, "complete": 203,
                  "pending_sam3_conditions": 8, "section_5_7": headings[6]}, indent=2))
