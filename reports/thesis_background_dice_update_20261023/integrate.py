"""Insert the verified background-Dice comparison into the existing thesis."""
from pathlib import Path
import collections
import csv
import difflib
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / 'reports/background_dice_archive_recovery_20261023'
THESIS = REPO / 'Template_tesi/thesis-revision.tex'
METRICS = ['foreground_dice', 'background_dice', 'dice_including_background']
MODELS = ['nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16']
MODEL_NAMES = {'nnunet': 'nnU-Net', 'swinunetr': 'SwinUNETR',
               'unetr_vit_b16': 'UNETR', 'basic_unet16': '3D U-Net',
               'MedSAM3': 'MedSAM3', 'SAM3': 'SAM3'}
CAMPAIGNS = ['GeoFWI3D_conventional', 'MSD_conventional', 'ADNI_conventional',
             'MSD_MedSAM3_extension', 'ADNI_MedSAM3']

def load_csv(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mean_by_fold(rows, metric):
    folds = sorted({r['fold'] for r in rows})
    return statistics.mean(statistics.mean(r[metric] for r in rows if r['fold'] == f)
                           for f in folds)

def arm_label(arm):
    if arm in {'dice', 'baseline'}:
        return 'Baseline'
    parts = [p for p in arm.split('_') if p != 'dice']
    letters = {'bands': 'B', 'edge': 'E', 'sdf': 'S', 'depth': 'D'}
    return '$' + '+'.join(letters[p] for p in parts) + '$'

rows = load_csv('background_dice_runs.csv')
for r in rows:
    for metric in METRICS:
        r[metric] = float(r[metric])
    expected = ((2 * r['foreground_dice'] + r['background_dice']) / 3
                if r['campaign'].startswith('MSD') else
                (r['foreground_dice'] + r['background_dice']) / 2)
    assert abs(expected - r['dice_including_background']) < 1e-12
assert len({(r['campaign'], r['run_id']) for r in rows}) == len(rows)

grouped = collections.defaultdict(list)
for r in rows:
    grouped[(r['campaign'], r['model'], r['arm'])].append(r)
conditions = {}
for key, rs in grouped.items():
    conditions[key] = {m: mean_by_fold(rs, m) for m in METRICS}
    conditions[key]['runs'] = len(rs)
    conditions[key]['balanced'] = (len(rs) == 9 and len({r['fold'] for r in rs}) == 3
                                   and all(sum(r['fold'] == f for r in rs) == 3
                                           for f in {r['fold'] for r in rs}))
for r in load_csv('background_dice_conditions.csv'):
    key = (r['campaign'], r['model'], r['arm'])
    for m in METRICS:
        assert abs(conditions[key][m] - float(r[m])) < 1e-12

bases = {}
selected = {}
comparisons = []
flips = []
for campaign in CAMPAIGNS:
    models = MODELS if campaign.endswith('_conventional') else ['MedSAM3']
    for model in models:
        keys = [k for k in conditions if k[:2] == (campaign, model)]
        assert all(conditions[k]['balanced'] for k in keys)
        base = next(k for k in keys if k[2] in {'dice', 'baseline'})
        best = max(keys, key=lambda k: conditions[k]['foreground_dice'])
        assert best == max(keys, key=lambda k: conditions[k]['dice_including_background'])
        assert conditions[best]['background_dice'] > conditions[base]['background_dice']
        bases[(campaign, model)] = base
        selected[(campaign, model)] = best
        base_cells = {(r['fold'], r['seed']): r for r in grouped[base]}
        for key in keys:
            if key == base:
                continue
            changes = []
            for r in grouped[key]:
                b = base_cells[(r['fold'], r['seed'])]
                change = {m: r[m] - b[m] for m in METRICS}
                changes.append(change)
                if change['foreground_dice'] * change['dice_including_background'] < 0:
                    flips.append({'campaign': campaign, 'model': model, 'arm': key[2],
                                  'fold': r['fold'], 'seed': r['seed'],
                                  **{m: v * 100 for m, v in change.items()}})
            delta = {m: (conditions[key][m] - conditions[base][m]) * 100 for m in METRICS}
            assert delta['foreground_dice'] * delta['dice_including_background'] > 0
            positive = {m: sum(x[m] > 0 for x in changes) for m in METRICS}
            if key == best:
                assert positive['foreground_dice'] == positive['dice_including_background']
                assert positive['foreground_dice'] == (8 if (campaign, model) ==
                       ('GeoFWI3D_conventional', 'unetr_vit_b16') else 9)
            comparisons.append({'campaign': campaign, 'model': model, 'arm': key[2],
                                **{'delta_' + m: v for m, v in delta.items()},
                                **{'positive_' + m: v for m, v in positive.items()}})
assert len(selected) == 14 and len(comparisons) == 180 and len(flips) == 6
assert sum(x['delta_foreground_dice'] > 0 for x in comparisons) == 169
assert all(x['campaign'] == 'GeoFWI3D_conventional' for x in flips)
assert max(abs(x['foreground_dice']) for x in flips) < .14
assert max(abs(x['dice_including_background']) for x in flips) < .03
medical_bands = [x for x in comparisons if x['campaign'] in
                 ['MSD_conventional', 'ADNI_conventional'] and x['arm'] == 'dice_bands']
assert len(medical_bands) == 8
assert all(x['positive_foreground_dice'] == x['positive_dice_including_background'] == 9
           for x in medical_bands)

table_records = []
def table_row(key, base, model=True):
    c = conditions[key]
    table_records.append({'campaign': key[0], 'model': key[1], 'arm': key[2],
                          **c, 'base_arm': base[2]})
    name = MODEL_NAMES[key[1]] if model else ''
    cells = [name, arm_label(key[2])] + [f'{100*c[m]:.2f}' for m in METRICS]
    cells += ['---' if key == base else f'{100*(c[m]-conditions[base][m]):+.2f}'
              for m in ['foreground_dice', 'dice_including_background']]
    return ' & '.join(cells) + r' \\'

TABLE_START = r'''\begin{table}[p]
\centering\small\setstretch{1.05}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}llrrrrr@{}}
\hline
Model & Objective & $D_{\mathrm{fg}}$ & $D_{\mathrm{bg}}$ & $D_{\mathrm{all}}$ & $\Delta_{\mathrm{fg}}$ & $\Delta_{\mathrm{all}}$ \\
\hline
'''
conv_rows = []
for campaign, name in [('GeoFWI3D_conventional', 'GeoFWI3D'),
                       ('MSD_conventional', 'MSD'), ('ADNI_conventional', 'ADNI')]:
    conv_rows.append(r'\multicolumn{7}{@{}l}{\textit{' + name + r'}}\\[.25em]')
    for model in MODELS:
        base = bases[(campaign, model)]
        conv_rows += [table_row(base, base), table_row(selected[(campaign, model)], base, False)]
    conv_rows.append(r'\hline')
conv_table = TABLE_START + '\n'.join(conv_rows) + r'''
\end{tabular}
\caption{Foreground, background and all-class Dice for the conventional
models. Scores are percentages; paired gains are percentage points.
Each baseline is followed by the configuration selected by its highest
foreground mean in the preceding results. Each row averages three seeds
within each of three folds and then weights folds equally. The selected
configuration is also highest under all-class Dice in every model/dataset
comparison. These tables show means; fold variation and the original
selection limitations remain as reported above. Gains use unrounded values.}
\label{tab:background-conventional}
\end{table}
'''
adapt_rows = []
for campaign, name in [('MSD_MedSAM3_extension', 'MSD: MedSAM3 extension'),
                       ('ADNI_MedSAM3', 'ADNI: MedSAM3')]:
    adapt_rows.append(r'\multicolumn{7}{@{}l}{\textit{' + name + r'}}\\[.25em]')
    base = bases[(campaign, 'MedSAM3')]
    adapt_rows += [table_row(base, base), table_row(selected[(campaign, 'MedSAM3')], base, False), r'\hline']
sam_keys = [('GeoFWI3D_SAM3', 'SAM3', arm) for arm in
            ['dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge', 'dice_bands_edge_sdf']]
assert {k for k in conditions if k[0] == 'GeoFWI3D_SAM3' and conditions[k]['balanced']} == set(sam_keys)
assert sum(conditions[k]['runs'] for k in conditions if k[0] == 'GeoFWI3D_SAM3') == 116
adapt_rows.append(r'\multicolumn{7}{@{}l}{\textit{GeoFWI3D: SAM3, complete conditions in the recovery snapshot}}\\[.25em]')
for k in sam_keys:
    adapt_rows.append(table_row(k, sam_keys[0], k == sam_keys[0]))
adapt_table = TABLE_START + '\n'.join(adapt_rows) + r'''
\hline
\end{tabular}
\caption{Complementary Dice measures for pretrained adaptation. MedSAM3
rows compare the matched baseline with the previously selected
configuration; the MSD baseline belongs to the extension campaign.
The SAM3 block includes every objective with all nine evaluations in the
23 October recovery snapshot, including negative effects. Its rows do
not establish the highest configuration of the unfinished full inventory.
All rows use three folds and three seeds, the same per-volume averaging
as Table~\ref{tab:background-conventional}, percentage scores and
percentage-point gains. The SAM3 snapshot is separate from the final
fourteen-objective aggregates reserved in Section~\ref{sec:sam3-results}.}
\label{tab:background-adaptation}
\end{table}
'''

section = r'''% BEGIN BACKGROUND DICE ANALYSIS -- 23 October 2026
% Source: reports/background_dice_archive_recovery_20261023/
% Generated and checked by reports/thesis_background_dice_update_20261023/integrate.py.
\FloatBarrier
\subsection{Foreground, Background and All-Class Dice}
\label{sec:background-dice-results}

The primary Dice scores emphasize the labelled target structures.
Evaluating background separately complements them by checking whether
foreground gains accompany improved agreement in the surrounding region.
The recovered per-volume confusion matrices permit both measurements
on the same saved predictions, without further training or inference.
Write $D_{i,\mathrm{fg}}$ for the primary volume score and
$D_{i,\mathrm{bg}}=D_{i,0}$ for background Dice. Equal weighting of
all classes gives
\begin{equation}
D_{i,\mathrm{all}}=
\begin{cases}
\displaystyle\frac{2D_{i,\mathrm{fg}}+D_{i,\mathrm{bg}}}{3},
 & \text{MSD},\\[.6em]
\displaystyle\frac{D_{i,\mathrm{fg}}+D_{i,\mathrm{bg}}}{2},
 & \text{ADNI and GeoFWI3D}.
\end{cases}
\label{eq:reported-all-class-dice}
\end{equation}
MSD retains separate anterior and posterior classes. The class scores
are calculated within each volume, followed by equal-volume, seed and
fold reductions as in Section~\ref{sec:evaluation-methods}; confusion
counts are not pooled across volumes to form a substitute Dice score.

The complete comparison covers all 504 conventional runs on each
dataset, the 108-run MSD MedSAM3 extension and all 126 ADNI MedSAM3
runs. The original 27 MSD MedSAM3 runs retain their completed
foreground results, but their per-volume records were not located.
Their background and all-class Dice therefore remain unreported, and
the extension baseline is not substituted for their original control.

Tables~\ref{tab:background-conventional} and
\ref{tab:background-adaptation} place the three measures alongside
one another. The conventional and MedSAM3 comparisons retain the
configurations already selected by foreground Dice, allowing the
effect of changing the evaluation measure to be examined directly.
''' + conv_table + '\n' + adapt_table + r'''
Every one of these fourteen selected configurations improves both
mean foreground and mean background Dice. Background means in their
baseline and selected rows range from 95.24\% to 99.76\%, leaving
less room for improvement than the foreground scores. For example,
ADNI MedSAM3 changes from 77.32\% to 80.85\% foreground Dice and
from 99.10\% to 99.19\% background Dice. Its all-class score rises
from 88.21\% to 90.02\%. This supports an improvement in agreement
across both regions, while the foreground measure retains sensitivity
to the target structure.

The higher absolute all-class scores result from the added contribution
of background. Their gains are smaller because background changes
relatively little: its inclusion reduces the foreground weight to
two-thirds on MSD and one-half on the binary tasks. The selected
MSD SwinUNETR gain changes from 11.97 to 8.27 points; the ADNI
MedSAM3 gain changes from 3.53 to 1.81 points. Both quantities describe
the same predictions under different class weights.

The conclusions also hold beyond the selected configurations. Across
all 180 complete constrained-objective comparisons with their matched
baselines, the same 169 mean effects are positive and the same eleven
are negative under foreground and all-class Dice. The highest-mean
objective is unchanged in all fourteen model/dataset comparisons.
The selected configurations retain their positive-pair counts: nine
of nine in thirteen comparisons and eight of nine for geological
UNETR. Bands alone still improve all 72 conventional medical
model/fold/seed cells under both measures.

This agreement does not imply identical ordering of every outcome.
Six of 1,620 individual paired differences change sign, all on
GeoFWI3D; their foreground magnitudes are below 0.14 points and
their all-class magnitudes below 0.03 points. Some secondary objective
rankings also change. These small differences leave the reported
mean benefits and negative findings intact, without establishing
statistical superiority between closely ranked objectives.

The SAM3 archive snapshot contains 116 of the planned 126 evaluations.
Only its six fully populated objective conditions, including baseline,
enter Table~\ref{tab:background-adaptation}; each has all three folds
and three seeds. Within this balanced subset, bands retain a positive
mean effect of 1.48 foreground points and 0.79 all-class points,
whereas $B+E+S$ retains a negative effect of 1.85 and 0.99 points,
respectively. The remaining objective conditions require their final
records before a complete comparison or preferred configuration can
be reported. This subset is excluded from the 180-comparison check.

Together, the measures show that the selected foreground improvements
persist when background participates in the class average. They provide
a broader assessment of overlap while preserving the original
foreground endpoint. This is a sensitivity analysis of the metric on
the same experiments, so it retains their selection and replication
limitations and supplies no additional evidence of surface accuracy
or downstream usefulness.
\FloatBarrier
% END BACKGROUND DICE ANALYSIS

'''

original = THESIS.read_text()
assert '% BEGIN BACKGROUND DICE ANALYSIS' not in original, 'Analysis already inserted.'
edited = original
def replace_once(old, new):
    global edited
    assert edited.count(old) == 1, ('ambiguous source match', old[:90])
    edited = edited.replace(old, new, 1)

replace_once(r'\subsection{What the Annotation Audit Establishes}',
             section + r'\subsection{What the Annotation Audit Establishes}')
replace_once('Background is excluded from these primary metrics. Volumes receive\nequal weight in a run mean.',
             'Background is excluded from these primary metrics. Volumes receive\n'
             'equal weight in a run mean. Section~\\ref{sec:background-dice-results}\n'
             'adds background and all-class Dice as complementary measures on\n'
             'the same predictions, retaining this order of averaging.')
replace_once('MSD, ADNI, MedSAM3, SAM3 and the MSD annotation audit. Numerical SAM3',
             'MSD, ADNI, MedSAM3 and SAM3. A complementary analysis compares\n'
             'foreground, background and all-class Dice before the MSD annotation\n'
             'audit. Numerical SAM3')
replace_once('they are not estimates from an incomplete subset.\nEvery objective',
             'they are not estimates from an incomplete subset. The metric-sensitivity\n'
             'analysis identifies its separate SAM3 snapshot and uses only fully\n'
             'populated conditions within it. Every objective')
replace_once('FN/FP counts. The complete 504-cell Dice inventory is available,\n'
             'but nine reused fold-0 baseline raw archives are absent from the\n'
             'local extraction. Consequently, complete nine-cell baseline FN/FP\n'
             'aggregates are not reported here, and no full-matrix baseline-relative\n'
             'error reduction is inferred from the incomplete counts. This is an\n'
             'error-analysis evidence gap, not an unfinished segmentation run.',
             'FN/FP counts. The appendix retains the error-count coverage of the\n'
             '22 October extraction. The nine reused fold-0 baseline evaluation\n'
             'records have since been recovered, completing the confusion-matrix\n'
             'evidence used in Section~\\ref{sec:background-dice-results}. Full-matrix\n'
             'baseline-relative FN/FP summaries remain to be incorporated into\n'
             'the appendix; all 504 final Dice results are already included.')
replace_once('Further analysis of the completed studies should recover the nine reused\n'
             'GeoFWI3D baseline raw archives for full FN/FP comparisons, add paired',
             'Further analysis of the completed studies should extend the geological\n'
             'FN/FP comparisons using the recovered baseline records, add paired')
replace_once('Dashes identify unavailable\ncomplete baseline error aggregates, not missing Dice results.',
             'Dashes retain the baseline error aggregates omitted from the\n'
             '22 October extraction. The subsequent recovery is documented in\n'
             'Section~\\ref{sec:background-dice-evidence}; all Dice results are complete.')
replace_once('Nine reused fold-0 baselines are available as saved summary references\n'
             'with source/archive identities, but their raw archives are not present\n'
             'locally. Two additional baseline extracts contain only part of their\n'
             'manifest-listed metadata. The audit records these limits and does not\n'
             'claim a new complete payload check of every baseline or of deleted\n'
             'checkpoint tensors. Complete baseline FN/FP aggregates remain a\n'
             'reporting gap; all 504 final Dice values and their paired gains are\n'
             'included.',
             'At that extraction, nine reused fold-0 baselines were represented\n'
             'only by saved summary references and two additional baseline extracts\n'
             'contained part of their manifest-listed metadata. The subsequent\n'
             'recovery described in Section~\\ref{sec:background-dice-evidence}\n'
             'supplies the nine missing evaluation records and checks all 504\n'
             'evaluation hashes. It does not reload deleted checkpoint tensors\n'
             'or establish completeness of every other archived payload. The\n'
             'appendix retains the earlier FN/FP coverage pending integration\n'
             'of the recovered baseline counts.')
evidence = r'''\subsection{Background-Dice Recovery}
\label{sec:background-dice-evidence}

The complementary analysis in Section~\ref{sec:background-dice-results}
uses \path{reports/background_dice_archive_recovery_20261023/}.
The retained inventory contains 1,862 run evaluations and 124,860
validation-volume records: 504 runs for each conventional dataset,
108 for the MSD MedSAM3 extension, 126 for ADNI MedSAM3 and 116
SAM3 evaluations in the recovery snapshot. Each evaluation matches
its manifest SHA-256 and reproduces its recorded foreground run mean.
Per-volume foreground and background scores are recomputed from the
saved confusion matrices; the all-class score follows
Equation~\ref{eq:reported-all-class-dice}. These checks use evaluation
metadata rather than rerunning models.

The directory preserves \path{background_dice_cases.csv},
\path{background_dice_runs.csv}, \path{background_dice_conditions.csv}
and \path{background_metric_comparison.csv}, together with verification
records, download receipts and the recovery script. The thesis tables
are generated and checked in
\path{reports/thesis_background_dice_update_20261023/}.
The original MSD MedSAM3 campaign remains separate: its aggregate
foreground results are preserved, but the searched archives did not
yield its 27 per-volume evaluation records. Its missing complementary
scores are not inferred from the later campaign.

'''
replace_once(r'\subsection{SAM3 Campaign}', evidence + r'\subsection{SAM3 Campaign}')
replace_once('% Existing dataset illustrations and annotation-compatibility audit are retained.',
             '% Background/all-class Dice integrated 23 October 2026 from verified evaluation archives.\n'
             '% See reports/thesis_background_dice_update_20261023/ for numeric checks and patch.\n'
             '% Existing dataset illustrations and annotation-compatibility audit are retained.')

with (HERE / 'table_values.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(table_records[0]))
    writer.writeheader()
    writer.writerows(table_records)
audit = {'source_before_sha256': hashlib.sha256(original.encode()).hexdigest(),
         'source_after_sha256': hashlib.sha256(edited.encode()).hexdigest(),
         'input_sha256': {n: sha(DATA / n) for n in
                          ['background_dice_runs.csv', 'background_dice_conditions.csv']},
         'complete_model_dataset_comparisons': len(selected),
         'constrained_objective_comparisons': len(comparisons),
         'positive_means_under_both_metrics': 169,
         'negative_means_under_both_metrics': 11,
         'individual_sign_changes': flips,
         'selected_rankings_unchanged': True,
         'selected_background_means_improve': True,
         'sam3_snapshot_evaluations': 116,
         'sam3_complete_conditions_including_baseline': 6,
         'all_class_identity_checked_per_run': len(rows)}
(HERE / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n')
(HERE / 'change.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True), edited.splitlines(True),
    fromfile=str(THESIS), tofile=str(THESIS))))
assert THESIS.read_text() == original, 'Source changed during preparation.'
THESIS.write_text(edited)
print(json.dumps({'edited': str(THESIS), 'table_rows': len(table_records),
                  'comparison_checks': len(comparisons), 'added_lines':
                  len(edited.splitlines()) - len(original.splitlines())}, indent=2))
