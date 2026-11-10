"""Integrate verified final SAM3 results in the existing thesis, in place."""
from pathlib import Path
import difflib
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
THESIS = ROOT / 'Template_tesi/thesis-revision.tex'
rows = json.loads((OUT / 'sam3_conditions.json').read_text())
by_arm = {r['arm']:r for r in rows}
base = by_arm['dice']
original = THESIS.read_text()
text = original

def replace(old, new):
    global text
    assert text.count(old) == 1, (text.count(old),old[:140])
    text = text.replace(old,new,1)

def between(start, end, new):
    global text
    a=text.index(start); b=text.index(end,a)
    text=text[:a]+new+text[b:]

def label(arm):
    if arm == 'dice': return 'Baseline'
    names={'bands':'B','edge':'E','sdf':'S','depth':'D'}
    return 'Baseline $+' + '+'.join(names[p] for p in arm.split('_')[1:]) + '$'

def pm(row,metric,scale=100,signed=False):
    value=format(row[metric]*scale, '+.2f' if signed else '.2f')
    return value + r' $\pm$ ' + f'{row[metric+"_fold_sd"]*scale:.2f}'

def table_rows(label_name, new):
    global text
    target=text.index('\\label{'+label_name+'}')
    start=max(text.rfind('\\begin{table}',0,target),text.rfind('\\begin{longtable}',0,target))
    end=text.index('\\end{table}',target) if '\\begin{table}' in text[start:start+15] else text.index('\\end{longtable}',target)
    part=text[start:end]
    lines=part.splitlines()
    indexes=[i for i,line in enumerate(lines) if line.startswith('Baseline')]
    assert len(indexes)==14, (label_name,len(indexes))
    assert indexes==list(range(indexes[0],indexes[-1]+1))
    lines[indexes[0]:indexes[-1]+1]=new
    text=text[:start]+'\n'.join(lines)+'\n'+text[end:]

replace('% SAM3: full study structure integrated 23 October 2026.\n% Numerical results and empirical interpretations remain explicit placeholders.\n% Progress counts are recorded in reports/thesis_sam3_update_20261023/.\n% Revision provenance: reports/thesis_sam3_structure_20261023/.',
r'''% SAM3: all 126 final runs verified and integrated 23 October 2026.
% Scores, fold effects, FN/FP and complementary Dice: reports/thesis_sam3_final_20261023/.
% Earlier structure and progress reports remain dated provenance snapshots.''')
text = re.sub(r'%[^\n]*drafting markers; pending cells are not measured values\.\n', '', text)
replace('\\newcommand{\\samPending}{\\textit{pending}}\n\\newcommand{\\samResult}[1]{\\textit{[SAM3: #1 to insert]}}',
        '% SAM3 numerical results are complete; remaining presentation gaps are marked locally.')
replace(r'''and 25 updates. \samResult{final Dice, paired change in percentage
points and consistency across the nine comparisons}.''',r'''and 25 updates. Its highest-mean configuration reaches 72.76\% Dice,
improving the matched baseline by 3.45 percentage points, with positive
changes in eight of nine comparisons.''')
replace('The experimental design comprises six studies: three conventional dataset\ncomparisons, two MedSAM3 adaptations and one SAM3 adaptation. The analysis',
r'''The experimental evaluation comprises six completed studies: three conventional
dataset comparisons, two MedSAM3 adaptations and one SAM3 adaptation.
Each includes the full fourteen-objective comparison. The analysis''')
replace('14 objectives; 126 planned runs; 200 updates/run.', '14 objectives; 126 runs; 200 updates/run.')
replace('Study design: 1,890 planned runs in 210 objective conditions,\ncomprising 630 geological and 1,260 medical runs. These are design totals;\nSAM3 numerical entries awaiting final aggregation are marked explicitly.',
r'''Completed study design: 1,890 objective runs in 210 conditions,
comprising 630 geological and 1,260 medical runs.''')
replace('Training uses NVIDIA A100 GPUs. Resource measurements record the GPU\nallocation, timing scope and recovery attempts.',
r'''The completed inventory contains 76 runs on Colab A100-SXM4-80GB
GPUs and 50 on A100 80GB PCIe GPUs allocated as 4g.40gb MIG instances.
All runs retain the same scientific source identity. Archived hashes
verify unchanged base parameters and matching initial adapters, support
arrays and sampling schedules within every fold/seed comparison.
Hardware allocation was not a fully crossed factor; timing comparisons
must retain their execution platform and measurement scope.
Resource records also distinguish recovery attempts from the final
25-update trajectory.''')
replace('The six-study design covers 1,890 planned objective runs:', 'The six-study design covers 1,890 completed objective runs:')
replace(r'''audit. Numerical SAM3
cells marked \emph{pending} reserve the final balanced aggregates;
they are not estimates from an incomplete subset. The metric-sensitivity
and error analyses identify their separate SAM3 archive snapshot and
use only fully populated conditions within it. Every objective is evaluated against a matched final-budget baseline.''',
r'''audit. All SAM3 tables use the complete 126-run inventory, including
the error and complementary Dice analyses. Every objective is evaluated
against a matched final-budget baseline.''')

quality=[]
for r in rows:
    quality.append(' & '.join([label(r['arm']),pm(r,'foreground_dice'),
        '0.00' if r['arm']=='dice' else pm(r,'delta',signed=True),
        '---' if r['arm']=='dice' else f"{r['positive']}/9",pm(r,'training_dice')])+r' \\')
table_rows('tab:sam3-geo-results',quality)
replace('three-seed condition. Pending entries await final verified aggregation;\nbaseline changes are zero by definition.',
        'three-seed condition, verified against the final archives;\nbaseline changes are zero by definition.')
between('The adapted baseline reaches \\samResult', 'Full-volume errors for all SAM3 variants',r'''The adapted baseline reaches 69.31\% validation Dice and 74.97\%
support Dice. Adding bands, edges and depth supervision gives the
highest validation mean, 72.76\%, with a paired gain of
$3.45\pm2.43$ percentage points across the three fold means.
The gain is positive in eight of nine runs. Its fold-mean changes are
+6.19, +2.57 and +1.58 points for folds 0, 1 and 3, respectively
(Table~\ref{tab:sam3-geo-folds}). Thus the mean improvement occurs
in every support partition, although its magnitude varies and the
fold-0, seed-2 run loses 2.74 points. The selected configuration's
support Dice is 77.41\%; selection by the highest validation mean
remains exploratory.

Ten of the thirteen constrained variants improve mean validation Dice.
Depth supervision alone gives a 2.98-point gain, positive in eight
runs, while bands alone give 1.48 points, positive in six. The
$B+D$ and $B+S+D$ combinations also exceed the baseline by more
than 2.8 points. These results associate several of the larger gains
with depth supervision under the present calibration and schedule;
they do not isolate the contribution of depth weighting from that
of the added classification loss.

Combining more terms does not guarantee a larger benefit. Adding
signed distance to $B+E+D$ lowers mean Dice from 72.76\% to
70.95\%. Signed distance alone and $E+D$ give small negative
mean changes, while $B+E+S$ loses 1.85 points and improves only
three runs. The small positive means of $B+S$ and $B+E$ also
occur in only three runs each, illustrating why an average gain
should be read alongside its fold decomposition and positive-pair count.

''')
replace(r'''\noindent\samResult{paired baseline/constrained prediction panels,
selected by a stated rule, illustrating the measured FN/FP pattern}.''',
r'''\noindent\textit{[Presentation to insert: paired SAM3 baseline/constrained
prediction panels selected by a stated rule. The numerical comparison
is complete; these spatial examples require attributable retained outputs.]}''')
replace(r'''\samResult{cross-study interpretation of the verified paired effects,
including any differences from the medical adaptation results}.''',
r'''As in the medical adaptation studies, spatial supervision can improve
the short-budget baseline. However, the preferred combination differs:
$B+E+D$ for SAM3, compared with $B+E+S+D$ on MSD and $E+S+D$
on ADNI for MedSAM3. This supports testing the constraint combination
within each learning setting rather than assuming one combination
transfers unchanged across tasks.''')
replace('Ordering uses unrounded\nmeans; pending conditions appear last.', 'Ordering uses unrounded\nmeans.')
between('Table~\\ref{tab:sam3-geo-errors} uses the verified archive snapshot', '\\begingroup\\footnotesize',r'''Table~\ref{tab:sam3-geo-errors} uses all 126 final evaluations,
with nine runs for every objective. Counts are reconstructed from the
same per-volume confusion matrices used for the Dice comparison.

''')
replace(r'''Each run averages errors over 96 held-out volumes; complete rows report mean $\pm$ sample SD across three seed-averaged fold means. Runs records coverage in the stated archive snapshot. Pending rows await complete nine-run evidence. Within each model, rows ascend by mean total error (FN $+$ FP). Pending conditions follow the complete rows.''',
r'''Each run averages errors over 96 held-out volumes; rows report mean $\pm$ sample SD across three seed-averaged fold means. All rows contain nine runs. Rows ascend by mean total error (FN $+$ FP).''')
error_rows=[' & '.join([label(r['arm']),pm(r,'fn',.001),pm(r,'fp',.001),'9/9'])+r' \\' for r in sorted(rows,key=lambda r:r['fn']+r['fp'])]
table_rows('tab:sam3-geo-errors',error_rows)
replace(r'''\noindent\samResult{interpret the FN/FP changes of the final selected
SAM3 objective against its matched baseline after the complete
nine-run conditions have been verified; distinguish fewer misses,
less foreground extension and any tradeoff}.''',r'''The baseline makes substantially more foreground misses than false
positive assignments. The highest-Dice variant, $B+E+D$, reduces
mean FN from 34.16 to 27.16 thousand voxels per volume, a 20.5\%
decrease, while FP rises from 3.70 to 6.99 thousand. Its improvement
therefore reflects greater recovery of salt at the cost of additional
foreground extension. Total incorrectly labelled voxels fall by 9.8\%.
The increased background Dice reported in
Section~\ref{sec:background-dice-results} is compatible with this
tradeoff: foreground misses are false positives for background,
so background overlap depends on both error directions.

By contrast, $B+E+S$ increases both mean FN and mean FP relative
to baseline, consistent with its negative Dice change. The error
counts describe the amount of misclassification over the full volume;
they do not establish whether the remaining errors are concentrated
on particular surfaces or connected components.''')

replace('dataset, 108 MSD MedSAM3 run records and all 126 ADNI MedSAM3 runs.',
        'dataset, 108 MSD MedSAM3 run records, all 126 ADNI MedSAM3 runs\nand all 126 SAM3 runs.')
bgstart=r'\multicolumn{7}{@{}l}{\textit{GeoFWI3D: SAM3, complete conditions in the recovery snapshot}}'
start=text.index(bgstart);end=text.index('\\hline\n\\end{tabular}',start)
bgrows=[r'\multicolumn{7}{@{}l}{\textit{GeoFWI3D: SAM3, all fourteen objectives}}\\[.25em]']
for r in rows:
    bgrows.append(' & '.join(['SAM3' if r['arm']=='dice' else '',label(r['arm']),
        *[f'{r[m]*100:.2f}' for m in ['foreground_dice','background_dice','dice_including_background']],
        *['---' if r['arm']=='dice' else f'{100*(r[m]-base[m]):+.2f}' for m in ['foreground_dice','dice_including_background']]])+r' \\')
text=text[:start]+'\n'.join(bgrows)+'\n'+text[end:]
replace(r'''The SAM3 block includes every objective with all nine evaluations in the
23 October recovery snapshot, including negative effects. Its rows do
not establish the highest configuration of the unfinished full inventory.
All rows use three folds and three seeds, the same per-volume averaging
as Table~\ref{tab:background-conventional}, percentage scores and
percentage-point gains. The SAM3 snapshot is separate from the final
fourteen-objective aggregates reserved in Section~\ref{sec:sam3-results}.''',
r'''The SAM3 block includes all fourteen completed objective conditions,
including negative effects; $B+E+D$ has the highest foreground and
all-class mean. All rows use three folds and three seeds, the same
per-volume averaging as Table~\ref{tab:background-conventional},
percentage scores and percentage-point gains.''')
replace('Every one of these fourteen selected configurations improves both',
        'Every one of the fifteen foreground-selected configurations improves both')
replace(r'''all 180 complete constrained-objective comparisons with their matched
baselines, the same 169 mean effects are positive and the same eleven
are negative under foreground and all-class Dice. The highest-mean
objective is unchanged in all fourteen model/dataset comparisons.
The selected configurations retain their positive-pair counts: nine
of nine in thirteen comparisons and eight of nine for geological
UNETR.''',
r'''all 193 complete constrained-objective comparisons with recoverable
background scores, the same 179 mean effects are positive and the
same fourteen are negative under foreground and all-class Dice.
The highest-mean objective is unchanged in all fifteen model/dataset
comparisons. The selected configurations retain their positive-pair
counts: nine of nine in thirteen comparisons and eight of nine for
geological UNETR and SAM3.''')
replace('Six of 1,620 individual paired differences change sign, all on',
        'Six of 1,737 individual paired differences change sign, all on')
between('The SAM3 archive snapshot contains 116', 'Together, the measures show',r'''For SAM3, the selected $B+E+D$ variant raises background Dice
from 97.62\% to 97.84\% and all-class Dice from 83.46\% to
85.30\%. Its 3.45-point foreground gain becomes a 1.84-point
all-class gain. All thirteen SAM3 mean effects retain their sign,
and no individual SAM3 paired effect changes sign when background
is included. In particular, the negative $B+E+S$ finding persists
under both measures. These checks use the complete campaign.

''')
replace('% SAM3 full-study templates await verified final numerical aggregates.\n% Conventional geological results remain complete; no partial SAM3 means inserted.',
        '% SAM3 final 126-run aggregates verified in reports/thesis_sam3_final_20261023/.\n% All conventional and adaptation objective inventories are complete.')
replace('extractions. SAM3 placeholders retain the stated reporting conventions\nfor final insertion.',
        'extractions. SAM3 includes all fourteen final conditions, with the\nsame seed-within-fold reduction used in the main results.')
foldrows=[' & '.join([label(r['arm']),*[f'{r["foreground_dice_fold_means"][f]*100:.2f}' for f in ['0','1','3']],
    *['0.00' if r['arm']=='dice' else f'{r["delta_fold_means"][f]*100:+.2f}' for f in ['0','1','3']]])+r' \\' for r in rows]
table_rows('tab:sam3-geo-folds',foldrows)
replace(r'''use the corresponding baseline within each fold. Pending cells reserve
complete three-seed reductions; they are not partial means. All fourteen
objectives appear regardless of the direction of their final changes.''',
r'''use the corresponding baseline within each fold. All 126 final runs
are included, and every objective is shown regardless of the direction
of its change.''')

between('The geological adaptation campaign is defined in', '\\subsection{MedSAM3 Campaigns}',r'''The geological adaptation campaign is defined in
\path{reports/sam3_geofwi_f013_s012_126_20261019/bundle/}.
Its immutable plan assigns 126 distinct tasks to fourteen objectives,
folds 0, 1 and 3, and seeds 0, 1 and 2. All tasks are fresh runs;
historical pilots serve only as protocol references. The final
collection is preserved in the campaign's
\path{split_cluster_20261022/final_results_20261023/} directory.

The independent thesis extraction in
\path{reports/thesis_sam3_final_20261023/} reconciles every planned
identifier with its final result and verifies 2,252 manifest-listed
metadata payload hashes. All required evaluation, completion, protocol,
training-summary, sampling and support-inventory records are available.
The 76 Colab ZIP archives also match their recovery-receipt hashes;
the 50 cluster runs retain archive acknowledgements. Each final run
records 25 updates and unchanged frozen-parameter hashes. Initial
adapter, base, support-array and sampling-schedule identities agree
across all fourteen arms within each of the nine fold/seed cells.

The extraction recomputes 12,726 volume records from their confusion
matrices: 630 support and 12,096 validation evaluations. Per-volume
class Dice, run-level support/validation means and full-validation
FN/FP reproduce the archived values. The fourteen objective means and
positive-pair counts also reproduce the final campaign summary.
Dispersion is recomputed across the three seed-averaged fold means,
consistent with the thesis; the upstream summary's standard deviation
instead describes the nine individual runs.

All 126 records retain the same scientific source identity. Hardware
records distinguish 76 full A100-SXM4-80GB allocations from 50 A100
PCIe MIG 4g.40gb allocations. Recorded training-loop elapsed time is
not an end-to-end adaptation measurement: initial setup, shared
calibration, evaluation and unretained work before recovery require
separate accounting. The final ledger records upper bounds of five
repeated update attempts for fold 0, seed 1, $B+E+S$, and one for
fold 1, seed 2, $B+E+S$. These costs are additional to the final
25-update trajectories and are not counted as extra scientific runs.

''')
between('The retained inventory contains 1,862', 'The directory preserves \\path{background_dice_cases.csv}',r'''The original recovery inventory contains 1,862 run evaluations and
124,860 validation-volume records. The final SAM3 supplement in
\path{reports/thesis_sam3_final_20261023/} replaces its 116-record
SAM3 snapshot with all 126 final evaluations. Together they cover
1,872 run evaluations and 125,820 validation-volume records:
504 runs for each conventional dataset, 108 for MSD MedSAM3,
126 for ADNI MedSAM3 and 126 for SAM3. Evaluation payloads match
their manifest hashes and reproduce their recorded foreground means.
Per-volume foreground and background scores are recomputed from the
saved confusion matrices; the all-class score follows
Equation~\ref{eq:reported-all-class-dice}. These checks use evaluation
metadata rather than rerunning models.

''')
replace(r'''are generated and checked in
\path{reports/thesis_background_dice_update_20261023/}.''',
r'''are generated and checked in
\path{reports/thesis_background_dice_update_20261023/}, with the
completed SAM3 rows and the 193-comparison sensitivity check updated in
\path{reports/thesis_sam3_final_20261023/}.''')
replace(r'''MedSAM3 evaluations and the 116-record SAM3 snapshot. It produces
211 condition rows, including the two applicable MSD baseline rows.
Of these, 203 have all nine fold/seed cells; the other eight are SAM3
conditions whose aggregate errors remain pending. Every complete row
uses the declared fold and seed inventory.''',
r'''MedSAM3 evaluations and the original 116-record SAM3 snapshot.
The final SAM3 extraction in \path{reports/thesis_sam3_final_20261023/}
replaces that snapshot with 126 evaluations and recomputes all
fourteen SAM3 error rows. The combined tables therefore contain
211 complete condition rows, including the two applicable MSD baseline
rows. Every row uses its declared nine-cell fold and seed inventory.''')
replace(r'''SAM3 numerical and interpretation fields are marked explicitly in the
abstract, Section~\ref{sec:sam3-results} and its fold table. They require
the final verified extraction and do not imply measured outcomes.
The geological four-model matrix and its FN/FP aggregates are complete. Presentation''',
r'''The SAM3 numerical inventory, fold decomposition, full-volume FN/FP
and foreground/background/all-class Dice analyses are complete.
The geological four-model matrix and its FN/FP aggregates are also
complete. Presentation''')
replace('All four medical campaigns above are complete within their declared\ninventories.',
        'All six studies are complete within their declared objective inventories.')

assert '\\samPending' not in text and '\\samResult' not in text
assert THESIS.read_text()==original, 'Source changed during integration; reread before editing.'
patch=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='thesis-revision.tex (before)',tofile='thesis-revision.tex (after)'))
(OUT/'change.patch').write_text(patch)
(OUT/'source_before.tex').write_text(original)
THESIS.write_text(text)
(OUT/'integration.json').write_text(json.dumps({'before_sha256':hashlib.sha256(original.encode()).hexdigest(),
    'after_sha256':hashlib.sha256(text.encode()).hexdigest(), 'source':str(THESIS),
    'sam3_tables_updated':['quality','errors','folds','background/all-class'],
    'numerical_placeholders_remaining':0},indent=2)+'\n')
print('Updated existing thesis; saved patch and audit in',OUT)
