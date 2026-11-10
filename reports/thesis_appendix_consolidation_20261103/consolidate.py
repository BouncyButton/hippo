"""Reduce printed appendices and preserve their full evidence in a supplement.

The six aggregate Dice tables are retained byte-for-byte. No training,
inference, original aggregate or main-text result chart is changed.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import shutil
import statistics as stats
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROJECT = ROOT / 'Template_tesi'
SOURCE = PROJECT / 'thesis-revision.tex'
OLD = (HERE / 'before.tex').read_text()
APP = OLD[OLD.index('\\appendix'):]
SUPP = HERE / 'supplement'
SUPP.mkdir(exist_ok=True)
provenance = []


def read_json(path):
    return json.loads((ROOT / path).read_text())


def read_csv(path):
    with (ROOT / path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def copy_evidence(source, destination):
    source = ROOT / source
    dest = SUPP / destination
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    assert source.read_bytes() == dest.read_bytes()
    provenance.append(dict(file=destination, source=str(source.relative_to(ROOT)),
                           sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))


TABLES = re.findall(r'\\begin\{table\}.*?\\end\{table\}', APP, re.S)


def table(label):
    matches = [t for t in TABLES if r'\label{'+label+'}' in t]
    assert len(matches) == 1, (label, len(matches))
    return matches[0]


def make_table(spec, header, rows, caption, label):
    return (r'\begin{table}[htbp]'+'\n'+
            r'\centering\footnotesize\setstretch{1.05}'+'\n'+
            r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}'+'\n'+
            r'\begin{tabular}{@{}'+spec+r'@{}}'+'\n'+
            '\\hline\n'+header+r' \\'+'\n\\hline\n'+
            '\n'.join(rows)+'\n\\hline\n\\end{tabular}\n'+
            '\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n')


# Preserve the old appendix as a source fragment, not a replacement thesis.
(SUPP / 'archived-appendix.tex').write_text(
    '% Historical appendix fragment before consolidation; not a standalone document.\n'
    '% Its table labels and discussion are preserved for traceability, not inclusion.\n'
    + APP.replace(r'\end{document}', ''))

# Source extracts. The 1,269 medical records include nine extra matched controls.
for src, dst in [
    ('reports/thesis_noaug_update_20261021/all_1269_runs.csv', 'medical/all-runs-including-matched-controls.csv'),
    ('reports/thesis_noaug_update_20261021/all_141_condition_summaries.csv', 'medical/all-condition-summaries.csv'),
    ('reports/thesis_geofwi_update_20261022/all_504_runs.csv', 'geological/conventional-runs.csv'),
    ('reports/thesis_geofwi_update_20261022/all_56_condition_summaries.csv', 'geological/conventional-conditions.csv'),
    ('reports/thesis_sam3_final_20261023/sam3_runs.csv', 'geological/sam3-final-runs.csv'),
    ('reports/thesis_sam3_final_20261023/sam3_cases.csv', 'geological/sam3-final-cases.csv'),
    ('reports/thesis_qualitative_predictions_20261103/final_verification.json', 'qualitative/verification.json'),
    ('reports/thesis_qualitative_predictions_20261103/recovered/figures/summary.json', 'qualitative/case-selection.json'),
    ('reports/thesis_appendix_consolidation_20261103/figure_manifest.json', 'illustrations/figure-manifest.json'),
]:
    copy_evidence(src, dst)
for name in ['GeoFWI3D', 'MSD', 'ADNI']:
    copy_evidence(f'reports/thesis_qualitative_predictions_20261103/recovered/figures/{name}_all_cases.csv',
                  f'qualitative/{name}-all-cases.csv')
    copy_evidence(f'Template_tesi/figures/qualitative/{name.lower()}-extreme-predictions.png',
                  f'qualitative/{name}-adverse-and-largest-gain.png')
    copy_evidence(f'Template_tesi/figures/compiled/{name.lower()}-constraint-example.pdf',
                  f'illustrations/{name}-full-constraint-example.pdf')

sam = read_csv('reports/thesis_sam3_final_20261023/sam3_runs.csv')
assert len(sam) == 126 and len({r['run_id'] for r in sam}) == 126
errors = read_json('reports/thesis_all_variant_errors_20261023/condition_errors.json')
errors = [r for r in errors if r['campaign'] != 'GeoFWI3D_SAM3']
for arm in dict.fromkeys(r['arm'] for r in sam):
    group = [r for r in sam if r['arm'] == arm]
    assert len(group) == 9 and len({(r['fold'], r['seed']) for r in group}) == 9
    row = dict(campaign='GeoFWI3D_SAM3', model='SAM3',
               arm='baseline' if arm == 'dice' else arm.removeprefix('dice_'),
               runs=9, complete=True, missing_cells=[], unit='mean validation voxels per volume')
    for metric in ['fn', 'fp']:
        fold_means = {f: stats.mean(float(r[metric]) for r in group if r['fold'] == f)
                      for f in sorted({r['fold'] for r in group})}
        row.update({metric: stats.mean(fold_means.values()), metric+'_fold_sd': stats.stdev(fold_means.values()),
                    metric+'_fold_means': fold_means})
    row.update(swaps=None, swaps_fold_sd=None, swaps_fold_means=None)
    errors.append(row)
assert len(errors) == 211 and all(r['complete'] and r['runs'] == 9 for r in errors)
(SUPP / 'errors').mkdir(exist_ok=True)
(SUPP / 'errors/all-condition-errors.json').write_text(json.dumps(errors, indent=2)+'\n')
write_csv(SUPP / 'errors/all-condition-errors.csv', [
    {k:v for k,v in r.items() if not k.endswith('_fold_means') and k != 'missing_cells'} for r in errors])
# Preserve run errors, replacing the incomplete historical SAM3 snapshot.
run_errors = read_json('reports/thesis_all_variant_errors_20261023/run_errors.json')
assert isinstance(run_errors, list)
run_errors = [r for r in run_errors if r['campaign'] != 'GeoFWI3D_SAM3']
for r in sam:
    run_errors.append(dict(r, fn=float(r['fn']), fp=float(r['fp']), swaps=None,
                          source='geological/sam3-final-runs.csv'))
(SUPP / 'errors/all-run-errors.json').write_text(json.dumps(run_errors, indent=2)+'\n')

# Final complementary Dice inventory, excluding earlier incomplete SAM3 rows.
bg = read_csv('reports/background_dice_archive_recovery_20261023/background_dice_runs.csv')
bg = [r for r in bg if r['campaign'] != 'GeoFWI3D_SAM3']
bg.extend(sam)
assert len(bg) == 1872
write_csv(SUPP / 'complementary-dice/all-runs.csv', bg)
conditions = []
for key in dict.fromkeys((r['campaign'], r['model'], r['arm']) for r in bg):
    group = [r for r in bg if (r['campaign'],r['model'],r['arm']) == key]
    assert len(group) == 9
    row = dict(zip(['campaign','model','arm'],key), runs=9)
    for metric in ['foreground_dice', 'background_dice', 'dice_including_background']:
        means = [stats.mean(float(r[metric]) for r in group if r['fold'] == f)
                 for f in sorted({r['fold'] for r in group})]
        row[metric] = stats.mean(means)
        row[metric+'_fold_sd'] = stats.stdev(means)
    conditions.append(row)
write_csv(SUPP / 'complementary-dice/all-conditions.csv', conditions)

technical = (HERE / 'technical_details.tex').read_text()
technical = re.sub(r'@@TABLE:([^@]+)@@', lambda m: table(m[1]), technical)

aggregate_labels = ['tab:geo-matrix', 'tab:msd-noaug-matrix', 'tab:adni-noaug-matrix',
                    'tab:sam3-geo-results', 'tab:medsam-msd-noaug', 'tab:medsam-adni-noaug']
experimental = r'''
\section{Supplementary Experimental Results}
\label{sec:extended-results}

These tables retain the complete objective comparisons and the checks
needed to assess the main results. Interpretation is given in Chapter~5.
Run-level records, full error counts and fold decompositions are in the
electronic supplement (Appendix~\ref{sec:evidence-register}).

\subsection{Complete Aggregate Dice Comparisons}
\label{sec:complete-result-tables}\label{sec:secondary-objective-results}

The six tables below include every planned objective, including negative
results. $B$, $E$, $S$ and $D$ denote boundary bands, edge consistency,
signed distance and depth weighting; every row also includes its native
supervised baseline. Scores are percentages and gains are percentage
points. Each condition has three folds and three seeds. Means average
seeds within each fold and then weight folds equally; dispersion is the
sample standard deviation of the three fold means. Highest means are
descriptive selections on the reported validation data, not independent
confirmation that one objective is superior to a close alternative.

MSD MedSAM3 preserves the original control for $B$ and $B+E$
($74.93\pm2.18$) and the separate control for the other eleven
constrained objectives ($76.34\pm1.64$). Both execution batches are
complete; their controls are not interchanged or counted as additional
scientific objectives.

'''
experimental += '\n\n'.join(table(label) for label in aggregate_labels)

# Selected configurations are already chosen in the main results.
models = [('nnunet','nnU-Net'),('swinunetr','SwinUNETR'),('unetr_vit_b16','UNETR'),('basic_unet16','3D U-Net')]
selected = []
rep_rows = []
for dataset, label in [('GeoFWI3D','tab:geo-best'),('MSD','tab:msd-noaug-best'),('ADNI','tab:adni-noaug-best')]:
    rep_rows.append(r'\multicolumn{6}{@{}l}{\textit{'+dataset+r'}}\\[.25em]')
    for key, name in models:
        line = next(l for l in table(label).splitlines() if l.startswith(name+' &'))
        cells = [c.strip().rstrip('\\').strip() for c in line.split('&')]
        letters = re.search(r'\$\+([^$]+)\$',cells[1])[1].split('+')
        arm = '_'.join({'B':'bands','E':'edge','S':'sdf','D':'depth'}[x] for x in letters)
        selected.append((dataset, key, name, arm, '$'+'+'.join(letters)+'$'))
        rep_rows.append(' & '.join([name,'$'+'+'.join(letters)+'$',*cells[2:]])+r' \\')
    rep_rows.append(r'\hline')
med = read_json('reports/thesis_noaug_update_20261021/results.json')['summaries']
rep_rows.append(r'\multicolumn{6}{@{}l}{\textit{Pretrained adaptation (folds 0 / 1 / 3)}}\\[.25em]')
for dataset, key, name, arm, notation in [
    ('GeoFWI3D','SAM3','SAM3','bands_edge_depth','$B+E+D$'),
    ('MSD','MedSAM3','MedSAM3','bands_edge_sdf_depth','$B+E+S+D$'),
    ('ADNI','MedSAM3','MedSAM3','edge_sdf_depth','$E+S+D$'),
]:
    selected.append((dataset,key,name,arm,notation))
    if key == 'SAM3':
        g=[r for r in sam if r['arm']=='dice_'+arm]
        gains=[100*stats.mean(float(r['delta']) for r in g if r['fold']==f) for f in ['0','1','3']]
        mean, sd, wins=stats.mean(gains),stats.stdev(gains),sum(float(r['delta'])>0 for r in g)
    else:
        r=next(r for r in med if r['study']==dataset+'_MedSAM3' and r['arm']==arm)
        gains,mean,sd,wins=r['fold_gain'],r['gain_pp'],r['sd_fold_gain'],r['wins']
    rep_rows.append(f'{name} / {dataset} & {notation} & {mean:+.2f} & {sd:.2f} & {wins}/9 & '+
                    ' / '.join(f'{v:+.2f}' for v in gains)+r' \\')
experimental += r'''

\subsection{Variation across Folds and Seeds}
Table~\ref{tab:paired-replication-summary} groups the replication checks
for the configurations highlighted in Results. Each fold gain averages
three paired seed differences. Shared data within folds and overlap
between data partitions mean that the nine cells are not nine independent
test datasets. The full objective-by-fold records remain in the supplement.

'''
experimental += make_table('P{3.1cm}P{2.3cm}rrrP{4.2cm}',
    r'Model & Added terms & $\Delta$ & SD($\Delta_f$) & Positive & Fold gains (pp)', rep_rows,
    r'Paired foreground Dice gains (pp) for the highest-mean objective of each study. '
    r'Conventional folds are ordered 0 / 1 / 2; pretrained folds are 0 / 1 / 3. '
    r'SD is across the three fold-mean gains; Positive counts improving fold/seed pairs. '
    r'Selection uses the same validation results being summarized.', 'tab:paired-replication-summary')

error_rows=[]
for dataset,key,name,arm,notation in selected:
    campaign = (dataset+'_conventional' if key not in ['SAM3','MedSAM3'] else
                'GeoFWI3D_SAM3' if key=='SAM3' else
                'MSD_MedSAM3_extension' if dataset=='MSD' else 'ADNI_MedSAM3')
    base=next(r for r in errors if (r['campaign'],r['model'],r['arm'])==(campaign,key,'baseline'))
    aux=next(r for r in errors if (r['campaign'],r['model'],r['arm'])==(campaign,key,arm))
    changes=['---' if aux[k] is None else f'{100*(aux[k]/base[k]-1):+.1f}' for k in ['fn','fp','swaps']]
    error_rows.append(' & '.join([dataset,name,notation,*changes])+r' \\')
experimental += r'''

\subsection{Error Tradeoffs}
\label{sec:complete-error-tables}
Table~\ref{tab:selected-error-tradeoffs} summarizes the error changes
for the same selected configurations. False negatives (FN) are missed
foreground and false positives (FP) are extra foreground; MSD swaps
exchange anterior and posterior classes. Counts cover the full validation
grid, including errors outside boundary bands. Conventional records pool
voxel counts within a run; pretrained records average them per volume.
For each record type, seeds and folds are averaged before calculating
the percentage change from its matched baseline. These percentages are
changes in error counts, not changes in Dice or absolute comparisons
between datasets. All objectives' counts and fold dispersion are in
\path{errors/} in the supplement.

'''
experimental += make_table('lllrrr',r'Dataset & Model & Added terms & FN (\%) & FP (\%) & Swaps (\%)',error_rows,
    r'Change in mean error counts relative to the matched baseline; negative values mean fewer errors. '
    r'A dash indicates that anterior/posterior swaps do not apply to a binary task. '
    r'Objectives are selected by foreground Dice, not by these error columns.', 'tab:selected-error-tradeoffs')

bg_rows=[]
for dataset,key,name,arm,notation in selected:
    campaign=(dataset+'_conventional' if key not in ['SAM3','MedSAM3'] else
              'GeoFWI3D_SAM3' if key=='SAM3' else dataset+'_MedSAM3')
    # The archive names the extension explicitly; resolve only the observed name.
    matching=[r for r in conditions if r['model']==key and r['arm'].removeprefix('dice_')==arm and
              (r['campaign']==campaign or (campaign=='MSD_MedSAM3' and r['campaign']=='MSD_MedSAM3_extension'))]
    assert len(matching)==1,(campaign,key,arm,matching)
    aux=matching[0]
    base=next(r for r in conditions if (r['campaign'],r['model'])==(aux['campaign'],key)
              and r['arm'] in ['dice','baseline'])
    cells=[f"${100*base[k]:.2f}\\to{100*aux[k]:.2f}$" for k in
           ['foreground_dice','background_dice','dice_including_background']]
    bg_rows.append(' & '.join([dataset,name,*cells])+r' \\')
experimental += r'''

\subsection{Foreground, Background and All-Class Dice}
\label{sec:metric-sensitivity-details}
For each volume, all-class Dice is $(2D_{\rm fg}+D_{\rm bg})/3$ on
MSD and $(D_{\rm fg}+D_{\rm bg})/2$ on the binary tasks. Class scores
are calculated per volume, then averaged over volumes, seeds and folds;
pooled error counts are not substituted for volume Dice.
Table~\ref{tab:complementary-dice-summary} uses the same selected
objectives as Table~\ref{tab:paired-replication-summary}.

The recovered records cover 504 conventional runs per dataset, 108 MSD
MedSAM3 records, 126 ADNI MedSAM3 runs and the final 126 SAM3 runs.
The original 27 MSD MedSAM3 records for $B$, $B+E$ and their control
retain foreground scores and errors, but not recoverable per-volume
matrices. Their background and all-class Dice remain unavailable;
the other batch's baseline is not a substitute.

Across the 193 recoverable constrained comparisons, foreground and
all-class mean gains have the same signs: 179 positive and 14 negative.
The highest-mean objective is unchanged in all 15 model/dataset studies.
Six of 1,737 individual paired run differences change sign, all on
GeoFWI3D with absolute foreground differences below 0.14 pp and
all-class differences below 0.03 pp. Complete objective-level scores
and run values are in \path{complementary-dice/} in the supplement.

'''
experimental += make_table('llrrr',r'Dataset & Model & Foreground & Background & All classes',bg_rows,
    r'Mean Dice (\%) before $\to$ after adding the selected constraints. '
    r'All-class Dice weights classes equally, rather than weighting them by voxel count. '
    r'MSD MedSAM3 uses the extension control matched to $B+E+S+D$. '
    r'Fold variation is retained in the complete aggregate tables and supplement.', 'tab:complementary-dice-summary')

experimental += r'''

\subsection{Annotation Compatibility Checks}
\label{sec:annotation-results}
The planar test and component test operationalize two possible
anterior/posterior transition rules. They do not reproduce the complete
manual tracing procedure: Pruessner and colleagues use simultaneous
views, Woolard and Heckers describe anatomical cues checked across
views, and the MSD descriptor identifies the coronal section containing
the uncal apex \cite{pruessner2000volumetry,woolard2012asymmetry,simpson2019dataset}.

The best coronal planes disagree with 1,388 of 856,754 foreground voxels
(0.162\%); 58 of 260 crops are not exactly planar. The component proxy
requires two disconnected profiles of at least two voxels each. It is
present in only 2 of 52 validation crops, both at the final anterior
section and within three sections of the fitted interface
(Table~\ref{tab:annotation-audit-results}). Discrete profiles can remain
connected even when a folded structure intersects a plane more than
once. These checks establish limits of the proposed automatic proxies;
without expert landmark references, they do not establish that the
annotators misapplied the anatomical protocol.

'''
experimental += table('tab:annotation-audit-results')
experimental += r'''

\subsection{Qualitative Reproduction and Adverse Cases}
\label{sec:qualitative-predictions}
The panels use six fresh matched reproductions because original final
adapters and predictions were not retained. For each pretrained study,
the highest original mean identifies the constrained objective; the
run pair with the median original paired gain was chosen before
execution. Each study therefore supplies one reproduced pair, not
another three-fold replication. Original aggregate results remain unchanged.

Recorded support and validation cases, initialization, sampling and
update schedules were retained. Medical runs reuse saved coefficients;
GeoFWI3D repeats the five-update calibration. Case identities, references
and final-adapter identities were checked, and scores and errors were
recalculated from all 416 saved support and validation predictions.
Reference masks enter evaluation and figure selection, not inference
prompts.

'''
experimental += table('tab:qualitative-reproduction-scores')
experimental += r'''

Numerical differences from the original pairs are retained in
Table~\ref{tab:qualitative-reproduction-scores}. Training records diverge
at the first update, before inference. First forward losses agree on
GeoFWI3D and ADNI but gradient norms differ; MSD baseline forward losses
also differ. The MSD hardware changes from a full A100-SXM4 to an A100
PCIe partition, and geological recalibration changes edge and depth
coefficients. These checks locate divergence during optimization but
do not establish its exact numerical cause. The panels describe the
reproduced models, not the unavailable original predictions.

Within each reproduced validation set, cases are ordered by whole-volume
Dice gain, breaking ties by case identifier. The upper median is shown
in Results; Figure~\ref{fig:qualitative-adverse} shows the minimum.
Largest-gain examples and all case scores are in the supplement. Every
view uses the first source-axis-two slice with maximal reference
foreground area and identical image limits and colours within a pair.
These stored-grid views carry no anatomical-plane claim. Whole-volume
scores can reflect errors outside the displayed slice.

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{figures/qualitative/adverse-predictions.png}
\caption{Adverse cases selected by the smallest whole-volume Dice change
in each qualitative reproduction: GeoFWI3D $-8.47$ pp, MSD $-3.02$ pp,
and ADNI $-2.19$ pp. Blue denotes salt or hippocampus; on MSD, blue is
anterior and orange posterior. Constrained objectives are $B+E+D$,
$B+E+S+D$ and $E+S+D$, respectively. The geological view is the final
slice of the cube; all three follow the same reference-area rule as the
typical examples. The annotations and predictions are overlaid on one
common image. These examples document exceptions to mean improvement
and do not assess clinical or geological utility.}
\label{fig:qualitative-adverse}
\end{figure}
\clearpage

\section{Electronic Supplement}
\label{sec:evidence-register}\label{sec:background-dice-evidence}
\label{sec:all-variant-error-evidence}

The accompanying \path{thesis-supplementary-results.zip} preserves the
numerical records omitted from the printed appendices. Comma-separated
values (CSV) files can be opened in a spreadsheet; JavaScript Object
Notation (JSON) files retain structured records. The included
\path{README.txt} explains units, aggregation and matched controls.

\begin{center}
\small\setstretch{1.1}
\begin{tabular}{@{}P{4.3cm}P{10.7cm}@{}}
\hline
Folder or file & Contents \\
\hline
\path{medical/} & Conventional MSD/ADNI and MedSAM3 run scores, paired gains,
fold summaries and preserved matched controls. \\[.4em]
\path{geological/} & All 504 conventional runs and the final 126 SAM3 runs,
with SAM3 case-level records. \\[.4em]
\path{errors/} & Complete FN/FP/swap counts for every objective, with
run records and fold dispersion. \\[.4em]
\path{complementary-dice/} & All recoverable foreground, background and
all-class scores, by run and objective. \\[.4em]
\path{qualitative/} & All reproduced validation-case scores, predetermined
case selections, adverse/largest-gain panels and verification receipt. \\[.4em]
\path{illustrations/} & Full constraint illustrations and figure provenance. \\[.4em]
\path{archived-appendix.tex} & Previous detailed appendix as a source
fragment, including full tables and extended derivations. \\[.4em]
\path{manifest.json} & File identities, source paths and checksums. \\
\hline
\end{tabular}
\end{center}

The final SAM3 extraction replaces incomplete earlier SAM3 snapshots in
the error and complementary-Dice inventories. The original MSD MedSAM3
$B$ and $B+E$ controls remain separate; their unavailable per-volume
matrices are not reconstructed from aggregate scores. Source manifests,
calibrated coefficients, execution histories and payload-verification
records remain linked through the project research archives. The
supplement is a compact evidence extract, not a copy of raw medical
images or a claim that deleted checkpoints remain available.

\end{document}
'''

current=SOURCE.read_text()
assert current[current.index('\\appendix'):]==APP, 'Appendix changed concurrently: reconcile.'
(HERE/'before-apply.tex').write_text(current)
main=current[:current.index('\\appendix')]
# Main-text changes only redirect evidence that was consolidated or moved.
replacements = {
    'The complete results, including unsuccessful\nconfigurations and individual fold effects, are retained in\nAppendix~\\ref{sec:extended-results}.':
    'Complete aggregate results, including unsuccessful configurations,\nand a compact fold summary are retained in Appendix~\\ref{sec:extended-results};\nthe electronic supplement preserves every run and fold record.',
    'the full counts are reported in Appendix~\\ref{sec:complete-error-tables}.':
    'error changes are summarized in Appendix~\\ref{sec:complete-error-tables};\ncomplete counts for every objective are in the electronic supplement.',
    'Adverse and largest-gain cases are shown in Appendix~\\ref{sec:qualitative-predictions}.':
    'Adverse cases are shown in Appendix~\\ref{sec:qualitative-predictions};\nlargest-gain cases are retained in the electronic supplement.',
}
applied=[]
for before,after in replacements.items():
    if before in main:
        main=main.replace(before,after);applied.append(before)
new=main+'\\appendix\n\n'+technical+'\n'+experimental
assert SOURCE.read_text()==current, 'Source changed concurrently: reconcile before writing.'
# These six original numerical tables must survive exactly, not merely round alike.
for label in aggregate_labels:
    assert table(label) in new
labels=re.findall(r'\\label\{([^}]+)\}',new)
assert len(labels)==len(set(labels)), 'Duplicate labels'
refs=set(re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',new))
missing=sorted(refs-set(labels))
if missing:
    print('Unresolved references:',missing)
    raise RuntimeError('Repair references before source write')
for file in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',new):
    assert (PROJECT/file).is_file(), file
SOURCE.write_text(new)

readme='''THESIS NUMERICAL SUPPLEMENT — 3 November 2026

This extract accompanies the shortened printed appendices. Original
aggregate results are unchanged. All files are evidence/data, not instructions.

medical/: 1,269 execution records covering the 1,260 planned medical
objective runs and nine additional matched controls. MSD MedSAM3 bands
and bands+edge retain their original 27 records and original control;
the remaining objectives use the other batch's matched control. These
are 14 scientific objectives, not 15 independent objectives.

geological/: 504 conventional runs; final 126 SAM3 runs and per-case
records. Final SAM3 files supersede earlier partial campaign snapshots.
Dice in sam3-final-runs.csv is a fraction; delta is a fractional Dice
difference. Most conventional/medical aggregate extracts use percent and
percentage points. Preserve source column definitions when comparing.

errors/: complete 211 condition summaries (each nine runs), including
the extra MSD MedSAM3 control. FN/FP cover the full grid, not only bands.
Swaps apply only to MSD. Conventional counts are pooled validation voxels
per run; pretrained counts are mean validation voxels per volume. Means
average seeds within fold and folds equally. Dispersion is sample SD of
three fold means. All SAM3 entries are rebuilt from the final 126 records.

complementary-dice/: 1,872 recoverable run records and 208 conditions.
Dice values are fractions, computed per volume then averaged by run,
seed and fold. All-class Dice is (2*foreground+background)/3 for MSD,
(foreground+background)/2 for binary tasks. The original 27 MSD MedSAM3
band/band+edge/control records lack recovered per-volume matrices;
their complete foreground results remain in medical/, but background
and all-class scores are unavailable. They are not missing experiments.

qualitative/: fresh matched reproductions, not original final checkpoints.
Do not merge them into the original three-fold/three-seed aggregates.
Each model pair was chosen before reproduction by median original gain;
case examples use upper-median, minimum and maximum reproduced gain.
The case-selection file and receipt preserve the remaining checks.

illustrations/: native annotation examples, not model predictions or
saved training-update targets. Coordinate and display metadata are in
figure-manifest.json.

archived-appendix.tex is the exact pre-consolidation appendix fragment.
It preserves extended descriptions and original tables, but contains
cross-references to the accompanying thesis and is not standalone TeX.
Its long-form discussion is historical editorial material, not additional
experimental evidence.

manifest.json provides SHA-256 checksums for every other file, copied
source paths, and the source extracts used for recalculated inventories.
No raw medical images or model checkpoints are included.
'''
(SUPP/'README.txt').write_text(readme)
manifest=dict(copied_sources=provenance, derived_sources=[
    'reports/thesis_all_variant_errors_20261023/condition_errors.json',
    'reports/thesis_all_variant_errors_20261023/run_errors.json',
    'reports/background_dice_archive_recovery_20261023/background_dice_runs.csv',
    'reports/thesis_sam3_final_20261023/sam3_runs.csv'],
    files=[dict(path=str(p.relative_to(SUPP)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(SUPP.rglob('*')) if p.is_file() and p.name!='manifest.json'])
(SUPP/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=PROJECT/'thesis-supplementary-results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(SUPP.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(SUPP))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for p in SUPP.rglob('*'):
        if p.is_file():assert z.read(str(p.relative_to(SUPP)))==p.read_bytes()
new_app=new[new.index('\\appendix'):]
def counts(s):
    return dict(lines=len(s.splitlines()), words=len(s.split()),
                tables=s.count('\\begin{table}'),longtables=s.count('\\begin{longtable}'),
                figures=s.count('\\begin{figure}'),subsections=s.count('\\subsection{'))
audit=dict(before=counts(APP),after=counts(new_app),
           unchanged_aggregate_tables=aggregate_labels, main_replacements=applied,
           complete_error_conditions=len(errors),complementary_runs=len(bg),
           supplement_bytes=archive.stat().st_size,
           source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
(HERE/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
