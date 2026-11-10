"""Present the principal findings graphically and relocate secondary detail."""
from pathlib import Path
import csv
import json
import re

ROOT = Path('/Users/filippofocaccia/Desktop/hippo')
HERE = ROOT / 'reports/thesis_results_readability_20261102'
TARGET = ROOT / 'Template_tesi/thesis-revision.tex'
source = (HERE / 'source_before.tex').read_text()
start = source.index(r'\section{Results and Analysis}')
end = source.index(r'\section{Conclusions}', start)
old_results = source[start:end]


def figure(label):
    pos = old_results.index('\\label{' + label + '}')
    left = old_results.rfind(r'\begin{figure}', 0, pos)
    right = old_results.index(r'\end{figure}', pos) + len(r'\end{figure}')
    return old_results[left:right]


def section_between(begin, finish):
    return old_results[old_results.index(begin):old_results.index(finish)]


plot_rows = json.loads((ROOT / 'reports/thesis_professor_revision_20261102/plot_data.json').read_text())

# A shared horizontal scale makes the baseline and constrained scores legible
# without repeating every score in the prose.
scratch = [r'\begin{figure}[htbp]', r'\centering\setstretch{1.05}',
           r'\begin{tikzpicture}[x=1cm,y=-1cm,font=\footnotesize]']
for panel, (title, data) in enumerate(zip(
        ['(a) GeoFWI3D', '(b) MSD', '(c) ADNI'], plot_rows[:3])):
    top = panel * 4.05
    scratch.append(rf'\node[anchor=west,font=\bfseries\footnotesize] at (0,{top:.3f}) {{{title}}};')
    scratch.append(rf'\node[anchor=center,font=\scriptsize,text=thesisGray] at (14.35,{top:.3f}) {{Gain (pp)}};')
    for tick in range(0, 101, 20):
        x = 3.05 + tick * .1
        scratch.append(rf'\draw[thesisGray!22,line width=.3pt] ({x:.3f},{top+.38:.3f}) -- ({x:.3f},{top+3.32:.3f});')
        scratch.append(rf'\node[font=\scriptsize,text=thesisGray] at ({x:.3f},{top+3.57:.3f}) {{{tick}}};')
    for row, (model, objective, baseline, base_sd, selected, selected_sd, delta) in enumerate(data['rows']):
        centre = top + .77 + row * .74
        scratch.append(rf'\node[anchor=east] at (2.85,{centre:.3f}) {{{model}}};')
        for value, sd, y, colour in [(baseline, base_sd, centre-.13, 'thesisBlue!85'),
                                    (selected, selected_sd, centre+.13, 'thesisOrange!85')]:
            x = 3.05 + .1 * value
            lo, hi = x - .1 * sd, x + .1 * sd
            scratch.append(rf'\fill[{colour}] (3.05,{y-.095:.3f}) rectangle ({x:.3f},{y+.095:.3f});')
            scratch.append(rf'\draw[thesisInk,line width=.45pt] ({lo:.3f},{y:.3f}) -- ({hi:.3f},{y:.3f});')
            for cap in [lo, hi]:
                scratch.append(rf'\draw[thesisInk,line width=.45pt] ({cap:.3f},{y-.065:.3f}) -- ({cap:.3f},{y+.065:.3f});')
        scratch.append(rf'\node at (14.35,{centre:.3f}) {{$+{delta:.2f}$}};')
scratch += [
    r'\node[text=thesisGray] at (8.05,12.1) {Foreground Dice (\%)};',
    r'\fill[thesisBlue!85] (2.1,12.57) rectangle (2.4,12.75);',
    r'\node[anchor=west] at (2.55,12.66) {Baseline};',
    r'\fill[thesisOrange!85] (6.1,12.57) rectangle (6.4,12.75);',
    r'\node[anchor=west] at (6.55,12.66) {With spatial supervision};',
    r'\end{tikzpicture}',
    r'''\caption{Training from scratch: baseline and the highest-mean tested
spatial objective for each model and dataset. Whiskers show the sample
standard deviation of three seed-averaged fold means; they are not
confidence intervals. Right-hand labels give paired gains from unrounded
scores. Foreground Dice averages anterior and posterior on MSD and measures
the single target class on the binary tasks. Panels share a score scale,
but their task and budget differences preclude a direct dataset ranking.
Complete objective scores and selected combinations are in
Appendix~\ref{sec:complete-result-tables}.}''',
    r'\label{fig:scratch-key-results}',
    r'\label{fig:geo-key-results}\label{fig:msd-key-results}\label{fig:adni-key-results}',
    r'\end{figure}']
scratch = '\n'.join(scratch)

adaptation = figure('fig:adaptation-key-results')
adaptation = adaptation.replace('Highest-mean tested objective', 'With spatial supervision')
caption_at = adaptation.index(r'\caption{')
adaptation = adaptation[:caption_at] + r'''\caption{Pretrained adaptation:
baseline and the highest-mean tested objective, with fold-level standard
deviations. The MSD baseline is matched to the execution batch of the
all-term variant. $B,E,S,D$ denote bands, edges, signed distance and depth
supervision. The studies use different support sizes and update budgets;
the figure compares each model with its own baseline. Full results are in
Appendix~\ref{sec:complete-result-tables}.}
\label{fig:adaptation-key-results}
\end{figure}'''

# Read unrounded archive aggregates: rounded table subtraction would turn
# the MSD background gain of 0.1509 pp into a misleading 0.16 pp.
with (ROOT / 'reports/background_dice_archive_recovery_20261023/background_dice_conditions.csv').open() as f:
    conditions = list(csv.DictReader(f))
sam = json.loads((ROOT / 'reports/thesis_sam3_final_20261023/sam3_conditions.json').read_text())
sam_base = next(x for x in sam if x['arm'] == 'dice')
sam_best = next(x for x in sam if x['arm'] == 'dice_bands_edge_depth')
def condition(campaign, arm):
    return next(x for x in conditions if x['campaign'] == campaign and x['arm'] == arm)
metrics = ['foreground_dice', 'background_dice', 'dice_including_background']
metric_data = []
for study, baseline, selected in [
        ('SAM3 / GeoFWI3D', sam_base, sam_best),
        ('MedSAM3 / MSD', condition('MSD_MedSAM3_extension', 'baseline'),
         condition('MSD_MedSAM3_extension', 'bands_edge_sdf_depth')),
        ('MedSAM3 / ADNI', condition('ADNI_MedSAM3', 'dice'),
         condition('ADNI_MedSAM3', 'dice_edge_sdf_depth'))]:
    metric_data.append({'study': study,
                        'baseline': {m: float(baseline[m]) * 100 for m in metrics},
                        'selected': {m: float(selected[m]) * 100 for m in metrics},
                        'gains_pp': {m: (float(selected[m])-float(baseline[m]))*100 for m in metrics}})
(HERE / 'chart_data.json').write_text(json.dumps({'scratch': plot_rows[:3], 'adaptation': plot_rows[3],
                                               'metric_sensitivity': metric_data}, indent=2)+'\n')

metric_chart = [r'\begin{figure}[htbp]\centering\setstretch{1.05}',
                r'\begin{tikzpicture}[x=1cm,y=.70cm,font=\footnotesize]',
                r'\node[anchor=west] at (0,6.9) {Dice gain over matched baseline (percentage points)};']
for tick in range(7):
    metric_chart += [rf'\draw[thesisGray!22] (0,{tick}) -- (14.6,{tick});',
                     rf'\node[anchor=east] at (-.12,{tick}) {{{tick}}};']
colours = ['thesisOrange!85', 'thesisGray!80', 'thesisBlue!85']
for group, data in enumerate(metric_data):
    centre = 2.5 + group * 4.65
    for offset, metric, colour in zip([-.85, 0, .85], metrics, colours):
        x = centre + offset
        value = data['gains_pp'][metric]
        metric_chart += [rf'\fill[{colour}] ({x-.32:.3f},0) rectangle ({x+.32:.3f},{value:.6f});',
                         rf'\node[anchor=south,font=\scriptsize] at ({x:.3f},{value+.08:.6f}) {{{value:.2f}}};']
    label = data['study'].replace(' / ', r'\\')
    metric_chart.append(rf'\node[anchor=north,align=center,text width=3.7cm] at ({centre:.3f},-.15) {{{label}}};')
for x, colour, label in [(1.8, colours[0], 'Foreground'), (6.2, colours[1], 'Background'), (10.6, colours[2], 'All classes')]:
    metric_chart += [rf'\fill[{colour}] ({x},-2.5) rectangle ({x+.30},-2.25);',
                     rf'\node[anchor=west] at ({x+.45},-2.375) {{{label}}};']
metric_chart += [r'\end{tikzpicture}', r'''\caption{The same adaptation
predictions assessed in three ways. Bars show mean gains for the objectives
in Figure~\ref{fig:adaptation-key-results}, calculated from the unrounded
archive aggregates. Including the already high-scoring background reduces
the gain in the class average, while preserving its positive direction.
These are complementary measurements of the same runs. The exact
aggregation, absolute scores and checks across all recoverable objectives
are in Appendix~\ref{sec:metric-sensitivity-details}.}''',
                 r'\label{fig:metric-sensitivity-key-results}', r'\end{figure}']
metric_chart = '\n'.join(metric_chart)

# Move the full metric calculation, coverage and secondary checks alongside
# the existing complementary-score tables; retain their labels and numbers.
metric_detail = section_between(r'\subsection{Foreground, Background and All-Class Dice}',
                               r'% END BACKGROUND DICE ANALYSIS')
metric_detail = metric_detail.replace(r'\subsection{Foreground, Background and All-Class Dice}', '')
metric_detail = metric_detail.replace(r'\label{sec:background-dice-results}',
                                     r'\label{sec:metric-sensitivity-details}')
metric_detail = metric_detail[:metric_detail.index('Together, the measures show')] + '\\FloatBarrier\n'

audit = section_between(r'\subsubsection{Annotation Compatibility and the Anterior/Posterior Transition}',
                        r'\subsubsection{Limitations of the Experimental Findings}')
audit = audit.replace(r'\subsubsection{Annotation Compatibility and the Anterior/Posterior Transition}',
                      r'\subsection{Annotation Compatibility and the Anterior/Posterior Transition}')

# Keep the detailed observations that explain individual objectives and
# exceptional runs in a clearly identified appendix, rather than losing them.
detail_parts = [r'\subsection{Additional Objective-Level Comparisons}',
                r'\label{sec:secondary-objective-results}',
                'The following observations supplement the principal comparisons in\n'
                r'Chapter~\ref{sec:results}. Full scores, dispersion and paired effects'+'\n'
                'are retained in the tables above.\n']
def paragraphs_between(first, last):
    return old_results[old_results.index(first):old_results.index(last)].strip()

detail_parts += [r'\subsubsection*{GeoFWI3D: consistency, near ties and negative outcomes}',
                paragraphs_between('All four terms give the highest mean for nnU-Net',
                                   r'Table~\ref{tab:geo-ledger}')]
detail_parts += [r'\subsubsection*{MSD: individual constraints}',
                'Boundary bands alone improve all four model means and all 36 matched\n'
                'comparisons. Mean gains are 7.55, 3.85, 3.89 and 4.83 points for\n'
                'nnU-Net, SwinUNETR, UNETR and the compact U-Net, respectively.\n',
                paragraphs_between('The inventory also contains negative findings.',
                                   r'\FloatBarrier'+'\n'+r'\paragraph{ADNI hippocampus segmentation.}')]
detail_parts += [r'\subsubsection*{ADNI: individual constraints}',
                'Boundary bands alone improve every model mean and all 36 matched\n'
                'comparisons. Gains are 0.73, 3.34, 2.99 and 3.81 points for nnU-Net,\n'
                'SwinUNETR, UNETR and the compact U-Net, respectively.\n',
                paragraphs_between('SDF alone has essentially zero mean effect on nnU-Net',
                                   r'\FloatBarrier'+'\n'+r'\subsection{Adapting Pretrained Models}')]
detail_parts += [r'\subsubsection*{SAM3: variation across objectives and support partitions}',
                paragraphs_between('The adapted baseline reaches 69.31',
                                   'Full-volume errors for all SAM3 variants')]
detail_parts += [r'\subsubsection*{MSD MedSAM3: matched execution batches}',
                paragraphs_between('The MSD MedSAM3 campaign evaluates fourteen objectives',
                                   r'\FloatBarrier'+'\n'+r'\paragraph{MedSAM3 on ADNI.}')]
detail_parts += [r'\subsubsection*{ADNI MedSAM3: closely ranked configurations}',
                paragraphs_between('On ADNI, $E+S+D$ has the highest mean',
                                   r'\FloatBarrier'+'\n'+r'\subsection{Error Analysis}')]
objective_details = '\n\n'.join(detail_parts)+'\n\n\\FloatBarrier\n'

error_detail = section_between(r'\subsection{Error Analysis}', r'% BEGIN BACKGROUND DICE ANALYSIS')
error_detail = error_detail.replace(r'\subsection{Error Analysis}', '')
error_detail = error_detail.replace(r'\label{sec:error-analysis}', r'\label{sec:error-details}')
error_detail = error_detail.replace(r'\label{sec:capped-errors-resources}', '')
error_detail = error_detail.replace('in Appendix~\\ref{sec:complete-error-tables}', 'below')

main = r'''\section{Results and Analysis}
\label{sec:results}

Spatial supervision improves the best observed mean Dice in every model
and dataset studied, but its benefit depends on the objective and training
setting. This chapter presents those principal comparisons and examines
what the improvements mean. The complete results, including unsuccessful
configurations and individual fold effects, are retained in
Appendix~\ref{sec:extended-results}.

Each figure compares a baseline with the objective that has the highest
mean validation Dice in the same study. Selection uses these validation
results, so the comparisons are descriptive rather than independent tests
of an optimal method. Dice is reported in percent and gains in percentage
points (pp); all gains use the corresponding matched baseline.

\subsection{Spatial Supervision When Training from Scratch}
\label{sec:geo-results}\label{sec:main-msd-results}\label{sec:adni-results}

Figure~\ref{fig:scratch-key-results} brings the three datasets together.
The consistent pattern is an improvement over each model's own baseline.
The size of that improvement varies considerably, even among models
trained on the same data.

__SCRATCH__

The medical comparisons illustrate why absolute accuracy and improvement
must be distinguished. On MSD, SwinUNETR gains almost twelve points, yet
the compact U-Net reaches the highest Dice, 80.00\%. A large gain can
therefore reflect the correction of a weak short-budget baseline without
producing the most accurate model. On ADNI, nnU-Net starts from the
strongest baseline and retains the highest score, although its improvement
is only about one point.

The geological results extend the positive pattern beyond medical images.
The compact U-Net again reaches the highest score, while nnU-Net benefits
most from the added supervision. This is evidence that the approach can
help in both domains under the tested procedures. It does not establish
that one domain benefits more: the geological and medical comparisons
use different budgets and task definitions.

No single constraint combination explains all these gains. Boundary
membership alone improves every model mean across the three datasets,
but other additions can reduce accuracy. For example, signed-distance
supervision alone lowers the MSD nnU-Net mean by about two points.
The useful conclusion is therefore that annotation geometry can improve
learning when its contribution is tested within the chosen training
recipe. Adding more terms is not reliably better. The objective-level
comparisons, near ties and negative outcomes are documented in
Appendix~\ref{sec:secondary-objective-results}.

\FloatBarrier
\subsection{Spatial Supervision During Pretrained Adaptation}
\label{sec:sam3-results}\label{sec:medsam-results}\label{sec:medsam-adni-results}

The adaptation experiments ask whether the same idea remains useful when
the model already has a pretrained representation. Figure~\ref{fig:adaptation-key-results}
shows improvements in all three studies: SAM3 reaches 72.76\% Dice on
GeoFWI3D, while MedSAM3 reaches 81.73\% on MSD and 80.85\% on ADNI.
The largest improvement is on MSD, where the gain over the matched
adapted baseline exceeds five percentage points.

__ADAPTATION__

Spatial supervision consequently remains useful beyond learning from
scratch. The improvement is present in every fold mean for each selected
configuration, although the SAM3 gain varies more across support
partitions. The medical gains are positive in every matched run; SAM3
has one negative pair. These patterns support a benefit under the
evaluated adaptation conditions, without implying that every new support
set would produce the same change.

The preferred objective again depends on the task. SAM3 benefits most
from bands, edges and depth supervision; including signed distance lowers
its mean. MedSAM3 on MSD reaches its highest mean with all four terms,
whereas ADNI has closely ranked alternatives. The studies also differ in
initialization, support size and update budget, so their scores cannot
isolate the effect of medical pretraining or establish a model ranking at
equal resources. For MSD, comparisons retain the control from each
variant's execution batch; the complete matched results are in
Table~\ref{tab:medsam-msd-noaug}.

\FloatBarrier
\subsection{What the Overlap Improvements Mean}
\label{sec:error-analysis}\label{sec:capped-errors-resources}
\label{sec:background-dice-results}

Higher Dice does not mean that every kind of error becomes less frequent.
Across the adaptation studies, the selected constraints recover more of
the target structure but also label more surrounding voxels as foreground.
On MSD, fewer anterior/posterior exchanges accompany this tradeoff.
The models trained from scratch show a less uniform pattern: some mainly
reduce missed foreground, others mainly reduce unwanted extension. These
differences matter when deciding which errors an application can tolerate;
the full counts are reported in Appendix~\ref{sec:complete-error-tables}.

Including background in the Dice average provides a complementary check
on the overlap gains. Figure~\ref{fig:metric-sensitivity-key-results}
shows that the selected adaptation objectives improve both foreground
and background Dice. The same holds for the selected conventional models.
Background scores already lie close to their ceiling, so their changes
are much smaller. Giving background equal class weight raises the
absolute average while reducing the size of the improvement.

__METRICS__

The main conclusion survives this change of measure: across all recoverable
comparisons, the sign of the mean effect and the highest-mean objective
in each study are unchanged. A few small individual effects and secondary
rankings differ. Appendix~\ref{sec:metric-sensitivity-details} gives the
full checks and archive coverage. The earlier MSD MedSAM3 bands and
bands-plus-edges comparisons retain complete foreground results, but lack
recoverable background scores.

These measurements describe agreement with the annotations. They do not
establish where boundary errors occur or whether the improvement is
useful downstream. Paired prediction panels and surface measurements
would be needed to assess those questions; the existing dataset images
illustrate the annotations rather than model performance.

\FloatBarrier
\subsection{Limits of the Findings}
\label{sec:limitations}\label{sec:experimental-limitations}

The results concern limited training budgets, prepared hippocampus crops
and synthetic geological volumes. They establish benefits within those
settings; persistence after substantially longer training and transfer
to uncropped medical images or measured seismic data remain open.
Small support sets and overlap between some support partitions also
limit conclusions about sensitivity to the annotated examples available.

The comparisons evaluate complete calibrated objectives. In particular,
depth supervision adds a class loss as well as spatial weighting, so
a matched unweighted control is needed to distinguish those contributions.
Closely ranked combinations should likewise be treated as alternatives
within the observed variation rather than evidence of a unique optimum.

The MSD annotation audit gives a related caution about the knowledge
encoded in a constraint. Simplified coronal-plane and disconnected-profile
rules do not reproduce every annotated anterior/posterior transition.
The thesis therefore supervises the stored class boundary rather than
imposing either proxy universally. The detailed checks and their relation
to the original annotation protocols are in
Appendix~\ref{sec:annotation-results}; they do not establish that the
original anatomical tracing was incorrect.

\clearpage

'''
main = main.replace('__SCRATCH__', scratch).replace('__ADAPTATION__', adaptation).replace('__METRICS__', metric_chart)
assert r'\begin{table}' not in main and r'\begin{longtable}' not in main
updated = source[:start] + main + source[end:]

anchor = r'\subsection{Complete Error Counts}\label{sec:complete-error-tables}'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, objective_details+'\n'+anchor+'\n'+error_detail)
anchor = r'\subsection{Complementary Dice Measures}'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, anchor+'\n'+metric_detail)
anchor = r'\subsection{Geological Development Comparison}'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, audit+'\n\\FloatBarrier\n\n'+anchor)
updated = updated.replace(r'Section~\ref{sec:annotation-results}', r'Appendix~\ref{sec:annotation-results}')
updated = updated.replace(r'Sections~\ref{sec:annotation-compatibility} and~\ref{sec:annotation-results}',
                          r'Section~\ref{sec:annotation-compatibility} and Appendix~\ref{sec:annotation-results}')

# Preserve every original numerical table exactly, including the two newly
# relocated tables. Check labels and reference integrity before saving.
def tables(text):
    return [m.group(0) for m in re.finditer(r'\\begin\{(table|longtable)\}(.*?)\\end\{\1\}', text, re.S)]
before_tables = tables(source)
assert all(t in updated for t in before_tables), 'A numerical table was changed or lost'
labels = re.findall(r'\\label\{([^}]+)\}', updated)
assert len(labels) == len(set(labels)), 'Duplicate labels'
references = re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', updated)
assert not set(references)-set(labels), set(references)-set(labels)
assert len(tables(updated)) == len(before_tables)
TARGET.write_text(updated)
(HERE / 'main_results.tex.txt').write_text(main)
(HERE / 'revision_checks.json').write_text(json.dumps({
    'original_results_tables': len(tables(old_results)),
    'revised_results_tables': len(tables(main)),
    'revised_results_figures': main.count(r'\begin{figure}'),
    'all_original_tables_preserved_verbatim': len(before_tables),
    'new_charts': ['12 conventional model/dataset comparisons in three panels',
                   'foreground/background/all-class gains for pretrained adaptation'],
    'preserved_chart': 'pretrained baseline versus selected objective',
    'appendix_moves': ['both remaining Results tables', 'objective-level findings and exceptions',
                       'full error discussion', 'metric equations and sensitivity details',
                       'annotation compatibility audit'],
    'labels_unique': True, 'unresolved_references': []
}, indent=2)+'\n')
print('Saved Results chapter: zero tables, three figures; all original tables preserved verbatim.')
