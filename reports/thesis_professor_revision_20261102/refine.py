from pathlib import Path
import re
p=Path('Template_tesi/thesis-revision.tex')
s=p.read_text()
def rep(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new)

# Fix chart legends after inspecting the rendered pages.
for m in list(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S)):
    if any('\\label{fig:'+k+'-key-results}' in m.group() for k in ['geo','msd','adni','adaptation']):
        f=m.group()
        new=f.replace(',-26)',',-47)').replace(',-22)',',-43)').replace(',-24)',',-45)')
        s=s.replace(f,new)
rep('ADNI hippocampus\\\\MRI, separate left/right crops\\\\2 classes: background / hippocampus',
    'ADNI hippocampus (MRI)\\\\2 classes: background /\\\\hippocampus')

# Retain the full implemented calibration specification in the technical appendix.
start=s.index('Loss magnitudes depend on their supports and reductions.')
end=s.index('\\subsubsection{Resource Accounting and Evaluation}',start)
calibration=s[start:end].strip()
s=s[:start]+r'''An auxiliary loss can dominate an update simply because its numerical
scale differs from the baseline. Calibration sets its coefficient by
comparing the size of their gradients on the labelled support cases.
This uses training annotations only and holds the measured network state
fixed. The coefficients are then retained during optimization.

The medical studies calibrate after a short baseline prefix: nine updates
for nnU-Net and 50 for the other models. In a combined objective, terms
are calibrated sequentially against the baseline and preceding terms.
The order is bands, edges, signed distance and depth. Consequently, a
term's coefficient can change when it is used in a different combination;
these experiments compare complete calibrated objectives. A median
gradient target sets the typical strength, while an upper-percentile
bound limits unusually large contributions. The equations and numerical
settings are given in Appendix~\ref{sec:detailed-calibration}.

Auxiliary terms enter gradually after the prefix. Signed distance either
shares that ramp or starts later, according to the fixed activation
policy in Table~\ref{tab:sdf-activation-policies}. This policy was developed
from earlier MSD validation experiments and retained for the reported
campaigns; it was not selected independently of those exploratory results.
Combinations containing both signed distance and depth use delayed
activation of the distance term.

'''+s[end:]

start=s.index('The medical budget was anchored to nnU-Net')
end=s.index('\\subsection{Adapting Pretrained Models}',start)
resource=s[start:end].strip()
s=s[:start]+r'''The medical ceiling corresponds to the measured cost of 90 nnU-Net
updates, approximately 31,976 billion floating-point operations (GFLOPs).
Exploratory learning curves informed this short reference allowance.
With ten support crops and one crop per update, the reference represents
nine passes through the training subset. nnU-Net has the most expensive
update among the four implementations, so the same allowance lets models
with cheaper updates perform more optimization steps.

Every medical model and objective receives that ceiling. The accounting
includes the forward pass, active training losses, backpropagation and
the optimizer update. A step is allowed only when its full cost fits in
the remaining budget. Reference-target construction, validation, data
loading and storage are excluded, and calibration is recorded separately.
This matches optimization work, not elapsed time or end-to-end computation.
Equal allowances also need not place the architectures equally close to
convergence.

The reported checkpoint is the final state admitted by the budget, with
no early stopping or selection of a validation-best checkpoint. Selecting
the highest-mean objective after the experiments is a separate, descriptive
comparison. All 504 runs in each medical study enter the aggregation.
Appendix~\ref{sec:detailed-resource-accounting} gives the exact ceiling
and arithmetic conventions; Table~\ref{tab:cap-training-exposure} reports
how the resulting update counts translate into passes through the data.
The same numerical medical ceiling is used on MSD and ADNI, while their
different label vocabularies remain part of the interpretation.

'''+s[end:]

start=s.index('The geological cap follows the same design principle')
end=s.index('\\subsection{MSD: Medical Segmentation Decathlon}',start)
geo=s[start:end].strip()
s=s[:start]+r'''The geological study uses a larger computation allowance because its
inputs are larger. The reference is the original measured cost of 152
nnU-Net updates on the $96^3$ grid, approximately 182,178 GFLOPs. This
corresponds to eight passes through the 19 support scenes under the
reference profile. The final recipes retain that numerical ceiling and
charge both calibration and optimization within it. Evaluation and recipe
development remain outside the budget. Medical and geological results
therefore do not constitute an equal-compute comparison across datasets.

Calibration measures gradients on all 19 support scenes. Its strengths
differ across architectures, and the nnU-Net and UNETR settings follow
exploratory comparisons. Auxiliary terms enter through fixed ramps, with
some signed-distance configurations delayed. Each constrained run is
compared with the baseline of the same frozen model recipe. The final
admissible checkpoint supplies the score. Appendix~\ref{sec:geological-budget-details}
records the exact ceiling, strengths and schedules, while
Table~\ref{tab:cap-training-exposure} gives the actual training exposure.

'''+s[end:]

# Avoid explaining training loss versus evaluation Dice separately for each dataset.
rep('\\subsection{Adapting Pretrained Models}\n\\paragraph{SAM3 on GeoFWI3D.}\n\\label{sec:sam3-methods}',r'''\subsection{Adapting Pretrained Models}

These studies adapt an existing image model through LoRA while keeping
its base weights frozen. SAM3 starts from the general checkpoint;
MedSAM3 additionally uses the released medical adapter. Each constrained
run starts from the same state and uses the same support data and
sampling schedule as its matched baseline.

The baseline training objective combines detection classification and
bounding-box losses, a presence loss, and focal and soft Dice mask losses.
Spatial terms are added to this complete objective. The reported Dice
scores instead measure overlap of the final discrete segmentations:
training-set Dice uses the support images and validation Dice uses
held-out images. Fixed text queries identify the requested structures;
reference masks never supply localization prompts at inference.

\paragraph{SAM3 on GeoFWI3D.}
\label{sec:sam3-methods}''')
start=s.index('The SAM3 baseline uses a composite training objective:')
end=s.index('Constrained variants retain',start)
s=s[:start]+s[end:]
rep(r'''The baseline retains the composite SAM3
training objective described in Section~\ref{sec:sam3-methods}, including
its focal and soft Dice mask losses. Each constrained variant adds its
spatial penalties to this objective. These losses determine the adapter
updates; the Dice scores reported after fitting are separate overlap
measurements.''', '''The baseline retains the composite training objective described above,
and each constrained variant adds its spatial penalties.''')
rep(r'''Both columns report segmentation overlap, not the composite loss or its
Dice-loss component. Section~\ref{sec:evaluation-methods} gives the
formula and the complementary background and all-class measures.''',
    r'''Section~\ref{sec:evaluation-methods} gives the formula and the
complementary background and all-class measures.''')
rep('''As in MSD, these are overlap scores computed after fitting, separate
from every component of the training objective. Per-volume scores and''',
    'Per-volume scores and')

details=(r'\subsection{Detailed Calibration Protocol}\label{sec:detailed-calibration}'+'\n\n'+calibration+'\n\n'
         +r'\subsection{Medical Resource Accounting}\label{sec:detailed-resource-accounting}'+'\n\n'+resource+'\n\n'
         +r'\subsection{Geological Budget and Schedules}\label{sec:geological-budget-details}'+'\n\n'+geo+'\n\n')
pos=s.index(r'\subsection{Model Settings and Training Exposure}')
s=s[:pos]+details+s[pos:]

# Expand common numerical/implementation acronyms before their first prose use.
s=s.replace('The class vocabulary determines which errors a constraint can detect.',
            'The class vocabulary determines which errors a constraint can detect.')
s=s.replace('reference map $y_i:\\Omega_i\\to\\mathcal C$ assigns one of $K$',
            'reference map $y_i:\\Omega_i\\to\\mathcal C$ assigns one of $K$')
s=s.replace('The scalar loss is', 'The scalar loss is')
# More readable appendix opening now that all complete tables are here.
start=s.index('These tables complement the validation means and error counts')
end=s.index(r'\subsection{Complete Dice Comparisons}',start)
s=s[:start]+r'''This appendix retains the full numerical comparisons behind the main
results figures. Scores are percentages, $\Delta$ is a paired change in
percentage points, and $n_+$ counts positive changes among the nine
fold/seed pairs. These pairs share data and are not nine independent
datasets. Standard deviations (SDs) describe the three seed-averaged fold
means. Training exposure is reported in
Table~\ref{tab:cap-training-exposure}; individual run records are indexed
in Appendix~\ref{sec:evidence-register}.

'''+s[end:]
p.write_text(s)
print('Saved layout fixes, concise methods and full protocol appendices.')
