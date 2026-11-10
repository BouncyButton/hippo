from pathlib import Path
import re, hashlib, json
p=Path('/Users/filippofocaccia/Desktop/hippo/Template_tesi/thesis-revision.tex')
s=p.read_text()
before=s
assert hashlib.sha256(s.encode()).hexdigest()=='2ab630846e80c9ed76833eef188ffb47dfb036c73fef34979b2b8201c24fe386'
def replace(old,new):
    global s
    assert s.count(old)==1, (old[:90],s.count(old))
    s=s.replace(old,new,1)
def span(start,end,new):
    global s
    a=s.index(start); b=s.index(end,a)
    old=s[a:b]; s=s[:a]+new+s[b:]; return old

start=s.index('The experiments address two learning situations.')
end=s.index('The contribution is a common LTN/logLTN formulation',start)
learning=s[start:end]
s=s[:start]+s[end:]
learning=learning.replace('(SAM3) to geological images and its medical specialization, MedSAM3, to\nhippocampus images','(SAM3) and its medical specialization, MedSAM3, to the target tasks')
replace('The investigation first evaluates these ideas on salt-body segmentation',learning+'The investigation first evaluates these ideas on salt-body segmentation')
replace('crops \\cite{adniStudy}. Figure~\\ref{fig:dataset-examples} illustrates the\nimages and masks used in these three settings.','crops \\cite{adniStudy}. SAM3 is adapted to the geological task and\nMedSAM3 to the two medical tasks. Figure~\\ref{fig:dataset-examples}\nillustrates the images and masks used in these three settings.')
replace(r"""The logarithmic formulation, logLTN, represents truth values through
their logarithms and allows the corresponding penalties to be evaluated
in that space \cite{badreddine2023logltn}.""",r"""The logarithmic formulation, logLTN, expresses truth degrees through
their logarithms: multiplying degrees of satisfaction becomes addition,
and their negative logarithms give penalties to minimize
\cite{badreddine2023logltn}.""")
span('The contribution is a common LTN/logLTN formulation',r'\begin{figure}[p]',r"""The contribution is a common LTN/logLTN formulation of these four
objectives and an assessment of when they improve segmentation. The
formulation builds on established ideas, including signed-distance
boundary supervision, and specifies the regions, statements and
reductions used by each objective. Its value lies in making those
choices explicit and testing their effects across models, datasets and
initializations. A separate annotation audit tests whether proposed
geometric rules for the MSD class division agree with the stored masks
before treating such rules as anatomical constraints.
Figure~\ref{fig:thesis-overview} summarizes the investigation; the
experimental protocols are given in Chapter~\ref{sec:methodology}.

""")
replace('Reference indicators and annotation-derived geometry provide fixed',r"""For example, suppose the model assigns probability $0.8$ to class $c$
at a voxel. The statement \emph{this voxel belongs to class $c$} then
has truth degree $0.8$, and its negative log-truth is
$-\log(0.8)\simeq0.223$. Raising the probability to $0.9$ reduces this
penalty to about $0.105$. Averaging such penalties gives an objective
that training can minimize. The statement identifies the desired
property; its grounding supplies the number used in the calculation.

Reference indicators and annotation-derived geometry provide fixed""")
replace(r"""logLTN carries out the corresponding operations in logarithmic truth
space \cite{badreddine2023logltn}. If""",r"""Multiplying many truth degrees can produce very small values.
logLTN works with their logarithms, so products become sums and these
quantities can be evaluated without first forming the product
\cite{badreddine2023logltn}. If""")
replace(r"""When those operations agree, so do their gradients. Logical notation
therefore clarifies what the model is asked to satisfy; empirical
comparisons establish whether that supervision helps. These constraints
remain soft objectives and do not guarantee satisfaction on the final
discrete prediction.""",r"""When those operations agree, so do their gradients. The implementation
evaluates the numerical losses directly. The experiments therefore test
the added spatial objectives; they do not compare a logical calculation
with an algebraically identical loss and attribute a difference to the
notation. Logical statements clarify what is supervised and where it
applies. They remain soft objectives, with no guarantee of satisfaction
on the final discrete prediction.""")
replace(r"""These quantities are fixed with respect to $\theta$ during an update.
Preprocessing keeps the image, labels and coordinate map consistent.
Where the conventional geological UNETR recipe transforms a training
example, its labels and spatial targets follow the same transformation.""",r"""These quantities are fixed with respect to $\theta$ during an update.
For example, erosion selects reference voxels before the band loss is
evaluated. Backpropagation differentiates the predictions at those
locations; it does not need to differentiate the erosion or the labels.
When a training example is spatially transformed, its image, labels,
coordinate map and derived targets are kept aligned.""")
span('For the principal MSD comparison, bands and edges use the hippocampal',r'\begin{figure}[htbp]',r"""The choice of $F$ determines which interfaces a membership statement
can describe. On MSD, $F=\{A,P\}$ represents the whole hippocampus,
whereas $F=\{A\}$ separates anterior hippocampus from both posterior
hippocampus and background. The first choice cannot distinguish an
internal class exchange that preserves the union. The experiments use
both kinds of domain, as specified in Chapter~\ref{sec:methodology}.
Figure~\ref{fig:constraint-domains} illustrates the four forms of
spatial supervision.

""")
span(r'The implementation adds the small constant $\varepsilon$ to each','The main MSD implementation averages case losses only when both bands',r"""Consider one inner voxel with $q_i=0.8$ and one outer voxel with
$q_i=0.2$. Both statements have truth degree $0.8$, and the balanced
loss is $-\log(0.8)/(1+\varepsilon)\simeq0.223$. If the outer
prediction instead assigns $0.8$ to foreground, its negated membership
statement has truth degree $0.2$, increasing the loss to
$-[\log(0.8)+\log(0.2)]/[2(1+\varepsilon)]\simeq0.916$.
Thus, satisfying the inner statement alone does not satisfy the
two-sided constraint.

Each side receives half the nominal weight even when its voxel count
differs. The small denominator constant retains a factor
$n_i^s/(n_i^s+\varepsilon)$ multiplying the negative mean log-truth
$-\ell_\theta(\Phi_i^s)$. This factor is kept for exact correspondence
with the implemented loss. Empty supports require a separate validity
rule; adding $\varepsilon$ does not make an empty mean meaningful.
The full satisfaction identity is given in
Appendix~\ref{sec:technical-details}.

""")
replace(r"""The local calculation is ordinary binary cross-entropy from grouped
float32 logits; focal and degree-based weights are disabled.""",r"""The local calculation is binary cross-entropy, evaluated stably from
the grouped logits. Its derivatives propagate through the probabilities
to the network parameters. Numerical settings are recorded in
Appendix~\ref{sec:band-implementation-details}.""")
replace(r"""Write $\delta_i(v)=q_i(v)-h_i(v)$ for the membership residual. The
transition discrepancy and its squared penalty are""",r"""For a fixed class group, write $\mathcal E_i^B=\mathcal E_{i,F}^B$
and $\delta_i(v)=q_i(v)-h_i(v)$. The transition discrepancy and its
squared penalty are""")
sdf=span(r'For $M_{i,c}=\{v\in\Omega_i:y_i(v)=c\}$, define','The local penalty combines class error and signed distance:',r"""Write $\varphi_{i,c}(v)$ for the fixed signed distance of voxel $v$
from class $c$: it is negative inside that class and positive outside.
Its magnitude expresses how far a misplaced class probability lies from
the interface. The implementation measures distance on the unit voxel
grid, clips it at eight voxels and divides by eight, bounding the
weights. The precise half-voxel convention and absent-class rules are
given in Appendix~\ref{sec:sdf-construction-details}. For readability,
write $p_{i,c}(v)=p_{\theta,i,c}(v)$ below.

""")
replace(r"""contributions. The annotation determines this spatial coordinate, while
the target remains the original semantic class. The construction thus
supervises anterior and posterior labels without identifying their
interface with a universal separating plane.""",r"""contributions. A reference plane contains the hippocampal long axis;
layers are measured across that plane, rather than along the long axis.
The annotation determines this coordinate, while each voxel keeps its
original class target. The plane therefore organizes the weights; it
does not define an anatomical anterior/posterior cut.""")
span('Task-specific variants also retain their own domains and reductions.',r'\clearpage',r"""The task-specific domains, crop conventions and sampled reductions
are specified with the experimental protocols in
Chapter~\ref{sec:methodology}. These choices matter when transferring a
constraint between datasets. At inference, the network uses its image
and any specified prompts; reference-derived spatial targets are needed
only during training.

""")
models=span('The compact model uses the Medical Open Network','The native baseline objectives are retained',r"""This selection compares a compact convolutional network, a configured
convolutional segmentation system and two ways of incorporating attention.
Keeping the same output grid and labelled cases makes their responses
to the added supervision comparable within the stated compute allowance.
Layer widths and other implementation settings are recorded in
Appendix~\ref{sec:conventional-implementation}.

""")
schedules=span('nnU-Net uses polynomial learning-rate decay with exponent 0.9 and no','Fourteen objectives cover the combinations in',r"""Each model follows a fixed learning-rate trajectory across its
objectives. If an extra loss reduces the number of updates affordable
under the cap, the schedule is not shortened to compensate. The
comparison therefore measures the accuracy reached with the complete
objective inside the allowance. Optimizer schedules and precision
settings are given in Appendix~\ref{sec:conventional-implementation}.

""")
a=s.index(r'\subsection{Adapting Pretrained Models}')
b=s.index(r'\subsection{Evaluation Measures}',a)
oldadapt=s[a:b]
fig=re.search(r'\\begin\{figure\}\[htbp\].*?\\end\{figure\}',oldadapt,re.S).group()
newfig=re.sub(r'\\caption\{.*?\\label\{fig:medsam3-workflow\}',lambda _:r"""\caption{MedSAM3 adaptation and inference on MSD. Training changes only
the medical LoRA factors using five support crops per fold. The two
fixed queries produce foreground responses, which are combined with
background to obtain mutually exclusive labels. Held-out masks are used
for evaluation and supply no localization prompts.}
\label{fig:medsam3-workflow}""",fig,flags=re.S)
newfig=newfig.replace('14 variants in one MSD campaign','14 objectives across two execution batches')
details=oldadapt.replace(r'\subsection{Adapting Pretrained Models}',r'\subsection{Pretrained Adaptation Settings and Execution Records}\label{sec:adaptation-implementation}',1)
details=details.replace(fig,'').replace('Figure~\\ref{fig:medsam3-workflow} illustrates the MSD prediction path.\n','')
for label in ['sec:sam3-methods','sec:medsam-methods','sec:medsam-adni-methods']:
    details=details.replace('\\label{'+label+'}','\\label{'+label+'-details}')
details=details.replace('126$ planned objective runs','126$ completed objective runs')
details=details.replace('Section~\\ref{sec:medsam-methods}','Section~\\ref{sec:medsam-methods-details}')
details=details.replace('Section~\\ref{sec:medsam-adni-methods}','Section~\\ref{sec:medsam-adni-methods-details}')
adapt=r"""\subsection{Adapting Pretrained Models}

The adaptation studies ask how much the same spatial supervision helps
when an image representation is already available. All three studies
freeze the SAM3 base weights and train only LoRA factors. The geological
study starts from fresh factors with a zero initial update; the medical
studies start from the released MedSAM3 adapter
\cite{carion2025sam3,liu2025medsam3}. Within each comparison, baseline and
constrained runs share the initial parameters, support images and
sampling schedule.

The baseline combines detection classification and bounding-box losses,
a presence loss, and focal and soft Dice mask losses. Spatial penalties
are added to that objective. Their coefficients are calibrated from
gradients on the support annotations. Each optimizer update uses two
adjacent-slice pairs, allowing edge supervision to compare predictions
across planes as well as within them. Reference supports are constructed
in three dimensions. Full-volume counts and sampling corrections
preserve the corresponding volume reductions in expectation; averaging
each sampled plane separately would give a different weighting.
Appendix~\ref{sec:sampled-pretrained-supervision} gives these reductions.

Inference uses image planes and fixed text queries. Mask, query and
presence confidences supply the foreground responses, which are
converted to mutually exclusive labels and reassembled into volumes.
Reference masks are used to score these predictions, never to provide
localization prompts. Each study evaluates the adapter at its final
update, without early stopping or validation-based checkpoint selection.
Dice is averaged within a run, then across seeds within each fold and
equally across folds.

\paragraph{SAM3 on GeoFWI3D.}\label{sec:sam3-methods}
Each fold uses five labelled support volumes, taken from the beginning
of its established support list, and 96 held-out volumes. The native
$96^3$ reflectivity volumes are mapped from $[-1,1]$ to $[0,1]$ for
image processing. The query \emph{salt body} produces one foreground
response; its complement represents background.

Every arm receives 25 updates, using the first 25 steps of the retained
200-update learning-rate schedule. The schedule is not compressed into
a shorter decay. Bands are calibrated at initialization; other selected
terms are calibrated after five prefix updates, during which their
weights are zero. Bands and edges use the salt mask, with two-step,
zero-padded morphology retaining crop-face support. SDF uses one
foreground field on the unit voxel grid. Depth weights use source-array
geometry, with no anatomical orientation assumption. These choices
distinguish this adaptation protocol from the conventional geological
recipes.

\paragraph{MedSAM3 on MSD.}\label{sec:medsam-methods}
Five support crops are used in each fold, with 46, 52 and 44 held-out
crops respectively. The 15 distinct support crops and 142 distinct
evaluation crops do not overlap. Fold selection followed earlier
exploratory outcomes, and participant identities are unavailable; the
crop separation therefore does not establish participant independence.
Images and masks are placed in their closest canonical NIfTI
orientation. Intensities are clipped and scaled between the first and
99th percentiles, then converted to eight-bit planes.

The fixed queries are \emph{anterior hippocampus} and
\emph{posterior hippocampus}. Their response log-odds, together with a
zero background logit, are normalized by softmax before argmax assigns
one class per voxel. Bands and edges are applied separately to
background, anterior and posterior, so they include the internal
interface. SDF uses the two foreground classes and depth uses physical
geometry. All variants receive 200 updates, with their selected
auxiliary terms ramping from the first update.

The fourteen objectives were completed across two execution batches.
The original bands and bands-plus-edge results retain their matched
baseline and calibration; the other eleven constrained variants use
the controls and calibration recorded with their batch. Those controls
are not pooled, and repeated baseline runs do not create additional
objective configurations.
Figure~\ref{fig:medsam3-workflow} shows the adaptation and prediction
path.

"""+newfig+r"""

\paragraph{MedSAM3 on ADNI.}\label{sec:medsam-adni-methods}
Each fold uses two labelled crops from two subjects and 54 held-out
crops from 27 other subjects. Folds 0 and 3 share their two support
crops; fold 1 shares one and changes the other. Only three distinct
support crops occur across the study. These folds vary the evaluation
partition, but do not provide three independent draws of a training set.

The medical initialization, 200-update schedule and intensity scaling
follow the MSD protocol. ADNI retains its stored array orientation and
uses the single query \emph{hippocampus}. The binary projection combines
the foreground response with its complement. Bands and edges supervise
the hippocampus mask, without a second background-mask average.
SDF uses unit array voxels, and depth uses source-axis geometry without
an inferred anatomical orientation.

These update budgets describe additional task adaptation and exclude
pretraining. They do not provide a FLOP match with the conventional
studies. Exact optimizer settings, calibration rules, fold and seed
identifiers, and execution environments are retained in
Appendix~\ref{sec:adaptation-implementation}. Runtime comparisons keep
their recorded scope; a training-loop timer is not an end-to-end
adaptation cost.

"""
s=s[:a]+adapt+s[b:]
replace('the results tables uses this second definition','the results figures and tables uses this second definition')
replace('fold/seed cells. Each table identifies its reduction explicitly. They are not\nconfidence intervals','fold/seed cells. Each figure or table identifies its reduction explicitly.\nThese standard deviations are not confidence intervals')
future=span('The demonstrated gains provide a foundation for extending spatial',r'\clearpage',r"""The first priority is an independent evaluation with model,
objective and data-selection choices fixed in advance. New participants
and imaging collections would test whether the gains extend beyond the
datasets used to develop these recipes. Varying annotation and compute
budgets, including longer training and matched data exposure, would
show when constraints accelerate learning and when they improve the
accuracy ultimately reached. An unweighted classification control is
also needed to separate depth weighting from the benefit of additional
class supervision.

Future experiments should retain a small, documented set of paired
baseline and constrained predictions. Cases should be selected by a
stated rule and include failures as well as improvements. Overlays,
surface-distance measures and expert inspection would help determine
whether higher overlap corresponds to better contours and internal
class assignments. The present aggregate records cannot answer those
questions on their own.

Hippocampal subfield datasets offer a further test of spatial knowledge.
MNI-HISUB25 and the CoBrA atlas contain finer anatomical subdivisions
\cite{kulaga2015subfields,winterburn2013atlas}; the dataset of
Jafari \emph{et al.}\ adds a different clinical population
\cite{jafari2011dataset}. None is evaluated here. Their annotation
protocols could support statements about shared interfaces, relative
position and continuity of thin structures. Such relations would
require expert validation and tolerances for anatomical variation.

For the MSD anterior/posterior division, expert landmark annotations
and multiplanar images could support an uncertain transition region
around the uncal apex. This would allow a comparison between
landmark-guided supervision and the existing classwise terms without
assuming that one fixed cut describes every mask.

Pretrained adaptation could be extended by varying the support-set
composition, adapter capacity and amount of neighbouring-slice context.
Accuracy and additional computation should be reported together.
Adaptive constraint weights are another possibility, but their value
would need controlled comparisons with the fixed calibration used here.
These studies would identify which changes add useful supervision and
which mainly add training cost.

""")
appendix=r"""
\subsection{Boundary-Loss Numerical Settings}
\label{sec:band-implementation-details}

The principal MSD implementation evaluates grouped logits and binary
cross-entropy in float32. Optional focal factors and degree-based
weights are disabled. Both full-volume bands must be nonempty for a
case to enter the mean; if no valid case remains, the code returns a
differentiable zero connected to the logits. The count constant is
$\varepsilon=10^{-6}$. The stabilization identity above retains its
factor in the loss and gradient.

\subsection{Signed-Distance Field Construction}
\label{sec:sdf-construction-details}

"""+sdf+r"""
\subsection{Conventional Model and Optimizer Settings}
\label{sec:conventional-implementation}

"""+models+schedules+details
needle=r'\clearpage'+'\n'+r'\section{Supplementary Experimental Results}'
replace(needle,appendix+'\n'+needle)
# Preserve the data tables and results chapter exactly.
tables=lambda text:re.findall(r'\\begin\{(?:table|longtable)\}.*?\\end\{(?:table|longtable)\}',text,re.S)
assert tables(before)==tables(s)
results=lambda text:text[text.index(r'\section{Results and Analysis}'):text.index(r'\section{Conclusions}')]
assert results(before)==results(s)
p.write_text(s)
Path('reports/thesis_professor_followup_20261103/edit_summary.json').write_text(json.dumps({'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(s.encode()).hexdigest(),'tables_unchanged':len(tables(s)),'results_chapter_unchanged':True,'moved_sdf_chars':len(sdf),'moved_adaptation_chars':len(details)},indent=2))
print('Saved revised source:',len(s.splitlines()),'lines; tables and results chapter unchanged.')

