from pathlib import Path
import re,json,hashlib
root=Path('reports/thesis_professor_revision_20261102')
p=Path('Template_tesi/thesis-revision.tex');original=p.read_text();s=original
assert original==(root/'source_before.tex').read_text(), 'Source changed; reconcile first'
def sub(old,new):
 global s
 assert s.count(old)==1,(old[:90],s.count(old))
 s=s.replace(old,new)
sub('\\section*{Abstract}\n\n'+s.split('\\section*{Abstract}\n\n',1)[1].split('\\clearpage',1)[0],r'''% Submission version: abstract retained here, excluded from the compiled thesis.
% Set \thesisincludeabstracttrue if the programme requires it in the PDF.
\newif\ifthesisincludeabstract
\thesisincludeabstractfalse
\ifthesisincludeabstract
\section*{Abstract}

'''+(root/'abstract.txt').read_text()+'\n\\clearpage\n\\fi\n')
# Avoid a blank page when abstract is excluded.
sub('\\fi\n\\clearpage\n\\setcounter{tocdepth}', '\\fi\n\\setcounter{tocdepth}')
a=s.index('Three-dimensional semantic segmentation turns');b=s.index('\\begin{figure}',a)
s=s[:a]+(root/'intro.txt').read_text()+s[b:]
# Move the existing real examples to the introduction; do not fabricate predictions.
figs=list(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S))
fig=next(m.group() for m in figs if r'\label{fig:dataset-examples}' in m.group())
s=s.replace(fig,'')
anchor='\\begin{figure}[htbp]\n\\centering\n\\setstretch{1.05}'
pos=s.index(anchor,s.index('\\section{Introduction}'));s=s[:pos]+fig+'\n\n'+s[pos:]
# Simplify the introductory overview's protocol details.
s=s.replace('4 architectures; 14 objectives\\\\Dataset-specific FLOP budgets','Convolutional and transformer models\\\\Matched computation within each dataset')
s=s.replace('MedSAM3: MSD / ADNI, 200 updates\\\\5 / 2 labelled support crops\\\\SAM3: GeoFWI3D, 25 updates\\\\5 labelled support volumes','SAM3 on geological images\\\\MedSAM3 on medical images\\\\Few labelled examples\\\\Short adaptation schedules')
sub('''Spatial supervision combines a segmentation model with a rule for comparing
its predictions to annotation-derived geometry. This chapter introduces
the probability representation, conventional objectives and model families
used in that comparison, then develops the logical vocabulary and relevant
prior work. Experiment-specific settings are collected in
Chapter~\\ref{sec:methodology}.

''','')
sub('For case $i$, let',r'''A digital image is an array of measurements. In a two-dimensional image,
each entry is a pixel; in a volume, each entry is a voxel. Multiple
channels can represent colour or different imaging measurements, although
the datasets used here have one input channel. Image classification
assigns a label to an entire image, object detection locates objects,
and semantic segmentation assigns a class at every location. Segmentation
therefore preserves the spatial information needed to describe a region's
size and shape.

For an image volume indexed by $i$ (a \emph{case}), let''')
sub('''mutually exclusive classes. Background has label zero. The one-hot value''','''mutually exclusive classes. The subscript $i$ allows native image dimensions
to vary across volumes; individual training recipes may prepare a fixed
input size. Background has label zero. The one-hot value''')
sub('''$z_{\\theta,i,c}(v)$, converted to probabilities by''','''$z_{\\theta,i,c}(v)$, which are unrestricted class scores. The softmax
operation converts them to nonnegative probabilities that sum to one:''')
sub('''The same notation covers predictions assembled from
two-dimensional slices once their outputs have been converted to a common
semantic probability field.''','')
sub('''Salt segmentation uses $\\mathcal C=\\{0,S\\}$ and binary hippocampus
segmentation uses $\\{0,H\\}$. MSD instead uses $\\{0,A,P\\}$ for background,
anterior and posterior hippocampus.''','''Salt segmentation uses $\\mathcal C=\\{0,S\\}$, where $S$ denotes salt.
Binary hippocampus segmentation uses $\\{0,H\\}$, where $H$ denotes
hippocampus. MSD uses $\\{0,A,P\\}$: $A$ and $P$ denote anterior and
posterior hippocampus, respectively.''')
sub(r'\ell_{\mathrm{CE},i}',r'\mathcal{L}_{\mathrm{CE}}(x_i,y_i;\theta)')
sub('''Cross-entropy penalizes low probability for the reference class:''','''A loss assigns a numerical penalty to disagreement between a prediction
and its annotation. Training adjusts the parameters to reduce that
penalty. We write losses as $\\mathcal L(x_i,y_i;\\theta)$; later family
subscripts abbreviate these inputs when they are fixed by the context.
Cross-entropy penalizes low probability for the reference class:''')
sub('''NIfTI headers store coordinate transforms and spatial units''','''Neuroimaging Informatics Technology Initiative (NIfTI) headers store
coordinate transforms and spatial units''')
sub('''The four architectures trained from scratch all connect an encoder to a
decoder that restores voxelwise predictions. Their encoders construct
context in different ways. Comparing them tests whether the same
output-level spatial objective remains useful when the underlying
representation and cost of optimization change.''','''The four architectures provide different ways to use the surrounding
image when classifying a voxel. A convolutional model builds context
through repeated local operations; a transformer can combine features
from more distant locations through attention. Testing both helps assess
whether a spatial objective is useful beyond a single model family.
All four connect an encoder, which extracts image features, to a decoder,
which converts those features into voxelwise predictions. The comparison
also includes two convolutional baselines: a compact U-Net and the more
elaborate nnU-Net training pipeline. It therefore probes sensitivity to
both representation and training recipe, rather than isolating encoder
architecture alone.''')
sub('''Logic Tensor Networks associate first-order expressions with numerical
groundings that can be differentiated during learning''','''Logic Tensor Networks combine neural predictions with logical statements
about the desired output. Their numerical interpretation, called a
\emph{grounding}, maps a statement to a truth degree in $[0,1]$; this map
is denoted by $\\mathcal G_\\theta$. It makes statements differentiable
with respect to the model parameters''')
sub('''The satisfaction of a compound statement depends on the selected logical
operators; it need not be a calibrated probability of the statement as
a whole.''','''The satisfaction of a compound statement depends on how its component
truth degrees are combined. It measures agreement with the chosen rule;
it is not automatically the probability that the entire rule is true.''')
a=s.index('For example, requiring foreground membership on an annotated inner band');b=s.index('Geometric errors can enter',a)
s=s[:a]+s[b:]
sub('''implementation evaluates that violation directly; it need not compute
an exponential and subsequently its logarithm.''','''implementation evaluates that violation directly, avoiding a redundant
exponential followed by a logarithm.''')
s=s.replace('et al.\\',r'\emph{et al.}\\') if False else s
# Italicize prose occurrences only, leaving bibliography entries intact.
a=s.index('% Embedded bibliography');s=s[:a].replace('et al.\\','\\emph{et al.}\\')+s[a:]
sub('''SAM combines an image encoder''','''The Segment Anything Model (SAM) combines an image encoder''')
sub('''Low-Rank Adaptation parameterizes''','''Low-Rank Adaptation (LoRA) parameterizes''')
sub('''For the six-neighbour structuring element''','''For a fixed class group $F$, write $M=M_{i,F}$, $B_i^s=B_{i,F}^s$,
$q_i(v)=q_{\\theta,i,F}(v)$ and $h_i(v)=h_{i,F}(v)$ to simplify notation.
For the six-neighbour structuring element''')
sub('''For nonempty sides with $n_i^s=|B_i^s|$, the implemented loss is''',r'''The symbol $\Phi_i^s$ names the statement for side $s$, rather than a
new model output. Its truth is obtained from the membership probability:
$\mathcal G_\theta(\mathsf{In}_F(i,v))=q_i(v)$ inside and
$\mathcal G_\theta(\lnot\mathsf{In}_F(i,v))=1-q_i(v)$ outside.
Taking the negative log and averaging over each side gives the two terms
below. For nonempty sides, define their voxel counts
$n_i^s=|B_i^s|$, with $s\in\{\mathrm{in},\mathrm{out}\}$. The implemented loss is''')
sub('''Reducing each side separately gives them equal nominal weight even when''',r'''The constant $\varepsilon$ stabilizes the count denominators. For a
nonempty side, its contribution is
$-\frac{n_i^s}{n_i^s+\varepsilon}\ell_\theta(\Phi_i^s)$;
the half-sum of these contributions is Equation~\ref{eq:balanced-band-loss}.
This explicitly connects the quantified statements to the training loss.
Reducing each side separately gives them equal nominal weight even when''')
sub('''Equal reference labels have target difference zero. Across an annotated
boundary, the signed target is $+1$ or $-1$.''','''The selected pairs include neighbours on the same side of the boundary
as well as neighbours on opposite sides. Equal reference labels have
target difference zero, so their predictions should agree. For an inner
voxel paired with an outer voxel, the predictions should approach one
and zero, respectively. Across such an annotated boundary, the signed
target difference is $+1$ or $-1$.''')
sub('''(b) Adjacent pairs within the band support; coloured segments join voxel
centres.''','''(b) Adjacent pairs within the band support: the black line is the
annotation boundary, while coloured segments join voxel centres. Orange
segments cross between labels; blue segments connect equal labels.''')
# Keep the reason for seed inventories explicit and concise (reviewer question).
sub('''Within each dataset, all four
architectures and all fourteen objectives share the same seed set,''','''The numeric seed values identify pseudorandom streams; equal numbers
across different models or datasets would not make their initial states
identical. Each study retains its recorded seed inventory for reproducibility.
Within each dataset, all four architectures and all fourteen objectives
share the same seed set,''')
# Define computational units before first prose use after introduction.
sub('''four-model studies match optimization FLOPs;''','''four-model studies match the number of floating-point operations (FLOPs)
used during optimization;''')
# Remove fine-grained headings where continuous prose suffices; retain labels.
remove_titles=['Images and Reference Masks','Three-Fold, Three-Seed Design','Inputs and Frozen Partitions','Source Images and Reference Labels','Preparation, Partitions and Spatial Geometry','Initialization, Support Data and Predictions','Objectives, Calibration and Update Schedule','Paired Evaluation and Resource Accounting','Initialization, Supports and Prediction','Spatial Objectives and Optimization','Evaluation and Execution Measurements','Supports, Preprocessing and Prediction','Spatial Objectives, Optimization and Evaluation','Segmentation Quality and Paired Effects']
for title in remove_titles:
 sub('\\subsubsection{'+title+'}\n','')
# Explicitly mark missing qualitative evidence without a visible drafting instruction.
s=re.sub(r'\\noindent\\textit\{\[Presentation(?: evidence)? to insert:.*?\]\}', '% Qualitative prediction panels remain unavailable; limitation stated in Results.',s,flags=re.S)
# Future Work is standard uncountable usage; use a heading that also resolves the suggested plural.
sub('\\subsection{Future Work}', '\\subsection{Future Research Directions}')
# Replace overlong conclusion with synthesis, leaving future research intact.
a=s.index('This thesis demonstrates that spatial information');b=s.index('\\subsection{Future Research Directions}',a)
s=s[:a]+r'''This thesis examined whether segmentation models can make better use of
limited annotations by explicitly supervising their spatial information.
The LTN/logLTN formulation connects four properties of the label maps to
trainable losses: membership near boundaries, agreement between
neighbouring labels, classwise distance and supervision across depth
layers. These terms require no additional manual labels and leave the
model's inference inputs unchanged.

The experiments support this approach under the evaluated data and
training budgets. Benefits occur in both geological and medical tasks,
and in models trained from scratch as well as adapted pretrained models.
Their size and the preferred objective depend on the learning setting.
The contribution is therefore a set of explicitly defined spatial
objectives and evidence of where they help, rather than a universally
preferred combination. The full comparisons, error tradeoffs and limits
of this evidence are discussed in Chapter~\ref{sec:results}.

'''+s[b:]
(root/'prose_stage.tex').write_text(s)
print('Prose stage prepared',len(s))
