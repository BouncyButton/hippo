from pathlib import Path
import re, json, difflib, hashlib

path=Path('/Users/filippofocaccia/Desktop/hippo/Template_tesi/thesis-revision.tex')
audit=path.parent.parent/'reports/thesis_section46_simplification_20261103'
audit.mkdir(exist_ok=True)
before=path.read_text()
(audit/'before.tex').write_text(before)
a=before.index(r'\subsection{Adapting Pretrained Models}')
b=before.index(r'\subsection{Evaluation Measures}',a)
c=before.index(r'\subsection{Pretrained Adaptation Settings}')
d=before.index(r'\subsection{Illustration Supports and Depth-Layer Counts}',c)
main=r'''\subsection{Adapting Pretrained Models}

These studies ask whether spatial supervision remains useful when a
model already has learned image representations. SAM3 is adapted to
salt segmentation, while MedSAM3 supplies the medical starting point
for hippocampus segmentation \cite{carion2025sam3,liu2025medsam3}.
All three studies keep the pretrained backbone fixed and train only
the low-rank adaptation (LoRA) parameters introduced in
Chapter~\ref{sec:background}.

Each study compares the baseline with the spatial objectives in
Table~\ref{tab:constraint-combinations}. A constrained run retains
the baseline loss and uses the same starting weights, support images
and sampling schedule as its matched baseline. This makes the added
spatial supervision the difference being examined within each pair.

Although the data are volumes, the models process image slices.
Neighbouring slices are sampled together during training so that
spatial constraints can also supervise relationships between slices.
At evaluation, the slice predictions are reassembled into a
three-dimensional segmentation. Fixed text queries name the target
structure; reference masks are used for scoring and do not provide
location prompts. Figure~\ref{fig:medsam3-workflow} summarizes this
workflow.

\begin{figure}[htbp]
\centering
\setstretch{1.05}
\begin{tikzpicture}[x=1cm,y=-1cm,font=\small,
  every node/.style={text=thesisInk},>=stealth]
\node[anchor=west,text=thesisGray,font=\footnotesize\bfseries]
  at (0,-0.35) {ADAPTATION};
\draw[fill=thesisPaleBlue,draw=thesisBlue,rounded corners=2pt]
  (0,0) rectangle (4.8,1.5);
\node[align=center,text width=4.4cm] at (2.4,0.75)
  {Labelled support volumes\\Images and reference masks};
\draw[fill=thesisPaleOrange,draw=thesisOrange,rounded corners=2pt]
  (6,0) rectangle (15.4,1.5);
\node[align=center,text width=9cm] at (10.7,0.75)
  {Train the adapters; keep the backbone fixed\\Baseline or baseline + spatial constraints};
\draw[thesisGray,->,line width=.8pt] (4.8,0.75) -- (6,0.75);
\draw[thesisGray,->,dashed,line width=.8pt]
  (10.7,1.5) -- (10.7,2.15) -- (7.7,2.15) -- (7.7,3.15);
\node[anchor=east,text=thesisGray,font=\footnotesize]
  at (10.6,1.88) {Trained adapters};
\node[anchor=west,text=thesisGray,font=\footnotesize\bfseries]
  at (0,2.8) {PREDICTION ON HELD-OUT VOLUMES};
\draw[fill=thesisPaleBlue,draw=thesisBlue,rounded corners=2pt]
  (0,3.15) rectangle (4.4,4.95);
\node[align=center,text width=4cm] at (2.2,4.05)
  {Held-out image slices\\Fixed text queries};
\draw[fill=thesisPaleOrange,draw=thesisOrange,rounded corners=2pt]
  (5.5,3.15) rectangle (9.9,4.95);
\node[align=center,text width=4cm] at (7.7,4.05)
  {Adapted model\\Slice predictions};
\draw[fill=thesisPaleBlue,draw=thesisBlue,rounded corners=2pt]
  (11,3.15) rectangle (15.4,4.95);
\node[align=center,text width=4cm] at (13.2,4.05)
  {Reassemble slices\\3D segmentation};
\draw[thesisGray,->,line width=.8pt] (4.4,4.05) -- (5.5,4.05);
\draw[thesisGray,->,line width=.8pt] (9.9,4.05) -- (11,4.05);
\end{tikzpicture}
\caption{Shared workflow for SAM3 and MedSAM3 adaptation. Support
annotations guide adapter training. The adapted model then segments
held-out image slices using fixed text queries, and the predictions
are assembled into a volume. Held-out annotations are used only to
evaluate the resulting segmentation.}
\label{fig:medsam3-workflow}
\end{figure}

\paragraph{SAM3 on GeoFWI3D.}\label{sec:sam3-methods}
Each fold uses five labelled support volumes and 96 held-out volumes.
Starting from SAM3 with fresh adapters, each objective receives
25 updates. The fixed query \emph{salt body} identifies the foreground,
giving a binary salt/background segmentation.

\paragraph{MedSAM3 on MSD.}\label{sec:medsam-methods}
Each fold uses five support crops and 200 updates, starting from the
released medical adapters. The fixed queries \emph{anterior hippocampus}
and \emph{posterior hippocampus} produce a segmentation with those two
classes and background. The three folds have 46, 52 and 44 held-out
crops respectively, disjoint from the support crops. Fold selection
was exploratory, and unavailable participant identities prevent a
check of participant-level independence.

The fourteen objectives span two execution batches. Each constrained
objective retains its own batch's matched baseline; baseline results
from the two batches are not pooled.

\paragraph{MedSAM3 on ADNI.}\label{sec:medsam-adni-methods}
Each fold uses two labelled crops from two subjects, 200 updates, and
54 held-out crops from 27 other subjects. The query \emph{hippocampus}
produces a binary segmentation. Support sets overlap between folds:
only three distinct support crops occur across the study. These folds
therefore test different evaluation partitions without providing
three independent draws of the training set.

These allowances measure additional task adaptation and exclude
pretraining. They are not matched in FLOPs to the studies from scratch.
Appendix~\ref{sec:adaptation-implementation} records the preprocessing,
prediction conversion, loss settings and training schedules.

'''
details=r'''\subsection{Pretrained Adaptation Settings}
\label{sec:adaptation-implementation}

All three studies freeze the SAM3 base weights and update 18,497,664
LoRA parameters. Geological adaptation uses fresh factors that initially
leave the base model unchanged; medical adaptation starts with the
released MedSAM3 factors. Every run restores its matched initial
weights and sampling schedule.

The pretrained baseline supervises object classification, bounding
boxes, target presence and segmentation masks. The mask losses combine
soft Dice and focal loss, a form of cross-entropy that downweights
confident correct predictions. Spatial terms are added to this
objective. Full-volume masks define their supports and normalization
counts before slice sampling, as specified in
Section~\ref{sec:sampled-pretrained-supervision}.

AdamW uses weight decay 0.01, gradient-norm clipping at one, 20 warmup
updates, peak learning rate $5\times10^{-5}$ and final rate
$5\times10^{-6}$ over a 200-update horizon. Two adjacent-plane pairs
are accumulated per update. Medical runs execute 200 updates; SAM3
executes the first 25 without compressing the learning-rate schedule.
These are target-specific allowances and exclude pretraining.

For inference, each query's foreground response combines mask-location
confidence, query matching and target presence. The class conversion
is binary for salt and ADNI and multiclass for MSD, as detailed below.

\paragraph{SAM3 on GeoFWI3D.}\label{sec:sam3-methods-details}
Each fold takes the first five volumes from its established support
list. Native $96^3$ reflectivity volumes are mapped from $[-1,1]$ to
$[0,1]$ for image processing. The \emph{salt body} response and its
complement define the foreground/background probabilities. Bands and
edges use the salt mask with two-step, zero-padded morphology, retaining
crop-face support. Signed distance uses one foreground field on the
unit voxel grid; depth uses source-array geometry without anatomical
orientation assumptions.

The five support volumes provide twenty calibration probes (four
anchors each). Bands are calibrated at initialization. Other selected
terms are calibrated after five prefix updates in order $B\to E\to S\to D$,
retaining the band coefficient. Median/95th-percentile targets are
0.1/0.5. Terms awaiting calibration have zero weight, then enter the
global ramp $\min\{1,t/20\}$; entry at update six gives factor 0.3.
The 25 updates present 100 slices and class queries. Final evaluation
covers five support and 96 held-out volumes. The original 126 runs
span full Colab A100-SXM4 GPUs and A100 PCIe MIG partitions;
hardware was not fully crossed with objectives.

\paragraph{MedSAM3 on MSD.}\label{sec:medsam-methods-details}
Across folds, the support and validation sets contain 15 and 142
distinct crops respectively, with no overlap between the two groups.
Images and labels are jointly reordered and, where necessary, flipped
into the closest right--anterior--superior orientation recorded by
their NIfTI transforms. Intensities are clipped and scaled between
the first and ninety-ninth percentiles, then converted to eight-bit
planes.

For each anatomical query, response probability $q$ is converted to
log-odds, $\log(q/(1-q))$. The two foreground scores and a zero
background logit are normalized by softmax; argmax assigns one class
per voxel. Bands and edges are averaged over background, anterior and
posterior masks, including the internal interface. Signed distance
uses the two foreground classes, and depth uses physical geometry.

The released medical factors initialize 916 trainable adapter tensors.
Five support crops and 200 updates give 800 slice and 1,600 foreground
query presentations. Twenty calibration probes measure gradients with
respect to LoRA parameters in sequential order $B\to E\to S\to D$,
using targets 0.1/0.5. Combination-specific coefficients are shared
across optimization seeds within each fold according to the saved
calibration records. $B$ and $B+E$ keep their original calibration and
matched baseline; the other eleven constrained objectives use the
separately audited execution batch. Controls are not interchanged,
and repeated baseline runs do not enlarge the objective inventory.
Every selected term uses $\min\{1,t/20\}$ from the first update,
without delayed signed-distance activation.

\paragraph{MedSAM3 on ADNI.}\label{sec:medsam-adni-methods-details}
Folds 0 and 3 share both support crops; fold 1 shares one and changes
the other. Subject separation from validation follows
Section~\ref{sec:medsam-adni-methods}. The medical initialization and
intensity scaling follow MSD, but ADNI retains its stored array
orientation. The \emph{hippocampus} response and its complement define
the binary probabilities. Bands and edges supervise the foreground
mask without a second background-mask average. Signed distance uses
unit array voxels, and depth uses source-axis geometry without inferred
anatomical orientation.

Two support crops provide eight calibration probes, with the same
medical optimizer, sequential calibration targets and activation ramp.
The 200 updates give 800 slice and 800 single-class-query presentations.
Final evaluation covers both support crops and all 54 held-out crops.

Training-loop timers exclude initialization, calibration, validation
and inference; they do not measure peak memory or end-to-end adaptation
cost. All three studies use final-update evaluation without early
stopping. Matching initialization and sampling does not guarantee
bitwise-identical CUDA execution, as the qualitative reproductions
document in Section~\ref{sec:qualitative-predictions}.

'''
after=before[:a]+main+before[b:c]+details+before[d:]
assert path.read_text()==before,'Concurrent source edit'
path.write_text(after)
(audit/'after.tex').write_text(after)
(audit/'changes.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before',tofile='after')))
def prose(t):
 t=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',t,flags=re.S)
 t=re.sub(r'\\(?:subsection|paragraph|label|ref)\{[^}]+\}','',t)
 return len(t.split())
record={'section46_prose_words_before':prose(before[a:b]),'section46_prose_words_after':prose(main),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(audit/'checks.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

