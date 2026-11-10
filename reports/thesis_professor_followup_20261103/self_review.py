from pathlib import Path
p=Path('Template_tesi/thesis-revision.tex');s=p.read_text()
a=s.index(r'\subsection{Boundary-Loss Numerical Settings}')
b=s.index(r'\subsection{Signed-Distance Field Construction}',a)
band=s[a:b].replace(r'\subsection{Boundary-Loss Numerical Settings}',r'\paragraph{Boundary-loss settings.}').replace('The stabilization identity above retains its','Equation~\\ref{eq:balanced-band-satisfaction} retains its')
s=s[:a]+s[b:]
target=r'\subsection{Relational and Signed-Distance Derivatives}'
s=s.replace(target,band+'\n'+target,1)
s=s.replace('14 objectives across two execution batches','14 objectives; two execution batches')
a=s.index(r'\subsection{Pretrained Adaptation Settings and Execution Records}')
b=s.index(r'\clearpage'+'\n'+r'\section{Supplementary Experimental Results}',a)
s=s[:a]+r"""\subsection{Pretrained Adaptation Settings and Execution Records}
\label{sec:adaptation-implementation}

Section~\ref{sec:sam3-methods} defines the SAM3 support data, input
preparation and binary prediction rule; Sections~\ref{sec:medsam-methods}
and \ref{sec:medsam-adni-methods} give the corresponding MedSAM3
protocols. This section records their exact optimizer, calibration and
execution settings. All three studies optimize 916 LoRA tensors
containing 18,497,664 parameters while keeping the base weights frozen.

\paragraph{SAM3 on GeoFWI3D.}\label{sec:sam3-methods-details}
Folds 0, 1 and 3 are crossed with seeds 0, 1 and 2 and fourteen
objectives, giving 126 objective runs. Each update accumulates two
adjacent-slice pairs. The 25 updates therefore give 100 slice and
single-class-query presentations, repeatedly drawn from five support
volumes. AdamW uses weight decay 0.01 and gradient-norm clipping at
one. Its retained 200-update learning-rate horizon has a 20-update
warmup and peak/final rates of $5\times10^{-5}$ and $5\times10^{-6}$.
Only the first 25 steps are executed.

Calibration uses four anchor pairs per support volume, giving twenty
probes. Bands are calibrated at initialization for each fold/seed cell.
Other selected terms are calibrated sequentially after five prefix
updates, in the order $B\to E\to S\to D$, retaining the band
coefficient when present. Each additional term targets a median
gradient ratio of 0.1 with a 95th-percentile cap of 0.5. Terms awaiting
calibration have zero weight during the prefix; their fixed calibrated
coefficients then follow the global ramp $\min\{1,t/20\}$. A term
entering at update 6 consequently receives ramp factor $6/20$.

The sampled salt losses use full-volume supports and counts. Bands
retain two-step zero-padded morphology, including crop faces.
The half-voxel signed field uses unit-grid distances clipped at eight
voxels and divided by eight; depth uses source-array geometry.
The sampled reductions are in
Appendix~\ref{sec:sampled-pretrained-supervision}.

Each arm restores the same base and initial adapters and uses its
matched sampling schedule. Evaluation after update 25 covers all five
support and 96 held-out volumes. Paired gains use only the SAM3
baseline from the same fold and seed.

The completed inventory contains 76 runs on Colab A100-SXM4-80GB
graphics processing units (GPUs) and 50 on A100 80GB PCIe GPUs allocated
as 4g.40gb Multi-Instance GPU (MIG) partitions. All retain the same
scientific source identity. Archived hashes verify unchanged base
parameters and matching initial adapters, support arrays and sampling
schedules within each comparison. Hardware allocation was not fully
crossed with the objectives, so timing records retain their execution
platform. Recovery attempts are distinguished from the final
25-update trajectory.

\paragraph{MedSAM3 on MSD.}\label{sec:medsam-methods-details}
Folds 0, 1 and 3 are crossed with seeds 17, 83 and 191. Fourteen
objectives give 126 completed objective configurations across the
nine fold/seed cells, with repeated matched controls retained in the
execution records. The starting factors are the released medical
adapter. Each 200-update run uses two adjacent-slice pairs per update,
giving 800 slice and 1,600 foreground-query presentations. These
repetitions do not increase the five-crop annotation budget.

AdamW uses 20 warmup updates, peak/final rates of $5\times10^{-5}$
and $5\times10^{-6}$, weight decay 0.01 and gradient-norm clipping
at one. Every selected auxiliary term ramps as $\min\{1,t/20\}$
from the first update, without the delayed SDF policy of the conventional
models. Support volumes and anchors are sampled uniformly. Bands,
SDF and depth use the anchor plane; edges also use its companion.
The full-volume denominators and sampling factors are specified in
Appendix~\ref{sec:sampled-pretrained-supervision}.

Calibration uses four anchors per support volume, giving twenty
slabs. Gradients are measured with respect to the LoRA parameters,
retaining vector interactions in the sequential order
$B\to E\to S\to D$. Equation~\ref{eq:constraint-calibrated-coefficient}
uses $(\eta_a,\kappa_a)=(0.1,0.5)$. Coefficients depend on the selected
combination and use support data only. Within each fold they are shared
across optimization seeds according to the saved calibration-source
records. The $B$ and $B+E$ variants retain their original band and edge
calibration and matched baseline; the other eleven constrained variants
use their separately audited execution batch. Coefficients and controls
are not substituted between these groups.

Bands and edges average the background, anterior and posterior masks.
SDF uses the foreground fields with half-voxel shifts, clipping at eight
units and division by eight on MSD's verified 1 mm grid. Depth uses
physical geometry. Native-grid inference and final-update evaluation
follow Section~\ref{sec:medsam-methods}.

\paragraph{MedSAM3 on ADNI.}\label{sec:medsam-adni-methods-details}
Folds 0, 1 and 3 and seeds 0, 1 and 2 give 126 runs across fourteen
objectives. Support selection follows the saved subject manifest and
alternates left/right crops; the overlap between support sets is
specified in Section~\ref{sec:medsam-adni-methods}. The medical
initialization and optimizer settings are the same as for MSD.
With a single foreground query, the 200 updates give 800 slice and
800 query presentations from two support crops.

Four anchors per support volume give eight calibration slabs.
The sequential gradient calibration uses
$(\eta_a,\kappa_a)=(0.1,0.5)$ with combination-specific coefficients
computed from support data only. Schedule hashes verify matched
sampling across arms. Bands and edges use the single hippocampus
mask and its complement, without a second background-mask mean.
The half-voxel SDF is clipped at eight unit array voxels and divided
by eight. Depth uses the source-axis geometric reference.
Evaluation uses the final 200-update adapter on both support crops
and all 54 held-out crops.

In both medical studies, volume scores and mean per-crop confusion
counts are averaged within runs, across seeds within a fold and
equally across folds. No early stopping or validation-dependent
checkpoint selection is used. Training-loop timers exclude
initialization, calibration, validation and inference, and provide no
peak-memory measurement. The adaptation allowance excludes pretraining.
Production CUDA execution does not establish bitwise determinism from
a common seed alone. Execution and calibration records are indexed in
Appendix~\ref{sec:evidence-register}.

"""+s[b:]
s=s.replace('Fourteen\nobjectives give 126 completed objective configurations across the\nnine fold/seed cells','Fourteen\nobjectives give 126 completed objective runs across the\nnine fold/seed cells')
p.write_text(s)
print('Saved appendix consolidation.')

