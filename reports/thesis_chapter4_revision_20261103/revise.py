from pathlib import Path
import difflib, hashlib, json

ROOT = Path('/Users/filippofocaccia/Desktop/hippo')
REPORT = ROOT / 'reports/thesis_chapter4_revision_20261103'
SOURCE = ROOT / 'Template_tesi/thesis-revision.tex'
old = (REPORT / 'chapter4_before.tex').read_text()
new = old

def once(before, after):
    global new
    assert new.count(before) == 1, before[:100]
    new = new.replace(before, after, 1)

def between(start, end, replacement):
    global new
    a = new.index(start)
    b = new.index(end, a)
    new = new[:a] + replacement + new[b:]

between('Each run trains and evaluates', r'\subsection{Design and Units of Comparison}', r'''The experiments test whether spatial supervision helps models learn from
small labelled datasets under limited training budgets. They cover two
medical datasets and a geological application, using both models trained
from scratch and pretrained models adapted to the task. Within each model,
training with constraints is compared with a baseline using the same
training configuration and examples. Comparisons between models also
reflect their different configurations. This chapter describes the data,
the training procedures and the measures used to assess the predictions.

''')

once(r'''A fold identifies a partition into training and validation examples;
validation examples are held out from parameter fitting. A seed initializes
the pseudorandom generator controlling training randomness. Repeating seeds
within a fold measures sensitivity to initialization and data presentation;
changing folds also changes the support and evaluation images. These
repetitions provide evidence about consistency under the available data,
while sharing parts of the same experimental setting.''', r'''A run trains and evaluates one model with a specified objective. Its
labelled training examples form the \emph{support set}. A fold identifies
a partition into training and validation examples, with validation held
out from parameter fitting. A seed initializes the pseudorandom generator
controlling training randomness. Repeating seeds within a fold tests
sensitivity to initialization and data presentation; using several folds
also tests different data partitions. The dataset descriptions below
specify where support examples overlap between folds.''')

once(r'''A checkpoint is a saved
model state. Evaluation uses the final state allowed by the budget in the
capped experiments, the final 200-update MedSAM3 adapter and the final
25-update SAM3 adapter, without validation-selected checkpoint replacement
or early stopping.''', r'''Evaluation uses the final saved model state, or \emph{checkpoint},
allowed by each training budget. Runs do not stop early or replace that
state with a checkpoint chosen for its validation score.''')

once(r'''Table~\ref{tab:experimental-design} records the six studies''', r'''Training from scratch is limited by floating-point operations (FLOPs),
which measure arithmetic work; one GFLOP is one billion operations.
Medical budgets cover optimization, with calibration recorded separately;
the geological budget includes both. Adaptation instead fixes the number
of updates to the pretrained model's adapters. These budgets support
comparisons within each study and do not make all six studies equal in cost.

Table~\ref{tab:experimental-design} records the six studies''')

once(r'''Table~\ref{tab:dataset-task-overview} summarizes the input types and label
vocabularies. Each dataset section below illustrates these with an image,
its reference mask and an overlay of the two. The figures show slices
through the native three-dimensional volumes, before model-specific
padding or cropping. Only MSD includes an internal anterior/posterior division.
The experiments evaluate segmentation, with no downstream diagnostic task.
Within the chosen viewing direction, each example shows the slice with
the largest labelled area. The colours are transparent in the overlays
so that the underlying image remains visible. Image and label panels use
the same view and preserve the proportions of the stored grid.''', r'''Table~\ref{tab:dataset-task-overview} summarizes the images and labels.
The examples in the following sections pair an image slice with its
reference mask and a transparent overlay. They use the native grids,
before model-specific preparation, and show the slice with the largest
labelled area in the chosen viewing direction. Only MSD includes an
internal anterior/posterior division. The experiments evaluate
segmentation, with no downstream diagnostic task.''')

between('For preprocessing, each reflectivity volume is divided by',
        r'\subsection{MSD: Medical Segmentation Decathlon}', r'''The salt annotation uses two fields supplied by the simulator: velocity
and relative geological time (RGT), a coordinate describing the ordering
of geological layers. A voxel is labelled as salt when its RGT is zero
and its velocity matches the scene's recorded salt value. Using RGT alone
would also include null background artifacts. The preparation checks that
every voxel with the recorded salt velocity lies in the zero-RGT region
and retains all such components, including disconnected ones. This avoids
treating the background artifacts as salt boundaries.
% Verified label rule: scripts/geofwi3d_data.py, salt_mask.
% Velocity equality uses np.float32(metadata.salt_vel), as in the source volume.

Each reflectivity image is divided by the 99.5th percentile of its
absolute intensities and clipped to $[-1,1]$. This normalization uses
only image values and preserves the native $96^3$ grid.

For training from scratch, folds 0, 1 and 2 each supply 19 labelled
training scenes and 96 validation scenes. The support set is a fixed
approximately 5\% subset of the eligible training pool. Training and
validation are disjoint within each fold, and the three validation sets
are pairwise disjoint. The support sets contain 54 distinct scenes across
their 57 entries. All models and objectives use these same partitions
and the seeds in Table~\ref{tab:experimental-design}. SAM3 uses the smaller
support sets specified in Section~\ref{sec:sam3-methods}.

''')

once(r'''The MSD collection contains 263 labelled MRI crops with background,
anterior and posterior hippocampus classes \cite{simpson2019dataset}.''', r'''The downloaded MSD hippocampus dataset used here contains 260 labelled
MRI crops with background, anterior and posterior hippocampus classes
\cite{simpson2019dataset}. All 260 image--label pairs are retained in the
local preparation and form the pool for the experiments and annotation
audit.
% Verified against Task04_Hippocampus.tar: descriptor and inventory both 260.
% Archive matches MONAI's published MD5; all local image/label bytes match.
% Audit: reports/thesis_chapter4_revision_20261103/msd_count_verification.json.''')

once(r'''whether one coronal separation reproduces the stored labels in all
260 crops.''', r'''whether one coronal separation reproduces the stored labels across the
complete 260-crop collection.''')
once(r'''reports the findings within the limitations and explains their
consequences for constraint design.''', r'''reports the findings and their consequences for constraint design.''')
once(r'''builder uses the original MINC-format image arrays to export paired crops.''', r'''preparation uses the original images in Medical Imaging NetCDF (MINC)
format, a format for storing medical image volumes, to export paired crops.''')

between('Crops are centred on the reference-mask bounding box,',
        r'\subsection{Four-Model Comparisons under a FLOP Ceiling}', r'''Crops are centred on the reference-mask bounding box, with identical
indices applied to the image. Their evaluation therefore assumes prior
hippocampal localization. For the capped experiments, images are
z-score normalized over nonzero intensities and padded to $64^3$ without
resizing. The exported crops do not retain the original spacing and
orientation metadata, so their spatial constraints use voxel coordinates.

Folds 0, 1 and 2 keep both sides of each subject together. Each has an
eligible training pool of 108 subjects (216 crops) and 27 validation
subjects (54 crops), with no within-fold subject overlap. Five training
subjects contribute both sides, giving ten support crops per fold shared
across models and seeds. MedSAM3 uses the separate two-crop support
protocol in Section~\ref{sec:medsam-adni-methods}.

''')

between(r'\subsubsection{Models and Baseline Objectives}',
        r'\begin{table}[htbp]', r'''\subsubsection{Models and Training Configurations}

The four models represent different approaches to volumetric
segmentation: a compact convolutional U-Net, the configured convolutional
system nnU-Net, UNETR with attention across image patches, and SwinUNETR
with attention within shifted local windows. Chapter~\ref{sec:background}
introduces these architectures. Testing all four examines whether the
benefit of spatial supervision depends on the model receiving it.

All start from random initialization and process one volume per update.
Within each dataset, they use the same support examples and prediction
grid: $64^3$ for MSD and ADNI, and $96^3$ for GeoFWI3D. MSD requires three
output classes; ADNI and GeoFWI3D require two. Model sizes, optimizer
settings and learning-rate schedules are recorded in
Appendix~\ref{sec:conventional-implementation}.

Table~\ref{tab:capped-baseline-losses} summarizes the medical baseline
objectives. SwinUNETR, UNETR and the compact U-Net use soft Dice over all
classes, implemented with the Medical Open Network for Artificial
Intelligence (MONAI) library. nnU-Net combines foreground Dice with
cross-entropy over all classes. On GeoFWI3D, these choices are retained
except for UNETR, which uses foreground Dice plus cross-entropy and
squared prediction terms in the Dice denominator. These differences
are held fixed within each baseline/constrained comparison.

''')

between('nnU-Net also retains its weighted losses at multiple output resolutions,',
        'Fourteen objectives cover the combinations in', r'''nnU-Net applies its baseline losses at several output resolutions,
resampling the reference labels to each grid. All auxiliary terms act
only on the primary, full-resolution prediction. Each model also keeps
the same learning-rate trajectory across its objectives. If an extra
loss leaves room for fewer updates under the cap, that trajectory is
not shortened to compensate.

The final geological UNETR configuration includes two changes developed
after its initial recipe struggled within the budget. First, training
uses right-angle rotations and reflections in the horizontal plane,
applied jointly to images and labels. These transformations preserve
the depth axis and require no interpolation. Second, the outer two
voxel layers are excluded from band and edge supervision. A salt body
can continue beyond the observed cube, so its intersection with a cube
face need not be an observed salt/background boundary. The full input,
prediction grid and baseline loss are retained. Both choices apply to
the UNETR baseline and its constrained variants; the other three
geological models retain their original crop-face convention.
Appendix~\ref{sec:geo-boundaries} specifies that convention, and
Appendix~\ref{sec:geo-development} records the development comparison.
Thus, cross-model differences concern complete training configurations,
including these choices, rather than architecture alone.

\subsubsection{Spatial Objectives}

''')

once(r'''selected term among $B,E,S$. Bands and edges supervise
the hippocampal union on MSD and the single foreground on ADNI; SDF
supervises each foreground class, and depth
uses the original reference classes. Preliminary MSD screening did not''', r'''selected term among $B,E,S$. These abbreviations refer to boundary
bands, edge consistency and signed distance; $D$ adds depth supervision.
Preliminary MSD screening did not''')
once(r'''Their exclusion is a
scope decision based on the preliminary comparisons; the main replication
provides evidence for the retained configurations. It consequently
supports comparisons among those configurations, rather than a test of every possible combination and interaction.''', r'''The study therefore compares the retained objectives; it does not test
every possible combination or interaction.''')

geometry = r'''Bands and edges supervise the combined anterior/posterior foreground
on MSD, the hippocampus on ADNI and salt on GeoFWI3D. Signed-distance
fields (SDFs) are constructed separately for each foreground class;
depth supervision retains the original reference classes. All
conventional models use two-step bands with six face-sharing neighbours.
The class groups determine which interfaces receive supervision, while
the geological UNETR exclusion changes where bands and edges apply.

Distance fields are clipped at eight voxels and normalized by eight.
Depth supervision balances class errors across layers two units thick
within the foreground and an eight-unit surrounding region. On MSD,
these units are millimetres; its verified isotropic spacing of
$1\,\mathrm{mm}$ makes them numerically equal to voxel units. ADNI and
GeoFWI3D use voxel units and orient the reference plane using the
image grid rather than an anatomical direction. The plane passes
through the foreground centre and contains its main axis, following
Chapter~\ref{sec:spatial-constraints}. Matching these numerical widths
does not establish equal physical distances across datasets.

For GeoFWI3D, source axis 0 supplies the plane's reference direction;
another grid axis is used if its projection is too small. UNETR's
coordinate mapping is restored after augmentation before computing
the depth weights. This annotation-derived coordinate is not a
measurement of geological age or a guaranteed vertical ordering of
layers. Numerical distance and boundary conventions are given in
Appendix~\ref{sec:sdf-construction-details}.
% Geo depth settings retained from the verified protocol: projection norm
% below 0.1 selects the least-parallel source axis, with axis-index tie break;
% sparse-stratum threshold 64 voxels. No inference of physical spacing.

'''
once(r'\subsubsection{Calibration and Activation}', geometry + r'\subsubsection{Calibration and Activation}')

between('An auxiliary loss can dominate an update simply because its numerical',
        r'\subsubsection{Resource Accounting and Evaluation}', r'''An auxiliary loss can dominate training simply because its numerical
scale differs from the baseline. Calibration chooses its coefficient
by comparing their gradients on the support examples, without changing
the network during this measurement. The resulting coefficients remain
fixed during training. All ten medical support crops, or all 19
geological support scenes, contribute to calibration.

The conventional models first train with the baseline alone for nine
updates in nnU-Net or 50 in the other architectures. Selected terms are
then calibrated in order: bands, edges, signed distance and depth.
Each is compared with the baseline and any preceding terms, so its
coefficient can change between combinations. Targets for the typical
and unusually large gradient contributions control the strength of
supervision. Their values differ by model and dataset; the geological
nnU-Net and UNETR settings followed exploratory comparisons.
Appendix~\ref{sec:detailed-calibration} gives the equations and settings.

After calibration, auxiliary contributions increase gradually to full
strength. Signed distance either enters with the other terms or starts
later, following Table~\ref{tab:sdf-activation-policies}. Combinations
containing both signed distance and depth use the later start. These
policies were developed from earlier MSD validation experiments and
retained for the reported campaigns. The comparisons therefore assess
complete calibrated objectives, including their activation schedules.

''')

between('The medical ceiling corresponds to the measured cost of 90 nnU-Net',
        r'\begin{figure}[htbp]', r'''A fixed computation allowance limits how much training each model can
receive. The medical ceiling is approximately 31,976 GFLOPs, the
measured cost of 90 nnU-Net updates. With ten support crops, this
reference represents nine passes through the training subset.
Exploratory learning curves informed this short allowance. Models
with cheaper updates can take more steps within it.

The geological study uses a larger allowance for its larger input
volumes: approximately 182,178 GFLOPs, defined from the original cost
of 152 nnU-Net updates, or eight passes through 19 support scenes.
The final training configurations retain that ceiling. Medical
calibration is recorded separately, whereas geological calibration
consumes part of the allowance. The budgets therefore differ in both
size and accounting scope across domains.

Training costs include the forward pass, active losses, backpropagation
and optimizer update. Another step is taken only if its full cost fits
in the remaining budget. Reference-target construction, validation,
data loading, storage and recipe development are outside these
allowances. The comparison matches the specified arithmetic work,
not elapsed time or end-to-end computation, and does not imply that
different architectures are equally close to convergence.
Figure~\ref{fig:flop-budget-example} illustrates these distinctions.

''')

between('The reported checkpoint is the final state admitted by the budget,',
        r'\subsection{Adapting Pretrained Models}', r'''Each of the three conventional studies contributes all 504
final-budget runs. Reporting the highest-mean objective afterwards
is a descriptive comparison, separate from checkpoint selection.
Appendix~\ref{sec:detailed-resource-accounting} gives the exact ceilings
and counting conventions. Table~\ref{tab:cap-training-exposure} shows
the resulting numbers of updates and passes through the support data.

''')

between('The adaptation studies ask how much the same spatial supervision helps',
        r'\paragraph{SAM3 on GeoFWI3D.}', r'''The adaptation studies ask whether spatial supervision also helps
when the model already has pretrained image representations. SAM3 and
MedSAM3 process two-dimensional image planes, although the annotations
and final evaluation concern three-dimensional volumes. During
training, neighbouring planes are sampled together so that the edge
term can compare predictions across slices as well as within them.
At evaluation, slice predictions are reassembled into a volume.

All three studies freeze the SAM3 base weights and update only the
low-rank adaptation (LoRA) factors introduced in Chapter~\ref{sec:background},
giving 18,497,664 trainable parameters. Geological adaptation starts
with fresh factors that initially leave the base model unchanged;
medical adaptation starts with the released MedSAM3 adapter
\cite{carion2025sam3,liu2025medsam3}. Every matched baseline and
constrained run restores the same base weights and initial factors,
and uses the same support images and sampling schedule.

The pretrained baseline supervises object classification, bounding
boxes, target presence and segmentation masks. Its mask losses combine
soft Dice with focal loss, a form of cross-entropy that gives less
weight to already confident correct predictions. Spatial penalties
are added to this objective and calibrated on the support annotations.
Each update uses two adjacent-slice pairs. Constraint regions and
normalization counts are defined on the full annotated volume, so
sampling preserves the intended weighting across it; the corresponding
reductions are given in Appendix~\ref{sec:sampled-pretrained-supervision}.

Inference uses image planes and fixed text queries. The model combines
its confidence in each mask location, its match to the query and the
target's presence to form foreground responses. These responses are
converted into mutually exclusive class labels. Reference masks are
used to score the predictions, never to supply localization prompts.

''')

between('Dice serves two distinct roles.', 'Voxelwise argmax defines the predicted label.', r'''Every reported Dice score measures overlap between a discrete
prediction and its reference annotation. This is hard Dice, distinct
from the differentiable soft Dice loss used during training.
Validation scores use held-out examples; training scores use the
support examples supplied during fitting.

''')

once(r'''Background is excluded from these primary metrics. Volumes receive
equal weight in a run mean. Section~\ref{sec:background-dice-results}
adds background and all-class Dice as complementary measures on
the same predictions, retaining this order of averaging.''', r'''Background is excluded from these primary metrics.
Section~\ref{sec:background-dice-results} also reports background and
all-class Dice on the same predictions.

To summarize a study, we first average Dice equally across validation
volumes in a run, then across the three seeds within each fold, and
finally across the three folds. The reported sample standard deviation
describes variation between those three fold averages. Training Dice
follows the same hierarchy using support volumes.''')

once(r'''Dice is reported on a percentage scale and differences in percentage
points. Reported standard deviations are sample standard deviations across
the three seed-averaged fold means; paired-gain dispersions use the three
fold-mean differences. Positive-change counts use all nine matched
fold/seed cells. Each figure or table identifies its reduction explicitly.
These standard deviations are not confidence intervals over independent datasets: seeds share data,
training pools overlap and configuration selection is exploratory.
Dispersion across cases is identified separately from dispersion
across fold/seed cells.''', r'''Dice is reported as a percentage and gains as percentage-point
differences. Paired gains follow the same averaging hierarchy, with
their standard deviation calculated across the three fold-mean
differences. Positive-change counts refer to the nine individual
matched runs. These standard deviations describe observed variation,
not confidence intervals over independent datasets: seeds share data,
training pools overlap and configuration selection is exploratory.
Any dispersion across individual volumes is identified separately.''')

between('A confusion matrix counts how often each reference class is assigned each',
        r'\clearpage', r'''Error counts complement Dice by distinguishing missed foreground
(false negatives, FN) from extra foreground (false positives, FP).
MSD also permits anterior-to-posterior and posterior-to-anterior swaps.
These four disjoint error categories account for every incorrectly
labelled voxel. They can be read from a confusion matrix, which records
reference classes against predicted classes; the pretrained-model
matrices use reference classes as rows. ADNI and GeoFWI3D are binary
tasks and have no anterior/posterior swaps.

All counts cover the full prediction grid, including errors outside
the boundary bands. Conventional studies sum errors across validation
volumes before averaging runs. MedSAM3 and SAM3 instead average errors
per validation volume, then across seeds and folds. Their tables state
these units because totals over a validation set cannot be compared
directly with averages per volume.

''')

current = SOURCE.read_text()
a = current.index(r'\section{Datasets and Experimental Methodology}')
b = current.index(r'\section{Results and Analysis}', a)
assert current[a:b] == old, 'Chapter changed during drafting; reconcile before applying.'
(REPORT / 'before_apply.tex').write_text(current)
(REPORT / 'chapter4_after.tex').write_text(new)
(REPORT / 'chapter4.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='chapter4_before', tofile='chapter4_after')))
SOURCE.write_text(current[:a] + new + current[b:])
print(json.dumps({'before_words': len(old.split()), 'after_words': len(new.split()),
                  'removed_lines': len(old.splitlines())-len(new.splitlines()),
                  'outside_chapter_unchanged': True,
                  'chapter_sha256': hashlib.sha256(new.encode()).hexdigest()}, indent=2))

