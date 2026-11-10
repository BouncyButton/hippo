from pathlib import Path
import re, difflib, json, hashlib

path = Path('/Users/filippofocaccia/Desktop/hippo/Template_tesi/thesis-revision.tex')
audit = path.parent.parent / 'reports/thesis_section45_simplification_20261103'
audit.mkdir(exist_ok=True)
before = path.read_text()
(audit / 'before.tex').write_text(before)
start = before.index(r'\subsection{Four-Model Comparisons under a FLOP Ceiling}')
end = before.index(r'\subsection{Adapting Pretrained Models}', start)
old = before[start:end]

def block_with_label(text, environment, label):
    blocks = re.findall(r'\\begin\{' + environment + r'\}.*?\\end\{' + environment + r'\}', text, re.S)
    matches = [x for x in blocks if r'\label{' + label + '}' in x]
    assert len(matches) == 1, label
    return matches[0]

combinations = block_with_label(old, 'table', 'tab:constraint-combinations')
baseline_table = block_with_label(old, 'table', 'tab:capped-baseline-losses')
budget_figure = block_with_label(old, 'figure', 'fig:flop-budget-example')
resource = old[old.index(r'\subsubsection{Resource Accounting and Evaluation}'):]
resource = resource.replace(
    'Figure~\\ref{fig:flop-budget-example} illustrates these distinctions.',
    'Appendix~\\ref{sec:detailed-resource-accounting} illustrates the accounting\n'
    'scope and gives the exact counting rules.')
resource = resource.replace(budget_figure + '\n\\FloatBarrier\n', '')
resource = resource.replace(
    'Appendix~\\ref{sec:detailed-resource-accounting} gives the exact ceilings\n'
    'and counting conventions. Table~\\ref{tab:cap-training-exposure} shows\n'
    'the resulting numbers of updates and passes through the support data.',
    'Table~\\ref{tab:cap-training-exposure} in the appendix records the resulting\n'
    'numbers of updates and passes through the support data.')
resource = re.sub(r'\n{3,}', '\n\n', resource)

intro = r'''\subsection{Four-Model Comparisons under a FLOP Ceiling}
\label{sec:main-msd-methods}

The comparison asks whether spatial supervision helps different models
learn from the same small set of annotated volumes. On each dataset,
nnU-Net, SwinUNETR, UNETR and the compact three-dimensional U-Net are
trained from scratch, using the architectures introduced in
Chapter~\ref{sec:background}. For each model, the baseline uses its
configured segmentation loss. The constrained versions retain that loss
and add selected spatial terms. Training examples, initialization and
the remaining model settings are matched to the corresponding baseline.

The fourteen objectives are listed in
Table~\ref{tab:constraint-combinations}. Here, $B$ denotes boundary bands,
$E$ edge consistency, $S$ signed distance and $D$ depth supervision;
$\varnothing$ means that none of $B,E,S$ is added. Reading across a row
shows the same combination without and with depth supervision. Edge
alone and edge plus signed distance were not carried forward after
preliminary MSD screening, so the table covers the retained objectives,
rather than every possible combination.

'''
bridge = r'''

The additional losses need appropriate weights: a term should not
dominate training simply because its numerical values are larger.
Calibration sets those weights using the training examples, and the
constraints are then introduced gradually. The detailed weighting rules
and activation schedules are given in
Appendix~\ref{sec:detailed-calibration}.

The training configurations differ between architectures, including
their baseline losses and the task-specific adjustments made to
geological UNETR. These choices remain fixed within each
baseline/constrained comparison. Cross-model scores therefore compare
complete training configurations. The loss settings, spatial conventions
and model-specific adjustments are recorded in
Appendices~\ref{sec:sdf-construction-details}--\ref{sec:conventional-implementation}.

'''
new = intro + combinations + bridge + resource
after = before[:start] + new + before[end:]

def replace_once(old_text, new_text):
    global after
    assert after.count(old_text) == 1, repr(old_text[:100])
    after = after.replace(old_text, new_text, 1)

replace_once(r'''\label{sec:sdf-construction-details}

For a nonempty''', r'''\label{sec:sdf-construction-details}

In the conventional comparisons, bands and edges supervise the combined
anterior/posterior foreground on MSD, hippocampus on ADNI and salt on
GeoFWI3D. They use two-step morphology with six face-sharing neighbours.
Signed-distance fields are classwise over foreground classes; depth
supervision retains the original reference classes.

For a nonempty''')

replace_once(r'''its construction uses source-array coordinates in voxel units.

\paragraph{Geological crop faces.}''', r'''its construction uses source-array coordinates in voxel units.
Depth layers are two units thick within an eight-unit surrounding region:
millimetres on MSD and source-array voxels on ADNI and GeoFWI3D. MSD's
verified isotropic 1 mm spacing makes the two measures numerically equal
there; matching numerical widths does not imply equal physical distances
across datasets.

For GeoFWI3D, source axis 0 supplies the plane's reference direction.
If its projection orthogonal to the foreground's principal axis has norm
below 0.1, the least-parallel grid axis is used, with axis-index tie
breaking. The plane passes through the foreground centre and contains
its principal axis. UNETR's source-coordinate mapping is restored after
augmentation before constructing depth weights. Strata with fewer than
64 voxels receive reduced total weight as defined in
Section~\ref{sec:depth-constraints}. This coordinate does not measure
geological age or guarantee vertical stratigraphic ordering.

\paragraph{Geological crop faces.}''')

replace_once(r'''The final UNETR recipe intersects both bands with $R_i^{(2)}$;
nnU-Net, SwinUNETR and the compact U-Net retain the zero-padded bands.''',
r'''The final UNETR recipe intersects both bands with $R_i^{(2)}$;
nnU-Net, SwinUNETR and the compact U-Net retain the zero-padded bands.
This exclusion avoids treating a salt body's continuation beyond the
observed cube as a salt/background interface. The full input and
prediction grid remain available to the baseline loss.''')

exclusion_comments = old[old.index('% EXCLUSION EVIDENCE:'):old.index(r'\begin{table}', old.index('% EXCLUSION EVIDENCE:'))]
replace_once(r'''\subsection{Calibration and Activation Schedules}
\label{sec:detailed-calibration}

Conventional''', r'''\subsection{Calibration and Activation Schedules}
\label{sec:detailed-calibration}\label{sec:calibration-methods}

The objective inventory in Table~\ref{tab:constraint-combinations} follows
preliminary MSD screening. Edge alone did not establish a consistent
incremental benefit over the baseline, nor edge plus signed distance
over signed distance alone. Those two configurations were omitted from
the subsequent replications. This screening does not establish their
ineffectiveness across other datasets or training conditions.
''' + exclusion_comments + r'''
Conventional''')

replace_once(r'''Depth is calibrated against preceding terms at full strength even when
SDF activates later.''', r'''Depth is calibrated against preceding terms at full strength even when
SDF activates later. The resulting coefficients remain fixed during the
run, but can differ between objective combinations.''')

replace_once(r'''\label{sec:implementation-tables}\label{sec:conventional-implementation}

The medical ceiling''', r'''\label{sec:implementation-tables}\label{sec:conventional-implementation}

Figure~\ref{fig:flop-budget-example} illustrates why a shared ceiling
permits different numbers of updates and distinguishes the two
calibration-accounting conventions.

''' + budget_figure + r'''

The medical ceiling''')

replace_once(r'''The compact MONAI BasicUNet uses encoder widths''', r'''All conventional models start from random initialization and process one
volume per update, on $64^3$ grids for MSD and ADNI and $96^3$ for GeoFWI3D.
Output heads have three classes on MSD and two on ADNI and GeoFWI3D.
Table~\ref{tab:capped-baseline-losses} records the medical training losses.
The Dice implementations use the Medical Open Network for Artificial
Intelligence (MONAI) library. GeoFWI3D retains these loss choices except
for UNETR, which uses foreground Dice plus cross-entropy, with squared
prediction terms in the Dice denominator.

''' + baseline_table + r'''

The compact MONAI BasicUNet uses encoder widths''')

replace_once(r'''Constraints apply only to the primary output; nnU-Net's other supervised
resolutions keep their baseline losses.''', r'''Constraints apply only to the primary, full-resolution output.
nnU-Net's other supervised resolutions keep their baseline losses, with
reference labels resampled to each grid.

Geological UNETR uses dropout 0.1 and initial learning rate
$6\times10^{-4}$. Its final recipe adds right-angle rotations and
reflections in the horizontal plane, jointly applied to images and
labels without interpolation and preserving the depth axis. These
settings were developed after the initial recipe struggled within the
budget. The same augmentation and model settings apply to its baseline
and constrained variants. Its two-voxel band/edge frame exclusion is
specified in Section~\ref{sec:geo-boundaries}; the other geological
models retain their original crop-face convention.
Section~\ref{sec:geo-development} records the earlier development check.''')

replace_once(r'''and constraint calculations retain float32.

\begin{table}''', r'''and constraint calculations retain float32. Each model keeps the same
learning-rate trajectory across objectives; fewer affordable updates
do not shorten its original schedule horizon.

\begin{table}''')

assert path.read_text() == before, 'Concurrent edit; reread source before applying'
path.write_text(after)
(audit / 'after.tex').write_text(after)
(audit / 'changes.diff').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), after.splitlines(True), fromfile='before', tofile='after')))
print(json.dumps({'section45_tokens_before':len(old.split()),
                  'section45_tokens_after':len(new.split()),
                  'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))

