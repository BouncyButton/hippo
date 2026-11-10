from pathlib import Path
import re,json
root=Path('reports/thesis_professor_revision_20261102');s=(root/'prose_stage.tex').read_text();before=(root/'source_before.tex').read_text()
# Retain full tables unchanged, relocate them by label.
moved={}
def take(label):
 global s
 for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',s,re.S):
  if '\\label{'+label+'}' in m.group():
   moved[label]=m.group();s=s[:m.start()]+s[m.end():];return m.group()
 raise ValueError(label)
for k in ['tab:geo-matrix','tab:geo-best','tab:msd-noaug-matrix','tab:msd-noaug-best','tab:adni-noaug-matrix','tab:adni-noaug-best','tab:sam3-geo-results','tab:medsam-msd-noaug','tab:medsam-adni-noaug','tab:background-conventional','tab:background-adaptation','tab:cap-training-exposure','tab:main-model-settings','tab:sdf-activation-policies']:
 take(k)
long=[]
pattern=r'\\begingroup\\footnotesize\\setstretch\{1\.0\}.*?\\endgroup'
for m in re.finditer(pattern,s,re.S):
 if '\\begin{longtable}' in m.group():long.append(m.group())
assert len(long)==6
for x in long:s=s.replace(x,'')
# Data for main plots are extracted from the preserved tables.
plot_data=[]
def pair_chart(label, rows, caption):
 out=[r'\begin{figure}[htbp]\centering',r'\begin{tikzpicture}[x=1cm,y=0.035cm,font=\footnotesize]',r'\node[anchor=west] at (0,112) {Validation Dice (\%)};']
 for y in range(0,101,20):
  out.append(f'\\draw[thesisGray!25] (0,{y}) -- (14.6,{y});\\node[anchor=east] at (-.15,{y}) {{{y}}};')
 for i,(name,method,base,bsd,val,vsd,gain) in enumerate(rows):
  x=1.65+i*13.0/len(rows)
  for dx,v,sd,col in [(-.43,base,bsd,'thesisBlue'),(.43,val,vsd,'thesisOrange')]:
   xx=x+dx
   out.append(f'\\fill[{col}!85] ({xx-.33:.3f},0) rectangle ({xx+.33:.3f},{v});')
   out.append(f'\\draw[thesisInk] ({xx:.3f},{v-sd:.3f}) -- ({xx:.3f},{v+sd:.3f});\\draw[thesisInk] ({xx-.12:.3f},{v+sd:.3f}) -- ({xx+.12:.3f},{v+sd:.3f});')
  out.append(f'\\node[font=\\scriptsize] at ({x:.3f},{max(base+bsd,val+vsd)+6:.3f}) {{$+{gain:.2f}$ pp}};')
  out.append(f'\\node[align=center,anchor=north,text width=3.2cm] at ({x:.3f},-4) {{{name}\\\\$+{method}$}};')
 out += [r'\fill[thesisBlue!85] (2,-26) rectangle (2.3,-22);\node[anchor=west] at (2.45,-24) {Baseline};',r'\fill[thesisOrange!85] (6,-26) rectangle (6.3,-22);\node[anchor=west] at (6.45,-24) {Highest-mean tested objective};',r'\end{tikzpicture}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure}']
 plot_data.append({'label':label,'rows':rows});return '\n'.join(out)+'\n'
for key,short,title in [('tab:geo-matrix','geo','GeoFWI3D'),('tab:msd-noaug-matrix','msd','MSD'),('tab:adni-noaug-matrix','adni','ADNI')]:
 t=moved[key];baseline=next(x for x in t.splitlines() if x.startswith('Baseline &'))
 bases=[tuple(map(float,x)) for x in re.findall(r'(\d+\.\d+)\\pm(\d+\.\d+)',baseline)]
 b=moved[{'geo':'tab:geo-best','msd':'tab:msd-noaug-best','adni':'tab:adni-noaug-best'}[short]]
 rows=[]
 for i,n in enumerate(['nnU-Net','SwinUNETR','UNETR','3D U-Net']):
  line=next(x for x in b.splitlines() if x.startswith(n+' &'))
  method=re.search(r'\$\+([^$]+)\$',line).group(1);gain=float(line.split('&')[2].strip())
  src=next(x for x in t.splitlines() if x.startswith('Baseline $+'+method+'$ &'))
  vals=[tuple(map(float,x)) for x in re.findall(r'(\d+\.\d+)\\pm(\d+\.\d+)',src)]
  rows.append((n,method,*bases[i],*vals[i],gain))
 chart=pair_chart('fig:'+short+'-key-results',rows,f'{title}: baseline and the highest-mean spatially supervised objective for each architecture. Bars show mean foreground Dice and whiskers show the sample standard deviation across three seed-averaged fold means. Gains use the unrounded paired results. Selection uses the reported validation data; it is descriptive. Full objective and paired-fold results are in Tables~\\ref{{{key}}} and~\\ref{{'+{'geo':'tab:geo-best','msd':'tab:msd-noaug-best','adni':'tab:adni-noaug-best'}[short]+'}.')
 # Insert at start of each dataset's results, after label.
 keylabel={'geo':'sec:geo-results','msd':'sec:main-msd-results','adni':'sec:adni-results'}[short]
 anchor='\\label{'+keylabel+'}';s=s.replace(anchor,anchor+'\n\n'+chart,1)
# Add a summary chart for adaptation, preserving MSD's matching batch.
adapt=[('SAM3 / GeoFWI3D','B+E+D',69.31,4.00,72.76,1.97,3.45),('MedSAM3 / MSD','B+E+S+D',76.34,1.64,81.73,1.43,5.38),('MedSAM3 / ADNI','E+S+D',77.32,1.95,80.85,2.05,3.53)]
chart=pair_chart('fig:adaptation-key-results',adapt,r'Pretrained adaptation: each selected objective is paired with its own baseline. The MSD control is from the execution batch matched to the all-term variant. Whiskers are the sample standard deviation across three fold means. Support sizes and update budgets differ across these studies, so the bars do not compare models at equal resources. Full results appear in Tables~\ref{tab:sam3-geo-results}, \ref{tab:medsam-msd-noaug} and~\ref{tab:medsam-adni-noaug}.')
s=s.replace(r'\label{sec:sam3-results}',r'\label{sec:sam3-results}'+'\n\n'+chart,1)
# Merge neighbouring dataset results without losing cross-reference targets.
s=s.replace(r'\subsection{GeoFWI3D: Four-Model Salt Segmentation}',r'\subsection{Training from Scratch}'+'\n'+r'\paragraph{Geological segmentation.}')
s=s.replace(r'\subsection{MSD under the Optimization FLOP Ceiling}',r'\paragraph{MSD hippocampus segmentation.}')
s=s.replace(r'\subsection{ADNI under the Same Optimization FLOP Ceiling}',r'\paragraph{ADNI hippocampus segmentation.}')
s=s.replace(r'\subsection{SAM3 Adaptation on GeoFWI3D}',r'\subsection{Adapting Pretrained Models}'+'\n'+r'\paragraph{SAM3 on GeoFWI3D.}')
s=s.replace(r'\subsection{MedSAM3 Adaptation on MSD}',r'\paragraph{MedSAM3 on MSD.}')
s=s.replace(r'\subsection{MedSAM3 Adaptation on ADNI}',r'\paragraph{MedSAM3 on ADNI.}')
# Opening explains the questions; inventories are already in Methods.
a=s.index('The six-study design covers');b=s.index('\\subsection{Training from Scratch}',a)
s=s[:a]+r'''The results address three questions: whether spatial supervision improves
accuracy within the available training budget, whether the benefit extends
to pretrained adaptation, and which segmentation errors change. The main
figures compare each baseline with its highest-mean tested objective;
Appendix~\ref{sec:complete-result-tables} retains all objective scores,
including negative outcomes. These selections use the reported validation
results and should be read as descriptive comparisons.

Dice scores are percentages and changes are percentage points. A gain of
five points means, for example, a change from 70\% to 75\% Dice. The
symbols $B,E,S,D$ denote boundary bands, edge consistency, signed distance
and depth supervision. Whiskers and $\pm$ values describe the sample
standard deviation of three seed-averaged fold means, not confidence
intervals. Every gain retains the baseline matched to its training run.

'''+s[b:]
# Relocated tables have explicit appendix context; charts receive first emphasis.
s=s.replace('analysis to synthetic reflectivity volumes. Table~\\ref{tab:geo-matrix}', 'analysis to synthetic reflectivity volumes. Figure~\\ref{fig:geo-key-results}\nshows the main comparison. Appendix Table~\\ref{tab:geo-matrix}')
s=s.replace('reports the complete objective inventory. Boundary bands improve every','reports the complete objective inventory in the appendix;\nFigure~\\ref{fig:msd-key-results} shows the principal comparison. Boundary bands improve every')
s=s.replace('crops per fold. Bands alone improve all four means','crops per fold (Figure~\\ref{fig:adni-key-results}). Bands alone improve all four means')
s=s.replace('labelled geological volumes and 25 updates. Table~\\ref{tab:sam3-geo-results}','labelled geological volumes and 25 updates. Figure~\\ref{fig:adaptation-key-results}\ncompares the adaptation results. Appendix Table~\\ref{tab:sam3-geo-results}')
# Shorten development-only narrative in main results; preserve in appendix.
a=s.index('An earlier UNETR development comparison');b=s.index('The geological comparison uses a larger cap',a)
development=s[a:b];s=s[:a]+s[b:]
# Rebuild error discussion around findings, leaving every original error table in appendix.
a=s.index('The following tables report errors');b=s.index('% BEGIN BACKGROUND DICE ANALYSIS',a)
s=s[:a]+r'''Higher Dice can result from recovering more of an object, reducing
foreground outside it, or both. A false negative (FN) is foreground
predicted as background; a false positive (FP) is background predicted as
foreground. MSD additionally distinguishes anterior/posterior swaps,
which preserve whole-hippocampus membership but assign the wrong internal
class. These categories count different mistakes, so improved overlap
does not imply that every error type decreases.

For the conventional geological models, the highest-mean nnU-Net objective
reduces both FN (34.4\%) and FP (11.4\%). SwinUNETR and the compact U-Net
mainly reduce FP, by 12.4\% and 26.9\%, while their FN counts rise
slightly. UNETR shows the reverse tradeoff: FN falls by 4.3\% and FP
rises by 2.9\%. Thus, even within one dataset, spatial supervision changes
foreground retention and extension differently across architectures.

The conventional medical results show a related pattern. On MSD, the
selected nnU-Net objective reduces missed foreground with a slight
increase in extension; the other models reduce extension while increasing
omissions. All four selected configurations reduce anterior/posterior
swaps. On ADNI, nnU-Net reduces both binary error types, while the other
models exchange fewer false positives for more false negatives.

Pretrained adaptation favours greater foreground recovery. SAM3's
highest-mean objective reduces FN from 34.16 to 27.16 thousand voxels
per volume, while FP rises from 3.70 to 6.99 thousand. Total incorrectly
labelled voxels nevertheless fall by 9.8\%. For MedSAM3 on MSD, the
selected objective reduces FN by 36.0\% and internal swaps by 20.7\%,
while FP rises by 40.7\%. On ADNI, FN falls by 24.0\% and FP rises
by 26.2\%. These changes explain why an overlap gain should be considered
together with the error balance relevant to the application.

All counts cover the full evaluated image, including locations beyond
the constraint bands. Tables~\ref{tab:geo-all-errors}--\ref{tab:medsam-adni-all-errors}
in Appendix~\ref{sec:complete-error-tables} report every objective, including
those with negative Dice changes. Conventional-model counts are pooled
within each run; adaptation counts are means per image. Their absolute
magnitudes therefore should not be compared across studies. The error
counts establish how much misclassification changes, but retained paired
prediction panels and surface measurements are insufficient to establish
where those changes occur. The dataset illustrations show annotations,
not evidence of predicted segmentation quality.

\FloatBarrier
'''+s[b:]
# Main text retains a compact three-metric table for adaptation; full table moved.
short=r'''\begin{table}[htbp]
\centering\small\setstretch{1.1}
\begin{tabular}{@{}llrrr@{}}
\hline
Study & Training & Foreground & Background & All classes \\
\hline
SAM3 / GeoFWI3D & Baseline & 69.31 & 97.62 & 83.46 \\
 & $+B+E+D$ & 72.76 & 97.84 & 85.30 \\
MedSAM3 / MSD & Matched baseline & 76.34 & 99.01 & 83.90 \\
 & $+B+E+S+D$ & 81.73 & 99.17 & 87.54 \\
MedSAM3 / ADNI & Baseline & 77.32 & 99.10 & 88.21 \\
 & $+E+S+D$ & 80.85 & 99.19 & 90.02 \\
\hline
\end{tabular}
\caption{Three views of the same predictions: mean Dice (\%) for each
adaptation baseline and the objective selected by foreground Dice.
All-class Dice gives equal weight to each class. Full conventional and
adaptation comparisons are retained in Tables~\ref{tab:background-conventional}
and~\ref{tab:background-adaptation}.}
\label{tab:dice-three-measures-summary}
\end{table}
'''
anchor='Every one of the fifteen foreground-selected configurations';s=s.replace(anchor,short+'\n'+anchor,1)
s=s.replace('Tables~\\ref{tab:background-conventional} and\n\\ref{tab:background-adaptation} place the three measures alongside\none another.', 'Table~\\ref{tab:dice-three-measures-summary} illustrates the three measures\nfor adaptation. The full comparisons are in Appendix Tables~\\ref{tab:background-conventional}\nand~\\ref{tab:background-adaptation}.')
# Append detailed tables in cohesive blocks before existing supplementary ledgers.
insert=r'''\subsection{Complete Dice Comparisons}\label{sec:complete-result-tables}
The following tables preserve all evaluated objectives and their matched
controls. Main-text figures summarize selected comparisons; these tables
retain the complete numerical record, including negative results.

'''
for k in ['tab:geo-matrix','tab:geo-best','tab:msd-noaug-matrix','tab:msd-noaug-best','tab:adni-noaug-matrix','tab:adni-noaug-best','tab:sam3-geo-results','tab:medsam-msd-noaug','tab:medsam-adni-noaug']:
 insert+=moved[k]+'\n\\FloatBarrier\n'
insert+=r'''\subsection{Complete Error Counts}\label{sec:complete-error-tables}
Counts include all evaluated voxels. Conventional-model counts pool the
validation images within each run; adaptation counts are means per image.
Each condition then averages seeds within folds and weights folds equally.
Dispersion is the sample standard deviation across the three fold means.
Both the mean and its dispersion use the scale in each column heading.
Rows ascend by mean total error within a model, using unrounded values.

'''
insert+='\n\n'.join(long)+'\n\\FloatBarrier\n'
insert+=r'\subsection{Complementary Dice Measures}'+'\n'+moved['tab:background-conventional']+'\n\\FloatBarrier\n'+moved['tab:background-adaptation']+'\n\\FloatBarrier\n'
insert+=r'\subsection{Geological Development Comparison}'+'\n'+development+'\n'
pos=s.index(r'\subsection{Geological Comparison}',s.index(r'\section{Supplementary Experimental Results}'))
s=s[:pos]+insert+s[pos:]
# Technical settings remain available near the implementation appendix.
insert=r'''\subsection{Model Settings and Training Exposure}\label{sec:implementation-tables}
These tables give the configured network sizes, activation policies and
training exposure underlying the main-text comparisons.

'''
for k in ['tab:main-model-settings','tab:sdf-activation-policies','tab:cap-training-exposure']:insert+=moved[k]+'\n\\FloatBarrier\n'
pos=s.index(r'\section{Supplementary Experimental Results}');s=s[:pos]+insert+'\n\\clearpage\n'+s[pos:]
# Readability / consistency: introduce SDF where the family is formally defined.
s=s.replace('A distance field gives prediction errors a geometric scale over the whole','A signed-distance field (SDF) gives prediction errors a geometric scale over the whole')
# Remove duplicate float barriers and excessive blank lines.
s=re.sub(r'(\\FloatBarrier\s*){2,}',r'\\FloatBarrier\n\n',s)
s=re.sub(r'\n{4,}','\n\n\n',s)
(root/'layout_stage.txt').write_text(s)
(root/'plot_data.json').write_text(json.dumps(plot_data,indent=2))
(root/'relocated_tables.json').write_text(json.dumps(list(moved)+[re.search(r'\\label\{([^}]+)\}',x).group(1) for x in long],indent=2))
print('Prepared',len(moved),'tables and',len(long),'long tables; figures',len(plot_data))
