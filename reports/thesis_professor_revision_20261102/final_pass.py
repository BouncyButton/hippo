from pathlib import Path
from itertools import product
import re, json

root = Path('reports/thesis_professor_revision_20261102')
s = (root / 'layout_stage.txt').read_text()

def replace(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)

# A small, fully specified example explains the score before the soft loss.
dice = [r'\begin{figure}[htbp]', r'\centering',
        r'\begin{tikzpicture}[x=.65cm,y=.65cm,font=\small]']
ref = {(0, 1), (1, 1), (0, 2), (1, 2)}
pred = {(0, 1), (1, 1), (1, 2), (2, 2)}
for shift, title, cells in [(0, 'Reference', ref), (6, 'Prediction', pred),
                            (12, 'Overlap', ref & pred)]:
    dice.append(f'\\node at ({shift+1.5},4) {{{title}}};')
    for x, y in product(range(3), repeat=2):
        color = 'thesisBlue!75' if (x, y) in cells else 'thesisPaleGray'
        dice.append(f'\\filldraw[fill={color},draw=white,line width=1pt] '
                    f'({shift+x},{y}) rectangle ({shift+x+1},{y+1});')
    dice.append(f'\\draw[thesisGray] ({shift},0) rectangle ({shift+3},3);')
    dice.append(f'\\node at ({shift+1.5},-.6) {{{len(cells)} foreground pixels}};')
dice += [r'\node at (7.5,-1.8) {$\displaystyle\mathrm{Dice}=\frac{2\times3}{4+4}=0.75=75\%$};',
         r'\end{tikzpicture}',
         r'\caption{Dice on a nine-pixel synthetic image. Each mask contains four foreground pixels, of which three coincide. One reference pixel is missed and one extra pixel is predicted. Background agreement does not enter this foreground score. The same calculation applies to voxel sets in three dimensions.}',
         r'\label{fig:dice-example}', r'\end{figure}']
replace('Dice-based losses assess regional overlap \\cite{milletari2016vnet}.\nThe linear-denominator soft Dice used in this study is',
        r'''Dice measures how much a predicted region overlaps its reference.
For two nonempty pixel or voxel sets, it is twice the size of their
intersection divided by the sum of their sizes. A score of one indicates
perfect agreement and zero indicates no overlap.
Figure~\ref{fig:dice-example} works through a nine-pixel example.

''' + '\n'.join(dice) + r'''

Training uses a continuous version of this overlap measure so that its
derivatives can guide learning \cite{milletari2016vnet}. The
linear-denominator soft Dice used in this study is''')

# Exact one-step 3D morphology, drawn from voxel sets rather than an artist's approximation.
cube = set(product(range(3), repeat=3))
offsets = [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
dilated = {tuple(v[k]+a[k] for k in range(3)) for v in cube for a in offsets}
eroded = {v for v in cube if all(tuple(v[k]+a[k] for k in range(3)) in cube for a in offsets)}
assert len(dilated) == 81 and len(eroded) == 1

def project(v):
    x,y,z=v
    return (.55*(x-y), .27*(x+y)+.65*z)

def solid(voxels, center, mode):
    # Show the -x, -y and +z exposed faces, ordered from back to front.
    specs = [((-1,0,0), [(0,0,0),(0,1,0),(0,1,1),(0,0,1)], 55),
             ((0,-1,0), [(0,0,0),(1,0,0),(1,0,1),(0,0,1)], 75),
             ((0,0,1), [(0,0,1),(1,0,1),(1,1,1),(0,1,1)], 35)]
    faces=[]
    for v in voxels:
        color = 'thesisBlue' if mode=='erode' else ('thesisOrange' if v not in cube else 'thesisGray')
        for n, corners, shade in specs:
            if tuple(v[k]+n[k] for k in range(3)) in voxels:
                continue
            coords=[tuple(v[k]+c[k] for k in range(3)) for c in corners]
            depth=sum(-a[0]-a[1]+a[2] for a in coords)/4
            points=[]
            for c in coords:
                x,y=project(c)
                points.append(f'({center+x:.3f},{y-1.785:.3f})')
            faces.append((depth, f'\\filldraw[fill={color}!{shade},draw=white,line width=.25pt] '+ ' -- '.join(points)+' -- cycle;'))
    return [f[1] for f in sorted(faces)]

morph = [r'\begin{figure}[htbp]',r'\centering',r'\begin{tikzpicture}[x=1cm,y=1cm,font=\small]']
for vox, center, mode, title, count in [
    (cube,2,'original','Reference region',27),
    (eroded,7.7,'erode','One erosion',1),
    (dilated,13.4,'dilate','One dilation',81)]:
    morph += [f'\\node at ({center},3.1) {{{title}}};'] + solid(vox,center,mode)
    morph += [f'\\node at ({center},-3.0) {{{count} voxel'+('' if count==1 else 's')+'};']
morph += [r'\end{tikzpicture}',
          r'\caption{Six-neighbour morphology in three dimensions. A solid $3\times3\times3$ region contains 27 voxels. One erosion retains only its centre, whose six face-sharing neighbours are all inside. One dilation adds the 54 face-adjacent exterior voxels (orange), giving 81 voxels; diagonal neighbours are not added in that step. This example assumes a surrounding background grid. The experiments repeat the operation twice to construct their bands.}',
          r'\label{fig:three-dimensional-morphology}', r'\end{figure}']
replace('treated as absent, corresponding to zero padding. Repeating each\noperation twice gives',
        'treated as absent, corresponding to zero padding.\nFigure~\\ref{fig:three-dimensional-morphology} illustrates one step on\na solid cube.\n\n'+'\n'.join(morph)+'\n\nRepeating each operation twice gives')

# Keep introductory figures accessible: no run inventories or units introduced only later.
replace('Organization of the investigation. Four-model comparisons cover\nGeoFWI3D, MSD and ADNI. The medical studies share an optimization-only\nceiling; the larger geological ceiling also charges calibration.\nPretrained adaptation evaluates fourteen objectives with MedSAM3 on MSD\nand ADNI and with SAM3 on GeoFWI3D, under separate fixed-update budgets.\nArrows indicate conceptual relationships rather than weight transfer.',
        'Organization of the investigation. The same spatial objectives are\ntested in geological and medical segmentation, using models trained from\nscratch and adapted pretrained models. Each study compares ordinary\nsupervised training with the addition of spatial supervision. Arrows\nindicate conceptual relationships rather than transfer of model weights.')
s=s.replace('Synthetic reflectivity\\\\Background / salt','Synthetic reflectivity\\\\2 classes: background / salt')
s=s.replace('Background / anterior /\\\\posterior','3 classes: background /\\\\anterior / posterior')
s=s.replace('MRI, separate left/right crops\\\\Background / hippocampus','MRI, separate left/right crops\\\\2 classes: background / hippocampus')
replace('encode parameter count or FLOPs.',
        'encode parameter count or the number of floating-point operations\n(FLOPs), which measures arithmetic work.')
replace('UNETR divides the volume into patches, embeds them as a sequence and',
        'UNETR (U-Net with a transformer encoder) divides the volume into\npatches, embeds them as a sequence and')
replace('SwinUNETR forms a hierarchy of transformer features',
        'SwinUNETR combines a shifted-window (Swin) transformer with a U-Net\ndecoder, forming a hierarchy of transformer features')
replace('supervision across spatial strata. Each section introduces the reason\nfor the construction, defines its numerical form and explains the errors\nto which it responds. Calibration and training schedules are specified\nin Chapter~\\ref{sec:methodology}; supporting derivations appear in\nAppendix~\\ref{sec:technical-details}.',
        'supervision across spatial layers. Their definitions specify both\nthe property to encourage and the locations on which it is evaluated.')
# No physical claim changes: this is merely an exact restatement of the implemented stabilizer.
replace('The constant $\\varepsilon$ stabilizes the count denominators.',
        'The implementation adds the small constant $\\varepsilon$ to each\ncount denominator. Empty supports are handled separately below, so this\nconstant is not a substitute for that validity check.')
s=s.replace('% Thesis revision, begun 9 October 2026; medical results updated 21 October 2026.',
            '% Thesis revision, begun 9 October 2026; professor feedback incorporated 2 November 2026.')
s=s.replace('% SAM3 numerical results are complete; remaining presentation gaps are marked locally.',
            '% Numerical results are complete; evidence limitations are stated in Results.')

# Validate references, unique labels and preservation of the relocated numerical tables.
labels=re.findall(r'\\label\{([^}]+)\}',s)
assert len(labels)==len(set(labels)), 'Duplicate labels'
refs=re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',s)
assert not (set(refs)-set(labels)), set(refs)-set(labels)
bibs=set(re.findall(r'\\bibitem\{([^}]+)\}',s))
cites={k for group in re.findall(r'\\cite\{([^}]+)\}',s) for k in group.split(',')}
assert not(cites-bibs), cites-bibs
before=(root/'source_before.tex').read_text()
for label in json.loads((root/'relocated_tables.json').read_text()):
    pattern=r'\\begin\{table\}.*?\\end\{table\}' if 'all-errors' not in label and label!='tab:sam3-geo-errors' else r'\\begingroup\\footnotesize\\setstretch\{1\.0\}.*?\\endgroup'
    old=next(m.group() for m in re.finditer(pattern,before,re.S) if '\\label{'+label+'}' in m.group())
    assert s.count(old)==1, 'Table changed: '+label
assert all(ord(c)>=32 or c in '\n\t' for c in s), 'Control characters'
(root/'final_stage.txt').write_text(s)
print('Prepared final source:',len(s),'characters;',len(labels),'unique labels; all 20 relocated tables unchanged.')
