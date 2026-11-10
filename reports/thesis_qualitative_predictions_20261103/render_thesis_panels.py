"""Render verified native-grid panels without changing the saved selection rule."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / 'Template_tesi/figures/qualitative'
OUT.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads(p.read_text())
assert read(ROOT / 'final_verification.json')['status'] == 'passed'
summaries = read(ROOT / 'recovered/figures/summary.json')
cmap = ListedColormap(['#2878b5', '#e3812e'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
outputs = []

def render(rows, name):
    n = len(rows)
    fig = plt.figure(figsize=(8.0, 2.65*n + .25), facecolor='white')
    grid = fig.add_gridspec(n*3, 4, height_ratios=[.34, 1.85, .43]*n,
                           left=.025, right=.975, top=.985, bottom=.025,
                           hspace=.02, wspace=.12)
    for i, (summary, example, heading) in enumerate(rows):
        folder = ROOT / 'recovered'
        base, aux = [folder / summary[k]['id'] for k in ['baseline', 'constrained']]
        records = {r['case']:r for r in read(base / 'evaluation.json')['records'] if r['split'] == 'validation'}
        artifact = records[example['case']]['artifact']
        with np.load(base / artifact, allow_pickle=False) as a, np.load(aux / artifact, allow_pickle=False) as b:
            image, truth, preds = a['image'], a['truth'], [a['prediction'], b['prediction']]
            assert np.array_equal(truth, b['truth']) and np.array_equal(image, b['image'])
            z = example['slice_zero_based']
            assert z == int(np.argmax((truth>0).sum(axis=(0,1))))
            masks = [None, truth[:,:,z].T, preds[0][:,:,z].T, preds[1][:,:,z].T]
            label = fig.add_subplot(grid[3*i,:]);label.axis('off')
            label.text(0, .8, heading, fontsize=11.5, weight='bold', va='center')
            for j, mask in enumerate(masks):
                ax = fig.add_subplot(grid[3*i+1,j])
                ax.imshow(image[:,:,z].T, cmap='gray', vmin=0, vmax=1, origin='lower', interpolation='nearest')
                if mask is not None:
                    ax.imshow(np.ma.masked_where(mask==0,mask), cmap=cmap, vmin=1, vmax=2,
                              alpha=.6, origin='lower', interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(['Input', 'Reference', 'Baseline', 'Constrained'][j], fontsize=10.5, pad=4)
                score = fig.add_subplot(grid[3*i+2,j]);score.axis('off')
                if j == 0:
                    score.text(.5,.7,'Slice '+str(z),ha='center',va='center',fontsize=10)
                elif j == 1:
                    labels = 'Anterior / posterior' if summary['dataset']=='MSD' else 'Salt' if summary['dataset']=='GeoFWI3D' else 'Hippocampus'
                    score.text(.5,.7,labels,ha='center',va='center',fontsize=10)
                elif j == 2:
                    score.text(.5,.7,f"Dice {100*example['baseline_dice']:.2f}%",ha='center',va='center',fontsize=10.5)
                else:
                    score.text(.5,.65,f"Dice {100*example['constrained_dice']:.2f}%\n({example['delta_pp']:+.2f} pp)",ha='center',va='center',fontsize=10.5,linespacing=1.3)
    path = OUT / (name + '.png')
    fig.savefig(path, dpi=250, facecolor='white');plt.close(fig)
    outputs.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        examples=[dict(dataset=s['dataset'],**e) for s,e,_ in rows]))

render([(s,s['examples'][0],f"({chr(97+i)}) {s['dataset']} | {s['examples'][0]['case']}") for i,s in enumerate(summaries)], 'typical-predictions')
for s in summaries:
    render([(s,s['examples'][index],f"{label} | {s['examples'][index]['case']}")
            for index,label in [(1,'Adverse case'),(2,'Largest gain')]], s['dataset'].lower()+'-extreme-predictions')
(ROOT/'thesis_panel_manifest.json').write_text(json.dumps(dict(
    source='Verified NPZ predictions; original all-case selection preserved.',
    display='Source-axis-2; first maximum reference foreground area; transpose; origin lower; equal voxel aspect. Grayscale 0..1; blue #2878b5, orange #e3812e, alpha0.6.',
    score='Whole-volume foreground Dice (mean anterior/posterior on MSD), not slice Dice.',outputs=outputs),indent=2)+'\n')
print(json.dumps([x['file'] for x in outputs],indent=2))
