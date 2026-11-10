"""Consolidate existing, verified illustrations without changing case selection."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'reports/thesis_appendix_consolidation_20261103'
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'thesis-geometry-mpl'))
spec = importlib.util.spec_from_file_location('constraint_examples', Path(__file__).with_name('dataset-constraint-examples.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plt, np = m.plt, m.np
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
items = [m.load_case(*case) for case in m.CASES]
fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.1), gridspec_kw={'width_ratios': [1, 1.5]})
for row, item in enumerate(items):
    left, right = axes[row]
    m.panel(left, item, 'edges', zoom=True)
    left.set_title(item['name'] + ': selected neighbouring pairs', fontsize=9)
    geo = item['geometry']
    bins, counts, masses = [np.asarray(geo[k]) for k in ['bin_values', 'bin_counts', 'bin_masses']]
    right.bar(bins, 100*counts/counts.sum(), width=.84, color='#b9c4cb', label='Region voxels')
    right.plot(bins, 100*masses, 'o-', ms=2.5, lw=1, color=m.ORANGE, label='Loss weight')
    right.set(xlabel='Depth-layer index', ylabel='Share (%)')
    right.xaxis.set_major_locator(m.MaxNLocator(nbins=6, integer=True))
    right.spines[['top', 'right']].set_visible(False)
    right.set_ylim(bottom=0)
    if row == 0:
        right.legend(frameon=False, fontsize=8)
fig.subplots_adjust(left=.065, right=.985, top=.96, bottom=.11, hspace=.60, wspace=.36)
fig.text(.07, .02, 'Pairs: blue = reference agreement; orange = reference transition.', fontsize=8)
pdf = ROOT / 'Template_tesi/figures/compiled/constraint-support-details.pdf'
fig.savefig(pdf)
fig.savefig(REPORT / 'constraint-support-details.png', dpi=180)
plt.close(fig)

# Load the original renderer's definitions, without executing its output calls.
source = ROOT / 'reports/thesis_qualitative_predictions_20261103/render_thesis_panels.py'
tree = ast.parse(source.read_text())
tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.Assert))]
namespace = {'__file__': str(source)}
exec(compile(tree, str(source), 'exec'), namespace)
summaries = namespace['summaries']
namespace['render']([(s, s['examples'][1], f"({chr(97+i)}) {s['dataset']} | {s['examples'][1]['case']}") for i, s in enumerate(summaries)], 'adverse-predictions')
manifest = dict(constraint_examples=[item['record'] for item in items],
                constraint_figure=str(pdf.relative_to(ROOT)),
                constraint_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                qualitative=namespace['outputs'],
                note='Original case-selection rules and verified arrays retained. Only presentation consolidated.')
(REPORT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(pdf)
