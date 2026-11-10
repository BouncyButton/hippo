"""Teaching panels: native-volume bands and a controlled 3-D SDF example.

Geometry and images are real. The equal probabilities in the SDF example are
chosen to demonstrate the loss, not obtained from an experimental prediction.
"""
from pathlib import Path
import importlib.util
import json

SOURCE = Path(__file__).resolve().parent / 'dataset-constraint-examples.py'
spec = importlib.util.spec_from_file_location('geometry_figures', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import torch

REPORT = m.ROOT / 'reports/thesis_constraint_teaching_examples_20261103'


def save(fig, name):
    fig.savefig(m.OUTPUT / f'{name}.pdf', dpi=300, facecolor='white')
    fig.savefig(REPORT / f'{name}.png', dpi=220, facecolor='white')
    plt.close(fig)


def bands(items):
    fig = plt.figure(figsize=(6.3, 4.9))
    gs = fig.add_gridspec(2, 3, left=.025, right=.985, bottom=.15,
                          top=.91, wspace=.14, hspace=.16)
    for col, item in enumerate(items):
        d = {k: item['plane'](v) for k, v in item['fields'].items()}
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(d['image'], cmap='gray', vmin=item['limits'][0], vmax=item['limits'][1],
                  origin='lower', interpolation='nearest', rasterized=True)
        m.contour(ax, d['mask'], 'white', .9)
        x, y, size = item['zoom']
        ax.add_patch(Rectangle((x-.5, y-.5), size, size, fill=False,
                               edgecolor='#f0da68', linewidth=1.))
        ax.set_title(item['name'], fontsize=10, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('Annotated foreground', fontsize=8, labelpad=3)
        ax = fig.add_subplot(gs[1, col])
        m.panel(ax, item, 'bands', zoom=True)
        ax.set_xlabel('Two-step bands: enlarged view', fontsize=7.6, labelpad=4)
    fig.legend(handles=[Patch(color=m.BLUE, label='Inner band: foreground target'),
                        Patch(color=m.ORANGE, label='Outer band: background target'),
                        Line2D([], [], color='#7b858d', lw=1, label='White contour: reference boundary')],
               loc='lower center', bbox_to_anchor=(.5, .006), frameon=False,
               fontsize=7.3, ncol=1, labelspacing=.35)
    save(fig, 'dataset-bands-teaching')


def find_pair(item):
    field = item['fields']['sdf'] * 8.
    center = np.argwhere(item['fields']['mask']).mean(0)
    rotation = item['geometry']['basis'][:, [2, 0, 1]]
    candidates = []
    for near in np.argwhere(np.isclose(field, .5)):
        for axis in range(3):
            for sign in (-1, 1):
                far = near.copy(); far[axis] += 3 * sign
                if (far < 0).any() or (far >= field.shape).any():
                    continue
                if not np.isclose(field[tuple(far)], 3.5):
                    continue
                local = (np.stack([near, far]) - center) @ rotation
                if not (local[:, :2] >= 0).all():
                    continue
                score = np.linalg.norm(local[0, :2]) + abs(local[0, 2])
                candidates.append((float(score), tuple(near), tuple(far)))
    assert candidates
    _, near, far = min(candidates)
    assert item['fields']['labels'][near] == item['fields']['labels'][far] == 0
    return np.array([near, far]), rotation


def sdf_example(item):
    points, rotation = find_pair(item)
    distances = np.array([item['fields']['sdf'][tuple(p)] for p in points]) * 8
    probabilities = np.array([.25, .25])
    penalties = probabilities * distances / 8
    assert np.allclose(distances, [.5, 3.5])
    assert np.allclose(penalties, [.015625, .109375])
    assert np.isclose(penalties[1] / penalties[0], 7)

    fig = plt.figure(figsize=(6.3, 4.15))
    ax = fig.add_axes([.005, .22, .60, .66], projection='3d', computed_zorder=False)
    m.volume_panel(ax, item, 'sdf')
    display = points @ rotation
    for p, marker in zip(display, ['o', 'D']):
        ax.scatter(*p, s=32, marker=marker, c=m.INK, edgecolors='white',
                   linewidths=.65, depthshade=False, zorder=5)
    ax.plot(*display.T, color=m.INK, linewidth=.7, linestyle=':', zorder=4)
    # Distinct leader labels are projected from the actual 3-D voxel centres.
    for p, text, loc in zip(display, ['Near', 'Far'], [(.16, .24), (.84, .14)]):
        x, y, _ = proj3d.proj_transform(*p, ax.get_proj())
        ax.annotate(text, xy=(x, y), xycoords='data', xytext=loc,
                    textcoords='axes fraction', ha='center', va='center', fontsize=8,
                    arrowprops=dict(arrowstyle='-', color=m.INK, lw=.7),
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=10)

    fig.text(.5, .97, 'Same probability error, different distances', ha='center', va='top', fontsize=11)
    fig.text(.5, .906, r'Two background voxels: reference $y=0$, illustrative prediction $p=0.25$',
             ha='center', va='top', fontsize=8)
    info = fig.add_axes([.60, .23, .385, .64]); info.set_axis_off()
    for y, name, d, penalty, marker in zip([.59, .10], ['Near', 'Far'], distances, penalties, ['o','D']):
        info.add_patch(FancyBboxPatch((.015, y), .96, .36, boxstyle='round,pad=.012',
                                      facecolor='#f5f7f9', edgecolor='#b7c1c9', linewidth=.6))
        info.plot(.08, y+.28, marker, color=m.INK, ms=5, transform=info.transAxes)
        info.text(.15, y+.28, name, fontsize=9, fontweight='bold', va='center')
        info.text(.07, y+.17, f'Distance value: {d:g} voxels', fontsize=8)
        info.text(.07, y+.07, rf'Penalty: $0.25\times {d:g}/8={penalty:.6f}$', fontsize=8)
    info.set(xlim=(0,1), ylim=(0,1))
    fig.text(.5, .19, 'The farther voxel receives seven times the local penalty.',
             ha='center', fontsize=9, color=m.INK)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1,1), cmap=m.SDF_CMAP),
                      cax=fig.add_axes([.21, .10, .58, .02]), orientation='horizontal', ticks=[-1,0,1])
    cb.ax.tick_params(labelsize=7, pad=2, length=2)
    cb.set_label('Normalized signed distance: blue inside; orange outside', fontsize=7.5, labelpad=2)
    save(fig, 'sdf-3d-penalty-example')
    return dict(dataset=item['name'], case=item['case'], probe_voxels=points.tolist(),
                field_values_voxels=distances.tolist(), illustrative_probabilities=probabilities.tolist(),
                reference_classes=[0,0], local_penalties=penalties.tolist(), penalty_ratio=7.,
                selection='near=0.5 and far=3.5 on an axis-aligned ray in the open cutaway quadrant; closest to class centroid in display basis; lexicographic tie breaking',
                probabilities_are_experimental_outputs=False)


def main():
    REPORT.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':8,
                         'mathtext.fontset':'stix', 'pdf.fonttype':42,
                         'text.color':m.INK, 'axes.labelcolor':m.INK})
    torch.set_num_threads(2)
    items=[m.load_case(*args) for args in m.CASES]
    bands(items)
    example=sdf_example(items[1])
    manifest=dict(cases=[x['record'] for x in items], sdf_demonstration=example,
                  source_sha256=m.sha(Path(__file__)), geometry_generator_sha256=m.sha(SOURCE),
                  figures={name:m.sha(m.OUTPUT/f'{name}.pdf') for name in ['dataset-bands-teaching','sdf-3d-penalty-example']})
    (REPORT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(example,indent=2))


if __name__ == '__main__':
    main()
