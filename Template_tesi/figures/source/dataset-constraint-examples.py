"""Illustrate annotation-derived supervision on three actual native volumes.

Run from the repository using .venv/bin/python. No predictions or results are
used. All fields/counts are computed in 3-D on the native, unaugmented grid;
this is an illustration of the rules, not a saved training-update tensor.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-geometry-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from scripts.depth_supervision.geometry import build_geometry, signed_distance
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands
from thesis.new_constraints.edge_consistency import signed_face_error
from thesis.new_constraints.signed_distance import signed_distance_maps

OUTPUT = ROOT / "Template_tesi/figures/compiled"
REPORT = ROOT / "reports/thesis_dataset_constraint_examples_20261103"
BLUE, ORANGE, INK = "#24699b", "#cd752a", "#27313c"
SDF_CMAP = LinearSegmentedColormap.from_list("signed", [BLUE, "#fcfcfc", ORANGE])
DEPTH_CMAP = LinearSegmentedColormap.from_list("weights", ["#f0f6fb", "#85b4d0", "#24699b", "#07344f"])
DEPTH_NORM = LogNorm(vmin=.5, vmax=100)
CASES = [
    ("MSD", "Dataset101_MSD", "hippocampus_001"),
    ("ADNI", "Dataset103_ADNI", "hippocampus_adni_10007_L"),
    ("GeoFWI3D", "Dataset203_GeoFWI3D", "model_0001"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(name, folder, case):
    image_path = ROOT / f"datasets/{folder}/imagesTr/{case}_0000.nii.gz"
    label_path = ROOT / f"datasets/{folder}/labelsTr/{case}.nii.gz"
    image_file, label_file = nib.load(image_path), nib.load(label_path)
    image = np.asarray(image_file.dataobj, dtype=np.float32)
    labels = np.asarray(label_file.dataobj, dtype=np.uint8)
    assert image.shape == labels.shape and np.allclose(image_file.affine, label_file.affine)
    assert np.isfinite(image).all() and set(np.unique(labels)) <= {0, 1, 2}
    mask = labels > 0
    inner_t, outer_t = build_boundary_bands(torch.from_numpy(mask)[None, None], steps=2)
    inner, outer = inner_t.numpy()[0, 0], outer_t.numpy()[0, 0]
    structure = ndimage.generate_binary_structure(3, 1)
    assert np.array_equal(inner, mask & ~ndimage.binary_erosion(mask, structure, iterations=2, border_value=0))
    assert np.array_equal(outer, ndimage.binary_dilation(mask, structure, iterations=2, border_value=0) & ~mask)
    support = inner | outer
    # Source-array coordinates for ADNI/GeoFWI3D, physical coordinates for MSD.
    affine = label_file.affine if name == "MSD" else np.eye(4)
    geometry = build_geometry(labels, affine, fields=False)
    # None of these examples needs the geological near-parallel-axis fallback.
    points = np.argwhere(mask) @ affine[:3, :3].T + affine[:3, 3]
    covariance = np.cov(points.T, bias=True)
    _, axes = np.linalg.eigh(covariance)
    projected = np.array([1., 0., 0.]) - axes[0, -1] * axes[:, -1]
    assert np.linalg.norm(projected) >= .1
    weights = geometry["weights"]
    roi = weights > 0
    assert np.isclose(weights.sum(), 1., atol=1e-6)
    for b, count, mass in zip(geometry["bin_values"], geometry["bin_counts"], geometry["bin_masses"]):
        chosen = roi & (geometry["bin_ids"] == b)
        assert chosen.sum() == count and np.isclose(weights[chosen].sum(), mass, atol=1e-6)
    sdf = signed_distance(labels == 1, (1., 1., 1.), 8.) / 8.
    original = signed_distance_maps(torch.from_numpy(labels)[None, None]).numpy()[0, 0]
    assert np.allclose(sdf, original)
    # Axis 2 leaves source axis 0 in the displayed plane, exposing depth layers.
    # Max foreground area and first-index tie break use reference labels alone.
    axis = 2
    index = int(mask.sum(axis=(0, 1)).argmax())
    plane = lambda value: np.take(value, index, axis=axis).T
    full_counts, crossings = [], []
    for direction in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[direction], b[direction] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        selected = support[a] & support[b]
        full_counts.append(int(selected.sum()))
        crossings.append(int((selected & (mask[a] != mask[b])).sum()))
    _, actual_count = signed_face_error(torch.from_numpy(mask.astype(np.float32))[None, None],
                                        torch.from_numpy(mask)[None, None], inner_t | outer_t)
    assert int(actual_count.item()) == sum(full_counts)
    # A deterministic boundary-centred magnification for the appendix edge panel.
    m2 = plane(mask)
    boundary = m2 & ~ndimage.binary_erosion(m2)
    h, w = m2.shape
    interior = np.zeros_like(boundary)
    interior[6:-6, 6:-6] = True
    candidates = np.argwhere(boundary & interior)
    center = np.argwhere(m2).mean(0)
    if not len(candidates):
        candidates = np.argwhere(boundary)
    chosen = candidates[np.argmin(((candidates - center) ** 2).sum(1))]
    size = 16
    y0 = int(np.clip(chosen[0] - size//2, 0, h-size))
    x0 = int(np.clip(chosen[1] - size//2, 0, w-size))
    limits = (-1., 1.) if name == "GeoFWI3D" else tuple(np.percentile(image[image != 0], [1, 99]))
    fields = dict(image=image, labels=labels, mask=mask, inner=inner, outer=outer,
                  support=support, sdf=sdf, relative_weights=weights * roi.sum(),
                  depth=geometry["signed_depth_mm"], bins=geometry["bin_ids"], roi=roi)
    record = dict(dataset=name, case=case, image=str(image_path.relative_to(ROOT)),
                  labels=str(label_path.relative_to(ROOT)), image_sha256=sha(image_path),
                  labels_sha256=sha(label_path), shape=list(labels.shape),
                  nifti_affine=label_file.affine.tolist(), depth_coordinate_map=affine.tolist(),
                  units="mm" if name == "MSD" else "voxels", plane_axis=axis,
                  slice_index_zero_based=index, display="transpose remaining axes; origin lower; no flips",
                  slice_rule="axis 2 plane with maximum foreground area; lowest index breaks ties",
                  grayscale_limits=list(limits), display_zoom_xywh=[x0, y0, size, size],
                  zoom_rule="interior in-plane reference boundary voxel nearest slice foreground centroid; row-major tie break",
                  band_steps=2, band_connectivity=6, observation_frame_excluded=False,
                  band_counts=[int(inner.sum()), int(outer.sum())],
                  edge_counts_per_axis=full_counts, crossing_edge_counts_per_axis=crossings,
                  sdf_class=1, sdf_units="voxels", sdf_clip=8., sdf_half_voxel_offset=.5,
                  depth_stratum_width=2., depth_roi_margin=8., depth_sparse_threshold=64,
                  depth_projected_axis_norm=float(np.linalg.norm(projected)),
                  depth_fallback_used=False, depth_normal=geometry["normal"].tolist(),
                  depth_center=geometry["center_mm"].tolist(),
                  depth_bins=geometry["bin_values"].tolist(), depth_counts=geometry["bin_counts"].tolist(),
                  depth_masses=geometry["bin_masses"].tolist(), depth_weight_sum=float(weights.sum()),
                  depth_roi_voxels=int(roi.sum()),
                  displayed_depth_scale="weight / mean positive 3-D weight; common log colour scale 0.5 to 100",
                  native_grid_illustration=True, prediction_or_result_data_used=False)
    print(name, 'slice', index, 'full 3-D pair count', sum(full_counts), 'weight sum', weights.sum())
    return dict(name=name, case=case, fields=fields, plane=plane, geometry=geometry,
                record=record, limits=limits, zoom=(x0, y0, size))


def contour(ax, field, color="white", lw=.6, linestyle="solid"):
    if field.any() and not field.all():
        ax.contour(field.astype(float), levels=[.5], colors=[color], linewidths=lw,
                   linestyles=linestyle)


def edges(support, foreground):
    segments, changed = [], []
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        ys, xs = np.nonzero(support[a] & support[b])
        delta = (foreground[a] != foreground[b])[ys, xs]
        for y, x, crossing in zip(ys, xs, delta):
            segments.append([(x, y), (x + (axis == 1), y + (axis == 0))])
            changed.append(bool(crossing))
    return np.asarray(segments), np.asarray(changed)


def panel(ax, item, kind, *, zoom=False, overview=False):
    d = {key: item["plane"](value) for key, value in item["fields"].items()}
    h, w = d["image"].shape
    ax.set(xlim=(-.5, w-.5), ylim=(-.5, h-.5), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b6bfc6"); spine.set_linewidth(.5)
    if kind in ("reference", "bands", "edges"):
        ax.imshow(d["image"], cmap="gray", vmin=item["limits"][0], vmax=item["limits"][1],
                  origin="lower", interpolation="nearest", rasterized=True)
        overlay = np.zeros((*d["image"].shape, 4))
        if kind == "reference":
            overlay[d["labels"] == 1] = to_rgba(BLUE, .75)
            overlay[d["labels"] == 2] = to_rgba(ORANGE, .75)
        elif kind == "bands":
            overlay[d["inner"]] = to_rgba(BLUE, .90)
            overlay[d["outer"]] = to_rgba(ORANGE, .90)
        else:
            # Light neutral wash separates pair colours from image intensities.
            overlay[:] = to_rgba("white", .70)
        ax.imshow(overlay, origin="lower", interpolation="nearest", rasterized=True)
        if kind == "edges":
            seg, cross = edges(d["support"], d["mask"])
            factor = 1.25 if zoom else (0.65 if item["name"] == "GeoFWI3D" else 1.)
            for keep, color, width in ((~cross, BLUE, .38), (cross, ORANGE, .9)):
                ax.add_collection(LineCollection(seg[keep], colors=color, linewidths=width*factor,
                                                rasterized=overview))
            contour(ax, d["mask"], INK, .48)
        else:
            contour(ax, d["mask"], "white", .6)
            if kind == "reference" and item["name"] == "MSD":
                contour(ax, d["labels"] == 1, "white", .65)
    elif kind == "sdf":
        ax.imshow(d["sdf"], cmap=SDF_CMAP, vmin=-1, vmax=1, origin="lower", interpolation="nearest", rasterized=True)
        contour(ax, d["mask"], INK, .55)
        if item["name"] == "MSD":
            contour(ax, d["labels"] == 1, INK, .7)
            for c, letter in ((1, "A"), (2, "P")):
                if (d["labels"] == c).any():
                    y, x = np.argwhere(d["labels"] == c).mean(0)
                    ax.text(x, y, letter, ha="center", va="center", fontsize=7 if overview else 9,
                            color=INK, bbox=dict(boxstyle="round,pad=.12", facecolor="white", edgecolor="none", alpha=.85))
    elif kind == "weights":
        values = np.ma.masked_where(~d["roi"], d["relative_weights"])
        ax.set_facecolor("#e1e4e7")
        ax.imshow(values, cmap=DEPTH_CMAP, norm=DEPTH_NORM, origin="lower", interpolation="nearest", rasterized=True)
        contour(ax, d["mask"], INK, .65)
        ax.contour(d["depth"], levels=[0], colors=[ORANGE], linestyles="dashed", linewidths=.7)
    elif kind == "layers":
        values = np.ma.masked_where(~d["roi"], d["depth"])
        ax.set_facecolor("#e1e4e7")
        limit = max(abs(float(values.min())), abs(float(values.max())))
        ax.imshow(values, cmap=SDF_CMAP, vmin=-limit, vmax=limit, origin="lower", interpolation="nearest", rasterized=True)
        levels = np.arange(np.floor(values.min()/2)*2, values.max()+2, 2)
        ax.contour(values, levels=levels, colors=["#60666b"], linewidths=.35)
        ax.contour(d["depth"], levels=[0], colors=[ORANGE], linestyles="dashed", linewidths=1.2)
        contour(ax, d["mask"], INK, .65)
    if zoom:
        x, y, n = item["zoom"]
        ax.set(xlim=(x-.5, x+n-.5), ylim=(y-.5, y+n-.5))
        ax.set_xticks(np.arange(x-.5, x+n, 1), minor=True)
        ax.set_yticks(np.arange(y-.5, y+n, 1), minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid(which="minor", lw=.2, color="#aab1b8", alpha=.6)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.pdf", dpi=300, facecolor="white")
    fig.savefig(REPORT / f"{name}.png", dpi=210, facecolor="white")
    plt.close(fig)


def voxel_faces(occupied, values):
    """Exact exposed faces of native voxels; no resampling or smoothing."""
    vertices, face_values = [], []
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        for sign in (-1, 1):
            neighbour = np.roll(occupied, -sign, axis=axis)
            border = [slice(None)] * 3
            border[axis] = -1 if sign == 1 else 0
            neighbour[tuple(border)] = False
            exposed = occupied & ~neighbour
            centers = np.argwhere(exposed).astype(float)
            if not len(centers):
                continue
            offsets = np.zeros((4, 3))
            offsets[:, axis] = .5 * sign
            offsets[:, others] = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
            vertices.append(centers[:, None, :] + offsets[None, :, :])
            face_values.append(values[exposed])
    return np.concatenate(vertices), np.concatenate(face_values)


def volume_panel(ax, item, kind):
    f = item['fields']
    g = item['geometry']
    # Orient both 3-D displays in the reference-plane basis: normal along x,
    # principal axis along y, transverse direction along z.
    rotation = g['basis'][:, [2, 0, 1]]
    if kind == 'reference':
        verts, values = voxel_faces(f['mask'], f['labels'])
        colors = np.array([to_rgba(BLUE) if v == 1 else to_rgba(ORANGE) for v in values])
    elif kind == 'sdf':
        # A cutaway through a 4-voxel exterior neighbourhood exposes signed
        # values inside the actual class mask. Quarter removed for display only.
        selected_class = f['labels'] == 1
        center = np.argwhere(selected_class).mean(0)
        grid = np.indices(selected_class.shape).reshape(3, -1).T
        local = ((grid - center) @ rotation).reshape((*selected_class.shape, 3))
        removed = (local[..., 0] >= 0) & (local[..., 1] >= 0)
        occupied = (f['sdf'] <= .5) & ~removed
        verts, values = voxel_faces(occupied, f['sdf'])
        colors = SDF_CMAP((values + 1) / 2)
    else:
        # Five strata selected uniformly over occupied bin indices, including
        # both extremes. Counts/weights still use every layer of the full ROI.
        verts_all, values_all = [], []
        bins = item['geometry']['bin_values']
        selected_bins = bins[np.linspace(0, len(bins)-1, min(5, len(bins))).round().astype(int)]
        midpoint = .5 * (float(bins[0]) + float(bins[-1]))
        normal = item['geometry']['normal']
        for b in selected_bins:
            selected = f['roi'] & (f['bins'] == b)
            v, value = voxel_faces(selected, f['relative_weights'])
            v += 3. * (float(b) - midpoint) * normal
            verts_all.append(v); values_all.append(value)
        verts, values = np.concatenate(verts_all), np.concatenate(values_all)
        colors = DEPTH_CMAP(DEPTH_NORM(values))
    verts = verts @ rotation
    # Gentle directional lighting preserves the scalar colours and exposes
    # surface orientation. Native voxel faces remain geometrically unchanged.
    normals = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    shade = .80 + .20 * np.abs(normals @ (np.array([.4, .3, .85]) / np.linalg.norm([.4, .3, .85])))
    colors[:, :3] *= shade[:, None]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=colors, edgecolors='none',
                                        linewidths=0, rasterized=True, zsort='average'))
    low, high = verts.min(axis=(0, 1)), verts.max(axis=(0, 1))
    extent = high-low
    ax.set(xlim=(low[0]-1, high[0]+1), ylim=(low[1]-1, high[1]+1), zlim=(low[2]-1, high[2]+1))
    ax.set_box_aspect(extent+2, zoom=1.03)
    ax.view_init(elev=24, azim=60)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    item['record']['sdf_3d_display'] = 'native voxel faces; field <= +4 voxels; positive normal/principal-axis quadrant about class centroid removed for display'
    item['record']['depth_3d_display'] = 'five uniformly spaced occupied strata including extremes; each shifted by 3*(bin-midrange bin)*normal; full-volume weights/counts unchanged'
    bins = g['bin_values']
    item['record']['depth_displayed_bins'] = bins[np.linspace(0, len(bins)-1, min(5, len(bins))).round().astype(int)].tolist()
    item['record']['3d_camera'] = dict(elevation=24, azimuth=60, projection='orthographic', basis_columns=['normal','principal','transverse'])


def overview(items):
    fig = plt.figure(figsize=(6.3, 7.0))
    gs = fig.add_gridspec(3, 2, left=.06, right=.99, bottom=.15, top=.94, wspace=.01, hspace=.09)
    kinds = ["sdf", "weights"]
    titles = ["Signed distance: 3-D cutaway", "Depth weights: five 3-D layers"]
    for row, item in enumerate(items):
        for col, kind in enumerate(kinds):
            ax = fig.add_subplot(gs[row, col], projection='3d')
            volume_panel(ax, item, kind)
            if row == 0:
                ax.set_title(titles[col], fontsize=8.3, pad=8)
            if col == 0:
                ax.text2D(-.07, .5, item["name"], transform=ax.transAxes, rotation=90,
                          ha="center", va="center", fontweight="bold", fontsize=9)
                ax.text2D(.5, -.02, "Anterior class" if row == 0 else "Hippocampus" if row == 1 else "Salt",
                          transform=ax.transAxes, ha='center', fontsize=7.2)
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=SDF_CMAP),
                      cax=fig.add_axes([.13, .068, .33, .014]), orientation="horizontal", ticks=[-1, 0, 1])
    cb1.ax.tick_params(labelsize=7, pad=2, length=2)
    cb1.set_label("Signed distance / 8 voxels", fontsize=7.2, labelpad=2)
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=DEPTH_NORM, cmap=DEPTH_CMAP),
                      cax=fig.add_axes([.61, .068, .33, .014]), orientation="horizontal", ticks=[.5, 1, 10, 100])
    cb2.ax.set_xticklabels(["0.5", "1", "10", "100"])
    cb2.ax.minorticks_off(); cb2.ax.tick_params(labelsize=7, pad=2, length=2)
    cb2.set_label("Voxel weight / mean positive weight", fontsize=7.2, labelpad=2)
    fig.text(.54, .112, "Full-volume calculations; cutaway and separated layer samples aid visibility", ha="center", fontsize=7)
    save(fig, "dataset-constraint-overview")


def detailed(item):
    fig = plt.figure(figsize=(6.3, 5.65))
    gs = fig.add_gridspec(2, 3, left=.025, right=.99, bottom=.23, top=.88, wspace=.12, hspace=.29)
    specs = [("reference", "(a) Intact 3-D annotation"), ("bands", "(b) Image and bands"),
             ("edges", "(c) Neighbour pairs: detail"), ("sdf", "(d) 3-D signed distance"),
             ("layers", "(e) Layer size and influence"), ("weights", "(f) 3-D depth weights")]
    for n, (kind, title) in enumerate(specs):
        if kind in ('reference', 'sdf', 'weights'):
            ax = fig.add_subplot(gs[n//3, n%3], projection='3d')
            volume_panel(ax, item, kind)
        elif kind == 'layers':
            ax = fig.add_subplot(gs[n//3, n%3])
            g = item['geometry']; bins = g['bin_values']
            ax.bar(bins, 100*g['bin_counts']/g['bin_counts'].sum(), width=.84,
                   color='#b9c4cb', label='Share of ROI voxels')
            ax.plot(bins, 100*g['bin_masses'], 'o-', ms=2, lw=.8, color=ORANGE,
                    label='Share of loss weight')
            ax.set_xlabel('Depth-layer index', fontsize=7)
            ax.set_ylabel('Share (%)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.spines[['top','right']].set_visible(False)
            ax.set_ylim(0, max(float((100*g['bin_counts']/g['bin_counts'].sum()).max()), float((100*g['bin_masses']).max()))*1.35)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.legend(fontsize=5.5, loc='upper left', frameon=False)
        else:
            ax = fig.add_subplot(gs[n//3, n%3]); panel(ax, item, kind, zoom=(kind == "edges"))
        ax.set_title(title, fontsize=8.2, pad=5)
        if kind == 'bands':
            x, y, size = item["zoom"]
            ax.add_patch(Rectangle((x-.5, y-.5), size, size, fill=False, edgecolor="#f0da68", linewidth=.85))
        if kind == "reference":
            text = "Blue: anterior; orange: posterior" if item["name"] == "MSD" else "Blue: hippocampus" if item["name"] == "ADNI" else "Blue: salt"
            ax.text2D(.5, -.01, text, transform=ax.transAxes, ha='center', fontsize=6.6)
        if kind == "sdf":
            ax.text2D(.5, -.01, "Anterior class" if item['name']=='MSD' else 'Negative inside; positive outside',
                      transform=ax.transAxes, ha='center', fontsize=6.7)
    fig.text(.5, .975, f'{item["name"]}: {item["case"]}', ha="center", va="top", fontsize=10.3)
    fig.text(.5, .932, f'Band and pair views: axis 2, slice {item["record"]["slice_index_zero_based"]}; geometry computed in 3-D',
             ha="center", va="top", fontsize=8, color="#6d7782")
    fig.legend(handles=[Patch(color=BLUE, label="Inner band / equal-label pair"),
                        Patch(color=ORANGE, label="Outer band / label-change pair")],
               loc="lower center", bbox_to_anchor=(.5, .115), ncol=1, frameon=False, fontsize=7,
               handlelength=1.5, labelspacing=.35)
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=SDF_CMAP),
                      cax=fig.add_axes([.065, .07, .37, .017]), orientation="horizontal", ticks=[-1, 0, 1])
    cb1.ax.tick_params(labelsize=7, pad=2, length=2)
    cb1.set_label("Signed distance / 8 voxels", fontsize=7, labelpad=2)
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=DEPTH_NORM, cmap=DEPTH_CMAP),
                      cax=fig.add_axes([.565, .07, .37, .017]), orientation="horizontal", ticks=[.5, 1, 10, 100])
    cb2.ax.set_xticklabels(["0.5", "1", "10", "100"]); cb2.ax.minorticks_off()
    cb2.ax.tick_params(labelsize=7, pad=2, length=2)
    cb2.set_label("Voxel weight / mean positive weight", fontsize=7, labelpad=2)
    save(fig, item["name"].lower() + "-constraint-example")


def main():
    REPORT.mkdir(exist_ok=True, parents=True)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK})
    torch.set_num_threads(2)
    items = [load_case(*args) for args in CASES]
    overview(items)
    for item in items:
        detailed(item)
    sources = [Path(__file__), ROOT / "scripts/depth_supervision/geometry.py",
               ROOT / "thesis/new_constraints/bands/outer_boundary.py",
               ROOT / "thesis/new_constraints/edge_consistency.py",
               ROOT / "thesis/new_constraints/signed_distance.py"]
    names = ["dataset-constraint-overview"] + [x[0].lower()+"-constraint-example" for x in CASES]
    manifest = dict(cases=[x["record"] for x in items],
                    sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    figures={name: sha(OUTPUT / f"{name}.pdf") for name in names},
                    checks="input alignment; 6-neighbour bands vs SciPy; SDF vs training implementation; 3-D pair counts vs training implementation; depth normalization and per-layer masses")
    (REPORT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
