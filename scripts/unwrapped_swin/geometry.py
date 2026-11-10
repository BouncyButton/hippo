"""Actual cut-shell harmonic charts and depth-preserving voxel rasterization.

The chart construction is the one used by the original visualization. MRI
points are splatted with retained source indices, not treated as an invertible
volume warp. Sampling collisions and intensity loss are explicitly audited.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import trimesh
import igl

from scripts.depth_supervision.geometry import longitudinal_plane, world_points
from .surface import boundary_loops, clip_surface, cut_edges, flatten, open_holes


def disk_pieces(mesh, normal):
    """Keep all faces, opening additional cuts when topology requires them."""
    loops = boundary_loops(mesh)
    if not loops:
        # A geometric plane cannot open a flat closed component: both clips
        # can retain the entire two-sided sheet. Open two edges topologically
        # instead. Duplicated seam vertices preserve every original triangle.
        path = mesh.faces[int(np.argmax(mesh.area_faces))].tolist()
        opened = cut_edges(mesh, path)
        if not boundary_loops(opened):
            raise ValueError('Closed component did not acquire a seam boundary')
        return [opened]
    return [mesh]


def open_disk(mesh):
    try:
        return open_holes(mesh)[0]
    except AssertionError:
        paths = igl.cut_to_disk(np.asarray(mesh.faces, dtype=np.int64))
        edges = {tuple(sorted((a, b))) for path in paths for a, b in zip(path[:-1], path[1:])}
        flags = np.array([[tuple(sorted((face[k], face[(k + 1) % 3]))) in edges
                           for k in range(3)] for face in mesh.faces])
        v, f, _ = igl.cut_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), flags)
        result = trimesh.Trimesh(v, f, process=False)
        if len(boundary_loops(result)) != 1 or result.euler_number != 1:
            raise ValueError('Additional seams did not produce a disk')
        return result


def flatten_pieces(mesh, long_axis, normal, depth=0):
    """Split numerically collapsed disk charts while retaining source triangles."""
    result = []
    for piece in disk_pieces(mesh, normal):
        opened = open_disk(piece)
        try:
            uv, _, checks = flatten(opened, long_axis)
        except AssertionError as error:
            if not str(error).startswith('Flipped or collapsed flat triangle:'):
                raise
            if len(opened.faces) < 2:
                raise ValueError('A source triangle cannot be parameterized') from error
            # Face partitioning introduces no new interpolated vertices or
            # sliver triangles. Connected pieces retain every original face.
            centres = opened.triangles_center
            _, _, axes = np.linalg.svd(centres - centres.mean(0), full_matrices=False)
            order = np.argsort(centres @ axes[0], kind='stable')
            for group in np.array_split(order, 2):
                faces = opened.faces[group]
                used, inverse = np.unique(faces, return_inverse=True)
                part = trimesh.Trimesh(opened.vertices[used], inverse.reshape(-1, 3), process=False)
                # Separate vertex fans that now touch at a point but share no edge.
                part = cut_edges(part, [])
                for connected in part.split(only_watertight=False, repair=False):
                    result.extend(flatten_pieces(connected, long_axis, normal, depth + 1))
        else:
            checks['partition_depth'] = depth
            result.append(dict(mesh=opened, uv=uv, checks=checks))
    if sum(len(c['mesh'].faces) for c in result) != len(mesh.faces):
        raise ValueError('Adaptive chart partition lost or duplicated triangles')
    return result


def surface_charts(prediction, affine):
    plane = longitudinal_plane(prediction, affine)
    verts, faces, _, _ = marching_cubes(np.pad((prediction > 0).astype(np.float32), 1), .5,
                                       allow_degenerate=False)
    mesh = trimesh.Trimesh(world_points(verts - 1, affine), faces, process=False)
    charts = []
    for side, sign in enumerate((-1, 1)):
        for piece, _ in clip_surface(mesh, sign * plane['normal'], plane['center_mm']):
            for chart in flatten_pieces(piece, plane['basis'][:, 0], plane['normal']):
                chart['side'] = side
                charts.append(chart)
    if not charts or abs(sum(c['mesh'].area for c in charts) - mesh.area) > 1e-5 * mesh.area:
        raise ValueError('Unfolding lost source surface area')
    return charts, plane


def raster(points, values, spacing, target_shape=None):
    """Trilinear normalized splat of native MRI samples onto a chart grid."""
    if target_shape is None:
        low = points.min(0) - spacing
        size = np.ceil((points.max(0) - low) / spacing).astype(int) + 3
        size = np.maximum(8, ((size + 7) // 8) * 8)
    else:
        size = np.array(target_shape[::-1])
        spacing = spacing.copy()
        spacing[:2] = np.maximum(np.ptp(points, axis=0)[:2] / (size[:2] - 3), .001)
        low = points.min(0) - spacing
    if np.prod(size) > 2_000_000 or size.max() > 256:
        raise ValueError(f'Chart exceeds declared grid budget: {size}')
    q = (points - low) / spacing
    lo = np.floor(q).astype(int)
    fraction = q - lo
    # Tensor dimensions are depth, v, u; grid_sample coordinates are u, v, depth.
    shape = tuple(size[::-1])
    total = np.zeros((len(values[0]), int(np.prod(size))), np.float32)
    count = np.zeros(int(np.prod(size)), np.float32)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                offset = np.array([a, b, c])
                weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
                idx = lo + offset
                linear = np.ravel_multi_index(idx[:, ::-1].T, shape)
                np.add.at(count, linear, weight)
                for channel in range(total.shape[0]):
                    np.add.at(total[channel], linear, values[:, channel] * weight)
    total /= np.maximum(count, 1e-8)[None]
    valid = count > 1e-7
    volume = np.concatenate([total.reshape((-1, *shape)), valid.reshape((1, *shape)).astype(np.float32)])
    grid = 2 * q / (size - 1) - 1
    nearest = np.ravel_multi_index(np.rint(q).astype(int)[:, ::-1].T, shape)
    reconstructed = ndi.map_coordinates(volume[0], q[:, ::-1].T, order=1, prefilter=False)
    support = ndi.map_coordinates(valid.reshape(shape).astype(float), q[:, ::-1].T, order=1, prefilter=False)
    checks = dict(grid_shape=[int(x) for x in shape], spacing=spacing.tolist(), samples=len(q),
                  rounded_cell_collision_fraction=float(1 - len(np.unique(nearest)) / len(q)),
                  return_support_fraction=float((support > .999).mean()),
                  intensity_rmse=float(np.sqrt(np.mean((reconstructed - values[:, 0]) ** 2))))
    return volume, grid.astype(np.float32), checks


def build_atlas(image, prediction, affine, fov, *, margin_mm=8., chart_step=.75):
    """No annotation argument: geometry and all input channels use prediction/MRI."""
    if image.shape != prediction.shape or not np.isfinite(image).all():
        raise ValueError('Invalid MRI/prediction grid')
    charts, plane = surface_charts(prediction, affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    roi = (ndi.distance_transform_edt(prediction == 0, sampling=spacing) <= margin_mm) & fov
    indices = np.argwhere(roi)
    xyz = world_points(indices, affine)
    local = (xyz - plane['center_mm']) @ plane['basis']
    side = (local[:, 2] >= 0).astype(int)
    ids = np.full(len(xyz), -1, np.int32)
    uv = np.zeros((len(xyz), 2))
    for half in (0, 1):
        selected = np.flatnonzero(side == half)
        vertices, faces, uvs, chart_ids = [], [], [], []
        offset = 0
        for chart_id, chart in enumerate(charts):
            if chart['side'] != half:
                continue
            mesh = chart['mesh']
            vertices.append(mesh.vertices); faces.append(mesh.faces + offset); uvs.append(chart['uv'])
            chart_ids.extend([chart_id] * len(mesh.faces)); offset += len(mesh.vertices)
        vertices, faces, uvs = np.concatenate(vertices), np.concatenate(faces), np.concatenate(uvs)
        _, face_id, closest = igl.point_mesh_squared_distance(xyz[selected], vertices, faces)
        bary = trimesh.triangles.points_to_barycentric(vertices[faces[face_id]], closest)
        if not np.isfinite(bary).all():
            raise ValueError('Nonfinite projection coordinates')
        uv[selected] = np.einsum('ni,nij->nj', bary, uvs[faces[face_id]])
        ids[selected] = np.asarray(chart_ids)[face_id]
    if (ids < 0).any():
        raise ValueError('Unassigned MRI sample')
    values = np.c_[image[tuple(indices.T)], local[:, 0] / 32., local[:, 2] / 16.].astype(np.float32)
    linear_indices = np.ravel_multi_index(indices.T, image.shape)
    pack, audit = {}, []
    count = 0
    for chart_id, chart in enumerate(charts):
        selected = ids == chart_id
        if not selected.any():
            continue
        # The plane control uses the SAME points and chart membership; only UV changes.
        for view, positions, step in (
            ('unfold', np.c_[uv[selected], np.abs(local[selected, 2])], np.array([chart_step, chart_step, 1.])),
            ('plane', np.c_[local[selected, :2], np.abs(local[selected, 2])], np.array([1., 1., 1.])),
        ):
            target = pack[f'unfold_{count}_input'].shape[1:] if view == 'plane' else None
            volume, grid, checks = raster(positions, values[selected], step, target_shape=target)
            prefix = f'{view}_{count}'
            pack[prefix + '_input'] = volume.astype(np.float16)
            pack[prefix + '_grid'] = grid
            pack[prefix + '_indices'] = linear_indices[selected].astype(np.int32)
            checks.update(view=view, chart=count, side=chart['side'])
            audit.append(checks)
        count += 1
    if len(np.unique(np.concatenate([pack[f'unfold_{i}_indices'] for i in range(count)]))) != len(indices):
        raise ValueError('Lost or duplicated native voxel membership')
    pack['charts'] = np.array(count)
    pack['roi_indices'] = linear_indices.astype(np.int32)
    pack['side_by_point'] = side.astype(np.uint8)
    pack['unfolded_points'] = np.c_[uv, np.abs(local[:, 2])].astype(np.float32)
    pack['chart_by_point'] = ids
    pack['center'] = plane['center_mm']
    pack['basis'] = plane['basis']
    report = dict(charts=count, roi_voxels=len(indices),
                  representation='Cut-shell harmonic UV plus absolute physical plane depth, separate sides and chart pieces.',
                  geometry_uses_annotation=False, all_source_faces_retained=True,
                  continuous_map_is_claimed_invertible=False,
                  unique_native_records=True, charts_audit=audit,
                  surface=[dict(side=c['side'], **c['checks']) for c in charts])
    return pack, report
