"""Surface-cut and harmonic-chart routines from the original visualization."""
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve

def boundary_loops(mesh):
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    assert counts.max() <= 2, "Non-manifold edge"
    neighbors = {}
    for a, b in edges[counts == 1]:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    assert all(len(n) == 2 for n in neighbors.values()), "Non-manifold boundary"
    remaining = set(neighbors)
    loops = []
    while remaining:
        start = min(remaining)
        current, previous, loop = start, None, []
        while True:
            remaining.remove(current)
            loop.append(current)
            following = next(n for n in neighbors[current] if n != previous)
            if following == start:
                break
            previous, current = current, following
        loops.append(np.array(loop))
    return sorted(loops, key=len, reverse=True)


def cut_edges(mesh, path):
    """Duplicate vertex fans along an edge path, retaining all source faces."""
    seams = {tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])}
    faces = mesh.faces
    parent = np.arange(faces.size)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_faces = {}
    for fi, face in enumerate(faces):
        for k in range(3):
            edge = tuple(sorted((face[k], face[(k + 1) % 3])))
            edge_faces.setdefault(edge, []).append(fi)
    for edge, adjacent in edge_faces.items():
        if edge in seams or len(adjacent) != 2:
            continue
        a, b = adjacent
        for vertex in edge:
            ca = a * 3 + int(np.flatnonzero(faces[a] == vertex)[0])
            cb = b * 3 + int(np.flatnonzero(faces[b] == vertex)[0])
            parent[root(ca)] = root(cb)
    roots = np.array([root(i) for i in range(faces.size)])
    unique, inverse = np.unique(roots, return_inverse=True)
    result = trimesh.Trimesh(mesh.vertices[faces.ravel()[unique]], inverse.reshape(-1, 3), process=False)
    assert np.array_equal(result.triangles, mesh.triangles)
    return result


def open_holes(mesh):
    """Join boundary loops with shortest seams, without filling any hole."""
    seams = []
    loops = boundary_loops(mesh)
    # These actual patches have genus zero. A higher-genus patch needs more cuts.
    assert mesh.euler_number == 2 - len(loops), "Patch contains an uncut handle"
    while len(loops) > 1:
        edges = mesh.edges_unique
        lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        graph = sparse.coo_matrix((np.repeat(lengths, 2), (edges.ravel(), edges[:, ::-1].ravel())), shape=(len(mesh.vertices),) * 2).tocsr()
        distances, predecessors, _ = dijkstra(graph, indices=loops[0], min_only=True, return_predecessors=True)
        target = np.concatenate(loops[1:])
        end = int(target[np.argmin(distances[target])])
        path = [end]
        while predecessors[path[-1]] >= 0:
            path.append(int(predecessors[path[-1]]))
        assert len(path) >= 2
        seams.append(mesh.vertices[path].copy())
        mesh = cut_edges(mesh, path)
        new_loops = boundary_loops(mesh)
        assert len(new_loops) == len(loops) - 1
        loops = new_loops
    assert mesh.euler_number == 1
    return mesh, seams


def clip_surface(mesh, normal, origin):
    """Clip source triangles, retaining a source triangle index for every piece."""
    vertices, faces, source_ids, lookup = [], [], [], {}

    def vertex_id(point):
        key = tuple(np.round(point, 9))
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(point)
        return lookup[key]

    for fi, triangle in enumerate(mesh.triangles):
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            da, db = (a - origin) @ normal, (b - origin) @ normal
            if da >= 0:
                polygon.append(a)
            if (da >= 0) != (db >= 0):
                polygon.append(a + da / (da - db) * (b - a))
        if len(polygon) < 3:
            continue
        ids = [vertex_id(p) for p in polygon]
        for k in range(1, len(ids) - 1):
            face = [ids[0], ids[k], ids[k + 1]]
            if len(set(face)) == 3:
                faces.append(face)
                source_ids.append(fi)
    clipped = trimesh.Trimesh(vertices, faces, process=False)
    adjacency = clipped.face_adjacency
    graph = sparse.coo_matrix((np.ones(len(adjacency) * 2), (adjacency.ravel(), adjacency[:, ::-1].ravel())), shape=(len(faces),) * 2)
    count, components = connected_components(graph, directed=False)
    result = []
    for component in range(count):
        mask = components == component
        used, inverse = np.unique(clipped.faces[mask], return_inverse=True)
        patch = trimesh.Trimesh(clipped.vertices[used], inverse.reshape(-1, 3), process=False)
        result.append((patch, np.array(source_ids)[mask]))
    return sorted(result, key=lambda pair: pair[0].area, reverse=True)


def flatten(mesh, long_axis):
    """Positive mean-value harmonic coordinates with a convex ellipse boundary."""
    loop, = boundary_loops(mesh)
    # Both surfaces start their outer boundary at the anterior end.
    top = np.argmax(mesh.vertices[loop] @ long_axis)
    loop = np.roll(loop, -top)
    lengths = np.linalg.norm(mesh.vertices[np.roll(loop, -1)] - mesh.vertices[loop], axis=1)
    angle = np.pi / 2 + 2 * np.pi * np.r_[0, np.cumsum(lengths[:-1])] / lengths.sum()
    uv = np.zeros((len(mesh.vertices), 2))
    uv[loop] = np.c_[np.cos(angle), np.sin(angle)]
    # Mean-value weights reduce the influence of tiny cut triangles. All weights
    # remain positive; with a simple convex boundary this is a valid disk map.
    row, col, weights = [], [], []
    for k in range(3):
        i, j, h = mesh.faces[:, k], mesh.faces[:, (k + 1) % 3], mesh.faces[:, (k + 2) % 3]
        a, b = mesh.vertices[j] - mesh.vertices[i], mesh.vertices[h] - mesh.vertices[i]
        la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        cosine = np.clip(np.sum(a * b, axis=1) / (la * lb), -1 + 1e-12, 1 - 1e-12)
        half_tangent = np.sqrt((1 - cosine) / (1 + cosine))
        row.extend([i, i]); col.extend([j, h]); weights.extend([half_tangent / la, half_tangent / lb])
    adjacency = sparse.coo_matrix((np.concatenate(weights), (np.concatenate(row), np.concatenate(col))), shape=(len(uv),) * 2).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    interior = np.setdiff1d(np.arange(len(uv)), loop)
    uv[interior] = spsolve(laplacian[interior][:, interior], -laplacian[interior][:, loop] @ uv[loop])
    aspect = 1.7 if mesh.area > 100 else 1.0
    uv *= np.sqrt(mesh.area / np.pi) * np.array([1 / np.sqrt(aspect), np.sqrt(aspect)])
    triangles = uv[mesh.faces]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    signed = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    if np.sum(signed) < 0:
        uv[:, 0] *= -1
        signed *= -1
    tolerance = max(np.finfo(float).tiny, 1e-12 * mesh.area / len(mesh.faces))
    assert signed.min() > tolerance, f"Flipped or collapsed flat triangle: {signed.min()}"
    ratio = signed / mesh.area_faces
    return uv, loop, dict(
        vertices=len(uv), triangles=len(mesh.faces), euler=int(mesh.euler_number),
        boundary_loops=1, flipped_triangles=int((signed <= 0).sum()),
        minimum_flat_triangle_area=float(signed.min()),
        minimum_triangle_area_tolerance=float(tolerance),
        area_ratio_p05=float(np.percentile(ratio, 5)), area_ratio_p95=float(np.percentile(ratio, 95)),
        original_area_mm2=float(mesh.area), flat_area=float(signed.sum()),
    )

