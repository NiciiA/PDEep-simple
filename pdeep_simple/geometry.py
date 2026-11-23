"""Small conforming meshes, domain sampling, and P1 interpolation."""
import numpy as np
from scipy.spatial import cKDTree, Delaunay

GEOMETRIES = ('square', 'rectangle', 'disk', 'ellipse', 'lshape', 'annulus', 'hole',
              'polygon', 'triangle', 'trapezoid', 'star', 'cross', 'dumbbell', 'crescent', 'two_holes')


def polygon_mesh(loops, spacing):
    """Recover polygon segments by bisection, then keep only domain triangles.

    Once every segment is a Delaunay edge, no retained triangle can bridge a
    concavity or a hole. Curves are represented by the supplied polygon loops.
    """
    from matplotlib.path import Path
    loops = [np.asarray(loop, float) for loop in loops]
    paths = [Path(np.concatenate([loop, loop[:1]])) for loop in loops]
    def inside(points):
        mask = paths[0].contains_points(points)
        for path in paths[1:]:
            mask &= ~path.contains_points(points)
        return mask
    boundary, segments = [], []
    for loop in loops:
        start = len(boundary)
        for a, b in zip(loop, np.roll(loop, -1, axis=0)):
            count = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
            boundary.extend(a + (b - a) * np.arange(count)[:, None] / count)
        end = len(boundary)
        segments.extend((i, start if i == end - 1 else i + 1) for i in range(start, end))
    boundary = np.asarray(boundary)
    low, high = loops[0].min(0), loops[0].max(0)
    x, y = np.meshgrid(np.arange(low[0] + spacing / 2, high[0], spacing),
                       np.arange(low[1] + spacing / 2, high[1], spacing))
    points = np.column_stack([x.ravel(), y.ravel()]); points = points[inside(points)]
    # Avoid duplicate/cocircular boundary-interior vertices.
    points = points[distance_to_boundary(dict(nodes=boundary, boundary_edges=np.asarray(segments)), points) > spacing * .15]
    nodes = np.concatenate([boundary, points]); segments = np.asarray(segments)
    for _ in range(16):
        triangles = Delaunay(nodes).simplices
        edges = {tuple(e) for e in np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                                          triangles[:, [2, 0]]]), axis=1)}
        missing = np.array([tuple(sorted(s)) not in edges for s in segments])
        if not missing.any():
            break
        split = segments[missing]; middle = np.arange(len(nodes), len(nodes) + len(split))
        nodes = np.concatenate([nodes, nodes[split].mean(1)])
        segments = np.concatenate([segments[~missing], np.column_stack([split[:, 0], middle]),
                                   np.column_stack([middle, split[:, 1]])])
    else:
        raise ValueError('could not recover geometry boundary; increase resolution or widen thin regions')
    triangles = triangles[inside(nodes[triangles].mean(1))]
    used, inverse = np.unique(triangles, return_inverse=True)
    return nodes[used], inverse.reshape(-1, 3)


def mesh(family, resolution=12, width=1.0, height=1.0, radius=0.5, parameters=None):
    if resolution < 4:
        raise ValueError("mesh resolution must be at least 4")
    n = resolution + resolution % 2  # L corner must coincide with grid lines.
    parameters = {} if parameters is None else parameters
    if family in ("square", "rectangle", "lshape"):
        x, y = np.meshgrid(np.linspace(0, width, n + 1), np.linspace(0, height, n + 1))
        nodes = np.column_stack([x.ravel(), y.ravel()])
        triangles = []
        for j in range(n):
            for i in range(n):
                if family == "lshape" and i >= n // 2 and j >= n // 2:
                    continue
                a = j * (n + 1) + i
                triangles.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]])
        triangles = np.asarray(triangles)
        used, inverse = np.unique(triangles, return_inverse=True)
        nodes, triangles = nodes[used], inverse.reshape(-1, 3)
    elif family in ("disk", "ellipse"):
        rings, sectors = n // 2, 4 * n
        angles = np.arange(sectors) * 2 * np.pi / sectors
        nodes = [[0., 0.]]
        for r in np.linspace(radius / rings, radius, rings):
            nodes.extend(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        nodes = np.asarray(nodes)
        triangles = [[0, 1 + j, 1 + (j + 1) % sectors] for j in range(sectors)]
        for k in range(1, rings):
            for j in range(sectors):
                a, b = 1 + (k - 1) * sectors + j, 1 + (k - 1) * sectors + (j + 1) % sectors
                c, d = a + sectors, b + sectors
                triangles.extend([[a, c, d], [a, d, b]])
        triangles = np.asarray(triangles)
        if family == 'ellipse':
            nodes[:, 1] *= parameters.get('axis_ratio', .7)
    elif family in GEOMETRIES:
        angles = np.arange(4 * n) * 2 * np.pi / (4 * n)
        circle = lambda r, center: np.column_stack([r * np.cos(angles), r * np.sin(angles)]) + center
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        holes = []
        if family == 'annulus':
            outer = circle(radius, [0., 0.])
            holes = [circle(radius * parameters.get('inner_ratio', .4), [0., 0.])]
        elif family in ('hole', 'two_holes'):
            outer = square * [width, height]
            r = parameters.get('hole_radius', .12) * min(width, height)
            centers = [[.5, .5]] if family == 'hole' else [[.28, .5], [.72, .5]]
            holes = [circle(r, np.array(center) * [width, height]) for center in centers]
        elif family == 'triangle':
            outer = np.array([[0., 0.], [1., 0.], [parameters.get('apex', .5), 1.]]) * [width, height]
        elif family == 'trapezoid':
            ratio = parameters.get('top_ratio', .6)
            outer = np.array([[0., 0.], [1., 0.], [(1 + ratio) / 2, 1.], [(1 - ratio) / 2, 1.]]) * [width, height]
        elif family in ('star', 'polygon'):
            count = int(parameters.get('points', 5)) if family == 'star' else int(parameters.get('sides', 6))
            theta = np.arange(2 * count if family == 'star' else count) * 2 * np.pi / (2 * count if family == 'star' else count)
            r = np.ones(len(theta)) * radius
            if family == 'star':
                r[1::2] *= parameters.get('inner_ratio', .5)
            outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        elif family == 'cross':
            a = parameters.get('arm_width', .35) / 2; lo, hi = .5 - a, .5 + a
            outer = np.array([[lo, 0], [hi, 0], [hi, lo], [1, lo], [1, hi], [hi, hi],
                              [hi, 1], [lo, 1], [lo, hi], [0, hi], [0, lo], [lo, lo]]) * [width, height]
        elif family == 'dumbbell':
            a = parameters.get('neck_width', .25) / 2; lo, hi = .5 - a, .5 + a
            outer = np.array([[0, 0], [.35, 0], [.35, lo], [.65, lo], [.65, 0], [1, 0],
                              [1, 1], [.65, 1], [.65, hi], [.35, hi], [.35, 1], [0, 1]]) * [width, height]
        elif family == 'crescent':
            # Difference of two intersecting disks, represented by one boundary loop.
            offset, inner = .45, .85
            x = (1 - inner ** 2 + offset ** 2) / (2 * offset)
            theta = np.arccos(x); phi = np.arctan2(np.sqrt(1 - x * x), x - offset)
            a = np.linspace(theta, 2 * np.pi - theta, 3 * n)
            b = np.linspace(2 * np.pi - phi, phi, 3 * n)[1:-1]
            outer = radius * np.concatenate([np.column_stack([np.cos(a), np.sin(a)]),
                                             np.column_stack([offset + inner * np.cos(b), inner * np.sin(b)])])
        nodes, triangles = polygon_mesh([outer] + holes, max(np.ptp(outer, axis=0)) / n)
    else:
        raise ValueError(f"unknown geometry {family}; choose {GEOMETRIES}")
    # Orient every element counterclockwise, then find topological boundary edges.
    p = nodes[triangles]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    triangles[cross < 0] = triangles[cross < 0][:, [0, 2, 1]]
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges[counts == 1]
    return dict(nodes=nodes, triangles=triangles, boundary_nodes=np.unique(boundary_edges),
                boundary_edges=boundary_edges)


def element_geometry(nodes, triangles):
    p = nodes[triangles]
    twice = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if np.any(twice <= 0):
        raise ValueError("mesh has non-positive element areas")
    gradients = np.stack([p[:, [1, 2, 0], 1] - p[:, [2, 0, 1], 1],
                          p[:, [2, 0, 1], 0] - p[:, [1, 2, 0], 0]], axis=-1)
    return twice / 2, gradients / twice[:, None, None]


def node_weights(case):
    area, _ = element_geometry(case['nodes'], case['triangles'])
    weights = np.zeros(len(case['nodes']))
    np.add.at(weights, case['triangles'].ravel(), np.repeat(area / 3, 3))
    return weights


def sample_points(case, count, rng):
    area, _ = element_geometry(case['nodes'], case['triangles'])
    elements = rng.choice(len(area), count, p=area / area.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    bary = np.column_stack([1 - uv.sum(axis=1), uv])
    points = (case['nodes'][case['triangles'][elements]] * bary[..., None]).sum(axis=1)
    return points


def locate(case, points):
    """Locate in the supplied connectivity, never a new Delaunay triangulation."""
    # Matplotlib's triangle finder respects the reentrant L boundary.
    import matplotlib.tri as mtri
    triangulation = mtri.Triangulation(*case['nodes'].T, case['triangles'])
    elements = triangulation.get_trifinder()(*np.asarray(points).T)
    for index in np.flatnonzero(elements < 0):
        # Float32 roundoff can put an exact disk boundary vertex just outside.
        p = case['nodes'][case['triangles']]
        matrices = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=-1)
        uv = np.linalg.solve(matrices, (np.asarray(points)[index] - p[:, 0])[..., None])[..., 0]
        candidate = np.column_stack([1 - uv.sum(axis=1), uv])
        quality = candidate.min(axis=1)
        best = int(quality.argmax())
        if quality[best] < -1e-6:
            raise ValueError("query outside the physical mesh")
        elements[index] = best
    vertices = case['triangles'][elements]
    p = case['nodes'][vertices]
    matrix = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=-1)
    uv = np.linalg.solve(matrix, (np.asarray(points) - p[:, 0])[..., None])[..., 0]
    bary = np.column_stack([1 - uv.sum(axis=1), uv]).clip(0, 1)
    return vertices, bary / bary.sum(axis=1, keepdims=True)


def interpolate(case, queries, field=None):
    """P1 in space, piecewise linear in physical time; stationary t must be zero."""
    queries = np.asarray(queries)
    times = case['times']
    if np.any(queries[:, 2] < 0) or np.any(queries[:, 2] > times[-1] + 1e-10):
        raise ValueError("query time outside stored interval (stationary equations use t=0)")
    vertices, bary = locate(case, queries[:, :2])
    if field is None:
        field = np.stack([case['solution'], case.get('velocity', np.zeros_like(case['solution']))], -1)
    if len(times) == 1:
        return np.einsum('qjc,qj->qc', field[0, vertices], bary)
    right = np.clip(np.searchsorted(times, queries[:, 2], side='right'), 1, len(times) - 1)
    left = right - 1
    fraction = (queries[:, 2] - times[left]) / (times[right] - times[left])
    u0 = np.einsum('qjc,qj->qc', field[left[:, None], vertices], bary)
    u1 = np.einsum('qjc,qj->qc', field[right[:, None], vertices], bary)
    return u0 * (1 - fraction[:, None]) + u1 * fraction[:, None]


def boundary_distance(case):
    return cKDTree(case['nodes'][case['boundary_nodes']]).query(case['nodes'])[0]


def distance_to_boundary(case, points=None):
    """Euclidean distance to boundary segments, including inner boundaries."""
    points = case['nodes'] if points is None else np.asarray(points)
    segments = case['nodes'][case['boundary_edges']]
    a, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
    result = np.empty(len(points))
    for start in range(0, len(points), 512):
        offset = points[start:start + 512, None] - a
        fraction = (offset * delta).sum(-1) / (delta * delta).sum(-1)
        closest = offset - np.clip(fraction, 0, 1)[..., None] * delta
        result[start:start + 512] = np.sqrt((closest * closest).sum(-1).min(-1))
    return result


def evaluation_grid(case, resolution=0):
    """Bounding-box grid masked by the actual mesh; zero selects FEM nodes."""
    if resolution == 0:
        return case['nodes'], node_weights(case), {}
    if resolution < 3:
        raise ValueError('evaluation grids must be at least 3')
    import matplotlib.tri as mtri
    low, high = case['nodes'].min(0), case['nodes'].max(0)
    x, y = np.linspace(low[0], high[0], resolution), np.linspace(low[1], high[1], resolution)
    xx, yy = np.meshgrid(x, y)
    tri = mtri.Triangulation(*case['nodes'].T, case['triangles'])
    valid = tri.get_trifinder()(xx, yy) >= 0
    edge_weights = np.ones(resolution); edge_weights[[0, -1]] = .5
    weights = np.outer(edge_weights, edge_weights) * np.prod((high - low) / (resolution - 1))
    points = np.column_stack([xx[valid], yy[valid]])
    return points, weights[valid], dict(grid_x=x, grid_y=y, grid_mask=valid)


def refine_mesh(case, levels=1):
    """Red refinement preserves exactly the original polygon, including its holes.

    Physical input fields remain the same continuous P1 functions; this does not
    invent an analytic continuation of the stored random fields.
    """
    refined = dict(case)
    for _ in range(levels):
        triangles = refined['triangles']; n = len(refined['nodes'])
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        ab, bc, ca = (inverse.reshape(3, -1) + n)
        a, b, c = triangles.T
        new_triangles = np.concatenate([np.stack([a, ab, ca], -1), np.stack([ab, b, bc], -1),
                                        np.stack([ca, bc, c], -1), np.stack([ab, bc, ca], -1)])
        names = ['nodes', 'source', 'initial_u', 'initial_v', 'coefficient_fields', 'basis']
        names += [key for key in refined if key.endswith('_basis')]
        for name in names:
            if name in refined:
                values = np.asarray(refined[name], dtype=np.float64)
                refined[name] = np.concatenate([values, values[unique].mean(axis=1)])
        refined['triangles'] = new_triangles
        if 'element_coefficient' in refined:
            refined['element_coefficient'] = np.tile(refined['element_coefficient'], 4)
        all_edges = np.sort(np.concatenate([new_triangles[:, [0, 1]], new_triangles[:, [1, 2]], new_triangles[:, [2, 0]]]), axis=1)
        all_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
        refined['boundary_edges'] = all_edges[counts == 1]
        refined['boundary_nodes'] = np.unique(refined['boundary_edges'])
    return refined
