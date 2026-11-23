"""Small physical and observation-based classical reconstruction methods."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from .data import known_inputs, case_paths, load_case
from .geometry import interpolate, node_weights
from .physics import solve, WAVES, LINEAR_ELLIPTIC, STATIONARY
from .geometry import distance_to_boundary

# Each worker retains at most four sparse operators/factorizations, across cases.
_OPERATORS = {}
METHODS = ('prior', 'rbf', 'ridge', 'bayes', 'grid', 'pod', 'gp', 'map')
MAP_STATS = ('map_success', 'map_status', 'map_iterations', 'map_initial_objective', 'map_objective',
             'map_objective_evaluations', 'map_forward_solves', 'map_failed_solves', 'map_message')


def pod_group(case):
    """Identify the physical polygon, ignoring collinear boundary subdivisions."""
    neighbors = {}
    for a, b in case['boundary_edges']:
        neighbors.setdefault(int(a), []).append(int(b)); neighbors.setdefault(int(b), []).append(int(a))
    remaining = set(neighbors); segments = []
    while remaining:
        start = min(remaining); loop = [start]; previous, current = start, neighbors[start][0]
        while current != start:
            loop.append(current)
            choices = neighbors[current]
            if len(choices) != 2:
                raise NotImplementedError('POD requires closed polygon boundary loops')
            previous, current = current, choices[0] if choices[0] != previous else choices[1]
        remaining.difference_update(loop)
        points = case['nodes'][loop]
        left, right = points - np.roll(points, 1, axis=0), np.roll(points, -1, axis=0) - points
        corner = abs(np.cross(left, right)) > 1e-10 * np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        points = np.round(points[corner], 10)
        segments.extend(tuple(sorted((tuple(map(float, a)), tuple(map(float, b)))))
                        for a, b in zip(points, np.roll(points, -1, axis=0)))
    boundary = hashlib.sha256(repr(sorted(segments)).encode()).hexdigest()
    return f'{case["meta"]["pde"]}/{case["meta"]["geometry"]}/{boundary}'


def fit_pod(data, destination, rank=32):
    """Fit once on training fields only; persist compatible groups in one NPZ."""
    destination = Path(destination).resolve()
    if destination.suffix != '.npz' or rank < 1:
        raise ValueError('POD needs positive rank and a --pod-model path ending in .npz')
    paths = case_paths(data, 'train')
    signature = dict(data=str(Path(data).resolve()), rank=rank,
                     training=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
        if metadata['signature'] != json.loads(json.dumps(signature)):
            raise ValueError('POD training split/rank changed; use a new --pod-model path')
        return str(destination)
    groups = {}
    for path in paths:
        case = load_case(path)
        if len(case['times']) == 1 and case['meta']['pde'] in STATIONARY:
            groups.setdefault(pod_group(case), []).append(path)
    arrays, descriptions = {}, {}
    for key, members in groups.items():
        if len(members) < 2:
            continue  # no centered training variation is learnable from one field
        case = load_case(members[0]); coords = np.column_stack([case['nodes'], np.zeros(len(case['nodes']))])
        snapshots = np.stack([interpolate(load_case(p), coords)[:, 0] for p in members])
        mean = snapshots.mean(0); root_weight = np.sqrt(node_weights(case))
        _, singular, vt = np.linalg.svd((snapshots - mean) * root_weight, full_matrices=False)
        count = min(rank, int(np.sum(singular > (singular[0] * 1e-10 if len(singular) else 0.))))
        prefix = f'g{len(descriptions)}'
        for name in ('nodes', 'triangles', 'times'):
            arrays[f'{prefix}_{name}'] = case[name]
        arrays[f'{prefix}_mean'] = mean
        arrays[f'{prefix}_modes'] = vt[:count].T / root_weight[:, None]
        descriptions[key] = dict(prefix=prefix, rank=count, training_cases=[p.name for p in members])
    arrays['metadata'] = np.array(json.dumps(dict(signature=signature, groups=descriptions,
        inner_product='lumped P1 spatial mass; physical coordinates, no domain rescaling')))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f'POD fitted {len(descriptions)} compatible stationary groups: {destination}')
    return str(destination)


@lru_cache(maxsize=2)
def load_pod(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays['metadata'] = json.loads(str(arrays['metadata']))
    return arrays


def exact_source_coordinates(basis, values, known, mean):
    """Isotropic Gaussian prior conditioned on exact nodal linear constraints."""
    if not known.any() or not basis.shape[1]:
        return mean.copy(), np.eye(basis.shape[1])
    matrix = basis[known]
    _, singular, vt = np.linalg.svd(matrix, full_matrices=len(matrix) < basis.shape[1])
    rank = np.sum(singular > singular.max() * 1e-7)
    mean0 = mean + np.linalg.lstsq(matrix, values[known] - matrix @ mean, rcond=1e-7)[0]
    return mean0, vt[rank:].T


def squared_exponential(x, y, lengthscale, amplitude):
    distance = ((x[:, None] - y[None]) / lengthscale) ** 2
    return amplitude ** 2 * np.exp(-.5 * distance.sum(-1))


def map_estimate(case, obs, values, masks, scalar, scalar_masks, maxiter=100):
    """Small parameter-space MAP; Gaussian coordinates and bounded uniform scalars."""
    meta = case['meta']; pde, tier = meta['pde'], meta['tier']
    if not scalar_masks[1]:
        raise NotImplementedError('MAP currently requires known beta; it does not infer hidden nonlinear coefficients')
    values = values.astype(float)  # preserve finite-difference perturbations, not float32 assignment
    initial, bounds, gaussian = [], [], []
    source_map = coefficient_map = None
    base_fixed, base_index = float(scalar[0]), None
    source_coordinates = coefficient_coordinates = None
    if masks.all() and scalar_masks.all():
        pass  # Fully specified Forward: one solve, no unknown parameter to optimize.
    elif tier == 'T2':
        if not masks[:, :3].all() or meta.get('coefficient_kind', 'constant') != 'constant':
            raise NotImplementedError('MAP T2 requires fully known source/initial fields and a constant unknown scalar')
        limits = np.asarray(meta['distributions']['coefficient'])
        low, high = limits if limits.ndim else (float(limits), float(limits))
        if low <= 0:
            raise NotImplementedError('MAP requires positive conductivity/diffusivity/speed bounds')
        base_fixed = float((low + high) / 2)
        if low != high:
            base_index = 0; initial.append(base_fixed); bounds.append((low, high)); gaussian.append(False)
    elif tier == 'T3' and pde == 'diffusion':
        if meta.get('coefficient_kind') != 'smooth' or 'coefficient_basis' not in case:
            raise NotImplementedError('MAP T3 supports the configured smooth, low-dimensional coefficient basis only')
        if not masks[:, :3].all():
            raise NotImplementedError('MAP T3 requires known source and initial fields')
        prior = meta['field_priors']['coefficient']
        if prior['distribution'] != 'gaussian':
            raise NotImplementedError('MAP T3 currently requires Gaussian coefficient amplitudes')
        limits = np.asarray(meta['distributions']['coefficient'])
        low, high = limits if limits.ndim else (float(limits), float(limits))
        base_fixed = float((low + high) / 2)
        if low <= 0:
            raise NotImplementedError('MAP T3 requires positive coefficient-scale bounds')
        known = masks[:, 3].astype(bool); basis = case['coefficient_basis'].astype(float)
        mean = np.full(basis.shape[1], prior['mean'])
        contrast = meta['coefficient_contrast']
        if contrast <= 0:
            raise NotImplementedError('MAP T3 requires nonzero coefficient contrast')
        if known.any():
            if low != high:
                raise NotImplementedError('MAP Hybrid with both unknown coefficient scale and exact field samples is unsupported')
            ratio = (values[:, 3] / base_fixed - 1) / contrast
            if np.any(abs(ratio[known]) >= 1):
                raise NotImplementedError('MAP cannot invert saturated exact tanh coefficient samples')
            raw = np.zeros(len(values)); raw[known] = np.arctanh(ratio[known])
            mean, null = exact_source_coordinates(basis, raw, known, mean)
        else:
            null = np.eye(basis.shape[1])
        coefficient_coordinates = (basis, mean, prior['std'] * null, contrast)
        initial.extend(np.zeros(null.shape[1])); bounds.extend([(None, None)] * null.shape[1]); gaussian.extend([True] * null.shape[1])
        if low != high:
            base_index = len(initial); initial.append(base_fixed); bounds.append((low, high)); gaussian.append(False)
    elif pde in LINEAR_ELLIPTIC + ('semilinear', 'p_laplacian') and tier in ('T1', 'G1', 'T4'):
        if not masks[:, 1:].all() or not scalar_masks.all():
            raise NotImplementedError('MAP source inversion requires the full operator and initial fields to be known')
        basis = case.get('source_basis', case['basis']).astype(float)
        prior = meta.get('field_priors', {}).get('source', dict(mean=meta.get('amplitude_prior_mean', 0.),
            std=meta.get('amplitude_prior_std', 0.), distribution='gaussian' if 'Gaussian' in meta['distributions']['amplitudes'] else 'unsupported'))
        mean = np.full(basis.shape[1], prior['mean']); known = masks[:, 0].astype(bool)
        mean0, null = exact_source_coordinates(basis, values[:, 0], known, mean)
        if prior['distribution'] == 'gaussian' and prior['std'] > 0:
            source_coordinates = (basis, mean0, prior['std'] * null)
            initial.extend(np.zeros(null.shape[1])); bounds.extend([(None, None)] * null.shape[1]); gaussian.extend([True] * null.shape[1])
        elif prior['distribution'] == 'uniform':
            if known.any() and 0 < null.shape[1] < basis.shape[1]:
                raise NotImplementedError('MAP partial uniform-amplitude constraints are unsupported')
            if not null.shape[1]:
                source_coordinates = (basis, mean0, np.empty((len(mean), 0)))
            else:
                width = np.sqrt(3) * prior['std']
                source_coordinates = (basis, np.zeros_like(mean), np.eye(len(mean)))
                initial.extend(mean); bounds.extend([(m - width, m + width) for m in mean]); gaussian.extend([False] * len(mean))
        else:
            raise NotImplementedError('MAP source amplitudes require a declared Gaussian or uniform prior')
    else:
        raise NotImplementedError('MAP supports T2 scalars, configured T3 diffusion coefficients, and stationary source inversion')
    initial = np.asarray(initial, float); gaussian = np.asarray(gaussian, bool)
    if len(initial) > 16:
        raise NotImplementedError(f'MAP supports at most 16 free parameters; this conditioning set has {len(initial)}')
    counters = dict(evaluations=0, solves=0, failed=0); last = {}
    def candidate(theta):
        nonlocal source_map, coefficient_map
        inputs = values.copy(); coefficient = base_fixed if base_index is None else float(theta[base_index])
        if source_coordinates is not None:
            basis, mean, directions = source_coordinates
            source_map = mean + directions @ theta[:directions.shape[1]]
            inputs[:, 0] = basis @ source_map
        if coefficient_coordinates is not None:
            basis, mean, directions, contrast = coefficient_coordinates
            coefficient_map = mean + directions @ theta[:directions.shape[1]]
            inputs[:, 3] = coefficient * (1 + contrast * np.tanh(basis @ coefficient_map))
        counters['solves'] += 1
        field = forward_field(case, inputs, coefficient)
        if not np.isfinite(field).all():
            raise RuntimeError('non-finite MAP forward solution')
        return field, coefficient
    def objective(theta):
        counters['evaluations'] += 1
        try:
            field, coefficient = candidate(theta)
        except (RuntimeError, np.linalg.LinAlgError):
            counters['failed'] += 1
            return np.inf
        last.update(field=field, coefficient=coefficient)
        predicted = interpolate(case, obs['queries'], field)[np.arange(len(obs['values'])), obs['quantity']]
        return .5 * np.sum(((predicted - obs['values']) / obs['noise']) ** 2) + .5 * np.sum(theta[gaussian] ** 2)
    initial_objective = objective(initial)
    if not np.isfinite(initial_objective):
        raise NotImplementedError('MAP initial forward solve failed; no prediction produced')
    result = None
    if len(initial) and len(obs['values']):
        result = minimize(objective, initial, method='L-BFGS-B', jac='3-point', bounds=bounds,
                          options=dict(maxiter=maxiter, finite_diff_rel_step=1e-5, ftol=1e-10, gtol=1e-6))
    theta = initial if result is None else result.x
    value = initial_objective if result is None else objective(theta)
    if not np.isfinite(value):
        raise NotImplementedError('MAP final forward solve/objective failed; no prediction produced')
    field, coefficient = last['field'], last['coefficient']
    prepared = dict(field=field, map_parameters=theta, map_success=int(result is None or result.success),
                    map_status=0 if result is None else int(result.status), map_iterations=0 if result is None else int(result.nit),
                    map_initial_objective=float(initial_objective), map_objective=float(value),
                    map_objective_evaluations=counters['evaluations'], map_forward_solves=counters['solves'],
                    map_failed_solves=counters['failed'], map_message='no optimization needed' if result is None else str(result.message),
                    posterior='none: MAP point estimate, no posterior uncertainty')
    if tier in ('T2', 'T3'):
        prepared['coefficient'] = coefficient
    if source_map is not None:
        prepared['source_amplitude_map'] = source_map.copy()
    if coefficient_map is not None:
        prepared['coefficient_amplitude_map'] = coefficient_map.copy()
    return prepared


def gaussian_posterior(matrix, values, noise, prior_std):
    if np.any(np.asarray(noise) <= 0):
        raise ValueError('Gaussian baseline requires positive observation standard deviation')
    weighted = matrix / np.asarray(noise)[:, None]
    precision = weighted.T @ weighted + np.eye(matrix.shape[1]) / prior_std ** 2
    factor = cho_factor(precision)
    covariance = cho_solve(factor, np.eye(matrix.shape[1]))
    mean = cho_solve(factor, weighted.T @ (values / noise))
    return mean, covariance


def forward_field(case, values, coefficient):
    meta = case['meta']; pde = meta['pde']
    if 'element_coefficient' in case:
        operator = case['element_coefficient'][:, None]
    elif pde == 'semilinear' or meta.get('coefficient_kind') == 'smooth':
        operator = values[:, 3]
    else:
        operator = coefficient
    if pde in WAVES:
        operator = operator ** 2
    result, _ = solve(case, pde, values[:, 0], values[:, 1], values[:, 2], operator,
                      meta['beta'], len(case['times']), meta['solver'].get('time_step', .002), cache=_OPERATORS)
    return np.stack([result['solution'], result.get('velocity', np.zeros_like(result['solution']))], -1)


def conditional_physical_mean(case, values, masks):
    mean = values.copy()
    for channel in range(3):
        name = ('source', 'initial_u', 'initial_v')[channel]
        basis = case.get(f'{name}_basis', case['basis']).astype(float)
        prior = case['meta'].get('field_priors', {}).get(name, {})
        if case['meta']['tier'] == 'T3' and not prior:
            basis = basis / np.sqrt(np.arange(1, basis.shape[1] + 1))
        missing = masks[:, channel] == 0
        if not missing.any():
            continue
        known = ~missing
        amplitudes = np.full(basis.shape[1], prior.get('mean', 0.), dtype=float)
        if not basis.shape[1]:
            mean[missing, channel] = 0.
            continue
        if known.any():
            if prior.get('distribution', 'gaussian') != 'gaussian' and np.linalg.matrix_rank(basis[known]) < basis.shape[1]:
                raise NotImplementedError('prior with partial, underdetermined non-Gaussian amplitudes is unsupported')
            amplitudes += np.linalg.lstsq(basis[known], values[known, channel] - basis[known] @ amplitudes, rcond=1e-6)[0]
        mean[missing, channel] = (basis @ amplitudes)[missing]
    return mean


def prepare_baseline(name, case, obs, mode, seed, cache=None, settings=None):
    """Do inference once per conditioning set, then query it on any grid in chunks."""
    cache = {} if cache is None else cache  # per case, released after its grids
    settings = {} if settings is None else settings
    if name in ('pod', 'gp') and (len(case['times']) != 1 or case['meta']['pde'] not in STATIONARY):
        raise NotImplementedError(f'{name} currently supports stationary scalar fields only')
    if name == 'pod':
        if not settings.get('pod_model'):
            raise NotImplementedError('POD requires a representation fitted on the training split')
        saved = load_pod(settings['pod_model']); group = saved['metadata']['groups'].get(pod_group(case))
        if group is None:
            raise NotImplementedError('POD has no compatible physical-domain/PDE group with at least two training cases; no domain rescaling is applied')
        prefix = group['prefix']
        mesh = {key: saved[f'{prefix}_{key}'] for key in ('nodes', 'triangles', 'times')}
        fields = np.column_stack([saved[f'{prefix}_mean'], saved[f'{prefix}_modes']])[None]
        sensors = interpolate(mesh, obs['queries'], fields)
        ridge = settings.get('pod_ridge', 1.)
        rank = group['rank']
        coefficients = (gaussian_posterior(sensors[:, 1:], obs['values'] - sensors[:, 0], obs['noise'], 1 / np.sqrt(ridge))[0]
                        if rank else np.empty(0))
        return dict(_pod_mesh=mesh, _pod_fields=fields, pod_coefficients=coefficients, pod_rank=rank,
                    posterior='none: training-only weighted POD plus noise-weighted ridge, observations only')
    if name == 'gp':
        lengthscale, amplitude = settings.get('gp_lengthscale', .2), settings.get('gp_amplitude', 1.)
        coordinates = obs['queries'][:, :2].copy(); factor, alpha = None, np.empty(0)
        if len(coordinates):
            matrix = squared_exponential(coordinates, coordinates, lengthscale, amplitude)
            matrix.flat[::len(coordinates) + 1] += obs['noise'] ** 2
            factor = cho_factor(matrix)
            alpha = cho_solve(factor, obs['values'])
        return dict(_gp_coordinates=coordinates, _gp_factor=factor, _gp_alpha=alpha,
                    _gp_lengthscale=lengthscale, _gp_amplitude=amplitude,
                    posterior='observation-only zero-mean spatial squared-exponential GP; kernel intervals are not guaranteed calibrated')
    values, masks, scalar, scalar_masks = known_inputs(case, mode, np.random.default_rng(seed))
    if name == 'map':
        return map_estimate(case, obs, values, masks, scalar, scalar_masks, settings.get('map_maxiter', 100))
    if name == 'prior':
        if case['meta']['tier'] == 'T3' and not masks[:, 3].all():
            raise NotImplementedError('prior over an unknown spatial coefficient field is unsupported')
        if case['meta']['pde'] == 'eikonal':
            known_speed = values[masks[:, 3].astype(bool), 3]
            bounds = np.asarray(case['meta']['distributions']['coefficient'])
            slowness = (1 / known_speed[0] if len(known_speed) else 1 / float(bounds) if bounds.ndim == 0 else
                        np.log(bounds[1] / bounds[0]) / (bounds[1] - bounds[0]) if bounds[1] != bounds[0] else 1 / bounds[0])
            u = distance_to_boundary(case) * slowness
            return dict(field=np.stack([u, np.zeros_like(u)], -1)[None])
        key = ('prior', mode)
        if key in cache:
            return cache[key]
        mean = conditional_physical_mean(case, values, masks)
        if case['meta']['tier'] == 'T2' and not scalar_masks[0]:
            bounds = case['meta']['distributions']['coefficient']
            low, high = bounds if np.ndim(bounds) else (bounds, bounds)
            coefficients = low + (np.arange(24) + .5) / 24 * (high - low)
            field = sum(forward_field(case, mean, a) for a in coefficients) / len(coefficients)
            prepared = dict(field=field, coefficient=float(np.mean(coefficients)))
        else:
            if case['meta']['tier'] == 'T4' and not masks[:, 0].all():
                determined = masks[:, 0].any() and np.linalg.matrix_rank(case['basis'][masks[:, 0].astype(bool)], tol=1e-5) == case['basis'].shape[1]
                if not determined and (masks[:, 0].any() or case['meta'].get('amplitude_prior_mean', 0.) != 0.):
                    raise NotImplementedError('nonlinear prior mean with partial undetermined or nonzero-mean forcing is unsupported')
            prepared = dict(field=forward_field(case, mean, scalar[0]))
        cache[key] = prepared
        return prepared
    if name in ('ridge', 'bayes'):
        meta = case['meta']
        if name == 'ridge' and meta['tier'] != 'T1':
            raise NotImplementedError('ridge is supported only for T1')
        if name == 'bayes':
            if meta['tier'] not in ('T1', 'G1') or meta['pde'] not in LINEAR_ELLIPTIC:
                raise NotImplementedError('bayes supports T1/G1 linear elliptic equations with a known operator and Gaussian source prior')
            law = meta.get('distributions', {}).get('amplitudes', '')
            prior_std = meta.get('amplitude_prior_std', 0.)
            if 'Gaussian' not in law or not np.isfinite(prior_std) or prior_std <= 0:
                raise NotImplementedError('bayes requires a generating Gaussian prior; non-Gaussian amplitudes are unsupported')
            if not scalar_masks[0] or not masks[:, 3].all():
                raise NotImplementedError('bayes requires a fully known elliptic operator')
        basis = case['basis'].astype(float)
        if basis.shape[1] == 0:
            prepared = dict(field=forward_field(case, values, scalar[0]), amplitude_mean=np.empty(0))
            if name == 'bayes':
                prepared.update(amplitude_covariance=np.zeros((0, 0)), posterior='deterministic zero-source elliptic model')
            return prepared
        if 'responses' not in cache:
            responses = []
            for column in basis.T:
                inputs = values.copy(); inputs[:, 0] = column
                responses.append(forward_field(case, inputs, scalar[0]))
            cache['responses'] = np.stack(responses, -1)
        responses = cache['responses']
        matrix = np.column_stack([interpolate(case, obs['queries'], responses[..., i])[:, 0]
                                  for i in range(basis.shape[1])])
        known = masks[:, 0].astype(bool)
        prior_mean = np.full(basis.shape[1], meta.get('amplitude_prior_mean', 0.))
        mean0, null = exact_source_coordinates(basis, values[:, 0], known, prior_mean)
        covariance = None
        if null.shape[1] == 0:
            mean = mean0
            covariance = np.zeros((len(mean0), len(mean0))) if name == 'bayes' else None
        elif name == 'bayes':
            mu, cov = gaussian_posterior(matrix @ null, obs['values'] - matrix @ mean0,
                                          obs['noise'], case['meta']['amplitude_prior_std'])
            mean, covariance = mean0 + null @ mu, null @ cov @ null.T
        else:
            h = matrix @ null
            ridge = np.linalg.solve(h.T @ h + 1e-4 * np.eye(h.shape[1]), h.T @ (obs['values'] - matrix @ mean0))
            mean = mean0 + null @ ridge
        prepared = dict(responses=responses, amplitude_mean=mean)
        if covariance is not None:
            prepared.update(amplitude_covariance=covariance,
                            posterior=f'exact for the stored FEM/interpolated {meta["tier"]} {meta["pde"]} Gaussian model, up to float32 rounding')
        return prepared
    if name == 'rbf':
        dims = 3 if len(case['times']) > 1 else 2
        fits = []
        for channel in range(2 if case['meta']['pde'] in WAVES else 1):
            keep = obs['quantity'] == channel
            if keep.sum() >= dims + 1:
                fits.append(RBFInterpolator(obs['queries'][keep, :dims], obs['values'][keep], kernel='gaussian',
                                            epsilon=3., degree=-1, smoothing=max(float(obs['noise'][keep][0]) ** 2, 1e-7)))
            else:
                fits.append(float(obs['values'][keep].mean()) if keep.any() else 0.)
        return dict(fits=fits, dims=dims)
    if name == 'grid':
        if case['meta']['tier'] != 'T2':
            raise NotImplementedError('coefficient grid is supported only for T2')
        if not masks[:, :3].all():
            raise NotImplementedError('T2 grid reference requires fully known forcing and initial states')
        bounds = case['meta']['distributions']['coefficient']
        low, high = bounds if np.ndim(bounds) else (bounds, bounds)
        grid = np.linspace(low, high, 41) if not scalar_masks[0] else np.array([scalar[0]])
        key = ('grid_fields', mode)
        if key not in cache:
            cache[key] = [forward_field(case, values, a) for a in grid]
        fields = cache[key]; likelihood = []
        for field in fields:
            observed = interpolate(case, obs['queries'], field)[np.arange(len(obs['values'])), obs['quantity']]
            likelihood.append(-.5 * np.sum(((observed - obs['values']) / obs['noise']) ** 2))
        weights = np.exp(np.array(likelihood) - max(likelihood)); weights /= weights.sum()
        field = sum(weight * field for weight, field in zip(weights, fields))
        return dict(field=field, coefficient=float(weights @ grid), coefficient_grid=grid,
                    coefficient_probability=weights)
    raise ValueError(f'unknown baseline {name}')


def baseline_values(prepared, case, queries):
    extra = {k: v for k, v in prepared.items() if k not in ('field', 'responses', 'fits', 'dims') and not k.startswith('_')}
    if 'field' in prepared:
        if 'amplitude_covariance' in prepared:
            extra['solution_std'] = np.zeros(len(queries))
        return interpolate(case, queries, prepared['field']), extra
    prediction = np.zeros((len(queries), 2))
    if '_pod_fields' in prepared:
        fields = interpolate(prepared['_pod_mesh'], queries, prepared['_pod_fields'])
        prediction[:, 0] = fields[:, 0] + fields[:, 1:] @ prepared['pod_coefficients']
    elif '_gp_coordinates' in prepared:
        cross = squared_exponential(queries[:, :2], prepared['_gp_coordinates'], prepared['_gp_lengthscale'], prepared['_gp_amplitude'])
        prediction[:, 0] = cross @ prepared['_gp_alpha']
        variance = np.full(len(queries), prepared['_gp_amplitude'] ** 2)
        if prepared['_gp_factor'] is not None:
            variance -= np.einsum('qi,iq->q', cross, cho_solve(prepared['_gp_factor'], cross.T))
        extra['solution_std'] = np.sqrt(np.maximum(variance, 0.))
    elif 'responses' in prepared:
        responses = prepared['responses']
        matrix = np.column_stack([interpolate(case, queries, responses[..., i])[:, 0]
                                  for i in range(responses.shape[-1])])
        prediction[:, 0] = matrix @ prepared['amplitude_mean']
        if 'amplitude_covariance' in prepared:
            extra['solution_std'] = np.sqrt(np.maximum(np.einsum('qi,ij,qj->q', matrix, prepared['amplitude_covariance'], matrix), 0))
    else:
        for channel, fit in enumerate(prepared['fits']):
            prediction[:, channel] = fit(queries[:, :prepared['dims']]) if callable(fit) else fit
    return prediction, extra


def predict_baseline(name, case, queries, obs, mode, seed, settings=None):
    return baseline_values(prepare_baseline(name, case, obs, mode, seed, settings=settings), case, queries)
