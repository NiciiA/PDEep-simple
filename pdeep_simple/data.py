"""Independent physical cases, explicit conditioning masks, and reusable queries."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from .geometry import mesh, node_weights, sample_points, locate, interpolate, boundary_distance, GEOMETRIES
from .physics import solve, STATIONARY, WAVES, PARABOLIC, LINEAR_ELLIPTIC

DEFAULT_PDES = {'T1': ('poisson',), 'G1': ('poisson', 'heat', 'wave'),
                'T2': ('poisson', 'heat'), 'T3': ('heat',), 'T4': ('semilinear',)}
SUPPORTED = dict(DEFAULT_PDES, G1=LINEAR_ELLIPTIC + PARABOLIC + WAVES,
                 T2=LINEAR_ELLIPTIC + PARABOLIC + WAVES,
                 T3=('heat', 'diffusion'), T4=('semilinear', 'p_laplacian', 'eikonal'))
# Keep the first four identities unchanged for existing checkpoints.
PDES = ('poisson', 'heat', 'wave', 'semilinear', 'diffusion', 'elliptic_adr',
        'reaction_diffusion', 'damped_wave', 'p_laplacian', 'eikonal')
MODES = ('forward', 'inverse', 'hybrid')


def basis_fields(case, count, rng, fixed=False):
    nodes = case['nodes']
    lo, span = nodes.min(axis=0), np.ptp(nodes, axis=0)
    xy = (nodes - lo) / span
    centers = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75]]) if fixed else rng.uniform(.1, .9, (count, 2))
    width = np.full(len(centers), .16) if fixed else rng.uniform(.07, .3, count)
    fields = np.exp(-((xy[:, None] - centers) ** 2).sum(-1) / (2 * width ** 2))
    # Boundary taper keeps initial states in H^1_0; each basis has unit domain RMS.
    distance = boundary_distance(case)
    fields *= np.minimum(1., distance / (.2 * span.min()))[:, None]
    weights = node_weights(case)
    fields /= np.maximum(np.sqrt((weights[:, None] * fields ** 2).sum(0) / weights.sum()), 1e-8)
    return fields


def make_case(config, entry):
    if config.get('settings') or entry['pde'] not in ('poisson', 'heat', 'wave', 'semilinear') or entry['geometry'] not in ('square', 'disk', 'lshape'):
        return configured_case(config, entry)
    rng = np.random.default_rng(entry['seed'])
    tier, pde, family = config['tier'], entry['pde'], entry['geometry']
    width, height, radius = rng.uniform(.8, 1.2), rng.uniform(.8, 1.2), rng.uniform(.4, .65)
    if tier == 'T1':
        width = height = 1.
    case = mesh(family, config['resolution'], width, height, radius)
    count = 32 if tier == 'T3' else 4
    basis = basis_fields(case, count, rng, tier == 'T1')
    sigma = 12. if pde == 'semilinear' else (3. if pde == 'poisson' else .5)
    amplitudes = rng.normal(0, sigma, basis.shape[1])
    if tier == 'T3':
        # Broad to fine structures, with damped high-frequency amplitudes.
        amplitudes /= np.sqrt(np.arange(1, count + 1))
    field = basis @ amplitudes
    n = len(case['nodes'])
    source, initial_u, initial_v = np.zeros(n), np.zeros(n), np.zeros(n)
    coefficient = 1.
    if pde in ('poisson', 'semilinear'):
        source = field
        if tier == 'T2':
            coefficient = rng.uniform(.5, 2.)
    else:
        initial_u = field
        coefficient = rng.uniform(.03, .2) if pde == 'heat' else rng.uniform(.5, 1.2)
        if pde == 'wave':
            velocity_amplitudes = rng.normal(0, .5, count)
            initial_v = basis @ velocity_amplitudes
    beta = float(rng.uniform(.5, 3.)) if tier == 'T4' else 0.
    coefficient_field = np.full(n, coefficient)
    if tier == 'T4':
        smooth = basis_fields(case, 8, rng) @ rng.normal(0, .4, 8)
        coefficient_field = .5 + 1.5 / (1 + np.exp(-smooth))
    operator_coefficient = coefficient_field if tier == 'T4' else coefficient ** (2 if pde == 'wave' else 1)
    result, solver_info = solve(case, pde, source, initial_u, initial_v,
                                operator_coefficient, beta, config['frames'], config['dt'])
    case.update(result, source=source, initial_u=initial_u, initial_v=initial_v,
                coefficient_fields=coefficient_field, basis=basis, amplitudes=amplitudes)
    if pde == 'wave':
        case['initial_v_amplitudes'] = velocity_amplitudes
    case['meta'] = dict(entry, tier=tier, coefficient=float(coefficient), beta=beta,
                        width=float(width), height=float(height), radius=float(radius),
                        amplitude_prior_std=sigma, solver=solver_info,
                        targets=['solution'] + (['velocity'] if pde == 'wave' else []) +
                        (['coefficient'] if tier == 'T2' else []),
                        distributions={'coefficient': [.03, .2] if pde == 'heat' else
                                       ([.5, 1.2] if pde == 'wave' else [.5, 2.]),
                                       'amplitudes': 'zero-mean Gaussian; T3 frequency damped'})
    if not all(np.isfinite(v).all() for k, v in case.items() if k != 'meta'):
        raise RuntimeError('non-finite case; not saved')
    return case


def sample_setting(value, rng, integer=False):
    """A scalar is fixed; [low, high] is uniform (inclusive for integers)."""
    bounds = np.asarray(value, dtype=float)
    if bounds.ndim == 0:
        result = float(bounds)
    elif bounds.shape == (2,) and bounds[0] <= bounds[1]:
        if integer:
            if np.any(bounds != bounds.astype(int)):
                raise ValueError('basis counts and polygon point counts must be integers')
            return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))
        result = float(rng.uniform(*bounds))
    else:
        raise ValueError(f'expected a fixed value or ordered [low, high], got {value}')
    if not np.isfinite(result) or integer and result != int(result):
        raise ValueError(f'invalid numeric setting {value}')
    return int(result) if integer else result


def read_dataset_config(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # Python 3.9/3.10; installed by pyproject.toml
    with Path(path).open('rb') as stream:
        settings = tomllib.load(stream)
    allowed = {'basis': {'count', 'amplitude_mean', 'amplitude_std', 'distribution', 'scale_by_count', 'width'},
               'conditioning': {'hybrid_known_source_fraction', 'hybrid_known_initial_fraction', 'hybrid_known_coefficient_fraction'},
               'numerics': {'resolution', 'frames', 'dt', 'newton_iterations', 'newton_tolerance', 'p_regularization'}}
    operator_keys = {'coefficient_kind', 'coefficient_basis', 'coefficient_contrast'}
    pde_keys = {'poisson': {'forcing_basis', 'conductivity'},
                'diffusion': {'forcing_basis', 'conductivity'} | operator_keys,
                'elliptic_adr': {'forcing_basis', 'diffusivity', 'advection', 'reaction'} | operator_keys,
                'heat': {'initial_basis', 'forcing_basis', 'diffusivity', 'forcing_frequency'} | operator_keys,
                'reaction_diffusion': {'initial_basis', 'forcing_basis', 'diffusivity', 'reaction', 'forcing_frequency'} | operator_keys,
                'wave': {'displacement_basis', 'velocity_basis', 'forcing_basis', 'wave_speed', 'forcing_frequency'} | operator_keys,
                'damped_wave': {'displacement_basis', 'velocity_basis', 'forcing_basis', 'wave_speed', 'damping', 'forcing_frequency'} | operator_keys,
                'semilinear': {'forcing_basis', 'beta', 'conductivity'} | operator_keys,
                'p_laplacian': {'forcing_basis', 'p'}, 'eikonal': {'speed'}}
    geometry_keys = dict(rectangle={'aspect_ratio'}, ellipse={'axis_ratio'}, triangle={'apex'}, trapezoid={'top_ratio'},
                         star={'points', 'inner_ratio'}, polygon={'sides'}, annulus={'inner_ratio'},
                         cross={'arm_width'}, dumbbell={'neck_width'}, hole={'hole_radius'}, two_holes={'hole_radius'})
    for section, values in settings.items():
        if section in ('pdes', 'geometries'):
            names = PDES if section == 'pdes' else GEOMETRIES
            for name, options in values.items():
                keys = pde_keys.get(name, set()) if section == 'pdes' else geometry_keys.get(name, set())
                if name not in names or set(options) - keys:
                    raise ValueError(f'unknown {section} settings for {name}')
        elif section not in allowed or set(values) - allowed[section]:
            raise ValueError(f'unknown dataset config section/settings: {section}')
    return settings


def configured_case(config, entry):
    rng = np.random.default_rng(entry['seed'])
    settings = config.get('settings', {}); tier, pde = config['tier'], entry['pde']
    basis_settings = settings.get('basis', {}); options = settings.get('pdes', {}).get(pde, {})
    geometry = entry['geometry']
    width, height, radius = rng.uniform(.8, 1.2), rng.uniform(.8, 1.2), rng.uniform(.4, .65)
    if tier == 'T1':
        width = height = 1.
    parameters = {key: sample_setting(value, rng, key in ('points', 'sides'))
                  for key, value in settings.get('geometries', {}).get(geometry, {}).items()}
    for key in ('axis_ratio', 'apex', 'top_ratio', 'inner_ratio', 'arm_width', 'neck_width'):
        if key in parameters and not .1 <= parameters[key] <= .95:
            raise ValueError(f'{geometry}.{key} must be between .1 and .95')
    if parameters.get('hole_radius', .12) <= 0 or parameters.get('hole_radius', .12) >= .2:
        raise ValueError('hole_radius must lie in (0,.2), relative to the shorter side')
    if not 3 <= parameters.get('points', 5) <= 12 or not 3 <= parameters.get('sides', 6) <= 12:
        raise ValueError('polygon/star point counts must be in 3..12')
    if geometry == 'rectangle':
        ratio = parameters.get('aspect_ratio', 1.4)
        if not .4 <= ratio <= 2.5:
            raise ValueError('rectangle aspect_ratio must be in .4..2.5')
        width, height = np.sqrt(width * height * ratio), np.sqrt(width * height / ratio)
    case = mesh(geometry, config['resolution'], width, height, radius, parameters)
    n = len(case['nodes']); zeros = np.zeros(n); priors = {}
    def field(name, count):
        count = sample_setting(count, rng, True)
        if not 0 <= count <= 512:
            raise ValueError('basis counts must be in 0..512')
        width_range = basis_settings.get('width', [.07, .3])
        widths = np.array([sample_setting(width_range, rng) for _ in range(count)])
        if np.any(widths <= 0):
            raise ValueError('Gaussian widths must be positive fractions of domain size')
        centers = sample_points(case, count, rng)
        span = np.ptp(case['nodes'], axis=0)
        values = np.exp(-(((case['nodes'][:, None] - centers) / span) ** 2).sum(-1) / (2 * widths ** 2))
        values *= np.minimum(1., boundary_distance(case) / (.2 * span.min()))[:, None]
        weights = node_weights(case)
        values /= np.maximum(np.sqrt((weights[:, None] * values ** 2).sum(0) / weights.sum()), 1e-8)
        std = float(basis_settings.get('amplitude_std', 12. if pde == 'semilinear' else (3. if pde in STATIONARY else .5)))
        mean = float(basis_settings.get('amplitude_mean', 0.))
        if std <= 0 or not np.isfinite([mean, std]).all():
            raise ValueError('amplitude mean/std must be finite, with std > 0')
        if basis_settings.get('scale_by_count', False):
            std /= np.sqrt(max(count, 1))
        law = basis_settings.get('distribution', 'gaussian')
        if law == 'gaussian':
            amplitudes = rng.normal(mean, std, count)
        elif law == 'uniform':
            amplitudes = rng.uniform(mean - np.sqrt(3) * std, mean + np.sqrt(3) * std, count)
        else:
            raise ValueError('basis.distribution supports gaussian or uniform')
        case.update({f'{name}_basis': values, f'{name}_amplitudes': amplitudes,
                     f'{name}_centers': centers, f'{name}_widths': widths})
        priors[name] = dict(count=count, mean=mean, std=std, distribution=law)
        return values @ amplitudes
    count = basis_settings.get('count', 32 if tier == 'T3' else 4)
    source = field('source', options.get('forcing_basis', count if pde in STATIONARY and pde != 'eikonal' else 0)) if pde != 'eikonal' else zeros.copy()
    initial_u = field('initial_u', options.get('displacement_basis' if pde in WAVES else 'initial_basis', count)) if pde not in STATIONARY else zeros.copy()
    initial_v = field('initial_v', options.get('velocity_basis', count)) if pde in WAVES else zeros.copy()
    coefficient_name = ('speed' if pde == 'eikonal' else 'wave_speed' if pde in WAVES else
                        'diffusivity' if pde in PARABOLIC or pde == 'elliptic_adr' else 'conductivity')
    default = [.03, .2] if pde in PARABOLIC else ([.5, 1.2] if pde in WAVES else [.5, 2.] if tier == 'T2' else 1.)
    coefficient_range = options.get(coefficient_name, default)
    coefficient = sample_setting(coefficient_range, rng)
    if coefficient <= 0:
        raise ValueError('conductivity/diffusivity/speed must be positive')
    if pde in ('poisson', 'p_laplacian') and tier != 'T2' and coefficient != 1.:
        raise ValueError('Poisson/p_laplacian use unit conductivity; select diffusion for variable conductivity')
    kind = options.get('coefficient_kind', 'smooth' if pde in ('diffusion', 'semilinear') and tier != 'T2' else 'constant')
    if kind not in ('constant', 'smooth', 'interface') or tier == 'T2' and kind != 'constant' or pde == 'eikonal' and kind != 'constant':
        raise ValueError('coefficient_kind supports constant/smooth/interface; T2 and Eikonal require constant')
    coefficient_fields = np.full(n, coefficient)
    contrast = sample_setting(options.get('coefficient_contrast', .5), rng)
    if not 0 <= contrast < .95:
        raise ValueError('coefficient_contrast must be in [0,.95)')
    if kind == 'smooth':
        raw = field('coefficient', options.get('coefficient_basis', count if tier == 'T3' else 8))
        coefficient_fields = coefficient * (1 + contrast * np.tanh(raw))
    elif kind == 'interface':
        threshold = case['nodes'][:, 0].mean()
        coefficient_fields = coefficient * (1 + contrast * np.where(case['nodes'][:, 0] < threshold, -1., 1.))
        element_x = case['nodes'][case['triangles'], 0].mean(1)
        case['element_coefficient'] = coefficient * (1 + contrast * np.where(element_x < threshold, -1., 1.))
    physical = {}
    for key, default_value in (('reaction', 0.), ('damping', 0.), ('p', 3.), ('forcing_frequency', 0.)):
        physical[key] = sample_setting(options.get(key, default_value), rng)
    physical['advection'] = [sample_setting(options.get('advection', 0.), rng) for _ in range(2)]
    beta = sample_setting(options.get('beta', [.5, 3.] if pde == 'semilinear' else 0.), rng)
    if min(beta, physical['reaction'], physical['damping'], physical['forcing_frequency']) < 0 or pde == 'p_laplacian' and physical['p'] < 2:
        raise ValueError('reaction, damping, beta, frequency must be nonnegative; p must be >=2')
    if pde != 'elliptic_adr':
        physical['advection'] = [0., 0.]
    if pde not in ('elliptic_adr', 'reaction_diffusion'):
        physical['reaction'] = 0.
    if pde != 'damped_wave':
        physical['damping'] = 0.
    if pde in STATIONARY:
        physical['forcing_frequency'] = 0.
    conditioning = {key: sample_setting(value, rng) for key, value in settings.get('conditioning', {}).items()}
    if any(not 0 <= value <= 1 for value in conditioning.values()):
        raise ValueError('Hybrid reveal fractions must lie in [0,1]')
    primary = 'source' if pde in STATIONARY else 'initial_u'
    basis = case.get(f'{primary}_basis', np.empty((n, 0)))
    prior = priors.get(primary, dict(mean=0., std=1., distribution='gaussian'))
    case['meta'] = dict(entry, tier=tier, coefficient=coefficient, beta=beta, width=float(width), height=float(height),
                        radius=float(radius), geometry_parameters=parameters, parameters=physical, field_priors=priors,
                        conditioning=conditioning, numerics=settings.get('numerics', {}), coefficient_kind=kind, coefficient_contrast=contrast,
                        amplitude_prior_std=prior['std'], amplitude_prior_mean=prior['mean'],
                        distributions=dict(coefficient=np.asarray(coefficient_range).tolist(),
                                           amplitudes='Gaussian' if prior['distribution'] == 'gaussian' else 'uniform'))
    operator = (case['element_coefficient'][:, None] if 'element_coefficient' in case else
                coefficient_fields if kind == 'smooth' else coefficient)
    if pde in WAVES:
        operator = operator ** 2
    result, info = solve(case, pde, source, initial_u, initial_v, operator, beta, config['frames'], config['dt'])
    case.update(result, source=source, initial_u=initial_u, initial_v=initial_v,
                coefficient_fields=coefficient_fields, basis=basis,
                amplitudes=case.get(f'{primary}_amplitudes', np.empty(0)))
    case['meta'].update(solver=info, targets=['solution'] + (['velocity'] if pde in WAVES else []) + (['coefficient'] if tier == 'T2' else []))
    if not all(np.isfinite(value).all() for key, value in case.items() if key != 'meta'):
        raise RuntimeError('non-finite configured case')
    return case


def save_case(path, case):
    arrays = {k: (v.astype(np.float32) if v.dtype.kind == 'f' and k not in ('nodes', 'times') else v)
              for k, v in case.items() if k != 'meta'}
    arrays['metadata'] = np.array(json.dumps(case['meta']))
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_case(path):
    with np.load(path, allow_pickle=False) as saved:
        case = {k: saved[k] for k in saved.files if k != 'metadata'}
        case['meta'] = json.loads(str(saved['metadata']))
    return case


def generate_one(config, entry, root):
    path = Path(root) / entry['file']
    if path.exists():
        existing = load_case(path)
        if existing['meta']['seed'] != entry['seed']:
            raise ValueError(f'existing case has a different seed: {path}')
        return 'existing'
    save_case(path, make_case(config, entry))
    return 'generated'


def generate(args):
    start = time.perf_counter()
    settings = read_dataset_config(args.config) if getattr(args, 'config', None) else {}
    numerics = settings.get('numerics', {})
    resolution = args.resolution if args.resolution is not None else numerics.get('resolution', 12)
    frames = args.frames if args.frames is not None else numerics.get('frames', 21)
    dt = args.dt if args.dt is not None else numerics.get('dt', .002)
    pdes = args.pdes.split(',') if args.pdes else list(DEFAULT_PDES[args.tier])
    geometries = args.geometries.split(',')
    if not pdes or any(p not in SUPPORTED[args.tier] for p in pdes):
        raise ValueError(f'{args.tier} supports only {SUPPORTED[args.tier]}')
    if any(g not in GEOMETRIES for g in geometries):
        raise ValueError(f'geometries: {GEOMETRIES}')
    if args.tier == 'T1' and geometries != ['square']:
        raise ValueError('T1 uses the fixed unit square; pass --geometries square')
    if len(set(pdes)) != len(pdes) or len(set(geometries)) != len(geometries):
        raise ValueError('duplicate PDE or geometry')
    if min(args.train, args.val, args.test) < 0 or args.train + args.val + args.test == 0:
        raise ValueError('case counts must be nonnegative, with at least one case')
    config = dict(generator_version=1, tier=args.tier, pdes=pdes, geometries=geometries, train=args.train,
                  val=args.val, test=args.test, resolution=resolution, frames=frames,
                  dt=dt, seed=args.seed)
    if settings:
        config['settings'] = settings
    if dt > .005 and any(p in WAVES + PARABOLIC for p in pdes):
        print('WARN: coarse internal dt requested; check time refinement before using these references')
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['config'] != config:
        raise ValueError('existing dataset has a different configuration; use a new --out directory')
    entries = []
    for split in ('train', 'val', 'test'):
        for pde in pdes:
            for geometry in geometries:
                for index in range(config[split]):
                    number = len(entries)
                    seed = int(np.random.SeedSequence([args.seed, number]).generate_state(1)[0])
                    entries.append(dict(file=f'{split}_{pde}_{geometry}_{index:05d}.npz',
                                        split=split, pde=pde, geometry=geometry, seed=seed, id=number))
    manifest = dict(version=1, config=config, cases=entries, failures=[])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    workers = min(4, max(1, (os.cpu_count() or 2) // 2)) if args.workers == 'auto' else int(args.workers)
    if workers < 1:
        raise ValueError('--workers must be positive or auto')
    generated = existing = 0
    def record(entry, result=None, error=None):
        nonlocal generated, existing
        if error:
            manifest['failures'].append(dict(file=entry['file'], error=str(error)))
            print(f"FAILED {entry['file']}: {error}", flush=True)
        else:
            generated += result == 'generated'; existing += result == 'existing'
    if workers == 1:
        for entry in entries:
            try:
                record(entry, generate_one(config, entry, root))
            except Exception as error:
                record(entry, error=error)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(generate_one, config, e, root): e for e in entries}
            for future in as_completed(futures):
                try:
                    record(futures[future], future.result())
                except Exception as error:
                    record(futures[future], error=error)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'Generated {generated}; existing {existing}; failed {len(manifest["failures"])}; '
          f'total {len(entries)}; {time.perf_counter() - start:.2f}s')
    if manifest['failures']:
        raise RuntimeError('dataset has failed solves; inspect manifest and rerun the same command')


def case_paths(root, split, geometries=None):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = [e for e in manifest['cases'] if e['split'] == split and
               (not geometries or e['geometry'] in geometries)]
    paths = [root / e['file'] for e in entries]
    if not paths:
        raise ValueError(f'no cases in split {split}')
    if any(not p.exists() for p in paths):
        raise ValueError('dataset incomplete; rerun its generation command')
    return paths


def normalization(cases):
    """Global training-distribution scales, never held-out target magnitudes."""
    totals = np.zeros(2); fields = np.zeros(4); count = 0
    for case in cases:
        count += 1
        w = node_weights(case); w /= w.sum()
        totals[0] += np.mean(case['solution'] ** 2 @ w)
        totals[1] += np.mean(case.get('velocity', np.zeros_like(case['solution'])) ** 2 @ w)
        for i, name in enumerate(('source', 'initial_u', 'initial_v', 'coefficient_fields')):
            fields[i] += np.dot(w, case[name] ** 2)
    return dict(output=np.maximum(np.sqrt(totals / count), .01).tolist(),
                fields=np.maximum(np.sqrt(fields / count), .01).tolist())


def known_inputs(case, mode, rng):
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode}')
    meta = case['meta']; pde = meta['pde']
    values = np.column_stack([case[n] for n in ('source', 'initial_u', 'initial_v', 'coefficient_fields')]).copy()
    masks = np.ones_like(values)
    scalar_values = np.array([meta['coefficient'], meta['beta']])
    scalar_masks = np.ones(2)
    fractions = meta.get('conditioning', {})
    if mode != 'forward':
        if meta['tier'] == 'T2':
            # Keep Poisson forcing known to avoid amplitude/conductivity ambiguity.
            masks[:, 3] = 0; scalar_masks[0] = 0
            if mode == 'hybrid' and pde in PARABOLIC:
                masks[:, 1] = rng.random(len(values)) < fractions.get('hybrid_known_initial_fraction', .5)
        elif (meta['tier'] == 'T3' and pde == 'diffusion') or pde == 'eikonal':
            masks[:, 3] = 0 if mode == 'inverse' else rng.random(len(values)) < fractions.get('hybrid_known_coefficient_fraction', .5)
            scalar_masks[0] = 0
        else:
            hidden = [0] if pde in STATIONARY else ([1, 2] if pde in WAVES else [1])
            if pde not in STATIONARY and meta.get('field_priors', {}).get('source', {}).get('count', 0):
                hidden = [0] + hidden
            for channel in hidden:
                fraction = fractions.get('hybrid_known_source_fraction' if channel == 0 else 'hybrid_known_initial_fraction', .5)
                masks[:, channel] = 0 if mode == 'inverse' else rng.random(len(values)) < fraction
    return values * masks, masks, scalar_values * scalar_masks, scalar_masks


def observations(case, count, noise, rng):
    xy = sample_points(case, count, rng)
    t = rng.uniform(0, 1, count) if len(case['times']) > 1 else np.zeros(count)
    queries = np.column_stack([xy, t])
    quantity = rng.integers(0, 2, count) if case['meta']['pde'] in WAVES else np.zeros(count, int)
    truth = interpolate(case, queries)[np.arange(count), quantity]
    values = truth + rng.normal(0, noise, count)
    return dict(queries=queries, quantity=quantity, values=values, noise=np.full(count, noise))


def query_draw(case, count, rng):
    xy = sample_points(case, count, rng)
    t = rng.uniform(0, 1, count) if len(case['times']) > 1 else np.zeros(count)
    queries = np.column_stack([xy, t])
    return queries, interpolate(case, queries)


def model_input(case, mode, obs, scales, rng, device='cpu'):
    values, masks, scalar, scalar_masks = known_inputs(case, mode, rng)
    n = len(case['nodes'])
    boundary = np.zeros(n); boundary[case['boundary_nodes']] = 1
    features = np.column_stack([case['nodes'], boundary, values / scales['fields'], masks,
                                np.tile(np.r_[scalar, scalar_masks], (n, 1))])
    edges = np.concatenate([case['triangles'][:, [0, 1]], case['triangles'][:, [1, 2]],
                            case['triangles'][:, [2, 0]]])
    edges = np.unique(np.concatenate([edges, edges[:, ::-1]]), axis=0)
    tokens = np.zeros((max(1, len(obs['values'])), 6), np.float32)
    padding = np.ones(len(tokens), bool)
    count = len(obs['values'])
    if count:
        norm = np.array(scales['output'])[obs['quantity']]
        tokens[:count] = np.column_stack([obs['queries'], obs['quantity'], obs['values'] / norm,
                                          obs['noise'] / norm])
        padding[:count] = False
    def tensor(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)
    result = dict(features=tensor(features), nodes=tensor(case['nodes']), basis=tensor(case['basis']),
                edges=tensor(edges, torch.long), observations=tensor(tokens),
                observation_mask=tensor(padding, torch.bool), pde=PDES.index(case['meta']['pde']),
                scalar_masks=tensor(scalar_masks), scalar=tensor(scalar),
                field_masks=tensor(masks), field_values=tensor(values))
    lo = np.r_[case['nodes'].min(0), case['times'][0]]
    duration = case['times'][-1] - case['times'][0] if len(case['times']) > 1 else 1.
    span = np.r_[np.ptp(case['nodes'], axis=0), duration]
    result['coordinate_bounds'] = tensor(np.stack([lo, span]))
    result['output_scales'] = tensor(scales['output'])
    result['field_scales'] = tensor(scales['fields'])
    # Geometry and fully revealed operator only: no target fields enter the modal branch.
    physical = case['meta'].get('parameters', {})
    if (case['meta']['pde'] == 'wave' and masks[:, 3].all() and
            not physical.get('damping', 0.) and not any(physical.get('advection', [0., 0.]))):
        result['wave_geometry'] = {key: case[key] for key in ('nodes', 'triangles', 'boundary_nodes')}
        speed = case['element_coefficient'][:, None] if 'element_coefficient' in case else values[:, 3]
        result['wave_operator'] = speed ** 2
        result['wave_reaction'] = physical.get('reaction', 0.)
        result['forcing_frequency'] = physical.get('forcing_frequency', 0.)
    if case['meta']['pde'] == 'wave':
        result['forcing_frequency'] = physical.get('forcing_frequency', 0.)
        result['quadrature_weights'] = node_weights(case)
    if 'field_priors' in case['meta']:
        definitions, channels, parameters = [], [], []
        for channel, name in enumerate(('source', 'initial_u', 'initial_v', 'coefficient')):
            if f'{name}_basis' in case:
                basis = case[f'{name}_basis']; definitions.append(basis)
                channels.extend([channel] * basis.shape[1])
                prior = case['meta']['field_priors'][name]
                parameters.extend([[10 + 3 * channel, prior['count'], 1.], [11 + 3 * channel, prior['mean'], 1.],
                                   [12 + 3 * channel, prior['std'], 1.], [40 + channel, float(prior['distribution'] == 'gaussian'), 1.]])
        result['basis'] = tensor(np.concatenate(definitions, axis=1) if definitions else np.empty((n, 0)))
        result['basis_channels'] = tensor(channels, torch.long)
        physical = case['meta']['parameters']
        parameters.extend([[0., PDES.index(case['meta']['pde']), 1.], [1., physical['reaction'], 1.],
                           [2., physical['damping'], 1.], [3., physical['p'], 1.],
                           [4., physical['advection'][0], 1.], [5., physical['advection'][1], 1.],
                           [6., physical['forcing_frequency'], 1.]])
        parameters.extend([[7., ('constant', 'smooth', 'interface').index(case['meta']['coefficient_kind']), 1.],
                           [8., case['meta']['coefficient_contrast'], 1.]])
        if 'element_coefficient' in case and masks[:, 3].all():
            parameters.extend(np.column_stack([case['nodes'][case['triangles']].mean(1), case['element_coefficient']]).tolist())
        result['parameters'] = tensor(parameters)
    return result


def query_input(case, queries, device='cpu', requires_grad=False):
    queries = np.asarray(queries)
    if np.any(queries[:, 2] < 0) or np.any(queries[:, 2] > case['times'][-1] + 1e-10):
        raise ValueError('query time outside physical interval; stationary equations require t=0')
    vertices, bary = locate(case, queries[:, :2])
    return dict(coords=torch.tensor(queries, dtype=torch.float32, device=device, requires_grad=requires_grad),
                vertices=torch.tensor(vertices, dtype=torch.long, device=device),
                bary=torch.tensor(bary, dtype=torch.float32, device=device))
