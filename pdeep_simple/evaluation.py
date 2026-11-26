"""Fixed held-out cases and draws, domain quadrature, CSV metrics and field files."""
import csv
import json
import os
import re
import subprocess
import sys
import multiprocessing
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch
from .data import (case_paths, load_case, normalization, observations, model_input, query_input,
                   known_inputs, case_scales, MODES, PDES)
from .geometry import node_weights, evaluation_grid, interpolate, refine_mesh, reference_case
from .physics import assemble, solve, helmholtz_operator, WAVES, PARABOLIC, LINEAR_ELLIPTIC
from .reference import quantities
from .geometry import element_geometry
from .training import load_checkpoint, model_from_config, select_device, sync
from .baselines import prepare_baseline, baseline_values, fit_pod, METHODS, MAP_STATS


def write_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def model_label(config):
    if config['model'] == 'B':
        return 'B-' + config.get('b_routing', 'dynamic').capitalize()
    if config['model'] == 'F':
        return 'F-' + config.get('f_variant', 'base').capitalize() + ('-Flow' if config.get('flow') else '')
    return config['model'] + ('-Flow' if config.get('flow') else '-Geo')


def time_weights(times):
    if len(times) == 1:
        return np.ones(1)
    w = np.zeros(len(times))
    w[:-1] += np.diff(times) / 2; w[1:] += np.diff(times) / 2
    return w / w.sum()


def error_metrics(prediction, reference, weights, scale):
    difference = prediction - reference
    if difference.ndim == np.ndim(weights) + 1:
        weights = np.asarray(weights)[..., None]  # Combined vector Euclidean L2; scalar calls unchanged.
    mse = float(np.sum(weights * difference ** 2) / weights.sum())
    return dict(mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.sum(weights * abs(difference)) / weights.sum()),
                relative_l2=float(np.sqrt(np.sum(weights * difference ** 2) /
                                      max(np.sum(weights * reference ** 2), 1e-20))),
                max_absolute=float(abs(difference).max()), normalized_rmse=float(np.sqrt(mse) / scale),
                normalized_mse=mse / scale ** 2)


def residual_diagnostics(case, field, times, mode, seed, coefficient):
    if case['meta']['pde'] == 'navier_stokes':
        # P1 projection of neural velocity, not the solver's weak P2 constraint.
        area, gradient = element_geometry(case['nodes'], case['triangles'])
        derivative = np.einsum('teiq,eik->teqk', field[:, case['triangles'], :2], gradient)
        divergence = derivative[:, :, 0, 0] + derivative[:, :, 1, 1]
        return dict(pde_residual_rmse=np.nan, kinematic_rmse=np.nan,
                    velocity_p1_divergence_rmse=float(np.sqrt(np.average(divergence ** 2, axis=1, weights=area).mean())),
                    pressure_mean_absolute=float(np.abs(np.average(field[:, :, 2], axis=1, weights=node_weights(case))).mean()))
    values, masks, scalar, scalar_masks = known_inputs(case, mode, np.random.default_rng(seed))
    pde = case['meta']['pde']; n = len(case['nodes'])
    result = dict(pde_residual_rmse=float('nan'), kinematic_rmse=float('nan'))
    if len(times) > 2 and pde in WAVES:
        du = np.gradient(field[:, :, 0], times, axis=0, edge_order=2)
        w = node_weights(case); w /= w.sum()
        result['kinematic_rmse'] = float(np.sqrt(np.mean((du[1:-1] - field[1:-1, :, 1]) ** 2 @ w)))
    if not masks[:, 0].all():
        return result
    if not scalar_masks[0] and not masks[:, 3].all() and coefficient is None:
        return result
    a = scalar[0] if scalar_masks[0] else coefficient
    if not masks[:, 3].all() and case['meta']['tier'] != 'T2':
        return result
    physical = case['meta'].get('parameters', {})
    operator = (case['element_coefficient'][:, None] if 'element_coefficient' in case else values[:, 3]) if masks[:, 3].all() else a
    mass, stiffness = (helmholtz_operator(case, operator, check_coercivity=False) if pde == 'helmholtz' else
                       assemble(case, operator ** (2 if pde in WAVES else 1), physical.get('advection'), physical.get('reaction', 0.)))
    lump = np.asarray(mass.sum(axis=1)).ravel()
    if pde == 'eikonal':
        return result  # a nonsmooth distance field has no elliptic weak residual
    residual = np.array([stiffness @ u - mass @ values[:, 0] * np.cos(2 * np.pi * physical.get('forcing_frequency', 0.) * t)
                         for u, t in zip(field[:, :, 0], times)])
    if pde == 'p_laplacian':
        _, gradients = element_geometry(case['nodes'], case['triangles'])
        epsilon = case['meta'].get('numerics', {}).get('p_regularization', 1e-4)
        for it, u in enumerate(field[:, :, 0]):
            grad = np.einsum('ei,eik->ek', u[case['triangles']], gradients)
            factor = (epsilon ** 2 + (grad * grad).sum(-1)) ** ((physical.get('p', 3.) - 2) / 2)
            _, matrix = assemble(case, factor[:, None])
            residual[it] = matrix @ u - mass @ values[:, 0]
    if pde == 'semilinear':
        residual += scalar[1] * lump * field[:, :, 0] ** 3
    if pde in PARABOLIC + WAVES:
        if len(times) < 3:
            return result
        dynamic = field[:, :, 0] if pde in PARABOLIC else field[:, :, 1]
        temporal = np.gradient(dynamic, times, axis=0, edge_order=2)
        residual += (mass @ temporal.T).T
        if pde in WAVES:
            residual += physical.get('damping', 0.) * (mass @ field[:, :, 1].T).T
        residual = residual[1:-1]
    free = np.setdiff1d(np.arange(n), case['boundary_nodes'])
    result['pde_residual_rmse'] = float(np.sqrt(np.mean((residual[:, free] / lump[free]) ** 2)))
    return result


def summarize(rows):
    keys = ('tier', 'pde', 'geometry', 'mode', 'observations', 'method', 'quantity', 'time',
            'geometry_generalization', 'grid', 'refine', 'samples', 'flow_steps', 'observation_noise', 'reference_times')
    groups = {}
    for row in rows:
        group = tuple(row.get(k, '') for k in keys)
        total, metrics = groups.setdefault(group, [0, {}])
        groups[group][0] += 1
        for key, value in row.items():
            if key in keys or key in ('case', 'seed'):
                continue
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            if np.isfinite(value):
                count, summed = metrics.get(key, (0, 0.))
                metrics[key] = count + 1, summed + value
    return [dict(zip(keys, group), cases=total, **{k: s / n for k, (n, s) in metrics.items()})
            for group, (total, metrics) in groups.items()]


def read_rows(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def append_rows(path, rows):
    if not rows:
        return
    path = Path(path); new = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows); stream.flush()


def numerical_reference(case, levels):
    case = reference_case(case)
    if not levels:
        return case
    if case['meta']['pde'] == 'navier_stokes':
        raise ValueError('For NS choose --solver-resolution during generation; eval --refine is scalar-only')
    previous = case; convergence = []
    meta = case['meta']; pde = meta['pde']
    for level in range(1, levels + 1):
        refined = refine_mesh(previous, 1)
        a = (refined['element_coefficient'][:, None] if 'element_coefficient' in refined else
             refined['coefficient_fields'] if pde == 'semilinear' or 'field_priors' in meta else meta['coefficient'])
        if pde in WAVES:
            a = a ** 2
        if pde == 'eikonal':
            a = meta['coefficient']
        result, _ = solve(refined, pde, refined['source'], refined['initial_u'], refined['initial_v'],
                          a, meta['beta'], len(case['times']), meta['solver'].get('time_step', .002) / 2 ** level,
                          time_end=float(case['times'][-1]) if len(case['times']) > 1 else 1.)
        refined.update(result)
        coords = np.concatenate([np.column_stack([refined['nodes'], np.full(len(refined['nodes']), t)]) for t in case['times']])
        old = interpolate(previous, coords).reshape(len(case['times']), -1, 2)
        weights = time_weights(case['times'])[:, None] * node_weights(refined)[None]
        for channel, name in enumerate(('solution', 'velocity') if pde in WAVES else ('solution',)):
            error = error_metrics(old[:, :, channel], refined[name], weights, 1.)
            convergence.append(dict(level=level, nodes=len(refined['nodes']), triangles=len(refined['triangles']),
                                    quantity='u' if channel == 0 else 'v', previous_level_rmse=error['rmse'],
                                    previous_level_relative_l2=error['relative_l2']))
        previous = refined
    refined['refinement_rows'] = convergence
    return refined


def prediction_values(model, context, case, coords, scales, chunk, latents=None,
                      save_samples=4, interval=.9, sample_reference=None, sample_weights=None):
    """Chunk coordinates, retaining only a few whole-field samples for figures.

    The same latents are reused in every chunk, time slice, and evaluation grid.
    No reference solution or numerical solve is used by this function.
    """
    device = context['global_h'].device
    scales = case_scales(scales, case)
    channels = 3 if case['meta']['pde'] == 'navier_stokes' else 2
    with torch.no_grad():
        latent_contexts = [model.with_latent(context, z) for z in latents] if latents is not None else [context]
    from .wave import initial_diagnostics
    diagnostics = initial_diagnostics(latent_contexts)
    size = (len(coords), channels)
    mean = np.empty(size, np.float32)
    std = np.full(size, np.nan, np.float32); lower = std.copy(); upper = std.copy()
    kept = np.empty((min(save_samples, len(latent_contexts)), *size), np.float32) if latents is not None else np.empty((0, *size), np.float32)
    sample_squared = np.zeros((len(latent_contexts), channels))
    with torch.no_grad():
        for start in range(0, len(coords), chunk):
            stop = min(start + chunk, len(coords))
            queries = query_input(case, coords[start:stop], device)
            fields = np.stack([model.decode(c, queries).cpu().numpy()[:, :channels] for c in latent_contexts])
            fields *= np.asarray(scales['output'], np.float32)
            if not np.isfinite(fields).all():
                raise RuntimeError('non-finite field prediction; no metrics saved for this case')
            # Keep identical F samples at exactly zero spread, including known Forward states.
            statistics = fields.astype(np.float64) if hasattr(getattr(model, 'base', model), 'f_variant') or hasattr(model, 'codec') else fields
            if sample_reference is not None:
                sample_squared += ((fields - sample_reference[None, start:stop]) ** 2 * sample_weights[None, start:stop, None]).sum(1)
            mean[start:stop] = statistics.mean(0)
            if latents is not None:
                kept[:, start:stop] = fields[:len(kept)]
                if len(fields) > 1:
                    std[start:stop] = statistics.std(0, ddof=1)
                    lower[start:stop], upper[start:stop] = np.quantile(statistics, [(1 - interval) / 2, (1 + interval) / 2], axis=0)
    result = dict(sample_std=std, lower=lower, upper=upper, sample_fields=kept, diagnostics=diagnostics)
    if sample_reference is not None:
        result['sample_rmse'] = np.sqrt(sample_squared / sample_weights.sum())
    return mean, result


def classical_values(prepared, case, coords, chunk, interval=.9):
    from scipy.special import ndtri
    mean = np.empty((len(coords), 2), np.float64)
    std = np.full_like(mean, np.nan)
    for start in range(0, len(coords), chunk):
        stop = min(start + chunk, len(coords))
        mean[start:stop], extra = baseline_values(prepared, case, coords[start:stop])
        if 'solution_std' in extra:
            std[start:stop, 0] = extra['solution_std']
    quantile = ndtri((1 + interval) / 2)
    return mean, dict(sample_std=std, lower=mean - quantile * std, upper=mean + quantile * std,
                      sample_fields=np.empty((0, len(coords), 2)))


def evaluate_case(path, options, scales, methods, out, save=False, model=None, model_config=None):
    """One independent physical case: all modes, sensors, methods and grids."""
    path = Path(path); out = Path(out)
    case = load_case(path); meta = case['meta']; pde = meta['pde']
    vector = pde == 'navier_stokes'; channels = 3 if vector else 2
    scales = case_scales(scales, case)
    if vector and not (model is not None and model_config and model_config.get('model') == 'A' and model_config.get('vector_output')):
        raise ValueError('Navier-Stokes evaluation requires an A --joint-conditioning --vector-output checkpoint')
    reference = numerical_reference(case, options['refine'])
    if options['refine']:
        write_rows(out / f'refinement_{path.stem}.csv', reference['refinement_rows'])
    indices = np.unique(np.linspace(0, len(case['times']) - 1, min(options['frames'], len(case['times']))).round().astype(int))
    times = case['times'][indices]; tw = time_weights(times)
    if (vector and len(case['times']) > 1 or options.get('filtered') and pde in WAVES or
            model_config and model_config.get('model') in ('E', 'F') and pde in PARABOLIC + WAVES):
        times = np.linspace(case['times'][0], case['times'][-1], options['frames'])
        tw = time_weights(times)
    draw_seed = int(np.random.SeedSequence([options['seed'], meta['seed']]).generate_state(1)[0])
    nodal_coords = np.concatenate([np.column_stack([case['nodes'], np.full(len(case['nodes']), t)]) for t in times])
    rows, skipped, cache = [], [], {}
    device = options.get('device', 'cpu') if model is not None else 'cpu'
    samples = options.get('samples', 32) if model_config and model_config.get('flow') else 0
    grids = [(g, evaluation_grid(case, g)) for g in options['grids']]
    is_b = bool(model_config and model_config.get('model') == 'B')
    nested = options.get('nested_observations', False)
    full_obs = observations(case, max(options['observations']), options['noise'],
                            np.random.default_rng(draw_seed + max(options['observations']))) if nested else None
    for mode in options['modes']:
        for count in options['observations']:
            obs = ({key: value[:count] for key, value in full_obs.items()} if nested else
                   observations(case, count, options['noise'], np.random.default_rng(draw_seed + count)))
            for method in methods:
                coefficient = None; extra = {}; latents = None
                sync(device); started = time.perf_counter()
                if model is None:
                    if method == 'grid' and mode == 'hybrid' and pde in PARABOLIC:
                        skipped.append(dict(case=path.name, mode=mode, observations=count, method=method,
                                            reason='grid reference requires fully known initial field'))
                        continue
                    try:
                        prepared = prepare_baseline(method, case, obs, mode, draw_seed, cache, options)
                    except NotImplementedError as error:
                        skipped.append(dict(case=path.name, mode=mode, observations=count, method=method, reason=str(error)))
                        continue
                    coefficient = prepared.get('coefficient')
                    extra = {k: v for k, v in prepared.items() if k not in ('field', 'responses', 'fits', 'dims') and not k.startswith('_')}
                else:
                    inputs = model_input(case, mode, obs, scales, np.random.default_rng(draw_seed), device)
                    with torch.no_grad():
                        context = model.encode(inputs)
                        if model_config and model_config.get('model') in ('E', 'F'):
                            basis = context.get('wave')
                            extra.update(modal_wave=pde in WAVES and basis is not None,
                                         initial_aware_wave=pde in WAVES and 'wave_initial' in context,
                                         wave_modes=len(basis['omega']) if basis is not None else 0,
                                         modal_evolution=basis is not None,
                                         initial_aware_evolution='wave_initial' in context,
                                         evolution_rule=context.get('evolution_kind', 'wave' if pde == 'wave' else 'coordinate'))
                        if samples:
                            latents = model.sample_latents(context, samples, options['flow_steps'], draw_seed)
                        if is_b:
                            from .prism_evaluation import posterior_diagnostics
                            b_metrics, b_arrays = posterior_diagnostics(model, context, latents, case, options['interval'])
                            extra.update(b_arrays)
                        if meta['tier'] == 'T2':
                            coefficient = float(torch.stack([model.with_latent(context, z)['log_coefficient'].exp()
                                                             for z in latents]).mean().cpu()) if samples and not getattr(model, 'initial_only', False) else float(context['log_coefficient'].exp().cpu())
                sync(device); setup_seconds = time.perf_counter() - started
                started = time.perf_counter()
                if model is None:
                    nodal, nodal_spread = classical_values(prepared, case, nodal_coords, options['chunk'], options['interval'])
                else:
                    nodal, nodal_spread = prediction_values(model, context, case, nodal_coords, scales, options['chunk'],
                                                            latents, options['save_samples'], options['interval'],
                                                            interpolate(reference, nodal_coords) if is_b else None,
                                                            (tw[:, None] * node_weights(case)[None]).ravel() if is_b else None)
                sync(device); nodal_seconds = time.perf_counter() - started
                nodal = nodal.reshape(len(times), -1, channels)
                extra.update(nodal_spread.get('diagnostics', {}))
                coefficient = extra.get('coefficient_prediction', coefficient)
                diagnostics = residual_diagnostics(case, nodal, times, mode, draw_seed, coefficient)
                if is_b: diagnostics.update(b_metrics)
                if model_config and model_config.get('model') in ('E', 'F'):
                    if any(model_config.get(key) for key in ('initial_latent', 'assimilation', 'infer_operator')):
                        diagnostics.setdefault('operator_inferred', 0.)
                    if model_config.get('assimilation') or model_config.get('model') == 'F':
                        diagnostics.setdefault('assimilation_applied', 0.)
                        # Keep CSV columns stable for zero sensors and non-wave cases.
                        for when in ('before', 'after'):
                            for quantity in ('normalized', 'noise_weighted', 'u', 'v'):
                                diagnostics.setdefault(f'observation_{quantity}_rmse_{when}', np.nan)
                    if model_config.get('model') == 'F':
                        diagnostics['refinement_steps'] = 0.
                        if model_config.get('f_variant') == 'observable':
                            diagnostics.update({key: np.nan for key in ('latent_mean_variance', 'latent_strong_directions',
                                                                       'latent_weak_directions', 'latent_effective_directions')})
                diagnostics.update({key: value for key, value in nodal_spread.get('diagnostics', {}).items()
                                    if key != 'coefficient_prediction'})
                if 'modal_wave' in extra:
                    diagnostics.update(modal_wave=int(extra['modal_wave']), wave_modes=extra['wave_modes'],
                                       modal_evolution=int(extra['modal_evolution']),
                                       initial_aware_evolution=int(extra['initial_aware_evolution']),
                                       evolution_rule=extra['evolution_rule'])
                boundary = np.sqrt(np.mean(nodal[:, case['boundary_nodes']] ** 2, axis=(0, 1)))
                initial = np.sqrt(np.average((nodal[0] - interpolate(reference, nodal_coords[:len(case['nodes'])])) ** 2,
                                             axis=0, weights=node_weights(case))) if len(case['times']) > 1 else np.full(channels, np.nan)
                if vector:
                    boundary[2] = initial[2] = np.nan  # Pressure has neither a no-slip BC nor an independent IC.
                generalization = ('unseen_family' if model_config and model_config.get('training_geometries') and
                                  meta['geometry'] not in model_config['training_geometries'] else 'new_case_and_instance')
                for grid, (points, spatial, grid_arrays) in grids:
                    coords = np.concatenate([np.column_stack([points, np.full(len(points), t)]) for t in times])
                    truth = interpolate(reference, coords).reshape(len(times), -1, channels)
                    coarse = interpolate(case, coords).reshape(truth.shape)
                    if grid == 0:
                        field, spread, elapsed = nodal, nodal_spread, setup_seconds + nodal_seconds
                    else:
                        sync(device); started = time.perf_counter()
                        if model is None:
                            prediction, spread = classical_values(prepared, case, coords, options['chunk'], options['interval'])
                        else:
                            prediction, spread = prediction_values(model, context, case, coords, scales, options['chunk'],
                                                                    latents, options['save_samples'], options['interval'],
                                                                    truth.reshape(-1, 2) if is_b else None,
                                                                    (tw[:, None] * spatial[None]).ravel() if is_b else None)
                        sync(device); elapsed = setup_seconds + time.perf_counter() - started
                        field = prediction.reshape(truth.shape)
                    weights = tw[:, None] * spatial[None]
                    std, lower, upper = [spread[k].reshape(truth.shape) for k in ('sample_std', 'lower', 'upper')]
                    for channel in range(len(quantities(case))):
                        base = dict(case=path.name, seed=draw_seed, tier=meta['tier'], pde=pde, geometry=meta['geometry'],
                                    mode=mode, observations=count, method=method, quantity=quantities(case)[channel],
                                    geometry_generalization=generalization, grid=grid, refine=options['refine'], samples=samples,
                                    flow_steps=options['flow_steps'] if samples else 0, valid_points=len(points),
                                    observation_noise=options['noise'], reference_times=','.join(f'{t:.12g}' for t in times),
                                    domain_area_quadrature=float(spatial.sum()))
                        common = dict(boundary_rmse=float(boundary[channel]), initial_rmse=float(initial[channel]),
                                      inference_seconds=elapsed, coefficient_prediction=coefficient if coefficient is not None else np.nan,
                                      coefficient_true=meta['coefficient'] if meta['tier'] == 'T2' else np.nan,
                                      coefficient_absolute=abs(coefficient - meta['coefficient']) if coefficient is not None else np.nan,
                                      **diagnostics, **{key: extra.get(key, '' if key == 'map_message' else np.nan) for key in MAP_STATS})
                        if model_config and model_config.get('model') in ('E', 'F') and model_config.get('initial_latent'):
                            initial_std = nodal_spread['sample_std'].reshape(nodal.shape)[0, :, channel]
                            common['initial_sample_std_mean'] = float(np.average(initial_std, weights=node_weights(case))) if np.isfinite(initial_std).all() else np.nan
                        for it in [None] + list(range(len(times))):
                            pred, target = (field[:, :, channel], truth[:, :, channel]) if it is None else (field[it, :, channel], truth[it, :, channel])
                            w = weights if it is None else spatial
                            lo, hi, sd = ((lower[:, :, channel], upper[:, :, channel], std[:, :, channel]) if it is None else
                                          (lower[it, :, channel], upper[it, :, channel], std[it, :, channel]))
                            available = np.isfinite(sd).all()
                            uncertainty = dict(sample_std_mean=float(np.average(sd, weights=w)) if available else np.nan,
                                               interval_coverage=float(np.average((target >= lo) & (target <= hi), weights=w)) if available else np.nan,
                                               interval_width=float(np.average(hi - lo, weights=w)) if available else np.nan,
                                               interval_probability=options['interval'])
                            if is_b:
                                scores = spread['sample_rmse'][:, channel]
                                uncertainty.update(sample_solution_rmse_all_times_mean=float(scores.mean()),
                                                   sample_solution_normalized_rmse_all_times_mean=float(scores.mean() / scales['output'][channel]),
                                                   sample_solution_rmse_all_times_min=float(scores.min()),
                                                   sample_solution_rmse_all_times_max=float(scores.max()))
                            ref_error = (coarse[:, :, channel] - truth[:, :, channel]) if it is None else (coarse[it, :, channel] - truth[it, :, channel])
                            rows.append(dict(base, time='all' if it is None else float(times[it]),
                                             **error_metrics(pred, target, w, scales['output'][channel]), **common, **uncertainty,
                                             reference_rmse=float(np.sqrt(np.average(ref_error ** 2, weights=w)))))
                    if save:
                        suffix = f'_g{grid}' if grid else ''
                        filename = f'{path.stem}_{mode}_n{count}_{method}{suffix}.npz'
                        kept = spread['sample_fields'].reshape(-1, *truth.shape)
                        arrays = dict(nodes=case['nodes'], triangles=case['triangles'], times=times, query_points=points,
                                      reference=truth.astype(np.float32), prediction=field.astype(np.float32),
                                      sample_std=std.astype(np.float32), lower=lower.astype(np.float32), upper=upper.astype(np.float32),
                                      sample_fields=kept.astype(np.float32), observation_queries=obs['queries'],
                                      observation_quantity=obs['quantity'], observation_values=obs['values'], observation_noise=obs['noise'],
                                      metadata=np.array(json.dumps(dict(meta, case=path.name, mode=mode, count=count, method=method, grid=grid,
                                                                        refine=options['refine'], samples=samples, interval=options['interval'],
                                                                        posterior=extra.get('posterior', 'empirical flow, uncalibrated' if samples else 'none'),
                                                                        pod_rank=extra.get('pod_rank'),
                                                                        **{key: model_config[key] for key in ('initial_latent', 'assimilation', 'infer_operator')
                                                                           if model_config and model_config.get(key)},
                                                                        **{key: model_config[key] for key in ('f_variant', 'f_steps', 'f_architecture')
                                                                           if model_config and model_config.get('model') == 'F' and key in model_config},
                                                                        **{key: model_config[key] for key in ('b_routing', 'b_info_rank', 'b_tau', 'b_info_updates', 'b_latent_dim')
                                                                           if is_b and key in model_config},
                                                                        **{key: extra[key] for key in ('modal_wave', 'wave_modes', 'initial_aware_wave',
                                                                                                      'modal_evolution', 'initial_aware_evolution', 'evolution_rule') if key in extra},
                                                                        **{key: extra[key] for key in MAP_STATS if key in extra}))))
                        if vector:
                            field_mask = inputs['field_masks'].cpu().numpy(); scalar_mask = inputs['scalar_masks'].cpu().numpy()
                            arrays.update(known_forcing_y_mask=inputs['forcing_y_mask'].cpu().numpy(),
                                known_field_names=np.array(['forcing_x', 'initial_velocity_x', 'initial_velocity_y', 'viscosity']))
                        else:
                            _, field_mask, _, scalar_mask = known_inputs(case, mode, np.random.default_rng(draw_seed))
                        arrays.update(known_field_mask=field_mask.astype(bool), known_scalar_mask=scalar_mask.astype(bool))
                        if latents is not None:
                            arrays['latents'] = latents.detach().cpu().numpy()
                        if is_b: arrays['sample_solution_rmse'] = spread['sample_rmse']
                        arrays.update(grid_arrays)
                        arrays.update({k: v for k, v in extra.items() if isinstance(v, np.ndarray)})
                        if 'amplitude_covariance' in extra or method == 'gp':
                            arrays['solution_std'] = std[:, :, 0].ravel()
                        np.savez_compressed(out / filename, **arrays)
    return rows, skipped


def baseline_worker(path, options, scales, methods, out, save):
    torch.set_num_threads(1)
    return evaluate_case(path, options, scales, methods, out, save)


def baseline_workers(args, paths):
    requested = min(4, max(1, (os.cpu_count() or 2) // 2)) if args.workers == 'auto' else int(args.workers)
    if requested < 1 or args.memory_gb <= 0:
        raise ValueError('workers and memory-gb must be positive')
    budget = args.memory_gb * 1024 ** 3
    try:
        budget = min(budget, os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') * .5)
    except (ValueError, OSError):
        if sys.platform == 'darwin':
            try:
                stats = subprocess.check_output(['vm_stat'], text=True, timeout=2)
                page_size = int(re.search(r'page size of (\d+)', stats).group(1))
                pages = sum(int(re.search(rf'^Pages {name}:\s+(\d+)', stats, re.M).group(1))
                            for name in ('free', 'inactive', 'speculative'))
                budget = min(budget, pages * page_size * .5)
            except (OSError, subprocess.SubprocessError, AttributeError):
                pass  # Explicit memory budget still applies if OS statistics are unavailable.
    nodes = frames = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as case:
            nodes = max(nodes, len(case['nodes'])); frames = max(frames, len(case['times']))
    grid = max([int(g) for g in args.grids.split(',')] if args.grids else [0])
    # Process/runtime, up to four sparse factorization fill-ins, T2 candidate fields,
    # refined reference, and chunked prediction/output arrays. No dense N-by-N FEM.
    estimate = (256 * 1024 ** 2 + 320 * nodes ** 1.5 + 3 * 41 * frames * nodes * 16 +
                max(nodes, grid ** 2) * min(frames, args.frames) * 256 + nodes * 4 ** args.refine * frames * 32 +
                32 * max(int(n) for n in args.observations.split(',')) ** 2 +
                8 * args.chunk * max(int(n) for n in args.observations.split(',')))
    if estimate > budget:
        raise ValueError(f'one baseline worker needs roughly {estimate / 1024**3:.2f} GiB; increase --memory-gb or reduce the case/grid size')
    workers = min(requested, len(paths), max(1, int(budget // estimate)))
    print(f'baseline workers={workers} (requested {args.workers}); estimated {estimate / 1024**2:.0f} MiB/worker; budget={budget / 1024**3:.2f} GiB')
    return workers


def evaluate(args, baseline=False):
    modes = args.modes.split(',')
    pdes = [p.strip() for p in args.pdes.split(',')] if getattr(args, 'pdes', None) else [args.pde] if args.pde else []
    if any(p not in PDES for p in pdes):
        raise ValueError(f'PDEs must be selected from {PDES}')
    overrides = {key: getattr(args, key) for key in ('assimilation', 'infer_operator') if getattr(args, key, None) is not None}
    filtered = not baseline and bool(pdes or args.max_cases is not None or args.geometries or overrides or
                                     modes != list(MODES) or args.split != 'test')
    base = Path(args.out or (Path(args.run) / ('evaluations' if filtered else 'evaluation')))
    base.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='selection_', dir=base)) if filtered else base
    save_cases = args.save_cases if args.save_cases is not None else (None if filtered else 3)
    grids = [int(g) for g in args.grids.split(',')] if args.grids else [0]
    if args.frames < 1 or args.chunk < 1 or (save_cases is not None and save_cases < 0) or any(g < 3 and g != 0 for g in grids):
        raise ValueError('positive frames/chunk, nonnegative save-cases, grids >=3 required')
    if not 0 <= args.refine <= 3 or args.refine and args.max_cases is None:
        raise ValueError('--refine supports levels 0..3 and requires explicit --max-cases')
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError('max-cases must be positive')
    device = select_device(args.device); torch.set_num_threads(args.threads)
    root = Path(args.data); manifest = json.loads((root / 'manifest.json').read_text())
    geometries = args.geometries.split(',') if args.geometries else []
    limit = args.max_cases if args.max_cases is not None else (8 if args.grids else None)
    paths = []
    for entry in manifest['cases']:
        if entry['split'] != args.split or geometries and entry['geometry'] not in geometries:
            continue
        path = root / entry['file']; pde = entry.get('pde')
        if pdes and pde is None:  # Older manifests may lack the PDE tag.
            with np.load(path, allow_pickle=False) as saved:
                pde = json.loads(str(saved['metadata']))['pde']
        if pdes and pde not in pdes:
            continue
        paths.append(path)
        if limit is not None and len(paths) == limit:
            break
    if not paths:
        raise ValueError('no matching evaluation cases')
    if any(not p.exists() for p in paths):
        raise ValueError('selected dataset cases are missing')
    counts = [int(n) for n in args.observations.split(',')]
    if any(m not in MODES for m in modes) or min(counts) < 0 or args.noise < 0:
        raise ValueError('valid modes and nonnegative counts/noise required')
    model = None; config = {}
    if not baseline:
        checkpoint = load_checkpoint(Path(args.run) / args.checkpoint, device)
        config = dict(checkpoint['config'], **overrides); scales = checkpoint['scales']
        if config.get('flow') and checkpoint.get('stage') != 'flow':
            raise ValueError('this checkpoint has not reached flow training; resume it to finish both stages')
        if args.samples < 1 or args.flow_steps < 1 or args.save_samples < 0 or not 0 < args.interval < 1:
            raise ValueError('positive samples/flow-steps, nonnegative save-samples and interval in (0,1) required')
        model = model_from_config(config, device); model.load_state_dict(checkpoint['model']); model.eval()
        methods = [model_label(config)]
    else:
        scales = (json.loads((Path(args.normalization_run) / 'config.json').read_text())['scales']
                  if args.normalization_run else normalization(load_case(p) for p in case_paths(args.data, 'train')))
        tier = json.loads((Path(args.data) / 'manifest.json').read_text())['config']['tier']
        methods = args.methods.split(',') if args.methods else ['prior', 'rbf'] + (['ridge', 'bayes'] if tier == 'T1' else (['grid'] if tier == 'T2' else []))
        if args.noise == 0 and set(methods).intersection({'bayes', 'pod', 'gp', 'map', 'grid'}):
            raise ValueError('These likelihood baselines require positive noise; noiseless neural evaluation and prior/rbf remain available')
        for method in methods:
            if method not in METHODS:
                raise ValueError(f'unknown baseline {method}; choose {METHODS}')
        if args.pod_rank < 1 or args.map_maxiter < 1 or not all(np.isfinite(value) and value > 0 for value in
                (args.pod_ridge, args.gp_lengthscale, args.gp_amplitude)):
            raise ValueError('positive POD rank/ridge, GP lengthscale/amplitude and MAP maxiter required')
        if 'pod' in methods:
            args.pod_model = fit_pod(args.data, args.pod_model or out / 'pod.npz', args.pod_rank)
    options = dict(vars(args), modes=modes, observations=counts, grids=grids, device=device, filtered=filtered, save_cases=save_cases,
                   samples=getattr(args, 'samples', 0), flow_steps=getattr(args, 'flow_steps', 0),
                   save_samples=getattr(args, 'save_samples', 0), interval=getattr(args, 'interval', .9))
    (out / 'evaluation.json').write_text(json.dumps(dict(options, methods=methods, scales=scales,
        model_options={key: config.get(key) for key in ('model', 'joint_conditioning', 'vector_output', 'initial_latent', 'assimilation', 'infer_operator', 'flow',
                                                       'f_variant', 'f_steps', 'align_weight', 'f_architecture',
                                                       'b_routing', 'b_info_rank', 'b_tau', 'b_info_updates', 'b_latent_dim')},
        cases=[str(p.resolve()) for p in paths], data=str(Path(args.data).resolve()),
        reference=('stored FEM fields: P1 scalar/pressure, P2 NS velocity; linear interpolation in physical time'
                   if not args.refine else 'refined FEM with prolonged original P1 physical inputs; observations unchanged'),
        uncertainty='empirical flow intervals are not claimed calibrated; T1/G1 Gaussian elliptic Bayes exact only for stored discrete Gaussian model'), indent=2))
    # Per-case CSVs and appended metrics survive interruption. Final order is case order.
    (out / 'metrics.csv').write_text(''); (out / 'skipped.csv').write_text('')
    ranks, save = {}, {}
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            pde = json.loads(str(saved['metadata']))['pde']
        rank = ranks.get(pde, 0); ranks[pde] = rank + 1
        save[path] = save_cases is None or rank < save_cases
    started = time.perf_counter(); completed = 0
    def record(path, result):
        nonlocal completed
        rows, skipped = result
        write_rows(out / f'case_{path.stem}.csv', rows)
        append_rows(out / 'metrics.csv', rows); append_rows(out / 'skipped.csv', skipped)
        completed += 1; elapsed = time.perf_counter() - started
        speed = completed / max(elapsed, 1e-9); eta = (len(paths) - completed) / speed
        print(f'{completed}/{len(paths)} {path.name}; {speed:.2f} cases/s; ETA {eta:.1f}s', flush=True)
    if baseline:
        workers = baseline_workers(args, paths)
        names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = '1'  # Spawn imports NumPy/SciPy only after these limits.
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                iterator = iter(paths); pending = {}
                for _ in range(min(workers, len(paths))):
                    path = next(iterator)
                    pending[pool.submit(baseline_worker, path, options, scales, methods, out, save[path])] = path
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        path = pending.pop(future); record(path, future.result())
                        following = next(iterator, None)
                        if following is not None:
                            pending[pool.submit(baseline_worker, following, options, scales, methods, out, save[following])] = following
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    else:
        for path in paths:
            record(path, evaluate_case(path, options, scales, methods, out, save[path], model, config))
    # Rewrite only the small tables in deterministic physical-case order.
    ordered = out / 'metrics_ordered.tmp.csv'; ordered.write_text('')
    for path in paths:
        append_rows(ordered, list(read_rows(out / f'case_{path.stem}.csv')))
    ordered.replace(out / 'metrics.csv')
    write_rows(out / 'summary.csv', summarize(read_rows(out / 'metrics.csv')))
    print(f'Completed {completed} cases in {time.perf_counter() - started:.2f}s; CSV metrics and fields: {out}')
