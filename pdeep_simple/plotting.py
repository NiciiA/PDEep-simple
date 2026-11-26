"""Ordinary Matplotlib inspection and reconstruction figures."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .physics import WAVES, STATIONARY
from .data import PDES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from .data import case_paths, load_case, observations
from .geometry import node_weights


def field_panel(ax, case, values, title, limits=None, cmap='viridis'):
    limits = {} if limits is None else dict(vmin=limits[0], vmax=limits[1])
    if 'grid_mask' in case:
        image = np.ma.array(np.zeros(case['grid_mask'].shape), mask=~case['grid_mask'])
        image[case['grid_mask']] = np.asarray(values)
        artist = ax.pcolormesh(case['grid_x'], case['grid_y'], image, shading='nearest', cmap=cmap, **limits)
    else:
        tri = mtri.Triangulation(*case['nodes'].T, case['triangles'])
        artist = ax.tripcolor(tri, np.asarray(values), shading='gouraud', cmap=cmap, **limits)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.figure.colorbar(artist, ax=ax, fraction=.046, pad=.04)


def dataset_summary(paths, out):
    stats, parameters, betas, coefficient_ranges, splits = [], [], [], [], []
    for path in paths:
        case = load_case(path); u = case['solution']; w = node_weights(case); w /= w.sum()
        stats.append(dict(case=path.name, pde=case['meta']['pde'], geometry=case['meta']['geometry'],
                          minimum=float(u.min()), maximum=float(u.max()), mean=float(np.mean(u @ w)),
                          rms=float(np.sqrt(np.mean(u ** 2 @ w))), std=float(u.std()),
                          nodes=len(case['nodes']), triangles=len(case['triangles'])))
        parameters.append(case['meta']['coefficient']); betas.append(case['meta']['beta'])
        coefficient_ranges.extend([float(case['coefficient_fields'].min()), float(case['coefficient_fields'].max())])
        splits.append(case['meta']['split'])
    summary = dict(cases=len(stats), pdes=dict(Counter(s['pde'] for s in stats)),
                   geometries=dict(Counter(s['geometry'] for s in stats)),
                   coefficient_range=[min(parameters), max(parameters)], beta_range=[min(betas), max(betas)],
                   coefficient_field_range=[min(coefficient_ranges), max(coefficient_ranges)],
                   splits=dict(Counter(splits)), cases_summary=stats)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, key in zip(axes, ('rms', 'maximum', 'minimum')):
        for pde in summary['pdes']:
            ax.hist([s[key] for s in stats if s['pde'] == pde], bins=min(12, max(3, len(stats) // 3)), alpha=.6, label=pde)
        ax.set_xlabel(f'solution {key}'); ax.set_ylabel('physical cases'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'dataset_summary.png', dpi=140); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases_summary'}, indent=2))


def inspect_data(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    paths = case_paths(args.data, args.split)
    manifest = json.loads((Path(args.data) / 'manifest.json').read_text())
    all_paths = [Path(args.data) / entry['file'] for entry in manifest['cases']]
    dataset_summary(all_paths, out)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(paths), min(args.random, len(paths)), replace=False).tolist() if args.random else [args.case]
    for index in chosen:
        if not 0 <= index < len(paths):
            raise ValueError(f'case index must be in [0,{len(paths) - 1}]')
        case = load_case(paths[index]); meta = case['meta']; times = case['times']
        frames = np.unique(np.linspace(0, len(times) - 1, min(4, len(times))).round().astype(int))
        obs = observations(case, args.observations, args.noise, np.random.default_rng(args.seed + meta['seed']))
        panels = [('Coefficient a / alpha / c', case['coefficient_fields'])]
        if meta['pde'] != 'eikonal':
            panels.insert(0, ('Source f', case['source']))
        if meta['pde'] not in STATIONARY:
            panels.insert(1, ('Initial u', case['initial_u']))
        if meta['pde'] in WAVES:
            panels.append(('Initial v', case['initial_v']))
        panels += [(f'u, t={times[i]:.3f}', case['solution'][i]) for i in frames]
        if meta['pde'] in WAVES:
            panels += [(f'v, t={times[i]:.3f}', case['velocity'][i]) for i in frames]
        count = len(panels) + 3
        fig, axes = plt.subplots(int(np.ceil(count / 4)), 4, figsize=(17, 3.7 * int(np.ceil(count / 4))))
        axes = axes.ravel()
        ax = axes[0]; ax.triplot(*case['nodes'].T, case['triangles'], lw=.4, color='.5')
        for a, b in case['boundary_edges']:
            ax.plot(*case['nodes'][[a, b]].T, color='crimson', lw=1)
        ax.set_title(f'{len(case["nodes"])} nodes, {len(case["triangles"])} triangles'); ax.set_aspect('equal')
        for ax, (title, values) in zip(axes[1:], panels):
            if title.startswith('Coefficient') and 'element_coefficient' in case:
                artist = ax.tripcolor(*case['nodes'].T, case['triangles'], facecolors=case['element_coefficient'], shading='flat')
                ax.set_aspect('equal'); ax.set_title('Mesh-cell coefficient'); ax.set_xlabel('x'); ax.set_ylabel('y')
                fig.colorbar(artist, ax=ax, fraction=.046, pad=.04)
            else:
                field_panel(ax, case, values, title)
        overlay = axes[1 + len(panels)]
        middle = len(times) // 2
        field_panel(overlay, case, case['solution'][middle], f'Observations (all times); u at t={times[middle]:.3f}')
        for channel, marker in ((0, 'o'), (1, '^')):
            keep = obs['quantity'] == channel
            if keep.any():
                overlay.scatter(*obs['queries'][keep, :2].T, facecolors='white', edgecolors='black',
                                marker=marker, s=22, label='u' if channel == 0 else 'v')
        if len(obs['values']):
            overlay.legend(fontsize=8, loc='lower right')
        histogram = axes[2 + len(panels)]
        histogram.hist(case['solution'].ravel(), bins=40, alpha=.8); histogram.set_title('Stored u values (unweighted)')
        for ax in axes[count:]:
            ax.axis('off')
        stats = dict(minimum=float(case['solution'].min()), maximum=float(case['solution'].max()),
                     mean=float(case['solution'].mean()), std=float(case['solution'].std()),
                     nodes=len(case['nodes']), triangles=len(case['triangles']))
        coefficient_text = (f'a(x)=[{case["coefficient_fields"].min():.3g}, {case["coefficient_fields"].max():.3g}]'
                            if meta['tier'] == 'T4' else f'coefficient={meta["coefficient"]:.4g}')
        physical = meta.get('parameters', {})
        detail = f'beta={meta["beta"]:.3g}'
        for key in ('reaction', 'damping', 'forcing_frequency'):
            if physical.get(key):
                detail += f', {key}={physical[key]:.3g}'
        if meta['pde'] == 'p_laplacian':
            detail += f', p={physical["p"]:.3g}'
        if meta['pde'] == 'elliptic_adr':
            detail += f', b={np.round(physical["advection"], 3).tolist()}'
        counts = ', '.join(f'{key}:K={value["count"]}' for key, value in meta.get('field_priors', {}).items())
        if not counts:
            counts = f'K={case["basis"].shape[1]}'
        fig.suptitle(f'{meta["tier"]} | {meta["pde"]} | {meta["geometry"]} | {coefficient_text}, '
                     f'{detail}; {counts}\n'
                     f'u min={stats["minimum"]:.4g}, max={stats["maximum"]:.4g}, '
                     f'mean={stats["mean"]:.4g}, std={stats["std"]:.4g}', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(out / f'{paths[index].stem}.png', dpi=130); plt.close(fig)
        print(paths[index].name, stats)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def stage_lines(axes, runs):
    boundaries = set()
    for run in runs:
        path = run / 'config.json'
        config = json.loads(path.read_text()) if path.exists() else {}
        if config.get('staged'):
            boundaries.add(config['stage_start'])
    for boundary in sorted(boundaries):
        for ax in axes:
            ax.axvline(boundary, color='.4', linestyle='--', linewidth=1,
                       label=f'physics after {boundary}')


def loss_plots(run, out):
    if json.loads((run / 'config.json').read_text()).get('flow'):
        from .comparison import multiple_loss_plots
        multiple_loss_plots([run], out)
        return
    train = read_csv(run / 'train.csv'); val = read_csv(run / 'validation.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for records, label in ((train, 'train'), (val, 'validation')):
        axes[0].plot([int(r['update']) for r in records], [float(r['field']) for r in records], label=label)
        axes[2].plot([int(r['update']) for r in records], [float(r['total']) for r in records], label=label)
    for key in ('aux', 'kinematic', 'weak', 'initial', 'boundary', 'initial_reconstruction', 'alignment'):
        values = np.array([float(r.get(key) or 0.) for r in train])
        if np.any(values):
            axes[1].plot([int(r['update']) for r in train], values, label=key)
    stage_lines(axes, [run])
    for ax, title in zip(axes, ('Normalized field MSE', 'Other loss terms (unweighted)', 'Weighted total')):
        ax.set_title(title); ax.set_xlabel('update'); ax.set_yscale('symlog', linthresh=1e-5)
        if ax.lines:
            ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'losses.png', dpi=150); plt.close(fig)
    field_loss_plots([run], out)


def field_loss_plots(runs, out):
    """Existing normalized field diagnostics, separated by PDE and flow stage."""
    logs = []
    for index, run in enumerate(runs):
        for filename, split in (('train.csv', 'train'), ('validation.csv', 'validation')):
            if not (run / filename).exists():
                continue
            rows = read_csv(run / filename)
            for stage in ('deterministic', 'ae', 'flow'):
                selected = [r for r in rows if r.get('stage', 'deterministic') == stage or
                            stage == 'deterministic' and r.get('stage') in ('field', 'physics')]
                if selected:
                    logs.append((index, run.name, split, stage, selected))
    def present(rows, key):
        return [r for r in rows if r.get(key) not in (None, '') and np.isfinite(float(r[key]))]
    def draw(ax, train_key, validation_key, quantity='', color_offset=0, colors_per_run=1):
        for index, name, split, stage, records in logs:
            key = train_key if split == 'train' else validation_key
            rows = present(records, key)
            if not rows or key == 'kinematic' and not any(float(r[key]) for r in rows):
                continue
            stage_label = '' if stage == 'deterministic' else ' AE' if stage == 'ae' else ' flow sample mean'
            style = '-' if split == 'train' else ':' if stage == 'flow' else '--'
            ax.plot([int(r['update']) for r in rows], [float(r[key]) for r in rows], style,
                    color=f'C{(colors_per_run * index + color_offset) % 10}', marker='o' if len(rows) == 1 else None,
                    label=f'{name} {split}{stage_label}{quantity}')
    def finish(ax, title, ylabel):
        ax.set_title(title); ax.set_xlabel('optimizer update'); ax.set_ylabel(ylabel)
        ax.set_yscale('symlog', linthresh=1e-5); ax.grid(alpha=.2)
        if not ax.lines:
            ax.text(.5, .5, 'Diagnostics not recorded', transform=ax.transAxes, ha='center')
        stage_lines([ax], runs)
        if ax.lines:
            ax.legend(fontsize=7, ncol=2 if len(ax.lines) > 6 else 1)
    pdes = [pde for pde in PDES if any(present(rows, f'field_{pde}' if split == 'train' else f'validation_{pde}')
                                     for _, _, split, _, rows in logs)]
    if pdes:
        columns = min(3, len(pdes)); rows = (len(pdes) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(6 * columns, 3.8 * rows))
        for ax, pde in zip(axes.flat, pdes):
            draw(ax, f'field_{pde}', f'validation_{pde}')
            finish(ax, pde, 'normalized field MSE')
        for ax in list(axes.flat)[len(pdes):]:
            ax.axis('off')
        fig.tight_layout(); fig.savefig(out / 'losses_by_pde.png', dpi=150); plt.close(fig)
    waves = [pde for pde in WAVES if pde in pdes]
    if waves:
        fig, axes = plt.subplots(1, len(waves) + 1, squeeze=False, figsize=(7 * (len(waves) + 1), 4))
        for ax, pde in zip(axes[0], waves):
            for index, quantity in enumerate(('u', 'v')):
                draw(ax, f'field_{pde}_{quantity}', f'validation_{pde}_{quantity}', f' {quantity}', index, 2)
            finish(ax, f'{pde}: displacement u / velocity v', 'normalized reconstruction MSE')
        draw(axes[0, -1], 'kinematic', 'kinematic')
        finish(axes[0, -1], 'Kinematic consistency (separate loss)', 'unweighted kinematic loss')
        fig.tight_layout(); fig.savefig(out / 'wave_losses.png', dpi=150); plt.close(fig)


def reconstruction_plots(directory, out, maximum):
    settings_path = directory / 'evaluation.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    filtered = settings.get('filtered', False)
    selected = {}
    for path in sorted(directory.glob('*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
        key = ((meta['case'], meta['mode'], meta['method']) if filtered else
               (meta['pde'], meta['geometry'], meta['method']))
        score = (meta['mode'] == 'inverse', meta['count'])
        if key not in selected or score > selected[key][0]:
            selected[key] = score, path
    for _, path in list(selected.values())[:maximum]:
        with np.load(path, allow_pickle=False) as saved:
            case = {k: saved[k] for k in saved.files}
        meta = json.loads(str(case['metadata']))
        case_label = f'{meta["case"]} / ' if filtered else ''
        times = case['times']; maximum_frames = 5 if filtered or meta['method'] == 'E-Flow' or meta['method'].startswith('F-') or meta.get('initial_aware_evolution') or meta.get('initial_latent') or meta.get('assimilation') or meta.get('infer_operator') else 4
        indices = np.unique(np.linspace(0, len(times) - 1, min(maximum_frames, len(times))).round().astype(int))
        for channel in range(2 if meta['pde'] in WAVES else 1):
            truth, prediction = case['reference'][:, :, channel], case['prediction'][:, :, channel]
            low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
            fig, axes = plt.subplots(len(indices), 3, squeeze=False, figsize=(12, 3.4 * len(indices)))
            for row, index in enumerate(indices):
                t = times[index]
                for col, (values, title) in enumerate(((truth[index], 'Ground truth'), (prediction[index], meta['method']),
                                                       (abs(prediction[index] - truth[index]), 'Absolute error'))):
                    limits = (0., np.max(abs(prediction - truth))) if col == 2 else (low, high)
                    if limits[0] == limits[1]:
                        limits = (limits[0], limits[1] + 1e-7)
                    field_panel(axes[row, col], case, values, f'{title}, t={t:.3f}', limits,
                                'magma' if col == 2 else 'viridis')
                    obs = case['observation_queries']
                    keep = (case['observation_quantity'] == channel) & (abs(obs[:, 2] - t) <= (.16 if len(times) > 1 else 1e-6))
                    axes[row, col].scatter(*obs[keep, :2].T, facecolors='none', edgecolors='white', linewidths=.8, s=22)
            quantity = 'u' if channel == 0 else 'v'
            fig.suptitle(f'{case_label}{meta["pde"]} / {meta["geometry"]} / {meta["mode"]} / n={meta["count"]} / {quantity}')
            fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out / f'{path.stem}_{quantity}.png', dpi=140); plt.close(fig)
        if meta['method'] == 'E-Geo' and meta['pde'] == 'wave':
            index = len(times) // 2
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for channel, quantity in enumerate(('displacement u', 'velocity v')):
                truth = case['reference'][index, :, channel]; prediction = case['prediction'][index, :, channel]
                limits = (min(truth.min(), prediction.min()), max(truth.max(), prediction.max()))
                for col, (values, title) in enumerate(((truth, 'FEM reference'), (prediction, 'E-Geo'),
                                                       (abs(prediction - truth), 'Absolute error'))):
                    field_panel(axes[channel, col], case, values, f'{title}: {quantity}',
                                limits if col < 2 else None, 'magma' if col == 2 else 'viridis')
            branch = (f'{meta["wave_modes"]} FEM modes + anchored residual' if meta.get('modal_wave') else
                      'initial-aware evolution' if meta.get('initial_aware_wave') else 'direct decoder')
            fig.suptitle(f'{case_label}E-Geo / {meta["mode"]} / t={times[index]:.3f} / {branch}')
            fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out / f'{path.stem}_e_wave.png', dpi=140); plt.close(fig)


def evaluation_directory(run):
    """Most recent default full/selected evaluation, or an explicit result directory."""
    candidates = [run, run / 'evaluation'] + list((run / 'evaluations').glob('selection_*'))
    candidates = [path for path in candidates if (path / 'evaluation.json').exists()]
    if candidates:
        return max(candidates, key=lambda path: (path / 'evaluation.json').stat().st_mtime_ns)
    return run / 'evaluation' if (run / 'evaluation').is_dir() else run


def comparison_plots(paths, out, metric='rmse'):
    records = []
    for path in paths:
        if path.is_dir():
            path = evaluation_directory(path) / 'metrics.csv'
        if path.exists():
            label = path.parent.parent.name if path.parent.name == 'evaluation' else path.parent.name
            if path.parent.parent.name == 'evaluations' and path.parent.name.startswith('selection_'):
                label = path.parent.parent.parent.name
            settings_path = path.parent / 'evaluation.json'
            settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
            rows = read_csv(path); times = {}
            for row in rows:
                if row['time'] != 'all':
                    times.setdefault(row['case'], set()).add(float(row['time']))
            for row in rows:
                row['reference_times'] = row.get('reference_times', ','.join(f'{t:.12g}' for t in sorted(times.get(row['case'], []))))
                row['observation_noise'] = str(row.get('observation_noise', settings.get('noise', .01)))
                if len(paths) > 1:
                    row['method'] = f'{label}/{row["method"]}'
                records.append(row)
    for grid, refine in sorted({(r.get('grid', '0'), r.get('refine', '0')) for r in records}):
        selected = [r for r in records if r.get('grid', '0') == grid and r.get('refine', '0') == refine]
        suffix = f'_g{grid}_ref{refine}' if grid != '0' or refine != '0' else ''
        comparison_panels(selected, out, suffix, metric)


def comparison_case(row):
    return tuple(row.get(k, '') for k in ('case', 'seed', 'tier', 'reference_times', 'observation_noise'))


def comparison_panels(records, out, suffix='', metric='rmse'):
    error_label = 'Normalized RMSE' if metric == 'normalized_rmse' else 'RMSE'
    if metric != 'rmse':
        suffix += f'_{metric}'
    all_rows = [r for r in records if r['time'] == 'all']
    if not all_rows:
        print('No evaluation metrics found for comparisons'); return
    # Keep modes and output quantities separate so comparisons remain interpretable.
    for mode in sorted({r['mode'] for r in all_rows}):
        for quantity in sorted({r['quantity'] for r in all_rows}):
            subset = [r for r in all_rows if r['mode'] == mode and r['quantity'] == quantity]
            if not subset:
                continue
            methods = sorted({r['method'] for r in subset})
            common_counts = set.intersection(*[{int(r['observations']) for r in subset if r['method'] == m} for m in methods])
            shared = []
            for count in common_counts:
                rows = [r for r in subset if int(r['observations']) == count]
                common_cases = set.intersection(*[{comparison_case(r) for r in rows if r['method'] == m} for m in methods])
                shared.extend(r for r in rows if comparison_case(r) in common_cases)
            subset = shared
            if not subset:
                print(f'No shared cases/counts for {mode}/{quantity}; comparison omitted')
                continue
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for method in sorted({r['method'] for r in subset}):
                rows = [r for r in subset if r['method'] == method]
                counts = sorted({int(r['observations']) for r in rows})
                axes[0].plot(counts, [np.mean([float(r[metric]) for r in rows if int(r['observations']) == n]) for n in counts], marker='o', label=method)
                maximum = max(counts); rows = [r for r in rows if int(r['observations']) == maximum]
                for ax, key, panel_metric in ((axes[1], 'pde', 'relative_l2'), (axes[2], 'geometry', metric)):
                    labels = sorted({r[key] for r in subset})
                    ax.plot(labels, [np.mean([float(r[panel_metric]) for r in rows if r[key] == label]) if any(r[key] == label for r in rows) else np.nan for label in labels], marker='o', label=method)
            for ax, xlabel, ylabel in zip(axes, ('observations', 'PDE (largest n)', 'geometry (largest n)'),
                                          (error_label, 'relative L2', error_label)):
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=.2)
            fig.suptitle(f'{mode}, {quantity}{suffix}; mean per-case errors on shared cases/counts')
            fig.tight_layout(); fig.savefig(out / f'comparisons_{mode}_{quantity}{suffix}.png', dpi=150); plt.close(fig)
            shared_keys = {(comparison_case(r), r['method'], r['observations']) for r in subset}
            timed = [r for r in records if r['time'] != 'all' and r['mode'] == mode and r['quantity'] == quantity
                     and r['pde'] not in STATIONARY
                     and (comparison_case(r), r['method'], r['observations']) in shared_keys]
            if not timed:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for method, pde in sorted({(r['method'], r['pde']) for r in timed}):
                rows = [r for r in timed if r['method'] == method and r['pde'] == pde]
                maximum = max(int(r['observations']) for r in rows)
                rows = [r for r in rows if int(r['observations']) == maximum]
                times = sorted({float(r['time']) for r in rows})
                ax.plot(times, [np.mean([float(r[metric]) for r in rows if float(r['time']) == t]) for t in times], marker='o', label=f'{method} / {pde}')
            ax.set_xlabel('physical time'); ax.set_ylabel(error_label); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(out / f'error_time_{mode}_{quantity}{suffix}.png', dpi=150); plt.close(fig)
    coefficient_rows = [r for r in all_rows if r['quantity'] == 'u' and np.isfinite(float(r['coefficient_prediction']))]
    if coefficient_rows:
        pdes = sorted({r['pde'] for r in coefficient_rows})
        fig, axes = plt.subplots(1, len(pdes), figsize=(5 * len(pdes), 4.5), squeeze=False)
        for ax, pde in zip(axes[0], pdes):
            for method in sorted({r['method'] for r in coefficient_rows}):
                rows = [r for r in coefficient_rows if r['method'] == method and r['pde'] == pde]
                if rows:
                    ax.scatter([float(r['coefficient_true']) for r in rows], [float(r['coefficient_prediction']) for r in rows], s=18, alpha=.6, label=method)
            limits = ax.get_xlim(); ax.plot(limits, limits, 'k--'); ax.set_title(pde)
            ax.set_xlabel('true coefficient'); ax.set_ylabel('predicted coefficient'); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f'coefficient_scatter{suffix}.png', dpi=150); plt.close(fig)


def plot(args):
    if args.runs:
        from .comparison import compare_runs
        compare_runs(args)
        return
    if not args.run:
        raise ValueError('plot requires --run or --runs')
    run = Path(args.run); directory = evaluation_directory(run)
    settings_path = directory / 'evaluation.json'
    filtered = json.loads(settings_path.read_text()).get('filtered', False) if settings_path.exists() else False
    maximum = args.max_fields if args.max_fields is not None else (None if filtered else 9)
    out = Path(args.out) if args.out else (directory if filtered else run) / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    for what in args.what.split(','):
        if what == 'losses':
            loss_plots(run, out)
        elif what in ('fields', 'uncertainty'):
            if what == 'fields':
                reconstruction_plots(directory, out, maximum)
            from .comparison import load_field, uncertainty_plot
            for path in sorted(directory.glob('*.npz'))[:maximum]:
                uncertainty_plot(load_field(path), out, path.stem)
        elif what == 'comparisons':
            comparison_plots([run] + [Path(p) for p in args.compare], out, args.metric)
        else:
            raise ValueError('plot --what accepts losses,fields,uncertainty,comparisons')
    print(f'Figures: {out}')
