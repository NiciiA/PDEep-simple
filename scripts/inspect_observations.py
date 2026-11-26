"""Inspect saved evaluation sensors, or reproduce their exact evaluation draw."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdeep_simple.cli import parser as project_parser
from pdeep_simple.data import load_case, observations, known_inputs, MODES
from pdeep_simple.physics import WAVES, PARABOLIC
from pdeep_simple.plotting import evaluation_directory


def evaluation_settings(run, case_name, count, selected):
    if selected:
        path = Path(selected)
        directory = path.parent if path.name == 'evaluation.json' else path
        return directory, json.loads((directory / 'evaluation.json').read_text())
    # Prefer the latest selection that actually includes this case and sensor count.
    candidates = [run, run / 'evaluation', *list((run / 'evaluations').glob('selection_*'))]
    candidates = sorted([p for p in candidates if (p / 'evaluation.json').exists()],
                        key=lambda p: (p / 'evaluation.json').stat().st_mtime_ns, reverse=True)
    for directory in candidates:
        settings = json.loads((directory / 'evaluation.json').read_text())
        names = [Path(p).stem for p in settings.get('cases', [])]
        if count in settings.get('observations', []) and (not names or case_name in names):
            return directory, settings
    directory = evaluation_directory(run)
    path = directory / 'evaluation.json'
    return directory, json.loads(path.read_text()) if path.exists() else {}


def sensor_data(directory, case_path, case, mode, count, seed, noise):
    files = sorted(directory.glob(f'{case_path.stem}_{mode}_n{count}_*.npz'))
    if files:
        path = files[0]  # Different query grids/methods share the same sensors.
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
            if meta['seed'] != case['meta']['seed'] or not np.array_equal(saved['nodes'], case['nodes']):
                raise ValueError(f'{path} belongs to a different case/mesh; choose matching --data/--evaluation')
            obs = {key: saved[name].copy() for key, name in (
                ('queries', 'observation_queries'), ('quantity', 'observation_quantity'),
                ('values', 'observation_values'), ('noise', 'observation_noise'))}
            field_mask = saved['known_field_mask'].copy() if 'known_field_mask' in saved else None
            scalar_mask = saved['known_scalar_mask'].copy() if 'known_scalar_mask' in saved else None
        source = f'saved sensors: {path}'
    else:
        # Same two RNGs as evaluation.evaluate_case: sensors and masks use distinct seeds.
        obs = observations(case, count, noise, np.random.default_rng(seed + count))
        field_mask = scalar_mask = None
        source = f'reproduced sensors: observation RNG seed {seed + count}'
    mask_source = 'saved' if field_mask is not None and scalar_mask is not None else 'reproduced'
    if field_mask is None or scalar_mask is None:
        _, fields, _, scalars = known_inputs(case, mode, np.random.default_rng(seed))
        field_mask = fields if field_mask is None else field_mask
        scalar_mask = scalars if scalar_mask is None else scalar_mask
    return obs, field_mask, scalar_mask, source, mask_source


def status(mask):
    known = int(np.count_nonzero(mask)); total = np.size(mask)
    label = 'hidden' if known == 0 else 'known' if known == total else 'partially known'
    return f'{label} ({known}/{total})'


def print_summary(case, mode, obs, fields, scalars, source, mask_source, quantities, noise):
    pde = case['meta']['pde']; times = obs['queries'][:, 2]
    print(f'\n{mode.upper()} — {source}')
    print(f'  Sensor observations actually passed: {len(times)} (padding/null tokens excluded)')
    for quantity, name in quantities.items():
        print(f'    quantity ID {quantity}, {name}: {np.count_nonzero(obs["quantity"] == quantity)}')
    print(f'    exactly t=0: {np.count_nonzero(times == 0.)}')
    for end in (.05, .10, .25):
        print(f'    t <= {end:.2f}: {np.count_nonzero(times <= end)}')
    if len(case['times']) > 1:
        edges = np.linspace(case['times'][0], case['times'][-1], 5)
        for start, end in zip(edges[1:-1], edges[2:]):
            print(f'    {start:.2f} < t <= {end:.2f}: {np.count_nonzero((times > start) & (times <= end))}')
    else:
        print('    stationary PDE: no evolution interval')
    print(f'    physical time min/max: {times.min():.9g}, {times.max():.9g}' if len(times) else '    physical time min/max: n/a')
    sigma = np.unique(obs['noise'])
    print('    supplied noise_sigma (physical units): ' + (', '.join(f'{s:.9g}' for s in sigma) if len(sigma) else f'n/a; configured sigma={noise:g}'))
    print(f'  Dense known inputs, separate from sensors ({mask_source} masks; field counts are mesh nodes):')
    if pde != 'eikonal':
        print(f'    forcing/source f: {status(fields[:, 0])}')
    if pde in WAVES:
        print(f'    initial displacement u0: {status(fields[:, 1])}')
        print(f'    initial velocity v0: {status(fields[:, 2])}')
    elif pde in PARABOLIC:
        label = 'temperature' if pde == 'heat' else 'solution'
        print(f'    initial {label} u0: {status(fields[:, 1])}')
    else:
        print('    initial conditions: not applicable (stationary PDE)')
    coefficient = ('wave speed c' if pde in WAVES else 'diffusivity D' if pde in PARABOLIC or pde == 'elliptic_adr'
                   else 'speed' if pde == 'eikonal' else 'conductivity')
    print(f'    {coefficient} field: {status(fields[:, 3])}')
    value = f'; value={case["meta"]["coefficient"]:.9g}' if scalars[0] else ''
    print(f'    {coefficient} scalar: {status(scalars[:1])}{value}')
    if pde == 'semilinear':
        print(f'    cubic coefficient beta: {status(scalars[1:2])}')
    parameters = case['meta'].get('parameters', {})
    keys = (['forcing_frequency'] if pde in PARABOLIC + WAVES else [])
    keys += {'damped_wave': ['damping'], 'reaction_diffusion': ['reaction'],
             'elliptic_adr': ['advection', 'reaction'], 'p_laplacian': ['p']}.get(pde, [])
    for key in keys:
        default = {'p': 3., 'advection': [0., 0.]}.get(key, 0.)
        print(f'    {key}: known, {parameters.get(key, default)} (supplied PDE parameter)')
    print('  First 10 sensor rows; values/noise are in physical units, t is PDE time:')
    print(f'    {"x":>12} {"y":>12} {"t":>12} {"quantity ID / field":<27} {"value":>14} {"noise_sigma":>12}')
    for xyz, quantity, value, sigma in zip(obs['queries'][:10], obs['quantity'][:10], obs['values'][:10], obs['noise'][:10]):
        label = f'{quantity}: {quantities[int(quantity)]}'
        print(f'    {xyz[0]:12.8g} {xyz[1]:12.8g} {xyz[2]:12.8g} {label:<27} {value:14.9g} {sigma:12.8g}')


def make_plots(case, records, quantities, out, case_name):
    start, end = float(case['times'][0]), float(case['times'][-1])
    limits = (start, end) if end > start else (start - .05, start + .05)
    bins = np.linspace(*limits, 21)
    colors = ['C0', 'C1'][:len(quantities)]
    fig, axes = plt.subplots(1, len(records), squeeze=False, sharex=True, sharey=True,
                             figsize=(5 * len(records), 4))
    for ax, (mode, obs) in zip(axes[0], records.items()):
        times = [obs['queries'][obs['quantity'] == q, 2] for q in quantities]
        ax.hist(times, bins=bins, stacked=True, color=colors, label=list(quantities.values()))
        ax.set(title=f'{mode.title()} — {len(obs["values"])} sensors', xlabel='Physical PDE time t', xlim=limits)
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Number of sensor observations')
    fig.suptitle(f'{case_name}: observation times')
    fig.tight_layout(); fig.savefig(out / 'observation_times.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(len(records), 2, squeeze=False, sharex=True, figsize=(11, 3.3 * len(records)))
    for row, (mode, obs) in enumerate(records.items()):
        for column, coordinate in enumerate(('x', 'y')):
            ax = axes[row, column]
            for color, (quantity, label) in zip(colors, quantities.items()):
                xyz = obs['queries'][obs['quantity'] == quantity]
                ax.scatter(xyz[:, 2], xyz[:, column], color=color, s=20, alpha=.7, label=label)
            ax.set(title=f'{mode.title()}: time versus {coordinate}', xlabel='Physical PDE time t',
                   ylabel=f'{coordinate}-coordinate (physical)', xlim=limits,
                   ylim=(case['nodes'][:, column].min(), case['nodes'][:, column].max()))
            ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f'{case_name}: sparse space-time sensors (not dense initial fields)')
    fig.tight_layout(); fig.savefig(out / 'space_time_observations.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--case', required=True, help='Dataset case stem or .npz filename')
    parser.add_argument('--observations', type=int, default=128)
    parser.add_argument('--modes', default='forward,hybrid,inverse')
    parser.add_argument('--evaluation', help='Specific evaluation folder or evaluation.json; otherwise latest matching selection')
    parser.add_argument('--seed', type=int, help='Evaluation seed, only overriding defaults when no saved settings exist')
    parser.add_argument('--noise', type=float, help='Physical noise sigma, only overriding defaults when no saved settings exist')
    parser.add_argument('--out', help='PNG folder; default EVALUATION/observation_inspection/CASE_nCOUNT')
    args = parser.parse_args()
    modes = args.modes.split(',')
    if args.observations < 0 or any(mode not in MODES for mode in modes):
        parser.error('nonnegative observations and forward/hybrid/inverse modes required')
    run = Path(args.run); case_path = Path(args.data) / Path(args.case).with_suffix('.npz')
    case = load_case(case_path)
    directory, settings = evaluation_settings(run, case_path.stem, args.observations, args.evaluation)
    defaults = project_parser().parse_args(['eval', '--data', args.data, '--run', args.run])
    seed = settings.get('seed', defaults.seed if args.seed is None else args.seed)
    noise = settings.get('noise', defaults.noise if args.noise is None else args.noise)
    if (args.seed is not None and args.seed != seed) or (args.noise is not None and args.noise != noise):
        parser.error('--seed/--noise conflict with saved evaluation settings; select the intended --evaluation')
    if noise < 0:
        parser.error('noise must be nonnegative')
    # This is the case-specific seed calculation in evaluation.evaluate_case, not the training seed.
    draw_seed = int(np.random.SeedSequence([seed, case['meta']['seed']]).generate_state(1)[0])
    print(f'Case: {case_path}; PDE={case["meta"]["pde"]}; tier={case["meta"]["tier"]}')
    print(f'Evaluation settings: {directory / "evaluation.json"}' if settings else 'No saved evaluation settings; reproducing eval CLI defaults/explicit overrides.')
    print(f'Evaluation seed={seed}; case seed={case["meta"]["seed"]}; combined draw seed={draw_seed}; noise={noise:g}')
    if settings and (args.observations not in settings.get('observations', []) or
                     settings.get('cases') and case_path.stem not in [Path(p).stem for p in settings['cases']]):
        print('Case/count not recorded in this selection: missing files will be an eval-compatible reproduction, not a completed evaluation.')
    quantities = ({0: 'displacement u', 1: 'velocity v'} if case['meta']['pde'] in WAVES else
                  {0: 'temperature u' if case['meta']['pde'] == 'heat' else 'solution u'})
    records = {}
    for mode in modes:
        obs, fields, scalars, source, mask_source = sensor_data(directory, case_path, case, mode, args.observations, draw_seed, noise)
        if settings and mode not in settings.get('modes', []):
            source += ' (mode not recorded in selection; reproduced with its settings)'
        print_summary(case, mode, obs, fields, scalars, source, mask_source, quantities, noise)
        records[mode] = obs
    if len(records) > 1 and all(all(np.array_equal(obs[k], next(iter(records.values()))[k]) for k in obs) for obs in records.values()):
        print('\nThese modes use identical sparse sensor rows; their known initial/physical-input masks differ.')
    out = Path(args.out) if args.out else directory / 'observation_inspection' / f'{case_path.stem}_n{args.observations}'
    out.mkdir(parents=True, exist_ok=True)
    make_plots(case, records, quantities, out, case_path.stem)
    print(f'\nFigures: {out / "observation_times.png"}\n         {out / "space_time_observations.png"}')


if __name__ == '__main__':
    main()
