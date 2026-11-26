"""Measure F-Iterative correction stages using saved evaluation sensors and masks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_observations import evaluation_settings
from pdeep_simple.data import load_case, model_input, MODES
from pdeep_simple.geometry import node_weights, interpolate
from pdeep_simple.physics import WAVES
from pdeep_simple.training import load_checkpoint, model_from_config, select_device
from pdeep_simple.evaluation import prediction_values, error_metrics, write_rows
from pdeep_simple.wave import observed_values
from pdeep_simple.plotting import field_panel


TIMES = np.array([0., .25, .5, .75, 1.])


def saved_measurements(directory, case_path, case, mode, count, settings):
    paths = sorted(directory.glob(f'{case_path.stem}_{mode}_n{count}_*.npz'))
    if not paths:
        raise ValueError(f'No saved sensors for {mode}, n={count} in {directory}; choose --evaluation. No sensors were sampled.')
    # Prefer a nodal file so its saved FEM reference can be reused directly.
    selected = None
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
            if selected is None or meta.get('grid', 0) == 0:
                selected = path
            if meta.get('grid', 0) == 0:
                break
    with np.load(selected, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata']))
        if meta['seed'] != case['meta']['seed'] or not np.array_equal(saved['nodes'], case['nodes']):
            raise ValueError('Saved evaluation and dataset case/mesh differ')
        obs = {key: saved[name].copy() for key, name in (
            ('queries', 'observation_queries'), ('quantity', 'observation_quantity'),
            ('values', 'observation_values'), ('noise', 'observation_noise'))}
        masks = saved['known_field_mask'].copy(), saved['known_scalar_mask'].copy()
        nodal = (np.array_equal(saved['query_points'], case['nodes']) and np.array_equal(saved['times'], TIMES))
        reference = saved['reference'].copy() if nodal else None
    if reference is None:
        if settings.get('refine', 0):
            raise ValueError('This refined evaluation needs a saved nodal FEM reference at the five requested times')
        coords = np.concatenate([np.column_stack([case['nodes'], np.full(len(case['nodes']), t)]) for t in TIMES])
        reference = interpolate(case, coords).reshape(5, -1, 2)
    return obs, masks, reference, selected


def rmse(prediction, reference, weights):
    return error_metrics(prediction, reference, weights, 1.)['rmse'] if len(reference) else float('nan')


def range_text(values):
    if not len(values):
        return 'n/a'
    return f'{np.min(values):.8g}' if np.min(values) == np.max(values) else f'{np.min(values):.8g}..{np.max(values):.8g}'


def normalization_report(scales, obs, channels):
    print('\nCorrection normalization (reported RMSE below is always in physical units):')
    print('  Residual = predicted_normalized - measurement/output_scale; sigma_normalized = noise_sigma/output_scale.')
    for channel, quantity in enumerate(channels):
        sigma = obs['noise'][obs['quantity'] == channel]; scale = scales['output'][channel]
        effective = np.sqrt(sigma ** 2 + 1e-6 * scale ** 2)
        print(f'  {quantity}: output_scale={scale:.9g}, physical sigma={range_text(sigma)}, '
              f'normalized sigma={range_text(sigma / scale)}, effective physical sigma={range_text(effective)}')
    print('  Data term: sum(residual_normalized^2 / (sigma_normalized^2 + 1e-6)).')
    print('  Equivalently: sum(residual_physical^2 / (noise_sigma^2 + 1e-6*output_scale^2)).')
    print('  Velocity has no extra amplitude multiplier; counts, physical errors and decoder sensitivities still affect its influence.')
    print('  Control: unit squared distance to the initial latent/log-parameter prior; gradient direction = .25*g/max(RMS(g),1).')
    print('  Backtracking tries factors 1, .5, .25 and stops if none lowers the regularized objective.')
    print('  When inferred, the log-operator direction is additionally clipped to +/- .25. Known inputs remain masked/fixed.')


def show_table(rows, columns):
    labels = [label for _, label in columns]
    values = [[f'{r[key]:.7g}' if isinstance(r[key], (float, np.floating)) else str(r[key]) for key, _ in columns] for r in rows]
    widths = [max(len(label), *(len(row[i]) for row in values)) for i, label in enumerate(labels)]
    print(' | '.join(label.ljust(width) for label, width in zip(labels, widths)))
    print('-+-'.join('-' * width for width in widths))
    for row in values:
        print(' | '.join(value.ljust(width) for value, width in zip(row, widths)))


def plots(case, mode, channels, metrics, initial, reference, out):
    fig, axes = plt.subplots(len(channels), 3, squeeze=False, figsize=(15, 3.5 * len(channels)))
    for channel, quantity in enumerate(channels):
        for ax, category, title in zip(axes[channel], ('sensor', 'field', 'initial'), ('Sensors', 'Whole field: mean over five times', 'Initial state')):
            ax.plot([r['stage'] for r in metrics], [r[f'{category}_{quantity}_rmse'] for r in metrics], 'o-', color=f'C{channel}')
            ax.set(title=f'{title}: {quantity}', xlabel='Correction stage', ylabel='RMSE (physical units)', xticks=[r['stage'] for r in metrics])
            ax.grid(alpha=.2)
    fig.suptitle(f'{mode.title()}: F-Iterative correction')
    fig.tight_layout(); fig.savefig(out / f'{mode}_correction_rmse.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(len(channels), 4, squeeze=False, figsize=(15, 3.8 * len(channels)))
    for channel, quantity in enumerate(channels):
        truth, before, after = reference[0, :, channel], initial[0, :, channel], initial[-1, :, channel]
        limits = (min(truth.min(), before.min(), after.min()), max(truth.max(), before.max(), after.max()))
        if limits[0] == limits[1]:
            limits = (limits[0], limits[1] + 1e-8)
        error = np.abs(after - truth)
        panels = [(truth, 'FEM reference'), (before, 'Stage 0'), (after, f'Stage {metrics[-1]["stage"]}'), (error, '|Final - FEM|')]
        for column, (values, label) in enumerate(panels):
            field_panel(axes[channel, column], case, values, f'{quantity}0: {label}',
                        (0., max(float(error.max()), 1e-8)) if column == 3 else limits,
                        'magma' if column == 3 else 'viridis')
    fig.suptitle(f'{mode.title()}: initial-state reconstruction at physical t=0')
    fig.tight_layout(); fig.savefig(out / f'{mode}_initial_reconstruction.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True); parser.add_argument('--run', required=True)
    parser.add_argument('--case', required=True); parser.add_argument('--observations', type=int, default=128)
    parser.add_argument('--modes', default='forward,hybrid,inverse')
    parser.add_argument('--evaluation', help='Specific evaluation folder/evaluation.json; defaults to latest matching selection')
    parser.add_argument('--checkpoint', choices=['best.pt', 'last.pt'], help='Default: checkpoint named by saved evaluation')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--chunk', type=int, default=256); parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', help='Default: RUN/correction_diagnostics/CASE_nCOUNT')
    args = parser.parse_args(); modes = args.modes.split(',')
    if args.observations < 0 or args.chunk < 1 or args.threads < 1 or any(m not in MODES for m in modes):
        parser.error('valid modes, nonnegative observations and positive chunk/threads required')
    run = Path(args.run); case_path = Path(args.data) / Path(args.case).with_suffix('.npz'); case = load_case(case_path)
    directory, settings = evaluation_settings(run, case_path.stem, args.observations, args.evaluation)
    if not settings:
        parser.error('saved evaluation.json is required; use --evaluation')
    device = select_device(args.device); torch.set_num_threads(args.threads)
    checkpoint_path = run / (args.checkpoint or settings.get('checkpoint', 'best.pt'))
    checkpoint = load_checkpoint(checkpoint_path); config = dict(checkpoint['config'])
    for key in ('assimilation', 'infer_operator'):
        if settings.get(key) is not None:
            config[key] = settings[key]
    if config['model'] != 'F' or config.get('f_variant') != 'iterative' or config.get('flow'):
        parser.error('this utility inspects a deterministic F-Iterative checkpoint')
    if case['meta']['pde'] not in ('heat', 'wave', 'damped_wave', 'reaction_diffusion'):
        parser.error('select a supported time-dependent PDE')
    model = model_from_config(config, device); model.load_state_dict(checkpoint['model']); model.eval()
    model.requires_grad_(False)  # State gradients still run inside refine; no parameter gradients are needed.
    scales = checkpoint['scales']; channels = ('u', 'v') if case['meta']['pde'] in WAVES else ('u',)
    seed = int(np.random.SeedSequence([settings['seed'], case['meta']['seed']]).generate_state(1)[0])
    coords = np.concatenate([np.column_stack([case['nodes'], np.full(len(case['nodes']), t)]) for t in TIMES])
    weights = node_weights(case)
    out = Path(args.out) if args.out else run / 'correction_diagnostics' / f'{case_path.stem}_n{args.observations}'
    out.mkdir(parents=True, exist_ok=True); time_rows = []
    print(f'Checkpoint: {checkpoint_path}, update {checkpoint["update"]}; device={device}')
    print(f'Evaluation: {directory}; stages 0..{model.f_steps}; correction enabled={model.assimilation}')
    print('Field RMSE = arithmetic mean of five spatial FEM-quadrature RMSEs at t=0,.25,.5,.75,1.')
    print('Early termination repeats the final state; repeated stages are explicitly marked unchanged.')
    for mode in modes:
        obs, masks, reference, source = saved_measurements(directory, case_path, case, mode, args.observations, settings)
        inputs = model_input(case, mode, obs, scales, np.random.default_rng(seed), device)
        if not np.array_equal(inputs['field_masks'].cpu().numpy(), masks[0]) or not np.array_equal(inputs['scalar_masks'].cpu().numpy(), masks[1]):
            raise ValueError('Reproduced conditioning differs from saved masks; use matching data/evaluation settings')
        if mode == modes[0]:
            normalization_report(scales, obs, channels)
        print(f'\n{mode.upper()} — saved sensors/masks: {source}', flush=True)
        with torch.no_grad():
            prior = model.encode(inputs, correct=False)
            final, stages = model.refine(prior, return_stages=True) if model.assimilation else (prior, [prior])
            accepted = len(stages) - 1
            print(f'Accepted {accepted}/{model.f_steps} corrections; noise_sigma={range_text(obs["noise"])} physical.', flush=True)
            metrics, splits, initial = [], [], []
            for stage in range(model.f_steps + 1):
                context = stages[stage] if stage < len(stages) else final
                field, _ = prediction_values(model, context, case, coords, scales, args.chunk)
                field = field.reshape(5, -1, 2); initial.append(field[0])
                sensors = observed_values(model, context).cpu().numpy() * np.asarray(scales['output'])[obs['quantity']] if len(obs['values']) else np.empty(0)
                label = 'Before correction' if stage == 0 else f'After step {stage}'
                note = 'initial' if stage == 0 else 'accepted' if stage <= accepted else 'unchanged (stopped/skipped)'
                row = dict(stage=stage, label=label, status=note, accepted_steps=min(stage, accepted))
                split = dict(stage=stage, label=label)
                errors = np.empty((5, len(channels)))
                for channel, quantity in enumerate(channels):
                    keep = obs['quantity'] == channel
                    row[f'sensor_{quantity}_rmse'] = rmse(sensors[keep], obs['values'][keep], np.ones(keep.sum()))
                    row[f'noise_{quantity}_sigma_min'] = float(obs['noise'][keep].min()) if keep.any() else float('nan')
                    row[f'noise_{quantity}_sigma_max'] = float(obs['noise'][keep].max()) if keep.any() else float('nan')
                    for it in range(5):
                        errors[it, channel] = rmse(field[it, :, channel], reference[it, :, channel], weights)
                    row[f'field_{quantity}_rmse'] = float(errors[:, channel].mean())
                    row[f'initial_{quantity}_rmse'] = float(errors[0, channel])
                    for part, mask in (('known', masks[0][:, channel + 1].astype(bool)), ('hidden', ~masks[0][:, channel + 1].astype(bool))):
                        split[f'{quantity}0_{part}_rmse'] = rmse(field[0, mask, channel], reference[0, mask, channel], weights[mask])
                metrics.append(row); splits.append(split)
                time_rows.extend(dict(mode=mode, stage=stage, time=float(t), **{f'field_{q}_rmse': float(errors[it, c]) for c, q in enumerate(channels)}) for it, t in enumerate(TIMES))
        columns = [('label', 'Stage')] + [(f'{kind}_{q}_rmse', f'{kind.title()} {q} RMSE') for kind in ('sensor', 'field', 'initial') for q in channels] + [('status', 'Update')]
        show_table(metrics, columns); write_rows(out / f'{mode}_metrics.csv', metrics)
        if mode == 'hybrid':
            print('Hybrid initial-state split (same FEM quadrature restricted to each saved mask):')
            print(', '.join(f'{q}0: {np.count_nonzero(masks[0][:, c + 1])} known / {np.count_nonzero(~masks[0][:, c + 1].astype(bool))} hidden' for c, q in enumerate(channels)))
            show_table(splits, [('label', 'Stage')] + [(f'{q}0_{part}_rmse', f'{q}0 {part} RMSE') for q in channels for part in ('known', 'hidden')])
            write_rows(out / 'hybrid_initial_splits.csv', splits)
        plots(case, mode, channels, metrics, np.stack(initial), reference, out)
    write_rows(out / 'field_by_time.csv', time_rows)
    print(f'\nCSV tables and two PNG plot types per mode: {out}')


if __name__ == '__main__':
    main()
