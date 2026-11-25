"""Same-case figures and separate deterministic, autoencoder and flow loss curves."""
import json
from pathlib import Path
import numpy as np
from .physics import WAVES, STATIONARY
from .data import PDES
import torch
import matplotlib.pyplot as plt
from .data import case_paths, load_case, normalization
from .training import load_checkpoint, model_from_config, select_device
from .evaluation import evaluate_case, write_rows
from .plotting import field_panel, read_csv, comparison_plots, field_loss_plots, stage_lines
from .baselines import METHODS


def load_field(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {k: saved[k] for k in saved.files}
    result['meta'] = json.loads(str(result['metadata']))
    return result


def sensors(ax, field, time, channel):
    obs = field['observation_queries']
    keep = (field['observation_quantity'] == channel) & (abs(obs[:, 2] - time) <= (.16 if len(field['times']) > 1 else 1e-6))
    ax.scatter(*obs[keep, :2].T, facecolors='none', edgecolors='white', linewidths=.8, s=20)


def uncertainty_plot(field, out, name):
    bayes = 'amplitude_covariance' in field
    gp = field['meta']['method'] == 'gp'
    if not bayes and not gp and not field['meta'].get('samples'):
        return
    count = min(4 if field['meta']['method'] == 'E-Flow' else 2, len(field['sample_fields']))
    for channel in range(2 if field['meta']['pde'] in WAVES else 1):
        truth, mean = field['reference'][:, :, channel], field['prediction'][:, :, channel]
        std = field['sample_std'][:, :, channel]
        limits = (min(truth.min(), mean.min()), max(truth.max(), mean.max()))
        fig, axes = plt.subplots(len(field['times']), 4 + count, squeeze=False,
                                 figsize=(3.2 * (4 + count), 3.2 * len(field['times'])))
        for it, t in enumerate(field['times']):
            panels = [(truth[it], 'FEM reference', limits), (mean[it], 'GP posterior mean' if gp else 'Posterior mean' if bayes else 'Sample mean', limits),
                      (std[it], 'GP posterior std (assumed kernel)' if gp else 'Gaussian posterior std' if bayes else 'Sample std (uncalibrated)',
                       (0., max(float(np.nanmax(std)), 1e-8)) if np.isfinite(std).any() else None),
                      (abs(mean[it] - truth[it]), 'Absolute error of mean', None)]
            panels += [(sample[it, :, channel], f'Sample {k + 1}', limits) for k, sample in enumerate(field['sample_fields'][:count])]
            for ax, (values, label, scale) in zip(axes[it], panels):
                if np.isfinite(values).any():
                    title = label if len(field['times']) == 1 else f'{label}\nphysical t={t:.3f}'
                    field_panel(ax, field, values, title, scale,
                                'magma' if label.startswith(('GP posterior std', 'Gaussian posterior std', 'Sample std', 'Absolute error')) else 'viridis')
                    sensors(ax, field, t, channel)
                else:
                    ax.text(.5, .5, 'Std needs at least two samples', ha='center'); ax.axis('off')
        quantity = 'u' if channel == 0 else 'v'
        description = 'GP kernel uncertainty; calibration not guaranteed' if gp else 'Gaussian posterior uncertainty' if bayes else 'one latent per complete sample'
        fig.suptitle(f'{name} / {field["meta"]["mode"]} / {quantity}; {description}')
        fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out / f'{name}_uncertainty_{quantity}.png', dpi=130); plt.close(fig)


def multiple_loss_plots(runs, out):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for run in runs:
        if not (run / 'train.csv').exists():
            continue
        for filename, style, split in (('train.csv', '-', 'train'), ('validation.csv', '--', 'validation')):
            for stage, column, panel in [('deterministic', 'field', 0), ('ae', 'field', 1), ('flow', 'fm', 2), ('flow', 'field', 3)]:
                if panel == 3 and split != 'validation':
                    continue
                rows = [r for r in read_csv(run / filename) if r.get(column) and
                        (r.get('stage', 'deterministic') == stage or
                         stage == 'deterministic' and r.get('stage') in ('field', 'physics'))]
                if rows:
                    axes[panel].plot([int(r['update']) for r in rows], [float(r[column]) for r in rows],
                                     style, label=f'{run.name} {split}')
    stage_lines([axes[0]], runs)
    titles = ('Deterministic field MSE (normalized)', 'AE field MSE (normalized)', 'Latent flow matching MSE', 'Flow sample-mean MSE (normalized)')
    for ax, title in zip(axes, titles):
        ax.set_title(title); ax.set_xlabel('total update'); ax.set_yscale('symlog', linthresh=1e-5); ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No applicable loss records', transform=ax.transAxes, ha='center')
    fig.tight_layout(); fig.savefig(out / 'loss_comparison.png', dpi=140); plt.close(fig)
    field_loss_plots(runs, out)


def solution_comparison(records, out, mode):
    reference = records[0][1]
    for channel in range(2 if reference['meta']['pde'] in WAVES else 1):
        truth = reference['reference'][:, :, channel]
        predictions = [f['prediction'][:, :, channel] for _, f in records]
        limits = (min([truth.min()] + [p.min() for p in predictions]), max([truth.max()] + [p.max() for p in predictions]))
        max_error = max(float(abs(p - truth).max()) for p in predictions)
        fig, axes = plt.subplots(2 * len(reference['times']), len(records) + 1, squeeze=False,
                                 figsize=(3.3 * (len(records) + 1), 5.7 * len(reference['times'])))
        for it, t in enumerate(reference['times']):
            field_panel(axes[2 * it, 0], reference, truth[it], f'FEM, physical t={t:.3f}', limits)
            sensors(axes[2 * it, 0], reference, t, channel)
            axes[2 * it + 1, 0].axis('off')
            for col, ((label, field), prediction) in enumerate(zip(records, predictions), 1):
                field_panel(axes[2 * it, col], field, prediction[it], label, limits)
                field_panel(axes[2 * it + 1, col], field, abs(prediction[it] - truth[it]), 'Absolute error', (0., max(max_error, 1e-8)), 'magma')
                sensors(axes[2 * it, col], field, t, channel)
        quantity = 'u' if channel == 0 else 'v'
        fig.suptitle(f'Same {reference["meta"]["pde"]}/{reference["meta"]["geometry"]} case / {mode} / {quantity}')
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out / f'solutions_{mode}_{quantity}.png', dpi=130); plt.close(fig)


def conditioning_comparison(label, fields, out):
    anchor = fields[0]; modes = [f['meta']['mode'] for f in fields]
    for channel in range(2 if anchor['meta']['pde'] in WAVES else 1):
        truth = anchor['reference'][:, :, channel]
        predictions = [f['prediction'][:, :, channel] for f in fields]
        limits = (min([truth.min()] + [p.min() for p in predictions]), max([truth.max()] + [p.max() for p in predictions]))
        differences = [p - predictions[0] for p in predictions[1:]]
        bound = max([float(abs(d).max()) for d in differences] + [1e-8])
        fig, axes = plt.subplots(len(anchor['times']), 2 * len(fields), squeeze=False,
                                 figsize=(3.2 * 2 * len(fields), 3.2 * len(anchor['times'])))
        for it, t in enumerate(anchor['times']):
            panels = [(truth[it], f'FEM t={t:.3f}', limits, 'viridis')]
            panels += [(p[it], mode, limits, 'viridis') for p, mode in zip(predictions, modes)]
            panels += [(d[it], f'{mode} - {modes[0]}', (-bound, bound), 'coolwarm') for d, mode in zip(differences, modes[1:])]
            for ax, (value, title, scale, cmap) in zip(axes[it], panels):
                field_panel(ax, anchor, value, title, scale, cmap)
        quantity = 'u' if channel == 0 else 'v'
        fig.suptitle(f'{label} / conditioning comparison / {quantity}')
        fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out / f'{label}_conditioning_{quantity}.png', dpi=130); plt.close(fig)


def compare_runs(args):
    runs = [Path(p) for p in args.runs]
    out = Path(args.out or 'runs/comparison'); out.mkdir(parents=True, exist_ok=True)
    requested = set(args.what.split(','))
    if requested - {'losses', 'fields', 'comparisons', 'conditioning', 'uncertainty'}:
        raise ValueError('plot --what accepts losses,fields,comparisons,conditioning,uncertainty')
    if 'losses' in requested:
        multiple_loss_plots(runs, out)
    if not args.data or not requested.intersection({'fields', 'conditioning', 'uncertainty'}):
        if 'comparisons' in requested:
            comparison_plots(runs, out, args.metric)
        print(f'Loss/metric comparisons: {out}; add --data for same-case fields')
        return
    if args.frames < 1 or args.chunk < 1 or args.samples < 1 or args.flow_steps < 1 or args.save_samples < 0 or not 0 < args.interval < 1:
        raise ValueError('positive frames/chunk/samples/flow-steps and interval in (0,1) required')
    paths = case_paths(args.data, args.split, [args.geometry] if args.geometry else None)
    paths = [p for p in paths if not args.pde or load_case(p)['meta']['pde'] == args.pde]
    if not 0 <= args.case < len(paths):
        raise ValueError('case index out of range for the selected PDE/geometry')
    path = paths[args.case]
    options = dict(frames=args.frames, seed=args.seed, modes=args.modes.split(','), observations=[args.observations],
                   noise=args.noise, grids=[args.grid], refine=0, device=select_device(args.device), chunk=args.chunk,
                   samples=args.samples, flow_steps=args.flow_steps, save_samples=args.save_samples, interval=args.interval)
    if any(m not in ('forward', 'inverse', 'hybrid') for m in options['modes']) or args.noise <= 0 or args.observations < 0:
        raise ValueError('valid modes, nonnegative observation count and positive noise required')
    combined, records = [], {}
    for index, run in enumerate(runs):
        run_options = dict(options)
        destination = out / f'{index:02d}_{run.name}'; destination.mkdir(exist_ok=True)
        if (run / 'best.pt').exists():
            checkpoint = load_checkpoint(run / 'best.pt', options['device']); config = checkpoint['config']
            if config.get('flow') and checkpoint.get('stage') != 'flow':
                raise ValueError(f'{run} has not finished its autoencoder stage')
            model = model_from_config(config, options['device']); model.load_state_dict(checkpoint['model']); model.eval()
            scales = checkpoint['scales']; methods = [config['model'] + ('-Flow' if config.get('flow') else '-Geo')]
        else:
            settings = json.loads((run / 'evaluation.json').read_text())
            config, model = {}, None
            if any(m not in METHODS for m in settings['methods']):
                raise ValueError(f'{run} contains neural evaluation output; pass its model run for new field predictions')
            scales = (settings['scales'] if 'scales' in settings else
                      normalization(load_case(p) for p in case_paths(args.data, 'train')))
            methods = args.methods.split(',') if args.methods else settings['methods']
            for key in ('pod_model', 'pod_rank', 'pod_ridge', 'gp_lengthscale', 'gp_amplitude', 'map_maxiter'):
                if key in settings:
                    run_options[key] = settings[key]
        rows, skipped = evaluate_case(path, run_options, scales, methods, destination, True, model, config)
        write_rows(destination / 'metrics.csv', rows); write_rows(destination / 'skipped.csv', skipped)
        for row in rows:
            row = dict(row, method=f'{run.name}/{row["method"]}')
            combined.append(row)
        for file in sorted(destination.glob('*.npz')):
            field = load_field(file); label = f'{run.name}_{field["meta"]["method"]}'
            records.setdefault(field['meta']['mode'], []).append((label, field))
            if requested.intersection({'fields', 'uncertainty'}):
                uncertainty_plot(field, out, f'{label}_{field["meta"]["mode"]}')
        del model
    write_rows(out / 'metrics.csv', combined)
    if 'fields' in requested:
        for mode, fields in records.items():
            solution_comparison(fields, out, mode)
    for label in sorted({label for fields in records.values() for label, _ in fields}):
        fields = [field for mode in options['modes'] for name, field in records.get(mode, []) if name == label]
        if requested.intersection({'fields', 'conditioning'}):
            conditioning_comparison(label, fields, out)
    (out / 'comparison.json').write_text(json.dumps(dict(case=str(path.resolve()), runs=[str(r.resolve()) for r in runs], **options), indent=2))
    print(f'Same-case field, conditioning, uncertainty and loss figures: {out}')
