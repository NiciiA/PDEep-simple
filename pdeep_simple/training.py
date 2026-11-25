"""Ordinary per-case PyTorch minibatches and portable checkpoints."""
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import (case_paths, load_case, normalization, model_input, observations,
                   query_draw, query_input, MODES, PDES)
from .losses import case_loss, parse_losses, validate_loss_options, LOSS_NAMES
from .models import build_model
from .physics import WAVES
from .geometry import interpolate


def select_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    return name


def load_checkpoint(path, device='cpu'):
    # Checkpoints contain only state dicts, numbers, strings, lists and dictionaries.
    return torch.load(path, map_location=device, weights_only=False)


def model_from_config(config, device='cpu', report=True):
    if config.get('flow') and config['model'] not in ('A', 'C'):
        raise ValueError('--flow currently supports A and C only')
    model = build_model(config['model'], config['size'], **{k: config.get(k) for k in
                        ('width', 'layers', 'heads', 'memory')}, report=report,
                        fourier_bands=config.get('fourier_bands', 6), wave_modes=config.get('wave_modes', 32))
    if config.get('flow'):
        from .flow import LatentFlow
        model = LatentFlow(model, config.get('latent_dim', 32), config.get('latent_points', 256))
        if report:
            print(f'with latent flow: {sum(p.numel() for p in model.parameters()):,} parameters')
    return model.to(device)


def csv_row(path, row):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    columns = list(row)
    if not new:
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            added = [key for key in row if key not in columns]
            if added:
                columns = columns + added
                temporary = path.with_suffix(path.suffix + '.tmp')
                with temporary.open('w', newline='') as destination:
                    writer = csv.DictWriter(destination, fieldnames=columns)
                    writer.writeheader(); writer.writerows(reader)
        if added:
            temporary.replace(path)
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow(row)


def draw_training_case(case, mode, count, config, scales, rng, device, training=True):
    obs = observations(case, count, config['noise'], rng)
    inputs = model_input(case, mode, obs, scales, rng, device)
    coords, target = query_draw(case, config['queries'], rng)
    initial_aware = config['model'] == 'E' and case['meta']['pde'] == 'wave'
    if training and initial_aware:
        initial_count = max(1, len(coords) // 4)
        coords[:initial_count, 2] = case['times'][0]
        target[:initial_count] = interpolate(case, coords[:initial_count])
    queries = query_input(case, coords, device, bool(config['weights']['kinematic']) and not initial_aware)
    return inputs, queries, torch.tensor(target, dtype=torch.float32, device=device)


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def train(args):
    run = Path(args.run); run.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config.pop('command', None)
    config['data'] = str(Path(args.data).resolve())
    config['modes'] = args.modes.split(',')
    config['observations'] = [int(s) for s in args.observations.split(',')]
    config['weights'] = parse_losses(args.loss)
    if any(m not in MODES for m in config['modes']):
        raise ValueError(f'modes must be selected from {MODES}')
    if min(config['observations']) < 0 or args.noise < 0 or min(args.batch, args.queries, args.updates) < 1:
        raise ValueError('invalid training counts or noise')
    if any(m != 'forward' for m in config['modes']) and max(config['observations']) == 0:
        raise ValueError('inverse/hybrid training requires a nonzero observation option')
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if (run / 'last.pt').exists() and not checkpoint:
        raise ValueError('run already contains a checkpoint; use --resume or a new --run')
    if checkpoint:
        # Restore scientific settings, but allow data/checkpoints to move machines.
        saved = checkpoint['config']
        if args.staged and not saved.get('staged'):
            raise ValueError('--staged cannot be added to a non-staged checkpoint')
        if args.stage_start is not None and args.stage_start != saved.get('stage_start'):
            raise ValueError('--resume restores the saved --stage-start; omit it or use the saved value')
        config = dict(saved, data=str(Path(args.data).resolve()), updates=args.updates,
                      device=args.device, run=str(run), resume=args.resume)
    if config.get('staged'):
        if config.get('flow'):
            raise ValueError('--staged is for deterministic training; it cannot be combined with --flow')
        if not checkpoint:
            config['stage_start'] = args.stage_start if args.stage_start is not None else max(1, args.updates // 5)
        if not 1 <= config['stage_start'] < config['updates']:
            raise ValueError('staged training needs 1 <= stage-start < total updates')
        print(f'Staged training: field=1 through update {config["stage_start"]}, '
              f'then --loss {config["loss"]}' + (' (restored from checkpoint)' if checkpoint else ''))
    elif args.stage_start is not None:
        raise ValueError('--stage-start requires --staged')
    if config.get('flow'):
        if not checkpoint:
            config['ae_updates'] = args.ae_updates if args.ae_updates is not None else min(500, max(1, args.updates // 3))
        if not 1 <= config['ae_updates'] < config['updates']:
            raise ValueError('flow training needs 1 <= ae-updates < total updates')
        if not config['weights']['field']:
            raise ValueError('the field autoencoder needs a positive field loss')
    device = select_device(config['device'])
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed']); random.seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    paths = case_paths(config['data'], 'train', config.get('geometries', '').split(',') if config.get('geometries') else None)
    cases = [load_case(p) for p in paths]
    val_cases = [load_case(p) for p in case_paths(config['data'], 'val', config.get('geometries', '').split(',') if config.get('geometries') else None)]
    scales = checkpoint['scales'] if checkpoint else normalization(cases)
    config['scales'] = scales
    logged_pdes = config.get('logged_pdes', list(PDES) if any(c['meta']['pde'] not in PDES[:4] for c in cases) else list(PDES[:4]))
    config['logged_pdes'] = logged_pdes
    config['training_geometries'] = sorted({c['meta']['geometry'] for c in cases})
    validate_loss_options(cases, config['modes'], config['weights'])
    model = model_from_config(config, device)
    stage = checkpoint.get('stage', 'ae') if checkpoint and config.get('flow') else ('ae' if config.get('flow') else 'deterministic')
    if config.get('flow'):
        model.set_stage(stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'])
    start_update, best = 0, float('inf')
    if checkpoint:
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start_update, best = checkpoint['update'], checkpoint['best']
        rng.bit_generator.state = checkpoint['rng_state']
        if not (run / 'best.pt').exists():
            best = float('inf')
            print('No best checkpoint in destination; selecting best among resumed updates')
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device == 'cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    train_latents = val_latents = None
    if stage == 'flow':
        with torch.no_grad():
            train_latents = torch.stack([model.reference_latent(case, scales) for case in cases])
            val_latents = torch.stack([model.reference_latent(case, scales) for case in val_cases])
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'device={device}; {len(cases)} train, {len(val_cases)} validation cases; output scales={scales["output"]}')
    started = time.perf_counter()
    for update in range(start_update + 1, config['updates'] + 1):
        active_config = config
        if config.get('staged'):
            stage = 'field' if update <= config['stage_start'] else 'physics'
            if stage == 'field':
                weights = dict.fromkeys(LOSS_NAMES, 0.); weights['field'] = 1.
                active_config = dict(config, weights=weights)
            if update == config['stage_start'] + 1:
                print('Stage 2: continuing with the selected reconstruction and physics losses', flush=True)
        if config.get('flow') and update == config['ae_updates'] + 1 and stage == 'ae':
            model.eval(); train_latents = model.calibrate(cases, scales); model.set_stage('flow')
            with torch.no_grad():
                val_latents = torch.stack([model.reference_latent(case, scales) for case in val_cases])
            stage, best = 'flow', float('inf')
            optimizer = torch.optim.AdamW(model.velocity.parameters(), lr=config['lr'])
            print('Stage 2: frozen field representation; conditional flow matching only', flush=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals = dict.fromkeys(LOSS_NAMES + ('total',), 0.)
        by_pde = {}; fm_total = 0.
        wave_fields = {pde: [0., 0., 0] for pde in WAVES}
        indices = rng.choice(len(cases), config['batch'], replace=config['batch'] > len(cases))
        for index in indices:
            case = cases[index]
            mode = rng.choice(config['modes'])
            options = config['observations'] if mode == 'forward' else [n for n in config['observations'] if n > 0]
            count = int(rng.choice(options))
            diagnostics = {}
            if stage == 'flow':
                obs = observations(case, count, config['noise'], rng)
                inputs = model_input(case, mode, obs, scales, rng, device)
                with torch.no_grad():
                    context = model.encode(inputs)
                loss = model.matching_loss(context, train_latents[index], rng)
                fm_total += float(loss.detach()) / config['batch']
                parts = dict.fromkeys(LOSS_NAMES, loss.detach() * 0)
            else:
                inputs, queries, targets = draw_training_case(case, mode, count, active_config, scales, rng, device)
                context = None
                if stage == 'ae':
                    context = model.with_latent(model.encode(inputs), model.reference_latent(case, scales))
                loss, parts = case_loss(model, case, inputs, queries, targets, scales, active_config['weights'], context, diagnostics)
                if diagnostics:
                    values = wave_fields[case['meta']['pde']]; points = len(targets)
                    values[0] += diagnostics['u'] * points; values[1] += diagnostics['v'] * points; values[2] += points
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at update {update}')
            (loss / config['batch']).backward()
            for name in parts:
                totals[name] += float(parts[name].detach()) / config['batch']
            totals['total'] += float(loss.detach()) / config['batch']
            by_pde.setdefault(case['meta']['pde'], []).append(float(parts['field'].detach()))
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('non-finite model gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        row = dict(update=update, **totals)
        row.update({f'field_{pde}': float(np.mean(by_pde[pde])) if pde in by_pde else ''
                    for pde in logged_pdes})
        if config.get('flow'):
            row.update(stage=stage, fm=fm_total if stage == 'flow' else '')
            if stage == 'flow':
                for name in LOSS_NAMES:
                    row[name] = ''
                for pde in logged_pdes:
                    row[f'field_{pde}'] = ''
        elif config.get('staged'):
            row['stage'] = stage
        row.update({f'field_{pde}_{channel}': values[i] / values[2] if values[2] else ''
                    for pde, values in wave_fields.items() for i, channel in enumerate(('u', 'v'))})
        csv_row(run / 'train.csv', row)
        boundary = config.get('flow') and update in (config['ae_updates'], config['ae_updates'] + 1)
        if update == 1 or boundary or update % config['validate_every'] == 0 or update == config['updates']:
            model.eval()
            validation = dict.fromkeys(LOSS_NAMES + ('total',), 0.)
            validation_fields = {pde: [0., 0] for pde in PDES}
            validation_waves = {pde: [0., 0., 0] for pde in WAVES}
            val_rng = np.random.default_rng(config['seed'] + 9001); val_fm = 0.
            # Recreate fixed draws each validation, over every validation physical case.
            with torch.set_grad_enabled(stage != 'flow' and bool(active_config['weights']['kinematic'])):
                for index, case in enumerate(val_cases):
                    mode = config['modes'][index % len(config['modes'])]
                    counts = config['observations'] if mode == 'forward' else [n for n in config['observations'] if n > 0]
                    inputs, queries, targets = draw_training_case(case, mode, max(counts), active_config, scales, val_rng, device, training=False)
                    diagnostics = {}
                    if stage == 'flow':
                        context = model.encode(inputs)
                        loss = model.matching_loss(context, val_latents[index], val_rng)
                        val_fm += float(loss) / len(val_cases)
                        latents = model.sample_latents(context, 4, config['flow_steps'], config['seed'] + index)
                        fields = [model.decode(model.with_latent(context, z), queries) for z in latents]
                        channels = 2 if case['meta']['pde'] in WAVES else 1
                        scale = torch.tensor(scales['output'], device=device)
                        prediction = torch.stack(fields).mean(0)
                        errors = prediction[:, :channels] - targets[:, :channels] / scale[:channels]
                        field_loss = errors.square().mean()
                        if channels == 2:
                            diagnostics.update(zip(('u', 'v'), errors.detach().square().mean(0).cpu().tolist()))
                        parts = dict.fromkeys(LOSS_NAMES, loss * 0); parts['field'] = field_loss
                    else:
                        context = None
                        if stage == 'ae':
                            context = model.with_latent(model.encode(inputs), model.reference_latent(case, scales))
                        loss, parts = case_loss(model, case, inputs, queries, targets, scales, active_config['weights'], context, diagnostics)
                    for name in parts:
                        validation[name] += float(parts[name].detach()) / len(val_cases)
                    validation['total'] += float(loss.detach()) / len(val_cases)
                    points = len(targets); values = validation_fields[case['meta']['pde']]
                    values[0] += float(parts['field'].detach()) * points; values[1] += points
                    if diagnostics:
                        values = validation_waves[case['meta']['pde']]
                        values[0] += diagnostics['u'] * points; values[1] += diagnostics['v'] * points; values[2] += points
            validation_row = dict(update=update, **validation)
            if config.get('flow'):
                validation_row.update(stage=stage, fm=val_fm if stage == 'flow' else '')
                if stage == 'flow':
                    for name in LOSS_NAMES:
                        if name != 'field':
                            validation_row[name] = ''
            elif config.get('staged'):
                validation_row['stage'] = stage
            validation_row.update({f'validation_{pde}': values[0] / values[1] if values[1] else ''
                                   for pde, values in validation_fields.items()})
            validation_row.update({f'validation_{pde}_{channel}': values[i] / values[2] if values[2] else ''
                                   for pde, values in validation_waves.items() for i, channel in enumerate(('u', 'v'))})
            csv_row(run / 'validation.csv', validation_row)
            selection_loss = val_fm if stage == 'flow' else validation['field']
            improved = selection_loss < best
            best = min(best, selection_loss)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
                         config=config, scales=scales, best=best, stage=stage, rng_state=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None)
            torch.save(state, run / 'last.pt')
            if improved:
                torch.save(state, run / 'best.pt')
            field_text = f'{totals["field"]:.5g}' if stage != 'flow' else 'not optimized in FM stage'
            print(f'update {update}/{config["updates"]}: field={field_text}, '
                  f'validation={validation["field"]:.5g}, total={totals["total"]:.5g}'
                  + ((f', stage={stage}' + (f', FM={fm_total:.5g}' if stage == 'flow' else '')) if config.get('flow') or config.get('staged') else ''), flush=True)
    metric = 'FM' if stage == 'flow' else 'field'
    print(f'training finished in {time.perf_counter() - started:.2f}s; best validation {metric}={best:.5g}')
