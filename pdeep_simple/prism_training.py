"""B's three stages, using the project's data, losses, CSV and checkpoint format."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .data import observations, model_input, query_draw, query_input
from .geometry import interpolate
from .physics import solve, WAVES
from .losses import case_loss, validate_loss_options
from .prism import complete_inputs
from .training import csv_row, model_from_config, select_device


def forward_loss(model, case, scales, queries, targets, weights):
    inputs = complete_inputs(case, scales, model.latent_mean.device)
    state = dict(fields=inputs['field_values'], coefficient=inputs['scalar'][0])
    candidate = model.forward_context(inputs, state)
    loss, parts = case_loss(model.forward_operator, case, candidate['inputs'], queries, targets,
                            scales, weights, candidate['forward'])
    return loss, parts, candidate


def sensitivity_loss(model, case, scales, coords):
    """One optional multiplicative coefficient direction; two existing FEM solves.

    Uses complete physical systems only in forward supervision. The p-Laplacian
    solver ignores conductivity, so no coefficient derivative is invented there.
    """
    if case['meta']['pde'] == 'p_laplacian':
        return model.latent_mean.new_zeros(()), False
    inputs = complete_inputs(case, scales, model.latent_mean.device)
    queries = query_input(case, coords, model.latent_mean.device)
    def neural(log_factor):
        factor = log_factor.exp()
        fields = torch.cat([inputs['field_values'][:, :3], inputs['field_values'][:, 3:] * factor], -1)
        modified = dict(inputs)
        if 'coefficient_samples' in inputs:
            samples = inputs['coefficient_samples']
            modified['coefficient_samples'] = torch.cat([samples[:, :2], samples[:, 2:] * factor], -1)
        candidate = model.forward_context(modified, dict(fields=fields, coefficient=inputs['scalar'][0] * factor))
        return model.decode(candidate, queries)
    point = model.latent_mean.new_zeros((), requires_grad=True)
    _, derivative = torch.autograd.functional.jvp(neural, point, torch.ones_like(point), create_graph=True)
    epsilon = .01; references = []
    for sign in (-1, 1):
        factor = np.exp(sign * epsilon)
        coefficient = case.get('element_coefficient', case['coefficient_fields']) * factor
        if 'element_coefficient' in case: coefficient = coefficient[:, None]
        if case['meta']['pde'] in WAVES:
            coefficient = coefficient ** inputs['operator_metadata']['wave_coefficient_power']
        if case['meta']['pde'] == 'eikonal': coefficient = case['meta']['coefficient'] * factor
        result, _ = solve(case, case['meta']['pde'], case['source'], case['initial_u'], case['initial_v'],
                          coefficient, case['meta']['beta'], len(case['times']), case['meta']['solver'].get('time_step', .002))
        references.append(interpolate(dict(case, **result), coords))
    expected = torch.as_tensor((references[1] - references[0]) / (2 * epsilon) / scales['output'],
                               dtype=derivative.dtype, device=derivative.device)
    channels = 2 if case['meta']['pde'] in WAVES else 1
    return (derivative[:, :channels] - expected[:, :channels]).square().mean(), True


def train_b(config, cases, validation_cases, checkpoint):
    repr_updates, forward_updates = config.get('b_repr_updates'), config.get('b_forward_updates')
    if repr_updates is None or forward_updates is None or min(repr_updates, forward_updates) < 1:
        raise ValueError('B requires explicit positive --b-repr-updates and --b-forward-updates')
    boundary = repr_updates + forward_updates
    if boundary >= config['updates'] or config.get('staged'):
        raise ValueError('B needs remaining posterior updates after representation/forward stages; do not use --staged')
    if not np.isfinite(config['b_sensitivity_weight']) or config['b_sensitivity_weight'] < 0:
        raise ValueError('B sensitivity weight must be finite and nonnegative')
    config['flow'] = True  # B is inherently generative; no LatentFlow wrapper.
    if not config['weights']['field']:
        raise ValueError('B forward training needs a positive field loss')
    device = select_device(config['device']); scales = config['scales']; run = Path(config['run'])
    validate_loss_options(cases, ['forward'], config['weights'])
    model = model_from_config(config, device)
    if checkpoint: model.load_state_dict(checkpoint['model'])
    def stage_at(update):
        return 'representation' if update <= repr_updates else 'forward' if update <= boundary else 'flow'
    start = checkpoint['update'] if checkpoint else 0
    stage = checkpoint['stage'] if checkpoint else stage_at(1)
    model.set_stage(stage)
    def optimizer_for_stage():
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'])
    optimizer = optimizer_for_stage(); best = checkpoint['best'] if checkpoint else float('inf')
    rng = np.random.default_rng(config['seed'])
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']); rng.bit_generator.state = checkpoint['rng_state']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device == 'cuda' and checkpoint.get('cuda_rng'): torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        if not (run / 'best.pt').exists(): best = float('inf')
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'B stages: representation={repr_updates}, forward={forward_updates}, posterior={config["updates"] - boundary}; '
          'whitening uses training cases only; flow validation uses CFM loss', flush=True)
    train_latents = None

    def step(case, generator, training):
        if stage == 'representation':
            loss, per_quantity = model.representation_loss(case, scales)
            return loss, dict(representation=float(loss.detach()),
                             **{f'repr_{q}': float(value.detach()) for q, value in zip(('f', 'u0', 'v0', 'a'), per_quantity)})
        if stage == 'forward':
            coords, target = query_draw(case, config['queries'], generator)
            if len(case['times']) > 1 and training:
                coords[:max(1, len(coords) // 4), 2] = 0.; target = interpolate(case, coords)
            queries = query_input(case, coords, device, bool(config['weights']['kinematic']))
            target = torch.as_tensor(target, dtype=torch.float32, device=device)
            loss, parts, candidate = forward_loss(model, case, scales, queries, target, config['weights'])
            channels = 2 if case['meta']['pde'] in WAVES else 1
            if not training:
                prediction = model.decode(candidate, queries)
                mse = (prediction[:, :channels] - target[:, :channels] / prediction.new_tensor(scales['output'][:channels])).square().mean(0)
                variance_sum[:channels].add_(mse.detach()); variance_count[:channels].add_(1)
            metrics = dict(field=float(parts['field'].detach()))
            if config['b_sensitivity_weight'] and training:
                sensitivity, supported = sensitivity_loss(model, case, scales, coords[:16])
                loss = loss + config['b_sensitivity_weight'] * sensitivity
                metrics.update(sensitivity=float(sensitivity.detach()), sensitivity_supported=int(supported))
            return loss, metrics
        mode = generator.choice(config['modes']); count = int(generator.choice(config['observations']))
        # Unlike deterministic inversion, B also learns unobserved conditional priors.
        noise = config['noise'] * generator.uniform(.5, 1.5) if training else config['noise']
        obs = observations(case, count, noise, generator)
        inputs = model_input(case, mode, obs, scales, generator, device)
        context = model.encode(inputs)
        target = train_latents[id(case)] if training else model.reference_latent(case, scales).detach()
        loss = model.matching_loss(context, target, generator)
        return loss, dict(fm=float(loss.detach()), information_seconds=context['information_seconds'])

    for update in range(start + 1, config['updates'] + 1):
        desired = stage_at(update)
        if desired != stage:
            if stage == 'representation': model.calibrate(cases, scales)
            stage = desired; model.set_stage(stage); optimizer = optimizer_for_stage(); best = float('inf')
            print(f'B stage: {stage}', flush=True)
        if stage == 'flow' and train_latents is None:
            with torch.no_grad():
                train_latents = {id(c): model.reference_latent(c, scales) for c in cases}
        model.train(); optimizer.zero_grad(set_to_none=True); metrics = {}; by_pde = {}; started = time.perf_counter()
        indices = rng.choice(len(cases), config['batch'], replace=config['batch'] > len(cases))
        for index in indices:
            case = cases[index]; loss, values = step(case, rng, True)
            if not torch.isfinite(loss): raise RuntimeError('non-finite B training loss')
            (loss / config['batch']).backward()
            values['total'] = float(loss.detach())
            for key, value in values.items(): metrics[key] = metrics.get(key, 0.) + value / config['batch']
            if 'field' in values: by_pde.setdefault(case['meta']['pde'], []).append(values['field'])
        parameters = [p for p in model.parameters() if p.requires_grad]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError('non-finite B gradient')
        torch.nn.utils.clip_grad_norm_(parameters, 5.); optimizer.step()
        row = dict(update=update, stage=stage, **metrics, seconds=time.perf_counter() - started)
        row.update({f'field_{pde}': float(np.mean(values)) for pde, values in by_pde.items()})
        csv_row(run / 'train.csv', row)
        if update == 1 or update in (repr_updates, boundary) or update % config['validate_every'] == 0 or update == config['updates']:
            model.eval(); val_rng = np.random.default_rng(config['seed'] + 9001); validation = {}
            variance_sum = model.surrogate_variance.new_zeros(2); variance_count = variance_sum.clone()
            with torch.no_grad():
                for case in validation_cases:
                    loss, values = step(case, val_rng, False); values['total'] = float(loss)
                    for key, value in values.items(): validation[key] = validation.get(key, 0.) + value / len(validation_cases)
            if stage == 'forward': model.surrogate_variance.copy_(variance_sum / variance_count.clamp_min(1))
            csv_row(run / 'validation.csv', dict(update=update, stage=stage, **validation))
            improved = validation['total'] < best; best = min(best, validation['total'])
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update, config=config,
                         scales=scales, best=best, stage=stage, rng_state=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None)
            torch.save(state, run / 'last.pt')
            if improved: torch.save(state, run / 'best.pt')
            print(f'update {update}/{config["updates"]}: B {stage}, train={metrics["total"]:.5g}, '
                  f'validation={validation["total"]:.5g}, information_seconds={metrics.get("information_seconds", 0.):.3f}', flush=True)
