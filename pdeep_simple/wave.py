"""Cached FEM modes, diffusion/wave evolution and optional wave assimilation."""
import hashlib
import math
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from .physics import assemble


def modal_basis(inputs, count, cache):
    geometry = inputs['wave_geometry']; operator = inputs['wave_operator']
    reaction = inputs['wave_reaction']; device = inputs['nodes'].device
    digest = hashlib.blake2b(digest_size=16)
    for array in (*geometry.values(), operator, np.asarray(reaction)):
        digest.update(np.ascontiguousarray(array).tobytes())
    key = (digest.digest(), count, str(device))
    if key not in cache:
        free = np.setdiff1d(np.arange(len(geometry['nodes'])), geometry['boundary_nodes'])
        if not len(free):
            cache[key] = None
            return None
        mass, stiffness = assemble(geometry, operator, reaction=reaction)
        m, k = mass[free][:, free], stiffness[free][:, free]
        if len(free) == 1:
            values = np.array([k[0, 0] / m[0, 0]])
            vectors = np.array([[1. / np.sqrt(m[0, 0])]])
        else:
            # ARPACK needs k < n; retain at most n-1 modes on very small meshes.
            values, vectors = eigsh(k, k=min(count, len(free) - 1), M=m, sigma=0., which='LM',
                                    v0=np.linspace(1., 2., len(free)), tol=1e-9)
        order = np.argsort(values); values, vectors = values[order], vectors[:, order]
        if np.any(values <= 0):
            raise ValueError('E-Geo needs a positive definite spatial operator')
        vectors /= np.sqrt(np.sum(vectors * (m @ vectors), axis=0))
        vectors *= np.where(vectors[np.abs(vectors).argmax(0), np.arange(len(values))] < 0, -1., 1.)
        phi = np.zeros((len(geometry['nodes']), len(values))); phi[free] = vectors
        cache[key] = {name: torch.as_tensor(array, dtype=inputs['nodes'].dtype, device=device)
                      for name, array in dict(phi=phi, projection=(mass @ phi).T, omega=np.sqrt(values)).items()}
    return cache[key]


def sinc(x):
    # Smooth at zero, including the second derivatives needed to train velocity.
    small = x.abs() < .01
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x.square() / 6 + x.pow(4) / 120, safe.sin() / safe)


def modal_evolution(omega, a, b, forcing, time, frequency=0.):
    """q'' + omega^2 q = forcing*cos(2*pi*frequency*t), with q(0)=a, q'(0)=b.

    The sinc product also handles resonance and constant forcing without a
    division by omega^2 - forcing_frequency^2.
    """
    angle = time * omega
    cosine, sine = angle.cos(), angle.sin()
    plus = (omega + 2 * math.pi * frequency) * time / 2
    minus = (omega - 2 * math.pi * frequency) * time / 2
    response = .5 * time.square() * sinc(plus) * sinc(minus)
    response_dt = .5 * time * (sinc(plus) * minus.cos() + plus.cos() * sinc(minus))
    q = a * cosine + b / omega * sine + forcing * response
    velocity = -a * omega * sine + b * cosine + forcing * response_dt
    return q, velocity


def diffusion_evolution(rates, initial, forcing, time, frequency=0., t0=0.):
    """q' + rates*q = forcing*cos(2*pi*frequency*(t0+t)).

    Reaction-diffusion in this dataset has R(u)=-rho*u, so its reduced
    reaction is exactly -rho*q; add rho to the diffusion rates once.
    """
    angle = rates * time
    decay = (-angle).exp()
    if frequency == 0:
        small = angle.abs() < .001
        safe = torch.where(small, torch.ones_like(angle), angle)
        response = time * torch.where(small, 1 - angle / 2 + angle.square() / 6 - angle.pow(3) / 24,
                                      -torch.expm1(-angle) / safe)
    else:
        clock = 2 * math.pi * frequency
        phase0 = clock * torch.as_tensor(t0, dtype=rates.dtype, device=rates.device)
        phase = clock * time + phase0
        # Closed-form cosine-forcing convolution, also valid at zero decay rate.
        response = (rates * (phase.cos() - decay * torch.cos(phase0)) +
                    clock * (phase.sin() - decay * torch.sin(phase0))) / (rates.square() + clock ** 2)
    return initial * decay + forcing * response


def damped_trajectory(omega, a, b, forcing, damping, frequency, t0, duration, steps=64):
    """One small exact linear step, reused on a time grid for the whole case.

    The augmented state is (q, q', F*cos(nu*t), F*sin(nu*t)). A 4x4
    exponential handles under/critical/overdamping and forcing resonance.
    """
    dt = duration / steps
    matrix = omega.new_zeros(len(omega), 4, 4)
    matrix[:, 0, 1] = 1
    matrix[:, 1, 0] = -omega.square()
    matrix[:, 1, 1] = -damping
    matrix[:, 1, 2] = 1
    clock = 2 * math.pi * frequency
    matrix[:, 2, 3] = -clock; matrix[:, 3, 2] = clock
    # MPS lacks some matrix-exponential operations; this small copy retains gradients.
    argument = matrix * dt
    transition = torch.matrix_exp(argument.cpu()).to(omega.device) if omega.device.type == 'mps' else torch.matrix_exp(argument)
    phase = clock * torch.as_tensor(t0, dtype=omega.dtype, device=omega.device)
    state = torch.stack([a, b, forcing * phase.cos(), forcing * phase.sin()], -1)
    states = [state[:, :2]]
    for _ in range(steps):
        state = (transition @ state[..., None]).squeeze(-1)
        states.append(state[:, :2])
    return torch.stack(states), dt


def trajectory_values(trajectory, dt, time):
    """Cubic Hermite q and its time derivative; no integration per query."""
    steps = len(trajectory) - 1
    position = (time[:, 0] / dt).clamp(0, steps)
    left = position.floor().long().clamp(max=steps - 1)
    s = (position - left)[:, None]
    q0, v0 = trajectory[left].unbind(-1)
    q1, v1 = trajectory[left + 1].unbind(-1)
    q = (2 * s.pow(3) - 3 * s.square() + 1) * q0 + (s.pow(3) - 2 * s.square() + s) * dt * v0
    q = q + (-2 * s.pow(3) + 3 * s.square()) * q1 + (s.pow(3) - s.square()) * dt * v1
    velocity = (6 * s.square() - 6 * s) / dt * (q0 - q1)
    velocity = velocity + (3 * s.square() - 4 * s + 1) * v0 + (3 * s.square() - 2 * s) * v1
    return q, velocity


def observation_inputs(context, inputs):
    from .geometry import locate
    obs = inputs['observations'][~inputs['observation_mask']]
    context['assimilation_observations'] = obs
    if 'assimilation_queries' not in context and len(obs):
        geometry = inputs.get('wave_geometry', inputs.get('evolution_geometry'))
        vertices, bary = locate(geometry, obs[:, :2].detach().cpu().numpy())
        context['assimilation_queries'] = dict(coords=obs[:, :3].detach(),
            vertices=torch.as_tensor(vertices, dtype=torch.long, device=obs.device),
            bary=torch.as_tensor(bary, dtype=obs.dtype, device=obs.device))


def observed_values(model, context):
    obs = context['assimilation_observations']
    prediction = model.decode(context, context['assimilation_queries'])
    return prediction[torch.arange(len(obs), device=obs.device), obs[:, 3].long()]


def speed_context(context, log_speed):
    return dict(context, log_coefficient=log_speed,
                wave=dict(context['wave'], omega=context['geometry_omega'] * log_speed.exp()))


def assimilate(model, context):
    """Conditional modal Gaussian update; one scalar GN step for inferred speed."""
    if context.get('f_refinement'):
        return model.refine(context)
    obs = context['assimilation_observations']
    context = dict(context, assimilation_applied=False)
    if not len(obs):
        return context
    prior = observed_values(model, context)
    context['observation_error_before'] = prior - obs[:, 4]
    if context.get('operator_inferred') and 'geometry_omega' in context:
        log_speed = context['log_coefficient']
        plus = observed_values(model, speed_context(context, log_speed + .001))
        minus = observed_values(model, speed_context(context, log_speed - .001))
        jacobian = (plus - minus) / .002
        variance = obs[:, 5].square() + 1e-6
        step = ((jacobian * (obs[:, 4] - prior) / variance).sum() /
                (1 + (jacobian.square() / variance).sum())).clamp(-.5, .5)
        for fraction in (1., .5):
            candidate = speed_context(context, log_speed + fraction * step)
            predicted = observed_values(model, candidate)
            if bool((((predicted - obs[:, 4]).square() - (prior - obs[:, 4]).square()) / variance).sum().detach() < 0):
                context = dict(candidate, assimilation_applied=True)
                prior = predicted
                break
    basis = context['wave']
    if basis is not None:
        inputs = context['wave_inputs']; scale = context['output_scales']
        phi, projection, omega = basis['phi'], basis['projection'], basis['omega']
        psi_u = phi * (1 - inputs['field_masks'][:, 1:2]) * scale[0]
        psi_v = phi * (1 - inputs['field_masks'][:, 2:3]) * scale[1]
        first_order = context.get('evolution_kind') in ('heat', 'reaction_diffusion')
        if first_order:
            psi_v = torch.zeros_like(psi_v)
        if bool(psi_u.any() or psi_v.any()):
            queries = context['assimilation_queries']
            def at_nodes(values):
                return (values[queries['vertices']] * queries['bary'][..., None]).sum(1)
            phi_obs, u_obs, v_obs = at_nodes(phi), at_nodes(psi_u), at_nodes(psi_v)
            a, b = projection @ psi_u, projection @ psi_v
            time = queries['coords'][:, 2:3] - context['wave_t0']
            if first_order:
                decay = (-time * context['diffusion_rates']).exp()
                displacement = torch.cat([u_obs + (phi_obs * (decay - 1)) @ a, torch.zeros_like(v_obs)], -1)
                velocity = torch.zeros_like(displacement)
            elif 'damped_trajectory' in context:
                zeros, ones = torch.zeros_like(omega), torch.ones_like(omega)
                duration = inputs['coordinate_bounds'][1, 2]
                ta, dt = damped_trajectory(omega, ones, zeros, zeros, inputs['damping'], 0., context['wave_t0'], duration)
                tb, _ = damped_trajectory(omega, zeros, ones, zeros, inputs['damping'], 0., context['wave_t0'], duration)
                ca, va = trajectory_values(ta, dt, time); cb, vb = trajectory_values(tb, dt, time)
                displacement = torch.cat([u_obs + (phi_obs * (ca - 1)) @ a,
                                          time * v_obs + (phi_obs * (cb - time)) @ b], -1)
                velocity = torch.cat([(phi_obs * va) @ a, v_obs + (phi_obs * (vb - 1)) @ b], -1)
            else:
                cosine, sine = (time * omega).cos(), (time * omega).sin()
                displacement = torch.cat([u_obs + (phi_obs * (cosine - 1)) @ a,
                                          time * v_obs + (phi_obs * (sine / omega - time)) @ b], -1)
                velocity = torch.cat([(phi_obs * (-omega * sine)) @ a,
                                      v_obs + (phi_obs * (cosine - 1)) @ b], -1)
            matrix = torch.where(obs[:, 3:4] == 0, displacement, velocity) / scale[obs[:, 3].long(), None]
            # Small double-precision system; MPS solves on CPU, with gradients retained.
            device = matrix.device if matrix.device.type != 'mps' else torch.device('cpu')
            h = matrix.to(device=device, dtype=torch.float64)
            noise = obs[:, 5].to(device=device, dtype=torch.float64)
            residual = (obs[:, 4] - prior).to(device=device, dtype=torch.float64)
            system = h @ h.T + torch.diag(noise.square() + 1e-6)
            delta = (h.T @ torch.linalg.solve(system, residual)).to(matrix)
            da, db = delta.chunk(2)
            context = dict(context, wave_initial=context['wave_initial'] + torch.stack([psi_u @ da, psi_v @ db], -1),
                           wave_a=context['wave_a'] + a @ da, wave_b=context['wave_b'] + b @ db,
                           assimilation_applied=True)
            if 'damped_trajectory' in context:
                context['damped_trajectory'] = context['damped_trajectory'] + ta * (a @ da)[None, :, None] + tb * (b @ db)[None, :, None]
    context['observation_error_after'] = observed_values(model, context) - obs[:, 4]
    return context


def initial_diagnostics(contexts):
    result = {}
    if not contexts or ('wave_inputs' not in contexts[0] and not contexts[0].get('operator_inferred')):
        return result
    result['operator_inferred'] = float(np.mean([c.get('operator_inferred', False) for c in contexts]))
    if any(c.get('operator_inferred') for c in contexts):
        result['coefficient_prediction'] = float(torch.stack([c['log_coefficient'].exp() for c in contexts]).mean().detach())
    if 'refinement_steps' in contexts[0]:
        result['refinement_steps'] = float(np.mean([c['refinement_steps'] for c in contexts]))
    if 'latent_covariance' in contexts[0]:
        covariances = [c['latent_covariance'].detach().cpu().double() for c in contexts]
        values = torch.stack([torch.linalg.eigvalsh(c) for c in covariances])
        result.update(latent_mean_variance=float(values.mean()),
                      latent_strong_directions=float((values < .5).sum(-1).double().mean()),
                      latent_weak_directions=float((values >= .5).sum(-1).double().mean()),
                      latent_effective_directions=float((1 - values).sum(-1).mean()))
    if 'assimilation_observations' in contexts[0]:
        result['assimilation_applied'] = float(np.mean([c.get('assimilation_applied', False) for c in contexts]))
        for when in ('before', 'after'):
            errors = [c[f'observation_error_{when}'].detach() for c in contexts if f'observation_error_{when}' in c]
            if errors:
                result[f'observation_normalized_rmse_{when}'] = float(torch.stack([e.square().mean() for e in errors]).mean().sqrt())
                obs = contexts[0]['assimilation_observations']; scale = contexts[0]['output_scales']
                variance = obs[:, 5].square() + 1e-6
                result[f'observation_noise_weighted_rmse_{when}'] = float(torch.stack([(e.square() / variance).mean() for e in errors]).mean().sqrt())
                for channel, name in enumerate(('u', 'v')):
                    keep = obs[:, 3] == channel
                    if bool(keep.any()):
                        result[f'observation_{name}_rmse_{when}'] = float(torch.stack([e[keep].square().mean() for e in errors]).mean().sqrt() * scale[channel])
    return result
