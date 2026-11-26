"""B-Geo: physical causes -> sensor likelihood geometry -> conditional transport.

Only reference_latent/representation_loss read complete training targets. The
posterior path receives masked inputs; the forward path receives a candidate
physical system and an empty sensor set. No numerical forward solve runs here.
"""
import time
import numpy as np
import torch
from torch import nn
from .models import AGeo, EGeo, PhysicalEncoder, Coordinates, AttentionBlock, mlp
from .flow import FieldEncoder
from .data import PDES, model_input, observations
from .physics import PARABOLIC, WAVES

PHYSICAL_FIELDS = ('source', 'initial_u', 'initial_v', 'coefficient_fields')


def complete_inputs(case, scales, device):
    """Supervision only: fully specified physical systems, with no sensors."""
    rng = np.random.default_rng(0)
    return model_input(case, 'forward', observations(case, 0, 0., rng), scales, rng, device)


def relevant_fields(inputs):
    pde = PDES[inputs['pde']]
    return (pde != 'eikonal', pde in PARABOLIC + WAVES, pde in WAVES, pde != 'p_laplacian')


def physical_scales(inputs):
    return inputs['field_scales']


def known_constant_coefficient(inputs):
    """An exact known entry fixes a declared constant coefficient everywhere."""
    if not inputs['physical_layout']['constant_coefficient']:
        return None
    if bool(inputs['scalar_masks'][0]):
        return inputs['scalar'][0]
    mask = inputs['field_masks'][:, 3].bool()
    return inputs['field_values'][mask, 3][0] if bool(mask.any()) else None


def observation_jacobian(predicted, state, block=8):
    """Only N_sensor x N_latent derivatives; bounded VJP batches on CUDA/CPU."""
    if state.device.type == 'mps':
        return torch.stack([torch.autograd.grad(y, state, retain_graph=True)[0] for y in predicted]).detach()
    rows = []
    for start in range(0, len(predicted), block):
        stop = min(start + block, len(predicted))
        basis = predicted.new_zeros((stop - start, len(predicted)))
        basis[torch.arange(stop - start), torch.arange(start, stop)] = 1
        rows.append(torch.autograd.grad(predicted, state, basis, is_grads_batched=True, retain_graph=True)[0].detach())
    return torch.cat(rows)


def information_matrix(jacobian, variance):
    return jacobian.T @ (jacobian / variance[:, None])


def information_subspace(jacobian, variance, rank, tau):
    """Top-rank approximation to W=G(G+tau I)^-1; small eigensolve on CPU.

    The observation Jacobian is exact; discarded information eigenvalues are
    treated as zero. This is a rank-truncation approximation, not a hard claim
    that the complementary directions are unidentifiable.
    """
    if not len(jacobian):
        return dict(vectors=jacobian.new_empty((jacobian.shape[1], 0)), values=jacobian.new_empty(0), weights=jacobian.new_empty(0))
    gram = information_matrix(jacobian.cpu().double(), variance.cpu().double())
    values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
    rank = min(rank, len(values), len(jacobian))
    values = values[-rank:].flip(0).clamp_min(0)
    vectors = vectors[:, -rank:].flip(1)
    # Canonical signs make the compact information tokens reproducible.
    signs = vectors[vectors.abs().argmax(0), torch.arange(rank)].sign()
    vectors = vectors * signs
    return dict(vectors=vectors.to(jacobian), values=values.to(jacobian),
                weights=(values / (values + tau)).to(jacobian))


def project_information(value, subspace):
    vectors = subspace['vectors']
    return ((value @ vectors) * subspace['weights']) @ vectors.T


def route_velocity(evidence, ambiguity, subspace):
    return ambiguity + project_information(evidence - ambiguity, subspace)


class PhysicalCodec(nn.Module):
    """Four semantic groups, a shared field encoder and nonlinear spatial decoder.

    Groups describe causes (f,u0,v0,a), never fixed observable/ambiguous indices.
    Boundary data and reaction/damping are known in the current datasets.
    """
    def __init__(self, width, latent_dim):
        super().__init__()
        self.group_dim = latent_dim // 4
        self.encoder = FieldEncoder(width, self.group_dim)
        self.coordinates = Coordinates(width)
        self.quantity = nn.Embedding(4, width)
        self.spatial = mlp(2 * width + self.group_dim, width, width)
        self.output = nn.Linear(width, 1)
        self.coefficient = mlp(width + self.group_dim, width, 1)

    def geometry(self, inputs):
        xy = (inputs['nodes'] - inputs['coordinate_bounds'][0, :2]) / inputs['coordinate_bounds'][1, :2]
        coords = torch.cat([xy, torch.zeros_like(xy[:, :1])], -1)
        h = self.coordinates(coords)
        weights = inputs['node_weights']; weights = weights / weights.sum()
        return coords, h, (h * weights[:, None]).sum(0)

    def encode(self, inputs):
        coords, _, _ = self.geometry(inputs)
        values = inputs['field_values'] / physical_scales(inputs)
        values = torch.cat([values[:, :3], inputs['field_values'][:, 3:].clamp_min(1e-8).log()], -1)
        groups = []
        for q in range(4):
            if not relevant_fields(inputs)[q]:
                groups.append(values.new_zeros(self.group_dim))
                continue
            pairs = torch.stack([values[:, q], values.new_full((len(values),), q / 3)], -1)
            groups.append(self.encoder(coords, pairs, inputs['node_weights']))
        return torch.cat(groups)

    def decode(self, latent, inputs, clamp_known=True):
        _, coords, pooled = self.geometry(inputs)
        groups = latent.reshape(4, self.group_dim)
        predicted = []
        for q in range(4):
            h = torch.cat([coords + self.quantity.weight[q], pooled.expand(len(coords), -1),
                           groups[q].expand(len(coords), -1)], -1)
            predicted.append(self.output(self.spatial(h))[:, 0])
        predicted = torch.stack(predicted, -1)
        coefficient = self.coefficient(torch.cat([pooled, groups[3]]))[0]
        lo, hi = inputs['physical_layout']['coefficient_bounds']
        def positive(value):
            return lo + (hi - lo) * value.sigmoid() if hi > lo else value.clamp(-12, 12).exp()
        scalar_coefficient = positive(coefficient)
        known_coefficient = known_constant_coefficient(inputs) if clamp_known else None
        if known_coefficient is not None:
            scalar_coefficient = known_coefficient
        field_coefficient = (scalar_coefficient.expand(len(coords)) if inputs['physical_layout']['constant_coefficient']
                             else predicted[:, 3].clamp(-12, 12).exp())
        free = 1 - inputs['features'][:, 2]
        fields = torch.stack([predicted[:, 0] * inputs['field_scales'][0],
                              predicted[:, 1] * inputs['field_scales'][1] * free,
                              predicted[:, 2] * inputs['field_scales'][2] * free, field_coefficient], -1)
        relevant = fields.new_tensor(relevant_fields(inputs), dtype=torch.bool)
        fields = torch.where(relevant[None], fields, inputs['field_values'])
        if clamp_known:
            fields = torch.where(inputs['field_masks'].bool(), inputs['field_values'], fields)
            scalar_coefficient = torch.where(inputs['scalar_masks'][0].bool(), inputs['scalar'][0], scalar_coefficient)
        if not inputs['physical_layout']['constant_coefficient'] and not bool(inputs['scalar_masks'][0]):
            weights = inputs['node_weights']; weights = weights / weights.sum()
            scalar_coefficient = (fields[:, 3] * weights).sum()
        return dict(fields=fields, coefficient=scalar_coefficient)


class RoutedFlow(nn.Module):
    """One shared hypothesis-attention backbone, two small velocity heads."""
    def __init__(self, width, layers, heads, latent_dim):
        super().__init__()
        self.group_dim = latent_dim // 4
        self.hypothesis = mlp(self.group_dim + 3, width, width)
        self.quantity = nn.Embedding(4, width)
        self.information = mlp(latent_dim + 2, width, width)
        self.blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(layers)])
        self.evidence = nn.Linear(width, self.group_dim)
        self.ambiguity = nn.Linear(width, self.group_dim)

    def forward(self, state, clock, context, subspace, physical_state=None):
        clock = torch.stack([clock, (torch.pi * clock).sin(), (torch.pi * clock).cos()])
        physical_state = state if physical_state is None else physical_state
        h = self.hypothesis(torch.cat([physical_state.reshape(4, -1), clock.expand(4, -1)], -1))
        h = h + self.quantity.weight + context['global_h']
        memory = context['memory']
        if len(subspace['values']):
            info = torch.cat([subspace['vectors'].T, subspace['values'].log1p()[:, None], subspace['weights'][:, None]], -1)
            memory = torch.cat([memory, self.information(info)])
        for block in self.blocks:
            h = block(h, torch.cat([memory, h]))
        return self.evidence(h).flatten(), self.ambiguity(h).flatten()


class BGeo(nn.Module):
    def __init__(self, width, layers, heads, memory, latent_dim=128, information_rank=16,
                 information_tau=1., routing='dynamic', information_updates=2, route_weight=.1):
        super().__init__()
        if latent_dim < 4 or latent_dim % 4 or information_rank < 1 or not np.isfinite(information_tau) or information_tau <= 0:
            raise ValueError('B needs latent dimension divisible by four, positive information rank and tau')
        if routing not in ('ordinary', 'fixed', 'dynamic') or information_updates < 1 or not np.isfinite(route_weight) or route_weight < 0:
            raise ValueError('invalid B routing settings')
        self.latent_dim, self.rank, self.tau = latent_dim, information_rank, information_tau
        self.routing, self.information_updates, self.route_weight = routing, information_updates, route_weight
        self.codec = PhysicalCodec(width, latent_dim)
        self.conditioner = PhysicalEncoder(width, heads, joint_conditioning=True)
        # Reuse E's initial anchoring and v=du/dt, with A's learned forward map.
        # AGeo.encode below deliberately bypasses E's FEM/modal dispatch.
        self.forward_operator = EGeo(width, layers, heads, memory)
        self.forward_operator.encoder = PhysicalEncoder(width, heads, joint_conditioning=True)
        self.velocity = RoutedFlow(width, layers, heads, latent_dim)
        self.initial_estimate = mlp(width, width, latent_dim)
        self.register_buffer('latent_mean', torch.zeros(latent_dim))
        self.register_buffer('latent_cholesky', torch.eye(latent_dim))
        self.register_buffer('whitening_ready', torch.tensor(False))
        self.register_buffer('surrogate_variance', torch.zeros(2))

    def whiten(self, value):
        # triangular solve on CPU is portable to MPS; gradients still propagate.
        if value.device.type == 'mps':
            return torch.linalg.solve_triangular(self.latent_cholesky.cpu(), (value - self.latent_mean).cpu()[..., None], upper=False)[..., 0].to(value)
        return torch.linalg.solve_triangular(self.latent_cholesky, (value - self.latent_mean)[..., None], upper=False)[..., 0]

    def unwhiten(self, value):
        return value @ self.latent_cholesky.T + self.latent_mean

    def reference_latent(self, case, scales):
        return self.codec.encode(complete_inputs(case, scales, self.latent_mean.device))

    def calibrate(self, cases, scales):
        with torch.no_grad():
            values = torch.stack([self.reference_latent(c, scales) for c in cases])
            cpu_values = values.cpu().double()
            centered = cpu_values - cpu_values.mean(0)
            covariance = centered.T @ centered / max(len(values) - 1, 1)
            regularization = max(float(covariance.diag().mean()) * 1e-3, 1e-5)
            chol = torch.linalg.cholesky(covariance + regularization * torch.eye(self.latent_dim, dtype=torch.float64))
            self.latent_mean.copy_(values.mean(0)); self.latent_cholesky.copy_(chol)
            self.whitening_ready.fill_(True)
        return values

    def representation_loss(self, case, scales):
        inputs = complete_inputs(case, scales, self.latent_mean.device)
        latent = self.codec.encode(inputs)
        state = self.codec.decode(latent, inputs, clamp_known=False)
        weights = inputs['node_weights']; weights = weights / weights.sum()
        errors = ((state['fields'] - inputs['field_values']) / physical_scales(inputs)).square()
        active = errors.new_tensor(relevant_fields(inputs))
        per_quantity = (errors * weights[:, None]).sum(0) * active
        return per_quantity.sum() / active.sum().clamp_min(1), per_quantity

    def decode_physical_state(self, latent, inputs):
        return self.codec.decode(self.unwhiten(latent), inputs)

    def candidate_inputs(self, inputs, state):
        values = state['fields']; masks = torch.ones_like(values)
        scalar = torch.stack([state['coefficient'], inputs['scalar'][1]])
        scalar_masks = torch.ones_like(scalar)
        if not inputs['physical_layout']['constant_coefficient']:
            # The full coefficient field specifies the operator. Its generator's
            # redundant amplitude is not an additional forward-model input.
            scalar = torch.stack([scalar[0] * 0, scalar[1]])
            scalar_masks = torch.stack([scalar_masks[0] * 0, scalar_masks[1]])
        features = torch.cat([inputs['features'][:, :3], values / inputs['field_scales'], masks,
                              torch.cat([scalar, scalar_masks]).expand(len(values), -1)], -1)
        # Explicit allow-list: no sensor values or cached numerical operators.
        keys = ('nodes', 'edges', 'basis', 'basis_channels', 'coordinate_bounds', 'output_scales',
                'field_scales', 'operator_metadata', 'node_weights', 'physical_layout', 'pde', 'coefficient_samples', 'triangles')
        candidate = {k: inputs[k] for k in keys if k in inputs}
        candidate.update(features=features, field_values=values, field_masks=masks, scalar=scalar, scalar_masks=scalar_masks,
                         observations=values.new_zeros((1, 6)), observation_mask=torch.ones(1, dtype=torch.bool, device=values.device))
        return candidate

    def forward_context(self, inputs, state):
        candidate = self.candidate_inputs(inputs, state)
        context = AGeo.encode(self.forward_operator, candidate)
        context['wave'] = None
        pde = PDES[inputs['pde']]
        if pde in PARABOLIC + WAVES:
            context.update(wave_initial=state['fields'][:, 1:3], output_scales=inputs['output_scales'],
                           wave_t0=inputs['coordinate_bounds'][0, 2], evolution_kind=pde)
        return dict(forward=context, physical=state, inputs=candidate, global_h=context['global_h'],
                    log_coefficient=state['coefficient'].log())

    def encode(self, inputs):
        if not bool(inputs['scalar_masks'][1]):
            raise ValueError('B has no decoder for hidden beta; current datasets supply beta')
        context = self.conditioner(inputs)
        context['memory'] = torch.cat([context['mesh'], context['pde'], context['obs'][~context['padding']]])
        known_coefficient = known_constant_coefficient(inputs)
        all_known = all(not active or bool(inputs['field_masks'][:, q].all()) or q == 3 and known_coefficient is not None
                        for q, active in enumerate(relevant_fields(inputs)))
        context.update(inputs=inputs, all_known=all_known,
                       log_coefficient=inputs['scalar'][0].clamp_min(1e-8).log(),
                       information_records=[], information_seconds=0., candidate_cache={})
        return context

    def with_latent(self, context, latent):
        key = tuple(latent.detach().cpu().tolist()) if not torch.is_grad_enabled() else None
        if key is not None and key in context['candidate_cache']:
            return context['candidate_cache'][key]
        inputs = context['inputs']
        if context['all_known']:
            coefficient = (inputs['scalar'][0] if bool(inputs['scalar_masks'][0]) else
                           (inputs['field_values'][:, 3] * inputs['node_weights']).sum() / inputs['node_weights'].sum())
            fields = inputs['field_values']
            known_coefficient = known_constant_coefficient(inputs)
            if known_coefficient is not None:
                coefficient = known_coefficient
                fields = torch.cat([fields[:, :3], coefficient.expand(len(fields), 1)], -1)
            physical = dict(fields=fields, coefficient=coefficient)
        else:
            physical = self.decode_physical_state(latent, inputs)
        result = self.forward_context(inputs, physical)
        if key is not None:
            context['candidate_cache'][key] = result
        return result

    def decode(self, context, queries):
        field = self.forward_operator.decode(context['forward'], queries)
        if PDES[context['inputs']['pde']] not in WAVES:
            field = torch.stack([field[:, 0], torch.zeros_like(field[:, 0])], -1)
        return field

    def forward(self, inputs, queries, latent=None):
        context = self.encode(inputs)
        if latent is None:
            latent = self.sample_latents(context, samples=1)[0]
        candidate = self.with_latent(context, latent)
        return dict(field=self.decode(candidate, queries), log_coefficient=candidate['log_coefficient'],
                    physical=candidate['physical'])

    def predict_observations(self, context, latent):
        inputs = context['inputs']; valid = ~inputs['observation_mask']; obs = inputs['observations'][valid]
        if not len(obs):
            return latent.new_empty(0)
        self.prepare_sensors(context)
        candidate = self.forward_context(inputs, self.decode_physical_state(latent, inputs))
        # Barycentric geometry is prepared once; physical times stay untouched.
        queries = dict(context['sensor_queries'], coords=obs[:, :3])
        prediction = self.decode(candidate, queries)
        return prediction[torch.arange(len(obs), device=obs.device), obs[:, 3].long()]

    def prepare_sensors(self, context):
        if 'sensor_queries' not in context:
            from .geometry import locate
            inputs = context['inputs']; obs = inputs['observations'][~inputs['observation_mask']]
            geometry = dict(nodes=inputs['nodes'].detach().cpu().numpy(), triangles=inputs['triangles'].cpu().numpy())
            vertices, bary = locate(geometry, obs[:, :2].detach().cpu().numpy()) if len(obs) else (np.empty((0, 3), int), np.empty((0, 3)))
            context['sensor_queries'] = dict(coords=obs[:, :3],
                vertices=torch.as_tensor(vertices, dtype=torch.long, device=obs.device),
                bary=torch.as_tensor(bary, dtype=obs.dtype, device=obs.device))

    def local_information(self, context, latent, record=False, flow_time=0., sample_index=-1):
        empty = dict(vectors=latent.new_empty((self.latent_dim, 0)), values=latent.new_empty(0), weights=latent.new_empty(0))
        inputs = context['inputs']; obs = inputs['observations'][~inputs['observation_mask']]
        if not len(obs) or context['all_known'] or self.routing == 'ordinary':
            return empty
        self.prepare_sensors(context)
        def synchronize():
            if latent.device.type == 'cuda': torch.cuda.synchronize()
            elif latent.device.type == 'mps': torch.mps.synchronize()
        synchronize(); started = time.perf_counter()
        with torch.enable_grad():
            state = latent.detach().requires_grad_(True)
            predicted = self.predict_observations(context, state)
            jacobian = observation_jacobian(predicted, state)
        variance = obs[:, 5].square() + self.surrogate_variance[obs[:, 3].long()] + 1e-6
        subspace = information_subspace(jacobian, variance, self.rank, self.tau)
        synchronize(); context['information_seconds'] += time.perf_counter() - started
        if record:
            context['information_records'].append(dict(state=latent.detach(), flow_time=flow_time, sample_index=sample_index, **subspace))
        return subspace

    def matching_loss(self, context, target, rng):
        z1 = self.whiten(target.detach())
        z0 = torch.as_tensor(rng.standard_normal(self.latent_dim), dtype=z1.dtype, device=z1.device)
        clock = z1.new_tensor(rng.random()); state = (1 - clock) * z0 + clock * z1
        estimate = self.initial_estimate(context['global_h'])
        anchor = estimate if self.routing == 'fixed' else state
        subspace = self.local_information(context, anchor)
        evidence, ambiguity = self.velocity(state, clock, context, subspace, self.unwhiten(state))
        prediction = (evidence + ambiguity) / 2 if self.routing == 'ordinary' else route_velocity(evidence, ambiguity, subspace)
        target_velocity = z1 - z0
        loss = (prediction - target_velocity).square().mean()
        if self.routing != 'ordinary':
            informed = project_information(evidence - target_velocity, subspace)
            remainder = ambiguity - target_velocity
            ambiguous = remainder - project_information(remainder, subspace)
            loss = loss + self.route_weight * (informed.square().mean() + ambiguous.square().mean())
        return loss + .05 * (estimate - z1).square().mean()

    def sample_latents(self, context, samples=32, steps=32, seed=123):
        if samples < 1 or steps < 1:
            raise ValueError('samples and flow-steps must be positive')
        if context['all_known']:
            return self.latent_mean.new_zeros((samples, self.latent_dim))
        if not bool(self.whitening_ready):
            raise ValueError('B posterior sampling requires completed representation/whitening training')
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        states = torch.randn(samples, self.latent_dim, generator=generator).to(self.latent_mean)
        refresh = set(np.linspace(0, steps, self.information_updates, endpoint=False).astype(int))
        with torch.no_grad():
            fixed = (self.local_information(context, self.initial_estimate(context['global_h']), record=True)
                     if self.routing == 'fixed' else None)
            for sample in range(samples):
                state = states[sample]; subspace = fixed
                for index in range(steps):
                    if subspace is None or self.routing == 'dynamic' and index in refresh:
                        anchor = self.initial_estimate(context['global_h']) if self.routing == 'fixed' else state
                        subspace = self.local_information(context, anchor, record=True, flow_time=index / steps, sample_index=sample)
                    clock = state.new_tensor(index / steps)
                    def velocity(z, s):
                        evidence, ambiguity = self.velocity(z, s, context, subspace, self.unwhiten(z))
                        return (evidence + ambiguity) / 2 if self.routing == 'ordinary' else route_velocity(evidence, ambiguity, subspace)
                    first = velocity(state, clock)
                    second = velocity(state + first / steps, clock + 1 / steps)
                    state = state + (first + second) / (2 * steps)
                states[sample] = state
        if not torch.isfinite(states).all():
            raise RuntimeError('non-finite B posterior sample')
        return states

    def set_stage(self, stage):
        self.zero_grad(set_to_none=True)
        for parameter in self.parameters(): parameter.requires_grad_(False)
        modules = dict(representation=[self.codec], forward=[self.forward_operator],
                       flow=[self.conditioner, self.velocity, self.initial_estimate])
        for module in modules[stage]:
            for parameter in module.parameters(): parameter.requires_grad_(True)
