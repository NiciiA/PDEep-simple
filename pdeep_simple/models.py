"""Shared physical encoders; attention memory (A) and residual FiLM field (C)."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PRESETS = {
    's': dict(width=128, layers=6, heads=4, memory=16),
    'm': dict(width=256, layers=7, heads=8, memory=24),
    'l': dict(width=512, layers=9, heads=8, memory=32),
    'xl': dict(width=768, layers=8, heads=12, memory=48),
    'xxl': dict(width=1024, layers=12, heads=16, memory=64),
}
# Typed PDE nodes: u, Laplacian, dt, dtt, coefficient, multiply, forcing, cubic, sum.
# Edges communicate operator dependencies, separate from spatial mesh edges.
PDE_GRAPHS = [
    ([0, 1, 4, 5, 6], [(0, 1), (1, 3), (2, 3), (3, 4)]),
    ([0, 1, 2, 4, 5, 6], [(0, 1), (0, 2), (1, 4), (3, 4), (4, 2), (5, 2)]),
    ([0, 1, 3, 4, 5, 6], [(0, 1), (0, 2), (1, 4), (3, 4), (4, 2), (5, 2)]),
    ([0, 1, 4, 5, 7, 4, 5, 8, 6],
     [(0, 1), (1, 3), (2, 3), (0, 4), (4, 6), (5, 6), (3, 7), (6, 7), (7, 8)]),
]
# Reuse the same primitive vocabulary/weights for the additional operators.
PDE_GRAPHS += [
    ([0, 1, 4, 5, 6], [(0, 1), (1, 3), (2, 3), (3, 4)]),  # diffusion
    ([0, 1, 4, 5, 1, 4, 5, 4, 5, 8, 6],
     [(0, 1), (1, 3), (2, 3), (0, 4), (4, 6), (5, 6), (0, 8), (7, 8), (3, 9), (6, 9), (8, 9), (9, 10)]),
    ([0, 1, 2, 4, 5, 4, 5, 8, 6],
     [(0, 1), (0, 2), (1, 4), (3, 4), (0, 6), (5, 6), (4, 7), (6, 7), (2, 7), (7, 8)]),
    ([0, 1, 2, 3, 4, 5, 4, 5, 8, 6],
     [(0, 1), (0, 2), (0, 3), (1, 5), (4, 5), (2, 7), (6, 7), (3, 8), (5, 8), (7, 8), (8, 9)]),
    ([0, 1, 4, 5, 1, 6], [(0, 1), (1, 3), (2, 3), (3, 4), (4, 5)]),  # p-Laplacian
    ([0, 1, 8, 4], [(0, 1), (1, 2), (2, 3)]),  # gradient norm = reciprocal speed
]


def mlp(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, outputs))


class GraphLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = mlp(2 * width + 2, width, width)
        self.update = mlp(2 * width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, nodes, edges):
        src, dst = edges.T
        messages = self.message(torch.cat([h[src], h[dst], nodes[src] - nodes[dst]], -1))
        pooled = torch.zeros_like(h).index_add(0, dst, messages)
        degree = torch.zeros(len(h), device=h.device).index_add(0, dst, torch.ones_like(dst, dtype=h.dtype))
        return self.norm(h + self.update(torch.cat([h, pooled / degree.clamp_min(1)[:, None]], -1)))


class OperatorGraphLayer(nn.Module):
    """Directed operand messages, with a distinct embedding for each input role."""
    def __init__(self, width):
        super().__init__()
        from .operator_graph import EDGE_ROLES
        self.roles = nn.Embedding(len(EDGE_ROLES), width)
        self.message = mlp(2 * width, width, width)
        self.update = mlp(2 * width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, edges):
        src, dst, role = edges.T
        messages = self.message(torch.cat([h[src], self.roles(role)], -1))
        pooled = torch.zeros_like(h).index_add(0, dst, messages)
        return self.norm(h + self.update(torch.cat([h, pooled], -1)))


class PhysicalEncoder(nn.Module):
    def __init__(self, width, heads, joint_conditioning=False):
        super().__init__()
        self.mesh_input = mlp(15, width, width)
        self.basis_input = mlp(3, width, width)
        self.graph = nn.ModuleList([GraphLayer(width) for _ in range(2)])
        self.pde_types = nn.Embedding(9, width)
        self.pde_message = nn.ModuleList([mlp(2 * width, width, width) for _ in range(2)])
        self.obs_input = mlp(6, width, width)
        self.null = nn.Parameter(torch.randn(1, width) * .02)
        self.obs_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.global_net = mlp(3 * width, width, width)
        self.joint_conditioning = joint_conditioning
        if joint_conditioning:
            from .operator_graph import NODE_TYPES
            self.pde_types = nn.Embedding(len(NODE_TYPES), width)
            self.pde_message = nn.ModuleList([OperatorGraphLayer(width) for _ in range(2)])
            self.pde_numbers = mlp(8, width, width)
            self.field_refs = nn.Embedding(5, width)
            self.basis_channels = nn.Embedding(4, width)

    def forward(self, inputs):
        basis = inputs['basis']
        xy = inputs['nodes'][:, None].expand(-1, basis.shape[1], -1)
        if basis.shape[1]:
            encoded_basis = self.basis_input(torch.cat([xy, basis[..., None]], -1))
            if 'basis_channels' in inputs:
                embedding = self.basis_channels if self.joint_conditioning else self.pde_types
                encoded_basis = encoded_basis + embedding(inputs['basis_channels'])[None]
            prior_features = encoded_basis.mean(1)
        else:
            prior_features = torch.zeros((len(basis), self.null.shape[-1]), device=basis.device)
        h = self.mesh_input(inputs['features']) + prior_features
        for layer in self.graph:
            h = layer(h, inputs['nodes'], inputs['edges'])
        if self.joint_conditioning:
            from .operator_graph import operator_graph
            graph = operator_graph(inputs)
            p = self.pde_types(graph['types']) + self.pde_numbers(graph['numbers']) + self.field_refs(graph['fields'])
            for layer in self.pde_message:
                p = layer(p, graph['edges'])
            if 'coefficient_samples' in inputs:
                # Preserve exact permitted element coefficients as spatial tokens,
                # explicitly associated with the graph's coefficient-field node.
                coefficient_node = p[graph['fields'] == 4][0]
                p = torch.cat([p, self.basis_input(inputs['coefficient_samples']) + coefficient_node])
        else:
            types, edges = PDE_GRAPHS[inputs['pde']]
            p = self.pde_types(torch.tensor(types, device=h.device))
            adjacency = torch.zeros(len(types), len(types), device=h.device)
            for i, j in edges:
                adjacency[i, j] = adjacency[j, i] = 1
            adjacency /= adjacency.sum(-1, keepdim=True).clamp_min(1)
            for layer in self.pde_message:
                p = p + layer(torch.cat([p, adjacency @ p], -1))
            if 'parameters' in inputs:
                p = torch.cat([p, self.basis_input(inputs['parameters'])])
        # Always-valid null token makes an empty observation set safe.
        obs = torch.cat([self.null, self.obs_input(inputs['observations'])], 0)
        padding = torch.cat([torch.zeros(1, dtype=torch.bool, device=h.device), inputs['observation_mask']])
        attended, _ = self.obs_attention(obs[None], obs[None], obs[None], key_padding_mask=padding[None],
                                         need_weights=True)
        obs = obs + attended[0]
        pooled_obs = obs[~padding].mean(0)
        global_h = self.global_net(torch.cat([h.mean(0), p.mean(0), pooled_obs]))
        return dict(mesh=h, pde=p, obs=obs, padding=padding, global_h=global_h,
                    observation_coords=inputs['observations'][:, :3])


class Coordinates(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('frequencies', 2 ** torch.arange(5, dtype=torch.float32) * math.pi)
        self.linear = nn.Linear(33, width)

    def forward(self, coords):
        angles = (coords[..., None] * self.frequencies).flatten(-2)
        return self.linear(torch.cat([coords, angles.sin(), angles.cos()], -1))


def local_features(context, queries):
    local_mesh = (context['mesh'][queries['vertices']] * queries['bary'][..., None]).sum(1)
    # A location-sensitive, permutation-invariant local observation summary.
    delta = queries['coords'][:, None] - context['observation_coords'][None]
    weight = torch.exp(-delta.square().sum(-1) / .12)
    weight = weight * (~context['padding'][1:])[None]
    local_obs = weight @ context['obs'][1:] / weight.sum(-1, keepdim=True).clamp_min(1e-6)
    global_h = context['global_h'].expand(len(local_mesh), -1)
    return local_mesh, local_obs, global_h


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = mlp(width, 2 * width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, h, memory, padding=None, bias=None, observation_start=0):
        if bias is None:
            out, _ = self.attention(h[None], memory[None], memory[None],
                                    key_padding_mask=None if padding is None else padding[None], need_weights=True)
            out = out[0]
        else:
            # Use this block's existing MHA projections and output. Add only a
            # Q-by-sensor bias slice, without allocating another Q-by-mesh mask.
            heads = self.attention.num_heads
            dimension = h.shape[-1] // heads
            weights = self.attention.in_proj_weight.chunk(3)
            offsets = self.attention.in_proj_bias.chunk(3)
            q, k, v = [F.linear(x, w, b).reshape(len(x), heads, dimension).transpose(0, 1)
                       for x, w, b in zip((h, memory, memory), weights, offsets)]
            logits = (q / math.sqrt(dimension)) @ k.transpose(-1, -2)
            end = observation_start + bias.shape[-1]
            logits[:, :, observation_start:end] = logits[:, :, observation_start:end] + bias
            if padding is not None:
                logits = logits.masked_fill(padding[None, None], float('-inf'))
            out = (logits.softmax(-1) @ v).transpose(0, 1).reshape_as(h)
            out = self.attention.out_proj(out)
        h = self.norm1(h + out)
        return self.norm2(h + self.ff(h))


class ObservationBias(nn.Module):
    """Per-head sensor logit bias; signed, scaled space/time and normalized noise."""
    def __init__(self, heads):
        super().__init__()
        self.quantity = nn.Embedding(2, 8)  # Existing IDs: scalar/u=0, wave v=1.
        self.network = mlp(13, 32, heads)

    def forward(self, context, coords):
        obs = context['observation_tokens']
        delta = (coords[:, None] - obs[None, :, :3]) / context['relative_scale']
        if not context['time_dependent']:
            delta = torch.cat([delta[..., :2], torch.zeros_like(delta[..., 2:3])], -1)
        # Smooth spatial norm also supports the optional derivative-based losses.
        distance = torch.sqrt(delta[..., :2].square().sum(-1, keepdim=True) + 1e-12)
        quantity = self.quantity(obs[:, 3].long()).expand(len(coords), -1, -1)
        uncertainty = obs[:, 5].clamp_min(1e-8).log()[None, :, None].expand(len(coords), -1, -1)
        relative = torch.cat([delta, distance, quantity, uncertainty], -1)
        return self.network(relative).permute(2, 0, 1)


class AGeo(nn.Module):
    def __init__(self, width, layers, heads, memory, joint_conditioning=False):
        super().__init__()
        self.joint_conditioning = joint_conditioning
        self.encoder = PhysicalEncoder(width, heads, joint_conditioning)
        self.tokens = nn.Parameter(torch.randn(memory, width) * .02)
        self.aggregate = AttentionBlock(width, heads)
        self.coordinates = Coordinates(width)
        self.local = nn.Linear(3 * width, width)
        self.decoder = nn.ModuleList([AttentionBlock(width, heads) for _ in range(layers)])
        self.output = mlp(width, width, 2)
        self.parameter = mlp(width, width, 1)
        if joint_conditioning:
            self.mesh_fusion = AttentionBlock(width, heads)
            self.observation_bias = ObservationBias(heads)
            self.register_buffer('_joint_version', torch.tensor(1))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        joint = prefix + '_joint_version' in state_dict
        if joint != self.joint_conditioning:
            raise ValueError('Original and joint-conditioning A checkpoints are incompatible; train the new variant from scratch')
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def encode(self, inputs):
        context = self.encoder(inputs)
        padding = None
        if self.joint_conditioning:
            from .data import PDES
            from .physics import PARABOLIC, WAVES
            context['physical_mesh'] = context['mesh']
            conditioning = torch.cat([context['pde'], context['obs']])
            padding = torch.cat([context['padding'].new_zeros(len(context['pde'])), context['padding']])
            context['mesh'] = self.mesh_fusion(context['mesh'], conditioning, padding)
            context['global_h'] = self.encoder.global_net(torch.cat([
                context['mesh'].mean(0), context['pde'].mean(0), context['obs'][~context['padding']].mean(0)]))
            # Only valid sensors enter decoder memory; the null stays valid.
            memory = torch.cat([context['mesh'], context['pde'], context['obs'][~context['padding']]])
            padding = None
            context.update(observation_tokens=inputs['observations'][~inputs['observation_mask']],
                           observation_start=len(context['mesh']) + len(context['pde']) + 1,
                           relative_scale=inputs['coordinate_bounds'][1].clamp_min(1e-8),
                           time_dependent=PDES[inputs['pde']] in PARABOLIC + WAVES)
        else:
            memory = torch.cat([context['mesh'], context['pde'], context['obs'][~context['padding']]])
        global_tokens = self.aggregate(self.tokens + context['global_h'], memory, padding)
        context['memory'] = torch.cat([memory, global_tokens])
        context['log_coefficient'] = self.parameter(context['global_h']).squeeze(-1)
        return context

    def decode(self, context, queries):
        bias = None
        if self.joint_conditioning:
            def interpolate(features):
                return (features[queries['vertices']] * queries['bary'][..., None]).sum(1)
            # Keep original local physics alongside the fused spatial features.
            # The fixed Gaussian sensor summary is replaced by relative attention.
            local = torch.cat([interpolate(context['physical_mesh']), interpolate(context['mesh']),
                               context['global_h'].expand(len(queries['coords']), -1)], -1)
            if len(context['observation_tokens']):
                bias = self.observation_bias(context, queries['coords'])
        else:
            local = torch.cat(local_features(context, queries), -1)
        h = self.coordinates(queries['coords']) + self.local(local)
        for block in self.decoder:
            h = block(h, context['memory'], bias=bias, observation_start=context.get('observation_start', 0))
        return self.output(h)

    def forward(self, inputs, queries):
        context = self.encode(inputs)
        return dict(field=self.decode(context, queries), log_coefficient=context['log_coefficient'])


class FiLMLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.modulation = nn.Linear(3 * width, 2 * width)
        self.residual = mlp(width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, condition):
        gamma, beta = self.modulation(condition).chunk(2, -1)
        return h + self.residual((1 + .1 * gamma) * self.norm(h) + beta)


class CGeo(nn.Module):
    def __init__(self, width, layers, heads, memory):
        super().__init__()
        self.encoder = PhysicalEncoder(width, heads)
        self.coordinates = Coordinates(width)
        self.decoder = nn.ModuleList([FiLMLayer(width) for _ in range(layers)])
        self.output = mlp(width, width, 2)
        self.parameter = mlp(width, width, 1)

    def encode(self, inputs):
        context = self.encoder(inputs)
        context['log_coefficient'] = self.parameter(context['global_h']).squeeze(-1)
        return context

    def decode(self, context, queries):
        condition = torch.cat(local_features(context, queries), -1)
        h = self.coordinates(queries['coords'])
        for layer in self.decoder:
            h = layer(h, condition)
        return self.output(h)

    def forward(self, inputs, queries):
        context = self.encode(inputs)
        return dict(field=self.decode(context, queries), log_coefficient=context['log_coefficient'])


class FourierCoordinates(nn.Module):
    def __init__(self, width, bands):
        super().__init__()
        self.register_buffer('frequencies', 2 * math.pi * 2 ** torch.arange(bands, dtype=torch.float32))
        self.linear = nn.Linear(3 + 6 * bands, width)

    def forward(self, coords):
        angles = (coords[..., None] * self.frequencies).flatten(-2)
        return self.linear(torch.cat([coords, angles.sin(), angles.cos()], -1))


class DGeo(AGeo):
    def __init__(self, width, layers, heads, memory, fourier_bands=6):
        super().__init__(width, layers, heads, memory)
        self.coordinates = FourierCoordinates(width, fourier_bands)

    def encode(self, inputs):
        context = super().encode(inputs)
        context['coordinate_bounds'] = inputs['coordinate_bounds']
        return context

    def decode(self, context, queries):
        lo, span = context['coordinate_bounds']
        coords = (queries['coords'] - lo) / span
        h = self.coordinates(coords) + self.local(torch.cat(local_features(context, queries), -1))
        for block in self.decoder:
            h = block(h, context['memory'])
        return self.output(h)


class EGeo(AGeo):
    def __init__(self, width, layers, heads, memory, wave_modes=32, initial_latent=None,
                 assimilation=False, infer_operator=False):
        super().__init__(width, layers, heads, memory)
        self.wave_modes = wave_modes
        self.mode_head = mlp(2 * width + 1, width, 3)
        self.initial_head = mlp(2 * width, width, 2)
        self.wave_cache = {}
        self.initial_latent = initial_latent
        self.assimilation, self.infer_operator = assimilation, infer_operator
        if initial_latent is not None:
            from .flow import InitialStateCodec
            self.initial_codec = InitialStateCodec(width, initial_latent)
            self.initial_condition = mlp(width, width, initial_latent)

    def encode(self, inputs, correct=True):
        context = super().encode(inputs)
        context['wave'] = None
        if inputs['pde'] == 2:
            return self.wave_context(context, inputs, correct=correct)
        from .data import PDES
        from .physics import PARABOLIC
        pde = PDES[inputs['pde']]
        if pde in PARABOLIC + ('damped_wave',):
            return self.evolution_context(context, inputs, pde)
        return context

    def evolution_context(self, context, inputs, pde):
        from .wave import modal_basis, damped_trajectory
        damped = pde == 'damped_wave'
        scale = inputs['output_scales']; global_h = context['global_h']
        predicted = self.initial_head(torch.cat([context['mesh'], global_h.expand(len(context['mesh']), -1)], -1)) * scale
        chosen = torch.where(inputs['field_masks'][:, 1:3].bool(), inputs['field_values'][:, 1:3], predicted)
        context.update(evolution_kind=pde, wave_initial=chosen, output_scales=scale,
                       wave_t0=inputs['coordinate_bounds'][0, 2], operator_inferred=False)
        operator = inputs.get('evolution_operator')
        factor = scale.new_ones(())
        if bool(inputs['scalar_masks'][0]):
            context['log_coefficient'] = inputs['scalar'][0].log()
        if operator is None and self.infer_operator and inputs['evolution_scalar_operator']:
            # T2 hides only a constant scalar: geometry modes need no hidden operator.
            operator = np.ones(len(chosen))
            if bool(inputs['field_masks'][:, 3].any()):
                coefficient = inputs['field_values'][:, 3].sum() / inputs['field_masks'][:, 3].sum()
            elif bool(inputs['scalar_masks'][0]):
                coefficient = inputs['scalar'][0]
            else:
                coefficient = context['log_coefficient'].exp()
                context['operator_inferred'] = True
            context['log_coefficient'] = coefficient.log()
            factor = coefficient if damped else coefficient.sqrt()
        if operator is None:
            return context  # Still initial-aware; learn evolution through the anchored residual.
        basis_inputs = dict(inputs, wave_geometry=inputs['evolution_geometry'], wave_operator=operator, wave_reaction=0.)
        basis = modal_basis(basis_inputs, self.wave_modes, self.wave_cache)
        if basis is None:
            return context
        basis = dict(basis, omega=basis['omega'] * factor)
        context['wave'] = basis
        projection, omega = basis['projection'], basis['omega']
        initial = projection @ chosen
        summary = torch.cat([projection @ context['mesh'], global_h.expand(len(omega), -1), omega.log1p()[:, None]], -1)
        forcing = (projection @ inputs['field_values'][:, 0] if bool(inputs['field_masks'][:, 0].all()) else
                   self.mode_head(summary)[:, 2] * inputs['field_scales'][0])
        context.update(wave_a=initial[:, 0], wave_b=initial[:, 1], wave_forcing=forcing,
                       forcing_frequency=inputs['forcing_frequency'])
        if damped:
            trajectory, dt = damped_trajectory(omega, initial[:, 0], initial[:, 1], forcing, inputs['damping'],
                inputs['forcing_frequency'], context['wave_t0'], inputs['coordinate_bounds'][1, 2])
            context.update(damped_trajectory=trajectory, damped_dt=dt)
        else:
            # Dataset reaction is a linear sink, whose mass-orthonormal projection is -rho*q.
            context['diffusion_rates'] = omega.square() + inputs['reaction']
        return context

    def wave_context(self, context, inputs, latent=None, correct=True):
        from .wave import modal_basis, assimilate, observation_inputs
        context = dict(context)
        scale = inputs['output_scales']
        global_h = context['global_h']
        if self.initial_latent is None:
            predicted_initial = self.initial_head(torch.cat([context['mesh'], global_h.expand(len(context['mesh']), -1)], -1)) * scale
        else:
            weights = torch.as_tensor(inputs['quadrature_weights'], dtype=scale.dtype, device=scale.device)
            geometry = self.initial_codec.geometry(inputs['nodes'], inputs['coordinate_bounds'], weights, inputs['features'][:, 2])
            latent = self.initial_condition(global_h) if latent is None else latent
            predicted_initial = self.initial_codec.decode(geometry, latent) * scale
        chosen = torch.where(inputs['field_masks'][:, 1:3].bool(), inputs['field_values'][:, 1:3], predicted_initial)
        context.update(wave_initial=chosen, output_scales=scale, wave_t0=inputs['coordinate_bounds'][0, 2])
        basis = modal_basis(inputs, self.wave_modes, self.wave_cache) if 'wave_operator' in inputs else None
        if self.initial_latent is not None or self.assimilation or self.infer_operator:
            context.update(wave_inputs=inputs, operator_inferred=False)
        if self.infer_operator and bool(inputs['scalar_masks'][0]):
            context['log_coefficient'] = inputs['scalar'][0].log()
        if basis is None and self.infer_operator and inputs.get('wave_scalar_operator'):
            geometry_inputs = dict(inputs, wave_operator=np.ones(len(chosen)), wave_reaction=0.)
            basis = modal_basis(geometry_inputs, self.wave_modes, self.wave_cache)
            if basis is not None:
                if bool(inputs['field_masks'][:, 3].any()):
                    speed = inputs['field_values'][:, 3].sum() / inputs['field_masks'][:, 3].sum()
                    context['log_coefficient'] = speed.log()
                else:
                    speed = context['log_coefficient'].exp()
                    context['operator_inferred'] = not bool(inputs['scalar_masks'][0])
                if bool(inputs['scalar_masks'][0]):
                    speed = inputs['scalar'][0]
                context['geometry_omega'] = basis['omega']
                basis = dict(basis, omega=basis['omega'] * speed)
        context['wave'] = basis
        if basis is not None:
            projection, omega = basis['projection'], basis['omega']
            summary = torch.cat([projection @ context['mesh'], global_h.expand(len(omega), -1), omega.log1p()[:, None]], -1)
            coefficients = self.mode_head(summary)
            initial = projection @ chosen
            a = initial[:, 0] if self.initial_latent is not None or bool(inputs['field_masks'][:, 1].all()) else coefficients[:, 0] * scale[0]
            b = initial[:, 1] if self.initial_latent is not None or bool(inputs['field_masks'][:, 2].all()) else coefficients[:, 1] * scale[1]
            forcing = (projection @ inputs['field_values'][:, 0] if bool(inputs['field_masks'][:, 0].all()) else
                       coefficients[:, 2] * inputs['field_scales'][0])
            context.update(wave_a=a, wave_b=b, wave_forcing=forcing, forcing_frequency=inputs['forcing_frequency'])
        if self.assimilation:
            observation_inputs(context, inputs)
            if correct:
                context = assimilate(self, context)
        return context

    def with_initial_latent(self, context, latent):
        if 'wave_inputs' not in context:
            return context
        raw = dict(context, log_coefficient=self.parameter(context['global_h']).squeeze(-1))
        return self.wave_context(raw, context['wave_inputs'], latent)

    def reference_latent(self, case, scales):
        if case['meta']['pde'] != 'wave':
            return next(self.parameters()).new_zeros(self.initial_latent)
        geometry, values, weights = self.reference_initial(case, scales)
        return self.initial_codec(geometry[0], values, weights)

    def reference_initial(self, case, scales):
        """Complete initial fields are targets here, never conditioning inputs."""
        from .geometry import node_weights
        device = next(self.parameters()).device
        nodes = torch.as_tensor(case['nodes'], dtype=torch.float32, device=device)
        bounds = torch.stack([torch.cat([nodes.amin(0), nodes.new_zeros(1)]),
                              torch.cat([nodes.amax(0) - nodes.amin(0), nodes.new_ones(1)])])
        weights = torch.as_tensor(node_weights(case), dtype=nodes.dtype, device=device)
        boundary = torch.zeros(len(nodes), device=device); boundary[case['boundary_nodes']] = 1
        geometry = self.initial_codec.geometry(nodes, bounds, weights, boundary)
        values = torch.as_tensor(np.column_stack([case['initial_u'], case['initial_v']]) / scales['output'], dtype=nodes.dtype, device=device)
        return geometry, values, weights

    def representation_loss(self, case, scales):
        geometry, target, weights = self.reference_initial(case, scales)
        latent = self.initial_codec(geometry[0], target, weights)
        error = (self.initial_codec.decode(geometry, latent) - target).square().mean(-1)
        return (error * weights).sum() / weights.sum()

    def decode(self, context, queries):
        from .wave import modal_evolution, diffusion_evolution, trajectory_values
        basis = context['wave']
        if 'wave_initial' not in context:
            return super().decode(context, queries)
        if context.get('evolution_kind') in ('heat', 'reaction_diffusion'):
            initial = (context['wave_initial'][queries['vertices'], 0] * queries['bary']).sum(1)
            time = queries['coords'][:, 2:3] - context['wave_t0']
            residual = super().decode(context, queries)[:, 0] * context['output_scales'][0]
            u = initial + time[:, 0] * residual
            if basis is not None:
                phi = (basis['phi'][queries['vertices']] * queries['bary'][..., None]).sum(1)
                q = diffusion_evolution(context['diffusion_rates'], context['wave_a'], context['wave_forcing'],
                                        time, context['forcing_frequency'], context['wave_t0'])
                u = u + (phi * (q - context['wave_a'])).sum(-1)
            return torch.stack([u, torch.zeros_like(u)], -1) / context['output_scales']
        # Evaluation uses no_grad externally; residual velocity still needs d/dt.
        gradients = torch.is_grad_enabled()
        with torch.enable_grad():
            coords = queries['coords']
            if not coords.requires_grad:
                coords = coords.detach().requires_grad_(True)
            residual = super().decode(context, dict(queries, coords=coords))[:, 0] * context['output_scales'][0]
            residual_dt = torch.autograd.grad(residual.sum(), coords, create_graph=gradients,
                                              retain_graph=gradients)[0][:, 2]
            initial = (context['wave_initial'][queries['vertices']] * queries['bary'][..., None]).sum(1)
            time = coords[:, 2:3] - context['wave_t0']
            u = initial[:, 0] + time[:, 0] * initial[:, 1] + time[:, 0].square() * residual
            v = initial[:, 1] + 2 * time[:, 0] * residual + time[:, 0].square() * residual_dt
            if basis is not None:
                phi = (basis['phi'][queries['vertices']] * queries['bary'][..., None]).sum(1)
                a, b = context['wave_a'], context['wave_b']
                if 'damped_trajectory' in context:
                    q, q_dt = trajectory_values(context['damped_trajectory'], context['damped_dt'], time)
                else:
                    q, q_dt = modal_evolution(basis['omega'], a, b, context['wave_forcing'], time,
                                              context['forcing_frequency'])
                u = u + (phi * (q - a - time * b)).sum(-1)
                v = v + (phi * (q_dt - b)).sum(-1)
            field = torch.stack([u, v], -1) / context['output_scales']
        return field if gradients else field.detach()


def build_model(kind='A', size='s', width=None, layers=None, heads=None, memory=None, report=True,
                fourier_bands=6, wave_modes=32, initial_latent=None, assimilation=None, infer_operator=False,
                f_variant='base', f_steps=3, joint_conditioning=False):
    config = PRESETS[size].copy()
    if kind in ('A', 'D', 'E', 'F') and size == 'l':
        config['layers'] = 8  # about 28M for A, nine FiLM layers give about 27M for C
    for name, value in dict(width=width, layers=layers, heads=heads, memory=memory).items():
        if value is not None:
            config[name] = value
    if min(config.values()) < 1 or config['width'] % config['heads']:
        raise ValueError('positive model dimensions required; width must be divisible by heads')
    if fourier_bands < 1 or wave_modes < 1:
        raise ValueError('fourier-bands and wave-modes must be positive')
    classes = dict(A=AGeo, C=CGeo, D=DGeo, E=EGeo)
    options = dict(fourier_bands=fourier_bands) if kind == 'D' else dict(wave_modes=wave_modes) if kind == 'E' else {}
    if joint_conditioning:
        if kind != 'A':
            raise ValueError('--joint-conditioning is only available for A-Geo')
        options['joint_conditioning'] = True
    if initial_latent is not None and initial_latent < 1:
        raise ValueError('initial-latent must be positive')
    if initial_latent is not None or assimilation or infer_operator:
        if kind not in ('E', 'F'):
            raise ValueError('initial-latent, assimilation and infer-operator are E/F-Geo options')
        options.update(initial_latent=initial_latent, assimilation=bool(assimilation), infer_operator=infer_operator)
    if kind == 'F':
        from .propagation import FGeo
        classes['F'] = FGeo
        options = dict(wave_modes=wave_modes, initial_latent=initial_latent or 32,
                       assimilation=True if assimilation is None else assimilation, infer_operator=infer_operator,
                       f_variant=f_variant, f_steps=f_steps)
    if kind not in classes:
        raise ValueError('available models are A, C, D, E and the new F; G/H were removed')
    model = classes[kind](**config, **options)
    if report:
        variant = ' joint-conditioning' if joint_conditioning else ''
        print(f'{kind}-Geo{variant} {size}: {sum(p.numel() for p in model.parameters()):,} parameters; {config}')
    return model
