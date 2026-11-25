"""Three small A/D variants: learned evolution, sinusoidal readout and kernels."""
import math
import numpy as np
import torch
from torch import nn
from .models import AGeo, DGeo, GraphLayer, mlp


def initial_state(context, inputs, head):
    features = torch.cat([context['mesh'], context['global_h'].expand(len(context['mesh']), -1)], -1)
    predicted = head(features) * inputs['output_scales']
    chosen = torch.where(inputs['field_masks'][:, 1:3].bool(), inputs['field_values'][:, 1:3], predicted)
    # Existing datasets prescribe homogeneous Dirichlet values on these nodes.
    return chosen * (1 - inputs['features'][:, 2:3])


class FGeo(AGeo):
    def __init__(self, width, layers, heads, memory, evolution_steps=32):
        super().__init__(width, layers, heads, memory)
        self.evolution_steps = evolution_steps
        operator_width = min(width, 64)
        self.initial_head = mlp(2 * width, width, 2)
        self.condition = mlp(2 * width, operator_width, operator_width)
        self.state_input = nn.Linear(operator_width + 4, operator_width)
        self.evolution = nn.ModuleList([GraphLayer(operator_width) for _ in range(2)])
        self.acceleration = mlp(operator_width, operator_width, 1)

    def encode(self, inputs):
        context = super().encode(inputs)
        if inputs['pde'] != 2:
            return context
        scale = inputs['output_scales']; t0, duration = inputs['coordinate_bounds'][:, 2]
        dt = duration / self.evolution_steps
        state = initial_state(context, inputs, self.initial_head)
        condition = self.condition(torch.cat([context['mesh'], context['global_h'].expand(len(state), -1)], -1))
        free = 1 - inputs['features'][:, 2]
        states = [state]
        for step in range(self.evolution_steps):
            time = t0 + step * dt
            forcing = inputs['field_values'][:, :1] * torch.cos(2 * math.pi * inputs['forcing_frequency'] * time)
            clock = state.new_full((len(state), 1), step / self.evolution_steps)
            h = self.state_input(torch.cat([condition, state / scale, clock, forcing / inputs['field_scales'][0]], -1))
            for layer in self.evolution:
                h = layer(h, inputs['nodes'], inputs['edges'])
            acceleration = self.acceleration(h).squeeze(-1) * scale[0] / duration.square()
            velocity = (state[:, 1] + dt * acceleration) * free
            displacement = (state[:, 0] + dt * velocity) * free
            state = torch.stack([displacement, velocity], -1)
            states.append(state)
        context.update(evolution_states=torch.stack(states), evolution_t0=t0, evolution_dt=dt, output_scales=scale)
        return context

    def decode(self, context, queries):
        if 'evolution_states' not in context:
            return super().decode(context, queries)
        position = ((queries['coords'][:, 2] - context['evolution_t0']) / context['evolution_dt']).clamp(0, self.evolution_steps)
        left = position.floor().long().clamp(max=self.evolution_steps - 1)
        fraction = (position - left)[:, None, None]
        states = context['evolution_states']; vertices = queries['vertices']
        values = states[left[:, None], vertices] * (1 - fraction) + states[left[:, None] + 1, vertices] * fraction
        return (values * queries['bary'][..., None]).sum(1) / context['output_scales']


class SineLayer(nn.Module):
    def __init__(self, inputs, outputs, omega, first=False):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.omega = omega
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, values):
        return torch.sin(self.omega * self.linear(values))


class GGeo(DGeo):
    def __init__(self, width, layers, heads, memory, fourier_bands=6, siren_omega=30.):
        super().__init__(width, layers, heads, memory, fourier_bands)
        self.output = nn.Sequential(SineLayer(width, width, siren_omega, first=True),
                                    SineLayer(width, width, siren_omega), nn.Linear(width, 2))
        with torch.no_grad():
            bound = math.sqrt(6 / width) / siren_omega
            self.output[-1].weight.uniform_(-bound, bound)


class HGeo(AGeo):
    def __init__(self, width, layers, heads, memory, green_sources=64):
        super().__init__(width, layers, heads, memory)
        self.green_sources = green_sources
        self.initial_head = mlp(2 * width, width, 2)
        self.source_head = mlp(2 * width, width, 1)
        kernel_width = min(width, 32)
        self.kernel_condition = nn.Linear(2 * width, kernel_width)
        self.kernel = mlp(2 * kernel_width + 8, kernel_width, 3)

    def encode(self, inputs):
        context = super().encode(inputs)
        if inputs['pde'] != 2:
            return context
        chosen = initial_state(context, inputs, self.initial_head)
        features = torch.cat([context['mesh'], context['global_h'].expand(len(chosen), -1)], -1)
        forcing = torch.where(inputs['field_masks'][:, :1].bool(), inputs['field_values'][:, :1],
                              self.source_head(features) * inputs['field_scales'][0]).squeeze(-1)
        # Known-operator kernels do not receive initial/forcing amplitudes or observations.
        operator = context
        if bool(inputs['field_masks'][:, 3].all()):
            operator_inputs = dict(inputs, features=inputs['features'].clone())
            operator_inputs['features'][:, 3:6] = 0
            operator_inputs['features'][:, 7:10] = 0
            operator_inputs.update(observations=torch.zeros_like(inputs['observations']),
                                   observation_mask=torch.ones_like(inputs['observation_mask']))
            operator = self.encoder(operator_inputs)
        condition = self.kernel_condition(torch.cat([operator['mesh'], operator['global_h'].expand(len(chosen), -1)], -1))
        weights = inputs['quadrature_weights']
        if self.green_sources >= len(weights):
            index = np.arange(len(weights)); selected_weights = weights
        else:
            # Fixed importance samples of the lumped FEM quadrature, independent of targets/RNG.
            index = np.random.default_rng(0).choice(len(weights), self.green_sources, replace=True, p=weights / weights.sum())
            selected_weights = np.full(self.green_sources, weights.sum() / self.green_sources)
        index = torch.as_tensor(index, dtype=torch.long, device=chosen.device)
        context.update(wave_initial=chosen, output_scales=inputs['output_scales'], coordinate_bounds=inputs['coordinate_bounds'],
                       green_condition=condition, green_source_condition=condition[index], green_source_nodes=inputs['nodes'][index],
                       green_initial=chosen[index], green_forcing=forcing[index], forcing_frequency=inputs['forcing_frequency'],
                       green_weights=torch.as_tensor(selected_weights, dtype=chosen.dtype, device=chosen.device),
                       green_free=1 - inputs['features'][:, 2])
        return context

    def response(self, context, queries, coords):
        lo, span = context['coordinate_bounds']; duration = span[2]
        time = coords[:, 2:3] - lo[2]
        x = (coords[:, :2] - lo[:2]) / span[:2]
        y = (context['green_source_nodes'] - lo[:2]) / span[:2]
        vertices, bary = queries['vertices'], queries['bary']
        condition = (context['green_condition'][vertices] * bary[..., None]).sum(1)
        count = len(y)
        pairs = torch.cat([x[:, None].expand(-1, count, -1), y[None].expand(len(x), -1, -1), x[:, None] - y[None],
                           condition[:, None].expand(-1, count, -1),
                           context['green_source_condition'][None].expand(len(x), -1, -1)], -1)
        def kernel_at(source_time):
            clock = torch.stack([time / duration, source_time / duration], -1).expand(-1, count, -1)
            return self.kernel(torch.cat([pairs, clock], -1))
        kernel = kernel_at(torch.zeros_like(time))
        scale = context['output_scales'][0]
        initial = context['green_initial'] / scale
        values = kernel[..., 0] * initial[None, :, 0]
        values = values + kernel[..., 1] * (time / duration) * duration * initial[None, :, 1]
        if bool(context['green_forcing'].any()):
            # S(t-s)=(t-s)*K(t,s): midpoint quadrature makes the forced term O(t^2).
            for fraction in (.25, .75):
                source_time = fraction * time
                forcing = context['green_forcing'][None] * torch.cos(2 * math.pi * context['forcing_frequency'] * (lo[2] + source_time))
                values = values + .5 * (1 - fraction) * kernel_at(source_time)[..., 2] * forcing * duration.square() / scale
        free = (context['green_free'][vertices] * bary).sum(1)
        return (values * context['green_weights'][None]).sum(1) * scale / duration.square() * free

    def decode(self, context, queries):
        if 'wave_initial' not in context:
            return super().decode(context, queries)
        gradients = torch.is_grad_enabled(); fields = []
        with torch.enable_grad():
            # Bound pairwise temporary tensors even for thousands of training queries.
            for start in range(0, len(queries['coords']), 128):
                q = {key: value[start:start + 128] for key, value in queries.items()}
                coords = q['coords']
                if not coords.requires_grad:
                    coords = coords.detach().requires_grad_(True)
                initial = (context['wave_initial'][q['vertices']] * q['bary'][..., None]).sum(1)
                time = coords[:, 2] - context['coordinate_bounds'][0, 2]
                u = initial[:, 0] + time * initial[:, 1] + time.square() * self.response(context, q, coords)
                v = torch.autograd.grad(u.sum(), coords, create_graph=gradients, retain_graph=gradients)[0][:, 2]
                fields.append(torch.stack([u, v], -1) / context['output_scales'])
        field = torch.cat(fields)
        return field if gradients else field.detach()
