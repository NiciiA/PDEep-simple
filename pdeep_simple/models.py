"""Shared physical encoders; attention memory (A) and residual FiLM field (C)."""
import math
import torch
from torch import nn

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


class PhysicalEncoder(nn.Module):
    def __init__(self, width, heads):
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

    def forward(self, inputs):
        basis = inputs['basis']
        xy = inputs['nodes'][:, None].expand(-1, basis.shape[1], -1)
        if basis.shape[1]:
            encoded_basis = self.basis_input(torch.cat([xy, basis[..., None]], -1))
            if 'basis_channels' in inputs:
                encoded_basis = encoded_basis + self.pde_types(inputs['basis_channels'])[None]
            prior_features = encoded_basis.mean(1)
        else:
            prior_features = torch.zeros((len(basis), self.null.shape[-1]), device=basis.device)
        h = self.mesh_input(inputs['features']) + prior_features
        for layer in self.graph:
            h = layer(h, inputs['nodes'], inputs['edges'])
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

    def forward(self, h, memory, padding=None):
        out, _ = self.attention(h[None], memory[None], memory[None],
                                key_padding_mask=None if padding is None else padding[None], need_weights=True)
        h = self.norm1(h + out[0])
        return self.norm2(h + self.ff(h))


class AGeo(nn.Module):
    def __init__(self, width, layers, heads, memory):
        super().__init__()
        self.encoder = PhysicalEncoder(width, heads)
        self.tokens = nn.Parameter(torch.randn(memory, width) * .02)
        self.aggregate = AttentionBlock(width, heads)
        self.coordinates = Coordinates(width)
        self.local = nn.Linear(3 * width, width)
        self.decoder = nn.ModuleList([AttentionBlock(width, heads) for _ in range(layers)])
        self.output = mlp(width, width, 2)
        self.parameter = mlp(width, width, 1)

    def encode(self, inputs):
        context = self.encoder(inputs)
        memory = torch.cat([context['mesh'], context['pde'], context['obs'][~context['padding']]])
        global_tokens = self.aggregate(self.tokens + context['global_h'], memory)
        context['memory'] = torch.cat([memory, global_tokens])
        context['log_coefficient'] = self.parameter(context['global_h']).squeeze(-1)
        return context

    def decode(self, context, queries):
        local = torch.cat(local_features(context, queries), -1)
        h = self.coordinates(queries['coords']) + self.local(local)
        for block in self.decoder:
            h = block(h, context['memory'])
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


def build_model(kind='A', size='s', width=None, layers=None, heads=None, memory=None, report=True):
    config = PRESETS[size].copy()
    if kind == 'A' and size == 'l':
        config['layers'] = 8  # about 28M for A, nine FiLM layers give about 27M for C
    for name, value in dict(width=width, layers=layers, heads=heads, memory=memory).items():
        if value is not None:
            config[name] = value
    if min(config.values()) < 1 or config['width'] % config['heads']:
        raise ValueError('positive model dimensions required; width must be divisible by heads')
    model = (AGeo if kind == 'A' else CGeo)(**config)
    if report:
        print(f'{kind}-Geo {size}: {sum(p.numel() for p in model.parameters()):,} parameters; {config}')
    return model
