"""A compact field autoencoder and conditional latent flow; no query-wise noise."""
import numpy as np
import torch
from torch import nn
from .models import Coordinates, mlp
from .data import query_draw


class FieldEncoder(nn.Module):
    def __init__(self, width, latent_dim):
        super().__init__()
        self.coordinates = Coordinates(width)
        self.points = mlp(3 * width + 2, width, width)
        self.pool = mlp(2 * width, width, latent_dim)

    def forward(self, coords, values, weights=None):
        h = self.coordinates(coords)
        # Coordinate/field products retain spatial structure under set pooling.
        features = torch.cat([h, h * values[:, :1], h * values[:, 1:2], values], -1)
        h = self.points(features)
        mean = h.mean(0) if weights is None else (h * weights[:, None]).sum(0) / weights.sum()
        return self.pool(torch.cat([mean, h.amax(0)]))


class InitialStateCodec(FieldEncoder):
    """Variable-mesh initial fields, with one geometry-conditioned spatial basis."""
    def __init__(self, width, latent_dim):
        super().__init__(width, latent_dim)
        self.latent_dim = latent_dim
        self.spatial_basis = mlp(2 * width, width, 2 * latent_dim)
        self.spatial_bias = mlp(2 * width, width, 2)

    def geometry(self, nodes, bounds, weights, boundary):
        coords = torch.cat([(nodes - bounds[0, :2]) / bounds[1, :2], torch.zeros_like(nodes[:, :1])], -1)
        h = self.coordinates(coords)
        pooled = (h * weights[:, None]).sum(0) / weights.sum()
        return coords, torch.cat([h, pooled.expand(len(h), -1)], -1), 1 - boundary

    def decode(self, geometry, latent):
        _, features, free = geometry
        basis = self.spatial_basis(features).reshape(-1, 2, self.latent_dim)
        return ((basis * latent).sum(-1) / np.sqrt(self.latent_dim) + self.spatial_bias(features)) * free[:, None]


class LatentFlow(nn.Module):
    def __init__(self, base, latent_dim=32, latent_points=256):
        super().__init__()
        if latent_dim < 1 or latent_points < 8:
            raise ValueError('latent-dim must be positive and latent-points at least eight')
        self.base = base
        width = base.coordinates.linear.out_features
        self.initial_only = hasattr(base, 'initial_codec')
        if self.initial_only:
            latent_dim = base.initial_latent
        else:
            self.field_encoder = FieldEncoder(width, latent_dim)
            self.latent_condition = nn.Linear(latent_dim, width)
            self.latent_basis = mlp(width, width, 2 * latent_dim)
        self.velocity = nn.Sequential(nn.Linear(latent_dim + width + 3, width), nn.SiLU(),
                                      nn.Linear(width, width), nn.SiLU(), nn.Linear(width, latent_dim))
        self.latent_dim, self.latent_points = latent_dim, latent_points
        self.register_buffer('latent_mean', torch.zeros(latent_dim))
        self.register_buffer('latent_scale', torch.ones(latent_dim))

    def encode(self, inputs):
        return self.base.encode(inputs, correct=False) if self.initial_only else self.base.encode(inputs)

    def with_latent(self, context, latent):
        """One latent conditions the complete physical field, including u and v."""
        if self.initial_only:
            return self.base.with_initial_latent(context, latent)
        result = dict(context)
        shift = self.latent_condition(latent)
        result['global_h'] = context['global_h'] + shift
        if 'memory' in context:  # A retains its defining attention decoder.
            result['memory'] = context['memory'] + shift
        result['latent'] = latent
        result['log_coefficient'] = self.base.parameter(result['global_h']).squeeze(-1)
        return result

    def decode(self, context, queries):
        if self.initial_only:
            return self.base.decode(context, queries)
        if 'latent' not in context:
            raise ValueError('flow decoding needs one latent for the whole field')
        h = self.base.coordinates(queries['coords'])
        basis = self.latent_basis(h).reshape(-1, 2, self.latent_dim)
        # An explicit spatial latent path supplements modulation of the A/C decoder.
        latent_field = (basis * context['latent']).sum(-1) / np.sqrt(self.latent_dim)
        return self.base.decode(context, queries) + latent_field

    def forward(self, inputs, queries, latent):
        context = self.with_latent(self.encode(inputs), latent)
        return dict(field=self.decode(context, queries), log_coefficient=context['log_coefficient'])

    def reference_latent(self, case, scales):
        """Training/AE validation only. Inference never calls this function."""
        if self.initial_only:
            return self.base.reference_latent(case, scales)
        from .geometry import interpolate
        device = self.latent_mean.device
        rng = np.random.default_rng([case['meta']['seed'], 731])
        coords, values = query_draw(case, self.latent_points, rng)
        if len(case['times']) > 1:
            coords[:self.latent_points // 4, 2] = 0.
            values = interpolate(case, coords)
        coords = torch.tensor(coords, dtype=torch.float32, device=device)
        values = torch.tensor(values / scales['output'], dtype=torch.float32, device=device)
        return self.field_encoder(coords, values)

    def calibrate(self, cases, scales):
        with torch.no_grad():
            z = torch.stack([self.reference_latent(case, scales) for case in cases])
            values = z[[i for i, case in enumerate(cases) if case['meta']['pde'] == 'wave']] if self.initial_only else z
            self.latent_mean.copy_(values.mean(0))
            self.latent_scale.copy_(values.std(0, unbiased=False).clamp_min(.05))
        return z

    def set_stage(self, stage):
        # Frozen AE gradients must not enter flow-stage gradient clipping.
        self.zero_grad(set_to_none=True)
        for parameter in self.parameters():
            parameter.requires_grad_(stage == 'ae')
        for parameter in self.velocity.parameters():
            parameter.requires_grad_(stage == 'flow')

    def flow_velocity(self, z, flow_time, context):
        # Flow time s is separate from physical time in queries['coords'][:, 2].
        s = flow_time.expand(len(z), 1)
        clock = torch.cat([s, (torch.pi * s).sin(), (torch.pi * s).cos()], -1)
        condition = context['global_h'].expand(len(z), -1)
        return self.velocity(torch.cat([z, clock, condition], -1))

    def matching_loss(self, context, target, rng):
        z1 = (target.detach() - self.latent_mean) / self.latent_scale
        z0 = torch.tensor(rng.standard_normal(self.latent_dim), dtype=z1.dtype, device=z1.device)
        flow_time = torch.tensor(rng.random(), dtype=z1.dtype, device=z1.device)
        zt = (1 - flow_time) * z0 + flow_time * z1
        return (self.flow_velocity(zt[None], flow_time, context)[0] - (z1 - z0)).square().mean()

    def sample_latents(self, context, samples=32, steps=32, seed=123):
        """Heun integration. Draw once, then reuse these latents for every query chunk."""
        if samples < 1 or steps < 1:
            raise ValueError('samples and flow-steps must be positive')
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        z = torch.randn(samples, self.latent_dim, generator=generator).to(self.latent_mean.device)
        h = 1. / steps
        with torch.no_grad():
            for index in range(steps):
                s = torch.tensor(index * h, dtype=z.dtype, device=z.device)
                first = self.flow_velocity(z, s, context)
                second = self.flow_velocity(z + h * first, s + h, context)
                z = z + h / 2 * (first + second)
        if not torch.isfinite(z).all():
            raise RuntimeError('non-finite latent flow sample')
        return z * self.latent_scale + self.latent_mean
