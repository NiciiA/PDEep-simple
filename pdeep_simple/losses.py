"""Small selectable objectives; weak losses use the actual P1 mesh matrices."""
import numpy as np
import torch
from .data import query_input, case_scales
from .physics import assemble, helmholtz_operator, STATIONARY, WAVES, PARABOLIC, time_dependent
from .geometry import element_geometry

LOSS_NAMES = ('field', 'aux', 'kinematic', 'weak', 'initial', 'boundary')


def parse_losses(text):
    weights = dict.fromkeys(LOSS_NAMES, 0.)
    for item in text.split(','):
        key, value = item.split('=')
        if key not in weights or float(value) < 0:
            raise ValueError(f'losses must be nonnegative; choose {LOSS_NAMES}')
        weights[key] = float(value)
    if not any(weights.values()):
        raise ValueError('at least one positive loss weight is required')
    return weights


def derivative(values, coords, channel):
    return torch.autograd.grad(values[:, channel].sum(), coords, create_graph=True, retain_graph=True)[0][:, 2]


def weak_available(case, inputs):
    return (case['meta']['pde'] not in ('eikonal', 'navier_stokes') and bool(inputs['field_masks'][:, 0].all()) and
            (bool(inputs['field_masks'][:, 3].all()) or case['meta']['tier'] == 'T2'))


def weak_residual(model, context, case, inputs, scales, time=.5):
    """Residual of the P1 projection of the neural field; centered temporal differences.

    This tests finitely many mesh functions, not the network's strong residual.
    Unknown T2 scalar coefficients use the positive auxiliary prediction.
    """
    device = inputs['nodes'].device
    pde = case['meta']['pde']
    scale = torch.tensor(scales['output'], device=device)
    step = min(.01, (case['times'][-1] - case['times'][0]) / 4) if time_dependent(case) else .01
    time = np.clip(time, case['times'][0] + step, case['times'][-1] - step) if time_dependent(case) else 0.
    times = [0.] if pde in STATIONARY else [time - step, time, time + step]
    fields = []
    for t in times:
        coords = np.column_stack([case['nodes'], np.full(len(case['nodes']), t)])
        fields.append(model.decode(context, query_input(case, coords, device))[:, :len(scale)] * scale)
    coefficient_known = bool(inputs['scalar_masks'][0])
    coefficient = inputs['scalar'][0] if coefficient_known else context['log_coefficient'].exp()
    if pde == 'helmholtz':
        known = bool(inputs['field_masks'][:, 3].all())
        mass, stiffness = (helmholtz_operator(case, case['element_coefficient'][:, None] if 'element_coefficient' in case else
                                            inputs['field_values'][:, 3].detach().cpu().numpy()) if known else assemble(case))
        mass = torch.as_tensor(mass.toarray(), dtype=torch.float32, device=device)
        stiffness = torch.as_tensor(stiffness.toarray(), dtype=torch.float32, device=device)
        u = fields[0][:, 0]
        residual = stiffness @ u - mass @ inputs['field_values'][:, 0]
        if not known: residual = residual - coefficient.square() * (mass @ u)
        free = np.setdiff1d(np.arange(len(u)), case['boundary_nodes'])
        return residual[free] / mass.sum(-1)[free].clamp_min(1e-10)
    if bool(inputs['field_masks'][:, 3].all()):
        operator = (case['element_coefficient'][:, None] if 'element_coefficient' in case else
                    inputs['field_values'][:, 3].detach().cpu().numpy())
        mass, stiffness = assemble(case, operator ** (2 if pde in WAVES else 1))
        factor = 1.
    else:
        mass, stiffness = assemble(case)
        factor = coefficient ** (2 if pde in WAVES else 1)
    # Dense small mesh matrices keep CPU/MPS/CUDA autograd portable and readable.
    mass = torch.tensor(mass.toarray(), dtype=torch.float32, device=device)
    stiffness = torch.tensor(stiffness.toarray(), dtype=torch.float32, device=device)
    lump = mass.sum(-1)
    u = fields[len(fields) // 2][:, 0]
    physical = case['meta'].get('parameters', {})
    _, extra = assemble(case, 0., physical.get('advection'), physical.get('reaction', 0.))
    extra = torch.tensor(extra.toarray(), dtype=torch.float32, device=device)
    forcing = inputs['field_values'][:, 0] * np.cos(2 * np.pi * physical.get('forcing_frequency', 0.) * times[len(times) // 2])
    residual = factor * (stiffness @ u) + extra @ u - mass @ forcing
    if pde == 'p_laplacian':
        area, gradients = element_geometry(case['nodes'], case['triangles'])
        gradients = torch.tensor(gradients, dtype=torch.float32, device=device)
        triangles = torch.tensor(case['triangles'], dtype=torch.long, device=device)
        grad = (u[triangles, None] * gradients).sum(1)
        epsilon = case['meta'].get('numerics', {}).get('p_regularization', 1e-4)
        power = (epsilon ** 2 + grad.square().sum(-1)) ** ((physical.get('p', 3.) - 2) / 2)
        local = torch.tensor(area, dtype=torch.float32, device=device)[:, None] * power[:, None] * (gradients * grad[:, None]).sum(-1)
        residual = torch.zeros_like(u).index_add(0, triangles.flatten(), local.flatten()) - mass @ forcing
    if pde in PARABOLIC:
        residual = residual + mass @ ((fields[2][:, 0] - fields[0][:, 0]) / (2 * step))
    if pde in WAVES:
        residual = residual + mass @ ((fields[2][:, 0] - 2 * u + fields[0][:, 0]) / step ** 2)
        residual = residual + physical.get('damping', 0.) * (mass @ fields[1][:, 1])
    if pde == 'semilinear':
        residual = residual + inputs['scalar'][1] * lump * u ** 3
    free = np.setdiff1d(np.arange(len(u)), case['boundary_nodes'])
    # Divide by test-function mass; errors then have PDE forcing units.
    return residual[free] / lump[free].clamp_min(1e-10)


def case_loss(model, case, inputs, queries, targets, scales, weights, context=None, diagnostics=None):
    scales = case_scales(scales, case)
    if context is None:
        context = model.encode(inputs)
    prediction = model.decode(context, queries)
    device = prediction.device
    scale = torch.tensor(scales['output'], device=device)
    vector = case['meta']['pde'] == 'navier_stokes'
    channels = 3 if vector else 2 if case['meta']['pde'] in WAVES else 1
    parts = {name: prediction.sum() * 0 for name in LOSS_NAMES}
    parts['field'] = ((prediction[:, :channels] - targets[:, :channels] / scale[:channels]) ** 2).mean()
    if diagnostics is not None and channels == 2:
        channel_errors = (prediction[:, :2].detach() - targets[:, :2] / scale[:2]).square().mean(0)
        diagnostics.update(zip(('u', 'v'), channel_errors.cpu().tolist()))
    if weights['aux'] and case['meta']['tier'] == 'T2':
        label = np.log(case['meta']['coefficient'])
        parts['aux'] = (context['log_coefficient'] - label) ** 2
    # E/F wave branches define v = du/dt exactly.
    if weights['kinematic'] and channels == 2 and 'wave_initial' not in context:
        du = derivative(prediction, queries['coords'], 0) * scale[0]
        parts['kinematic'] = ((du / scale[1] - prediction[:, 1]) ** 2).mean()
    if weights['weak'] and weak_available(case, inputs):
        residual = weak_residual(model, context, case, inputs, scales)
        # Fixed training-distribution forcing scale, not per-case target scale.
        residual_scale = max(scales['fields'][0], scales['output'][0])
        parts['weak'] = (residual / residual_scale).square().mean()
    if weights['initial'] and time_dependent(case):
        coords = np.column_stack([case['nodes'], np.zeros(len(case['nodes']))])
        initial = model.decode(context, query_input(case, coords, device))
        initial_channels = 2 if vector else channels
        mask = inputs['field_masks'][:, 1:1 + initial_channels]
        expected = inputs['field_values'][:, 1:1 + initial_channels] / scale[:initial_channels]
        if mask.sum() > 0:
            parts['initial'] = ((initial[:, :initial_channels] - expected).square() * mask).sum() / mask.sum()
    if weights['boundary']:
        nodes = case['nodes'][case['boundary_nodes']]
        t = np.full(len(nodes), (case['times'][0] + case['times'][-1]) / 2)
        values = model.decode(context, query_input(case, np.column_stack([nodes, t]), device))
        parts['boundary'] = values[:, :2 if vector else channels].square().mean()
    total = sum(weights[key] * parts[key] for key in weights)
    return total, parts


def validate_loss_options(cases, modes, weights):
    pdes = {c['meta']['pde'] for c in cases}
    if weights['kinematic'] and not pdes.intersection(WAVES):
        raise ValueError('kinematic loss requires at least one wave case')
    if weights['initial'] and not any(time_dependent(c) for c in cases):
        raise ValueError('initial loss requires a time-dependent case')
    if weights['aux'] and not any(c['meta']['tier'] == 'T2' for c in cases):
        print('aux: omitted; this dataset has no T2 scalar-parameter targets')
    if weights['kinematic'] and len(pdes) > 1:
        print('kinematic: wave cases only')
    if weights['initial'] and modes == ['inverse'] and not any(c['meta']['tier'] == 'T2' and c['meta']['pde'] in PARABOLIC + WAVES for c in cases):
        raise ValueError('initial loss has no known initial values in these inverse cases')
    if weights['initial']:
        print('initial: known initial nodes only; omitted on stationary or fully hidden initial fields')
    if weights['weak']:
        if 'navier_stokes' in pdes:
            raise ValueError('The scalar weak loss does not implement mixed NS momentum/continuity; use field, initial and boundary losses')
        if pdes == {'eikonal'}:
            raise ValueError('Eikonal has no elliptic weak loss here; use field and boundary losses')
        stationary = pdes.intersection(STATIONARY)
        if stationary and all(m == 'inverse' for m in modes) and pdes <= stationary and all(c['meta']['tier'] != 'T2' for c in cases):
            raise ValueError('weak loss needs known forcing; inverse stationary cases hide the forcing')
        print('weak: actual FEM test functions; cases with any hidden source nodes are omitted')
