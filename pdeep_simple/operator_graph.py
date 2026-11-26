"""Small expression graphs for the equations in physics.solve (operand -> operator).

Field references are 0=none, 1=f, 2=u0, 3=v0, 4=coefficient. Numerical
features use only masked model inputs and explicitly known solver settings.
These graphs describe equations; they do not execute a symbolic PDE solver.
"""
import math
import torch


NODE_TYPES = ('function', 'field', 'scalar', 'constant', 'gradient', 'divergence',
              'laplacian', 'dt', 'dtt', 'add', 'subtract', 'multiply', 'power',
              'norm', 'dot', 'vector', 'cos', 'time', 'at_time', 'trace', 'equation')
EDGE_ROLES = ('argument', 'left', 'right', 'coefficient', 'field', 'exponent',
              'x', 'y', 'residual', 'initial', 'boundary', 'time')


def navier_stokes_graph(unsteady=False, forcing_frequency=0.):
    """Vector momentum, incompressibility and no-slip; optional velocity IC."""
    types, edges, names = [], [], []
    def node(kind, name, operands=()):
        index = len(types); types.append(NODE_TYPES.index(kind)); names.append(name)
        edges.extend((i, index, EDGE_ROLES.index(role)) for i, role in operands)
        return index
    ux = node('function', 'velocity_x'); uy = node('function', 'velocity_y'); pressure = node('function', 'pressure')
    velocity = node('vector', 'velocity', [(ux, 'x'), (uy, 'y')])
    nu = node('scalar', 'viscosity')
    laplacian = node('laplacian', 'laplacian_velocity', [(velocity, 'argument')])
    viscous = node('multiply', 'viscosity_times_laplacian', [(nu, 'coefficient'), (laplacian, 'field')])
    zero = node('constant', 'zero')
    negative = node('subtract', 'negative_viscous_term', [(zero, 'left'), (viscous, 'right')])
    gradient = node('gradient', 'velocity_gradient', [(velocity, 'argument')])
    convection = node('dot', 'velocity_dot_gradient_velocity', [(velocity, 'left'), (gradient, 'right')])
    momentum = node('add', 'viscous_plus_convection', [(negative, 'left'), (convection, 'right')])
    gradp = node('gradient', 'pressure_gradient', [(pressure, 'argument')])
    momentum = node('add', 'plus_pressure_gradient', [(momentum, 'left'), (gradp, 'right')])
    if unsteady:
        derivative = node('dt', 'velocity_time_derivative', [(velocity, 'argument')])
        momentum = node('add', 'plus_time_derivative', [(derivative, 'left'), (momentum, 'right')])
    fx = node('field', 'forcing_x'); fy = node('field', 'forcing_y')
    forcing = node('vector', 'forcing', [(fx, 'x'), (fy, 'y')])
    if unsteady and forcing_frequency:
        frequency = node('scalar', 'angular_frequency')
        time = node('time', 'physical_time')
        phase = node('multiply', 'forcing_phase', [(frequency, 'coefficient'), (time, 'field')])
        cosine = node('cos', 'forcing_cosine', [(phase, 'argument')])
        forcing = node('multiply', 'time_dependent_forcing', [(cosine, 'coefficient'), (forcing, 'field')])
    momentum = node('subtract', 'momentum_residual', [(momentum, 'left'), (forcing, 'right')])
    node('equation', 'momentum_equals_zero', [(momentum, 'residual'), (zero, 'right')])
    incompressibility = node('divergence', 'divergence_velocity', [(velocity, 'argument')])
    node('equation', 'incompressibility', [(incompressibility, 'residual'), (zero, 'right')])
    trace = node('trace', 'boundary_velocity', [(velocity, 'argument')])
    node('equation', 'no_slip', [(trace, 'boundary'), (zero, 'right')])
    if unsteady:
        ix = node('field', 'initial_velocity_x'); iy = node('field', 'initial_velocity_y')
        initial = node('vector', 'initial_velocity', [(ix, 'x'), (iy, 'y')])
        at_zero = node('at_time', 'velocity_at_zero', [(velocity, 'argument'), (zero, 'time')])
        node('equation', 'initial_velocity_constraint', [(at_zero, 'left'), (initial, 'initial')])
    return dict(types=types, edges=edges, names=names)


def operator_graph(inputs):
    from .data import PDES
    from .physics import PARABOLIC, WAVES
    pde = PDES[inputs['pde']]
    if pde == 'navier_stokes':
        frequency = inputs['operator_metadata'].get('forcing_frequency', 0.)
        graph = navier_stokes_graph(inputs.get('time_dependent', False), frequency); device = inputs['scalar'].device
        numbers = inputs['scalar'].new_zeros((len(graph['types']), 8))
        index = graph['names'].index('viscosity'); known = inputs['scalar_masks'][0]
        numbers[index, 0] = torch.asinh(inputs['scalar'][0] * known); numbers[index, 1] = known
        refs = torch.zeros(len(graph['types']), dtype=torch.long, device=device)
        refs[index] = 4; refs[graph['names'].index('forcing_x')] = 1; refs[graph['names'].index('forcing_y')] = 5
        numbers[graph['names'].index('forcing_x'), 2] = inputs['field_masks'][:, 0].mean()
        numbers[graph['names'].index('forcing_y'), 2] = inputs['forcing_y_mask'].mean()
        for channel, name in enumerate(('initial_velocity_x', 'initial_velocity_y'), 1):
            if name in graph['names']:
                i = graph['names'].index(name); refs[i] = channel + 1
                numbers[i, 2] = inputs['field_masks'][:, channel].mean()
        if 'angular_frequency' in graph['names']:
            i = graph['names'].index('angular_frequency')
            numbers[i, 0] = math.asinh(2 * math.pi * frequency); numbers[i, 1] = 1.
        numbers[graph['names'].index('zero'), 1] = 1.
        return dict(types=torch.tensor(graph['types'], device=device), numbers=numbers, fields=refs,
                    edges=torch.tensor(graph['edges'], device=device), names=graph['names'])
    settings = inputs['operator_metadata']
    reference = inputs['scalar']
    types, numbers, fields, edges = [], [], [], []

    def scalar(value):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    def node(kind, operands=(), value=0., known=0., field=-1):
        index = len(types)
        types.append(NODE_TYPES.index(kind)); fields.append(field + 1)
        fraction = inputs['field_masks'][:, field].mean() if field >= 0 else scalar(0.)
        prior = settings['priors'].get(('source', 'initial_u', 'initial_v', 'coefficient')[field], {}) if field >= 0 else {}
        # Scalar values are scaled invertibly with asinh; field values stay on
        # the mesh. A partial field never supplies a hidden scalar or mean.
        numbers.append(torch.stack([torch.asinh(scalar(value) * scalar(known)), scalar(known), fraction,
                                    torch.asinh(scalar(prior.get('count', 0))),
                                    torch.asinh(scalar(prior.get('mean', 0))),
                                    torch.asinh(scalar(prior.get('std', 0))),
                                    scalar(prior.get('distribution') == 'gaussian'), scalar(bool(prior))]))
        edges.extend((operand, index, EDGE_ROLES.index(role)) for operand, role in operands)
        return index

    def constant(value):
        return node('constant', value=value, known=1.)

    def parameter(name):
        return node('scalar', value=settings[name], known=1.)

    def unary(kind, value):
        return node(kind, [(value, 'argument')])

    def binary(kind, left, right):
        return node(kind, [(left, 'left'), (right, 'exponent' if kind == 'power' else 'right')])

    def product(coefficient, field):
        return node('multiply', [(coefficient, 'coefficient'), (field, 'field')])

    u = node('function')
    grad = unary('gradient', u)
    zero = constant(0.)
    if pde == 'p_laplacian':
        # -div((epsilon^2 + |grad u|^2)^((p-2)/2) grad u) = f.
        squared = binary('add', binary('power', parameter('p_regularization'), constant(2.)),
                         binary('power', unary('norm', grad), constant(2.)))
        exponent = product(constant(.5), binary('subtract', parameter('p'), constant(2.)))
        spatial = unary('divergence', product(binary('power', squared, exponent), grad))
        residual = binary('subtract', zero, spatial)
    else:
        coefficient = node('field', value=inputs['scalar'][0], known=inputs['scalar_masks'][0], field=3)
        if pde == 'eikonal':
            # physics.solve returns distance_to_boundary / positive constant speed.
            residual = binary('subtract', unary('norm', grad), binary('power', coefficient, constant(-1.)))
        else:
            diffusion = binary('power', coefficient, constant(settings['wave_coefficient_power'])) if pde in WAVES else coefficient
            spatial = unary('laplacian', u) if pde == 'helmholtz' else unary('divergence', product(diffusion, grad))
            if pde in PARABOLIC:
                residual = binary('subtract', unary('dt', u), spatial)
            elif pde in WAVES:
                residual = binary('subtract', unary('dtt', u), spatial)
            else:
                residual = binary('subtract', zero, spatial)
            if pde == 'helmholtz':
                residual = binary('subtract', residual, product(binary('power', coefficient, constant(2.)), u))
            if pde in ('elliptic_adr', 'reaction_diffusion'):
                residual = binary('add', residual, product(parameter('reaction'), u))
            if pde == 'elliptic_adr':
                advection = node('vector', [(parameter('advection_x'), 'x'), (parameter('advection_y'), 'y')])
                residual = binary('add', residual, binary('dot', advection, grad))
            if pde == 'damped_wave':
                residual = binary('add', residual, product(parameter('damping'), unary('dt', u)))
            if pde == 'semilinear':
                beta = node('scalar', value=inputs['scalar'][1], known=inputs['scalar_masks'][1])
                residual = binary('add', residual, product(beta, binary('power', u, constant(3.))))
            if pde not in ('poisson', 'diffusion', 'elliptic_adr', 'heat', 'reaction_diffusion',
                           'wave', 'damped_wave', 'semilinear', 'helmholtz', 'darcy'):
                raise ValueError(f'no joint-conditioning expression graph for {pde}')
    if pde != 'eikonal':
        forcing = node('field', field=0)
        if pde in PARABOLIC + WAVES:
            phase = product(product(constant(2 * math.pi), parameter('forcing_frequency')), node('time'))
            forcing = product(forcing, unary('cos', phase))
        residual = binary('subtract', residual, forcing)

    constraints = [(residual, 'residual'), (zero, 'right')]
    if pde in PARABOLIC + WAVES:
        initial_u = node('at_time', [(u, 'argument'), (zero, 'time')])
        constraints.append((binary('subtract', initial_u, node('field', field=1)), 'initial'))
        if pde in WAVES:
            initial_v = node('at_time', [(unary('dt', u), 'argument'), (zero, 'time')])
            constraints.append((binary('subtract', initial_v, node('field', field=2)), 'initial'))
    # All current numerical datasets impose homogeneous Dirichlet boundaries.
    constraints.append((binary('subtract', unary('trace', u), zero), 'boundary'))
    node('equation', constraints)
    return dict(types=torch.tensor(types, dtype=torch.long, device=reference.device),
                numbers=torch.stack(numbers), fields=torch.tensor(fields, dtype=torch.long, device=reference.device),
                edges=torch.tensor(edges, dtype=torch.long, device=reference.device))
