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


def operator_graph(inputs):
    from .data import PDES
    from .physics import PARABOLIC, WAVES
    pde = PDES[inputs['pde']]
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
            spatial = unary('divergence', product(diffusion, grad))
            if pde in PARABOLIC:
                residual = binary('subtract', unary('dt', u), spatial)
            elif pde in WAVES:
                residual = binary('subtract', unary('dtt', u), spatial)
            else:
                residual = binary('subtract', zero, spatial)
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
                           'wave', 'damped_wave', 'semilinear'):
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
