"""Fine numerical references and dense output, independent of neural mesh tokens."""
import numpy as np
from .geometry import refine_mesh, interpolate, evaluation_grid, element_geometry
from .physics import solve, WAVES


def quantities(case):
    if case['meta']['pde'] == 'navier_stokes': return ('velocity_x', 'velocity_y', 'pressure')
    if case['meta']['pde'] == 'darcy': return ('pressure',)
    return ('u', 'v') if case['meta']['pde'] in WAVES else ('u',)


def darcy_flux(case):
    """Elementwise constitutive flux -K_e grad(p_h), not an H(div) reconstruction."""
    _, gradients = element_geometry(case['nodes'], case['triangles'])
    pressure_gradient = np.einsum('ei,eij->ej', case['solution'][0, case['triangles']], gradients)
    coefficient = case.get('element_coefficient', case['coefficient_fields'][case['triangles']].mean(1))
    return -coefficient[:, None] * pressure_gradient


def prepare_reference(case, config):
    model_resolution = config['resolution'] + config['resolution'] % 2
    levels = config.get('solver_refinements', 0)
    if config.get('solver_resolution'):
        levels = max(0, int(np.ceil(np.log2(config['solver_resolution'] / model_resolution))))
    fine = refine_mesh(case, levels) if levels else case
    pde = case['meta']['pde']
    if levels:
        if pde == 'navier_stokes':
            from .navier_stokes import solve_navier_stokes
            unsteady = case['meta'].get('navier_stokes_mode', 'steady') == 'unsteady'
            result, info = solve_navier_stokes(fine, fine['source_vector'], case['meta']['coefficient'], unsteady=unsteady,
                frames=config['frames'], dt=config['dt'], time_end=config.get('time_end', 1.),
                initial_parameters=case['meta'].get('initial_velocity_parameters'),
                forcing_frequency=case['meta'].get('parameters', {}).get('forcing_frequency', 0.))
        else:
            coefficient = (fine['element_coefficient'][:, None] if 'element_coefficient' in fine else fine['coefficient_fields'])
            if pde in WAVES:
                exponent = 1 if case['meta']['tier'] == 'T4' and 'parameters' not in case['meta'] else 2
                coefficient = coefficient ** exponent
            if pde == 'eikonal': coefficient = case['meta']['coefficient']
            result, info = solve(fine, pde, fine['source'], fine['initial_u'], fine['initial_v'], coefficient,
                                 case['meta']['beta'], config['frames'], config['dt'], time_end=config.get('time_end', 1.))
        fine.update(result)
        # Labels at model nodes are restrictions of the fine solution. Neither
        # target reference arrays nor dense grid points become mesh input tokens.
        coords = np.concatenate([np.column_stack([case['nodes'], np.full(len(case['nodes']), t)]) for t in fine['times']])
        restricted = interpolate(fine, coords).reshape(len(fine['times']), len(case['nodes']), -1)
        case['solution'] = restricted[:, :, 0]
        if pde in WAVES: case['velocity'] = restricted[:, :, 1]
        if pde == 'navier_stokes':
            case['solution_fields'] = restricted
            if unsteady:
                case['velocity_components'] = restricted[:, :, :2]; case['pressure'] = restricted[:, :, 2]
                case['initial_velocity'] = restricted[0, :, :2].copy()
                case.pop('initial_velocity_p2', None)
            else:
                case['velocity_components'] = restricted[0, :, :2]; case['pressure'] = restricted[0, :, 2]
            case.pop('velocity_p2', None)
        keys = ('nodes', 'triangles', 'boundary_nodes', 'boundary_edges', 'times', 'solution', 'velocity',
                'source', 'initial_u', 'initial_v', 'coefficient_fields', 'element_coefficient', 'solution_fields',
                'velocity_components', 'pressure', 'velocity_p2', 'source_vector', 'boundary_velocity',
                'initial_velocity', 'initial_velocity_p2')
        for key in keys:
            if key in fine: case['reference_' + key] = fine[key]
        case['meta']['solver'] = info
    if pde == 'darcy':
        flux = darcy_flux(fine)
        case['reference_flux' if levels else 'flux'] = flux
        case['meta']['flux_representation'] = 'cell-mean constitutive -K_e grad(p_h); not an H(div)-conservative reconstruction'
    if levels or config.get('evaluation_resolution'):
        case['meta']['resolutions'] = dict(model=model_resolution, solver=model_resolution * 2 ** levels,
             solver_requested=config.get('solver_resolution', model_resolution * 2 ** levels),
             solver_refinements=levels, evaluation=config.get('evaluation_resolution', 0),
             model_nodes=len(case['nodes']), solver_nodes=len(fine['nodes']),
             physical_fields=('P1 forcing prolonged; analytic velocity IC projected into solver P2 divergence-free space, restricted to model vertices'
                              if pde == 'navier_stokes' and case['meta'].get('navier_stokes_mode') == 'unsteady' else
                              'model-mesh P1 nodal fields prolonged exactly; cell coefficients inherited under refinement'))
        case['meta']['quantities'] = list(quantities(case))
    if config.get('evaluation_resolution'):
        points, weights, arrays = evaluation_grid(fine, config['evaluation_resolution'], uniform=True)
        values = np.stack([interpolate(fine, np.column_stack([points, np.full(len(points), t)]))[:, :len(quantities(case))]
                           for t in fine['times']])
        case.update(evaluation_points=points, evaluation_weights=weights, evaluation_values=values,
                    evaluation_grid_x=arrays['grid_x'], evaluation_grid_y=arrays['grid_y'], evaluation_mask=arrays['grid_mask'])
    return case
