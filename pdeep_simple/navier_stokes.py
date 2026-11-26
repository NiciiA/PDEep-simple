"""Optional low-Re steady/unsteady incompressible Taylor-Hood P2/P1 FEM.

No-slip on the whole boundary, integral p dx = 0, Picard linearization.
Assembly uses scikit-fem; sparse solves use SciPy. No pressure penalty or
unstable equal-order velocity/pressure discretization is substituted.
"""
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve


def require_backend():
    try:
        import skfem
    except ImportError as error:
        raise ValueError('Navier-Stokes requires the optional FEM backend: pip install -e ".[navier-stokes]"; no substitute labels are generated') from error
    return skfem


def solve_navier_stokes(case, forcing, viscosity, *, unsteady=False, frames=21, dt=.002,
                       time_end=1., initial_parameters=None, forcing_frequency=0.):
    if unsteady:
        return solve_unsteady(case, forcing, viscosity, frames, dt, time_end, initial_parameters, forcing_frequency)
    fem = require_backend()
    from skfem.helpers import dot, grad
    from skfem.models.poisson import vector_laplace, mass
    from skfem.models.general import divergence
    if viscosity <= 0 or not np.isfinite(viscosity):
        raise ValueError('Navier-Stokes viscosity must be positive')
    mesh = fem.MeshTri(case['nodes'].T, case['triangles'].T, sort_t=False)
    velocity_basis = fem.Basis(mesh, fem.ElementVector(fem.ElementTriP2()), intorder=5)
    pressure_basis = velocity_basis.with_element(fem.ElementTriP1())
    force = np.stack([pressure_basis.interpolate(forcing[:, q]) for q in range(2)])

    @fem.LinearForm
    def load(v, w):
        return dot(force, v)

    @fem.BilinearForm
    def advection(u, v, w):
        return dot(np.einsum('ij...,j...->i...', grad(u), w.velocity), v)

    diffusion = viscosity * fem.asm(vector_laplace, velocity_basis)
    continuity = fem.asm(divergence, velocity_basis, pressure_basis)
    pressure_mass = fem.asm(mass, pressure_basis)
    mean = np.asarray(pressure_mass.sum(1)).ravel()
    gauge = csr_matrix(mean[:, None])
    rhs = np.r_[fem.asm(load, velocity_basis), np.zeros(pressure_basis.N + 1)]
    boundary = velocity_basis.get_dofs().all()
    total = velocity_basis.N + pressure_basis.N + 1
    free = np.setdiff1d(np.arange(total), boundary)
    solution = np.zeros(total); relative = np.inf
    for iteration in range(40):
        advecting = velocity_basis.interpolate(solution[:velocity_basis.N])
        matrix = diffusion + fem.asm(advection, velocity_basis, velocity=advecting)
        system = bmat([[matrix, -continuity.T, None], [continuity, None, gauge], [None, gauge.T, None]], format='csr')
        updated = np.zeros(total); updated[free] = spsolve(system[free][:, free], rhs[free])
        nonlinear = diffusion + fem.asm(advection, velocity_basis, velocity=velocity_basis.interpolate(updated[:velocity_basis.N]))
        residual = np.r_[nonlinear @ updated[:velocity_basis.N] - continuity.T @ updated[velocity_basis.N:-1] - rhs[:velocity_basis.N],
                         continuity @ updated[:velocity_basis.N] + mean * updated[-1],
                         mean @ updated[velocity_basis.N:-1]]
        relative = np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1e-14)
        solution = updated
        if np.isfinite(solution).all() and relative < 1e-9:
            break
    else:
        raise RuntimeError('Low-Re Navier-Stokes Picard iteration did not converge; increase viscosity or reduce forcing')
    coefficients = solution[:velocity_basis.N]
    pressure = solution[velocity_basis.N:-1]
    velocity = coefficients[velocity_basis.nodal_dofs].T
    local = coefficients[velocity_basis.element_dofs].reshape(6, 2, -1).transpose(2, 0, 1)
    reynolds = np.max(np.linalg.norm(coefficients.reshape(-1, 2), axis=1)) * np.max(np.ptp(case['nodes'], axis=0)) / viscosity
    if reynolds > 5:
        raise ValueError(f'Navier-Stokes generated Re={reynolds:.3g}; this controlled dataset requires Re<=5')
    # Keep the actual P2 polynomial, not a P1 interpolation of vertex velocities.
    fields = np.column_stack([velocity, pressure])
    info = dict(method='scikit-fem Taylor-Hood P2/P1 Picard', backend_version=fem.__version__,
                iterations=iteration + 1, relative_residual=float(relative), reynolds=float(reynolds),
                pressure_integral=float(mean @ pressure), weak_divergence_norm=float(np.linalg.norm(continuity @ coefficients)),
                pressure_gauge='integral p dx = 0', velocity_boundary='homogeneous no-slip')
    return dict(times=np.array([0.]), solution=velocity[None, :, 0], solution_fields=fields[None],
                velocity_components=velocity, pressure=pressure, velocity_p2=local), info


def solve_unsteady(case, forcing, viscosity, frames, dt, time_end, initial_parameters, frequency):
    """Backward Euler + converged Picard, with mixed incompressibility each step.

    The initial curl field is projected into the discrete no-slip/divergence-free
    space. Initial pressure comes from a constrained acceleration solve, not from
    an independent pressure initial condition or the projection multiplier.
    """
    fem = require_backend()
    from skfem.helpers import dot, grad
    from skfem.models.poisson import vector_laplace, mass
    from skfem.models.general import divergence
    from scipy.sparse.linalg import factorized
    if frames < 2 or dt <= 0 or time_end <= 0 or viscosity <= 0:
        raise ValueError('Unsteady NS needs frames>=2, positive dt/time-end/viscosity')
    mesh = fem.MeshTri(case['nodes'].T, case['triangles'].T, sort_t=False)
    vb = fem.Basis(mesh, fem.ElementVector(fem.ElementTriP2()), intorder=5)
    pb = vb.with_element(fem.ElementTriP1())

    @fem.BilinearForm
    def vector_mass(u, v, w): return dot(u, v)

    @fem.BilinearForm
    def advection(u, v, w):
        return dot(np.einsum('ij...,j...->i...', grad(u), w.velocity), v)

    force = np.stack([pb.interpolate(forcing[:, q]) for q in range(2)])
    @fem.LinearForm
    def body_force(v, w): return dot(force, v)

    parameters = initial_parameters
    def initial_curl(x):
        lo = case['nodes'].min(0); span = np.ptp(case['nodes'], axis=0)
        a, b = (x - lo[:, None, None]) / span[:, None, None]
        fx, fy = a ** 2 * (1 - a) ** 2, b ** 2 * (1 - b) ** 2
        dx, dy = 2 * a * (1 - a) * (1 - 2 * a), 2 * b * (1 - b) * (1 - 2 * b)
        s = np.zeros_like(a); sx = s.copy(); sy = s.copy()
        for coefficient, (m, n) in zip(parameters['coefficients'], ((1, 1), (2, 1), (1, 2))):
            s += coefficient * np.sin(m * np.pi * a) * np.sin(n * np.pi * b)
            sx += coefficient * m * np.pi * np.cos(m * np.pi * a) * np.sin(n * np.pi * b)
            sy += coefficient * n * np.pi * np.sin(m * np.pi * a) * np.cos(n * np.pi * b)
        scale = 64 * parameters['amplitude']
        return np.array([scale * fx * (dy * s + fy * sy) / span[1],
                         -scale * fy * (dx * s + fx * sx) / span[0]])

    @fem.LinearForm
    def initial_load(v, w): return dot(initial_curl(w.x), v)

    m = fem.asm(vector_mass, vb); k = viscosity * fem.asm(vector_laplace, vb)
    b = fem.asm(divergence, vb, pb); pm = fem.asm(mass, pb)
    mean = np.asarray(pm.sum(1)).ravel(); gauge = csr_matrix(mean[:, None])
    load = fem.asm(body_force, vb)
    free = np.setdiff1d(np.arange(vb.N + pb.N + 1), vb.get_dofs().all())
    def system(a):
        return bmat([[a, -b.T, None], [b, None, gauge], [None, gauge.T, None]], format='csr')
    projection = factorized(system(m)[free][:, free].tocsc())
    def mass_solve(rhs):
        result = np.zeros(vb.N + pb.N + 1)
        result[free] = projection(np.r_[rhs, np.zeros(pb.N + 1)][free])
        return result
    velocity = mass_solve(fem.asm(initial_load, vb))[:vb.N]
    def convection(u): return fem.asm(advection, vb, velocity=vb.interpolate(u))
    acceleration = mass_solve(load - (k + convection(velocity)) @ velocity)
    pressure = acceleration[vb.N:-1]
    times = np.linspace(0., time_end, frames)
    substeps = int(np.ceil((times[1] - times[0]) / dt)); h = (times[1] - times[0]) / substeps
    snapshots, polynomials = [], []
    max_divergence = max_gauge = max_point_divergence = max_reynolds = max_residual = 0.
    def save_snapshot():
        nonlocal max_divergence, max_gauge, max_point_divergence, max_reynolds
        nodal = velocity[vb.nodal_dofs].T
        snapshots.append(np.column_stack([nodal, pressure]))
        polynomials.append(velocity[vb.element_dofs].reshape(6, 2, -1).transpose(2, 0, 1))
        max_divergence = max(max_divergence, np.linalg.norm(b @ velocity))
        max_gauge = max(max_gauge, abs(mean @ pressure))
        derivative = vb.interpolate(velocity).grad
        max_point_divergence = max(max_point_divergence, np.sqrt(np.sum((derivative[0, 0] + derivative[1, 1]) ** 2 * vb.dx)))
        max_reynolds = max(max_reynolds, np.max(np.linalg.norm(velocity.reshape(-1, 2), axis=1)) * np.ptp(case['nodes'], axis=0).max() / viscosity)
    save_snapshot()
    for frame in range(1, frames):
        for step in range(substeps):
            clock = times[frame - 1] + (step + 1) * h
            rhs = np.r_[m @ velocity / h + load * np.cos(2 * np.pi * frequency * clock), np.zeros(pb.N + 1)]
            iterate = velocity.copy()
            for iteration in range(30):
                a = m / h + k + convection(iterate)
                matrix = system(a); result = np.zeros(len(rhs))
                result[free] = spsolve(matrix[free][:, free], rhs[free])
                iterate = result[:vb.N]
                residual = system(m / h + k + convection(iterate)) @ result - rhs
                relative = np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1e-14)
                if np.isfinite(result).all() and relative < 1e-9: break
            else:
                raise RuntimeError('Unsteady NS Picard step did not converge; reduce dt/initial amplitude/forcing')
            velocity = iterate; pressure = result[vb.N:-1]; max_residual = max(max_residual, relative)
        save_snapshot()
    if max_reynolds > 5: raise ValueError('Unsteady NS controlled setup requires discrete Re<=5; reduce initial amplitude/forcing')
    fields = np.stack(snapshots); local = np.stack(polynomials)
    return dict(times=times, solution=fields[:, :, 0], solution_fields=fields,
                velocity_components=fields[:, :, :2], pressure=fields[:, :, 2], velocity_p2=local,
                initial_velocity=fields[0, :, :2].copy(), initial_velocity_p2=local[0].copy()), dict(
        method='Taylor-Hood P2/P1 backward Euler with converged Picard', backend_version=fem.__version__,
        time_step=float(h), internal_steps=(frames - 1) * substeps, time_end=float(time_end),
        relative_residual=float(max_residual), weak_divergence_norm=float(max_divergence),
        divergence_l2=float(max_point_divergence), pressure_integral=float(max_gauge), reynolds=float(max_reynolds),
        pressure_gauge='integral p(t) dx = 0 at every snapshot', velocity_boundary='homogeneous no-slip',
        initial_pressure='constrained acceleration solve; not an independent IC')


def known_physical_inputs(case, mode, rng):
    """Steady hides forcing; unsteady hides velocity IC, with known force/nu."""
    if mode not in ('forward', 'inverse', 'hybrid'):
        raise ValueError('unknown conditioning mode')
    forcing = case['source_vector'].copy(); mask = np.ones_like(forcing, dtype=bool)
    initial = case.get('initial_velocity', np.zeros_like(forcing)).copy(); initial_mask = np.ones_like(initial, dtype=bool)
    unsteady = case['meta'].get('navier_stokes_mode', 'steady') == 'unsteady'
    if not unsteady: initial_mask[:] = False
    if mode != 'forward':
        name = 'hybrid_known_initial_fraction' if unsteady else 'hybrid_known_source_fraction'
        fraction = case['meta'].get('conditioning', {}).get(name, .5)
        target = initial_mask if unsteady else mask
        target[:] = False if mode == 'inverse' else rng.random(target.shape) < fraction
    return dict(forcing=forcing * mask, forcing_mask=mask, viscosity=case['meta']['coefficient'],
                initial_velocity=initial * initial_mask, initial_mask=initial_mask,
                viscosity_known=True, boundary_velocity=case['boundary_velocity'], boundary_known=True)
