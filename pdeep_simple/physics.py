"""P1 FEM: consistent mass, implicit Euler, Newmark, lumped cubic reaction."""
import hashlib
import math
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized, spsolve
from .geometry import element_geometry, distance_to_boundary

STATIONARY = ('poisson', 'diffusion', 'elliptic_adr', 'semilinear', 'p_laplacian', 'eikonal', 'helmholtz', 'darcy', 'navier_stokes')
WAVES = ('wave', 'damped_wave')
PARABOLIC = ('heat', 'reaction_diffusion')
LINEAR_ELLIPTIC = ('poisson', 'diffusion', 'elliptic_adr', 'darcy')


def time_dependent(case):
    pde = case['meta']['pde']
    return pde in PARABOLIC + WAVES or pde == 'navier_stokes' and case['meta'].get('navier_stokes_mode', 'steady') == 'unsteady'


def helmholtz_operator(case, wavenumber, check_coercivity=True):
    """K - integral k_h^2 phi_i phi_j; exact P1 polynomial mass integration.

    Stay below 0.8 of the bounding-box Dirichlet eigenvalue lower bound.
    This conservative coercive regime excludes resonances; it is real Helmholtz,
    not the positive-reaction screened Poisson equation.
    """
    mass, stiffness = assemble(case)
    values = np.asarray(wavenumber)
    span = np.ptp(case['nodes'], axis=0)
    lower_bound = np.pi ** 2 * np.sum(1 / span ** 2)
    if np.min(values) <= 0 or check_coercivity and np.max(values) ** 2 >= .8 * lower_bound:
        raise ValueError('Helmholtz requires 0 < max(k)^2 < 0.8*pi^2*(1/Lx^2+1/Ly^2); reduce wavenumber/contrast')
    area, _ = element_geometry(case['nodes'], case['triangles'])
    if values.ndim == 0:
        reaction = values ** 2 * mass
    else:
        if values.ndim == 2:
            local = values[:, 0, None, None] ** 2 * area[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))
        else:
            # Integral lambda_i lambda_j lambda_a lambda_b = area*prod(alpha!)/360.
            integrals = np.empty((3, 3, 3, 3))
            for index in np.ndindex(integrals.shape):
                integrals[index] = math.prod(math.factorial(index.count(j)) for j in range(3)) / 360
            nodal = values[case['triangles']]
            local = np.einsum('ea,eb,ijab,e->eij', nodal, nodal, integrals, area)
        triangles = case['triangles']
        rows = np.repeat(triangles, 3, axis=1).ravel(); cols = np.tile(triangles, (1, 3)).ravel()
        reaction = coo_matrix((local.ravel(), (rows, cols)), shape=mass.shape).tocsr()
    return mass, stiffness - reaction


def assemble(case, coefficient=1., advection=None, reaction=0.):
    triangles, nodes = case['triangles'], case['nodes']
    area, gradients = element_geometry(nodes, triangles)
    a = np.asarray(coefficient)
    element_a = a[:, 0] if a.ndim == 2 else a[triangles].mean(axis=1) if a.ndim else a
    stiffness = area[:, None, None] * np.einsum('eik,ejk->eij', gradients, gradients)
    stiffness *= np.asarray(element_a)[..., None, None]
    mass = area[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))
    stiffness += reaction * mass
    if advection is not None:
        # Integral phi_i (b dot grad phi_j), for constant b on each element.
        stiffness += area[:, None, None] / 3 * np.einsum('eji,i->ej', gradients, advection)[:, None, :]
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    shape = (len(nodes), len(nodes))
    return (coo_matrix((mass.ravel(), (rows, cols)), shape=shape).tocsr(),
            coo_matrix((stiffness.ravel(), (rows, cols)), shape=shape).tocsr())


def p_laplacian(case, load, free, p, settings):
    """Damped Newton for the convex, epsilon-regularized P1 energy (p >= 2)."""
    if p < 2:
        raise ValueError('p_laplacian requires p >= 2')
    epsilon = settings.get('p_regularization', 1e-4)
    if epsilon <= 0:
        raise ValueError('p_regularization must be positive')
    area, gradients = element_geometry(case['nodes'], case['triangles'])
    triangles = case['triangles']; n = len(case['nodes'])
    rows = np.repeat(triangles, 3, axis=1).ravel(); cols = np.tile(triangles, (1, 3)).ravel()
    _, stiffness = assemble(case)
    u = np.zeros(n); u[free] = spsolve(stiffness[free][:, free], load)
    def terms(values, jacobian=False):
        grad = np.einsum('ei,eik->ek', values[triangles], gradients)
        squared = epsilon ** 2 + (grad * grad).sum(-1)
        power = squared ** ((p - 2) / 2)
        projected = np.einsum('eik,ek->ei', gradients, grad)
        residual = np.zeros(n)
        np.add.at(residual, triangles.ravel(), (area[:, None] * power[:, None] * projected).ravel())
        energy = np.sum(area / p * squared ** (p / 2)) - values[free] @ load
        if not jacobian:
            return energy
        local = np.einsum('eik,ejk->eij', gradients, gradients) * power[:, None, None]
        local += (p - 2) * squared[:, None, None] ** ((p - 4) / 2) * projected[:, :, None] * projected[:, None, :]
        matrix = coo_matrix(((area[:, None, None] * local).ravel(), (rows, cols)), shape=(n, n)).tocsr()
        return residual[free] - load, matrix[free][:, free], energy
    tolerance = settings.get('newton_tolerance', 1e-9) * max(1., np.linalg.norm(load))
    for iteration in range(settings.get('newton_iterations', 80)):
        residual, jacobian, energy = terms(u, True)
        norm = np.linalg.norm(residual)
        if norm <= tolerance:
            return u, dict(newton_iterations=iteration, nonlinear_residual=float(norm), p_regularization=epsilon)
        step = spsolve(jacobian, -residual); rate = 1.
        while rate >= 2 ** -24:
            trial = u.copy(); trial[free] += rate * step
            if terms(trial) <= energy + 1e-4 * rate * (residual @ step):
                u = trial; break
            rate /= 2
        else:
            raise RuntimeError('p_laplacian Newton line search failed')
    raise RuntimeError('p_laplacian Newton solver did not converge')


def solve(case, pde, source, initial_u, initial_v=None, coefficient=1., beta=0.,
          frames=21, dt=0.002, cache=None, time_end=1.):
    parameters = case.get('meta', {}).get('parameters', {})
    settings = case.get('meta', {}).get('numerics', {})
    reaction = parameters.get('reaction', 0.)
    advection = parameters.get('advection', [0., 0.])
    damping = parameters.get('damping', 0.)
    frequency = parameters.get('forcing_frequency', 0.)
    if pde == 'helmholtz':
        mass, operator = helmholtz_operator(case, coefficient)
        free = np.setdiff1d(np.arange(len(case['nodes'])), case['boundary_nodes'])
        load = (mass @ source)[free]
        values = spsolve(operator[free][:, free], load)
        residual = operator[free][:, free] @ values - load
        relative = np.linalg.norm(residual) / max(np.linalg.norm(load), 1e-14)
        if not np.isfinite(values).all() or relative > 1e-8:
            raise RuntimeError('Helmholtz discrete solve failed its residual check')
        solution = np.zeros((1, len(case['nodes']))); solution[0, free] = values
        return dict(times=np.array([0.]), solution=solution), dict(method='P1 FEM Helmholtz, zero Dirichlet', relative_residual=float(relative))
    if pde == 'eikonal':
        if np.ndim(coefficient) or coefficient <= 0:
            raise ValueError('eikonal supports positive constant speed only')
        u = distance_to_boundary(case) / coefficient
        return dict(times=np.array([0.]), solution=u[None]), dict(method='distance to boundary segments / constant speed')
    factors = {}
    if cache is not None:
        digest = hashlib.blake2b(digest_size=16)
        for array in (case['nodes'], case['triangles'], case['boundary_nodes'], np.asarray(coefficient),
                      np.asarray(advection), np.asarray(reaction)):
            digest.update(np.ascontiguousarray(array).tobytes())
        key = digest.digest()
        if key not in cache:
            mass, stiffness = assemble(case, coefficient, advection, reaction)
            free = np.setdiff1d(np.arange(len(case['nodes'])), case['boundary_nodes'])
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[key] = (mass, stiffness, free, mass[free][:, free], stiffness[free][:, free], {})
        mass, stiffness, free, m, k, factors = cache[key]
    else:
        mass, stiffness = assemble(case, coefficient, advection, reaction)
        free = np.setdiff1d(np.arange(len(case['nodes'])), case['boundary_nodes'])
        m, k = mass[free][:, free], stiffness[free][:, free]
    load = (mass @ source)[free]  # consistent integration of a nodal P1 forcing
    n = len(case['nodes'])
    info = {}
    if pde == 'p_laplacian':
        u, info = p_laplacian(case, load, free, parameters.get('p', 3.), settings)
        return dict(times=np.array([0.]), solution=u[None]), info
    if pde in LINEAR_ELLIPTIC + ('semilinear',):
        if cache is not None:
            if 'elliptic' not in factors:
                factors['elliptic'] = factorized(k.tocsc())
            u = factors['elliptic'](load)
        else:
            u = spsolve(k, load)
        if pde == 'semilinear':
            lump = np.asarray(mass.sum(axis=1)).ravel()[free]
            for iteration in range(settings.get('newton_iterations', 40)):
                residual = k @ u + beta * lump * u ** 3 - load
                if np.linalg.norm(residual) < settings.get('newton_tolerance', 1e-10) * max(1., np.linalg.norm(load)):
                    break
                jacobian = k + diags(3 * beta * lump * u ** 2)
                step = spsolve(jacobian, -residual)
                rate = 1.
                while rate > 1 / 1024:
                    trial = u + rate * step
                    new = k @ trial + beta * lump * trial ** 3 - load
                    if np.linalg.norm(new) < np.linalg.norm(residual):
                        break
                    rate /= 2
                u = trial
            else:
                raise RuntimeError("semilinear Newton solver did not converge")
            info = dict(newton_iterations=iteration,
                        nonlinear_residual=float(np.linalg.norm(k @ u + beta * lump * u ** 3 - load)))
        solution = np.zeros((1, n)); solution[0, free] = u
        if not np.isfinite(solution).all():
            raise RuntimeError('non-finite elliptic solution')
        return dict(times=np.array([0.]), solution=solution), info
    if frames < 2 or dt <= 0 or time_end <= 0:
        raise ValueError("dynamics need frames>=2, dt>0 and time-end>0")
    times = np.linspace(0, time_end, frames)
    steps = int(np.ceil((times[1] - times[0]) / dt))
    h = (times[1] - times[0]) / steps
    u = np.asarray(initial_u)[free].copy()
    solution = np.zeros((frames, n)); solution[0, free] = u
    if pde in PARABOLIC:
        key = ('heat', h)
        if key not in factors:
            if len(factors) >= 4:
                factors.clear()
            factors[key] = factorized((m + h * k).tocsc())
        step_solve = factors[key]
        for frame in range(1, frames):
            for step in range(steps):
                time = times[frame - 1] + (step + 1) * h
                u = step_solve(m @ u + h * load * np.cos(2 * np.pi * frequency * time))
            solution[frame, free] = u
        result = dict(times=times, solution=solution)
    elif pde in WAVES:
        v = np.zeros_like(u) if initial_v is None else np.asarray(initial_v)[free].copy()
        acceleration = spsolve(m, load - k @ u - damping * (m @ v))
        velocity = np.zeros_like(solution); velocity[0, free] = v
        key = ('wave', h, damping)
        if key not in factors:
            if len(factors) >= 4:
                factors.clear()
            factors[key] = factorized(((1 + h / 2 * damping) * m + h ** 2 / 4 * k).tocsc())
        step_solve = factors[key]
        for frame in range(1, frames):
            for step in range(steps):
                predicted_u = u + h * v + h ** 2 / 4 * acceleration
                predicted_v = v + h / 2 * acceleration
                time = times[frame - 1] + (step + 1) * h
                acceleration = step_solve(load * np.cos(2 * np.pi * frequency * time) - k @ predicted_u - damping * (m @ predicted_v))
                u = predicted_u + h ** 2 / 4 * acceleration
                v = predicted_v + h / 2 * acceleration
            solution[frame, free], velocity[frame, free] = u, v
        result = dict(times=times, solution=solution, velocity=velocity)
    else:
        raise ValueError(f"unsupported PDE: {pde}")
    if not all(np.isfinite(a).all() for a in result.values()):
        raise RuntimeError("non-finite numerical trajectory")
    return result, dict(time_step=h, internal_steps=steps * (frames - 1))
