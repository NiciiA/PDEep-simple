"""A few transparent numerical checks, not a qualification system."""
import json
from pathlib import Path
import numpy as np
import torch
from .geometry import mesh, element_geometry, node_weights, interpolate
from .physics import solve
from .data import make_case, normalization, observations, model_input, query_input
from .models import build_model, PRESETS
from .losses import case_loss, parse_losses


def relative_error(case, actual, expected):
    w = node_weights(case)
    return float(np.sqrt(np.sum((actual - expected) ** 2 * w) / max(np.sum(expected ** 2 * w), 1e-20)))


def run_checks(args):
    torch.set_num_threads(args.threads); torch.manual_seed(7)
    results = []
    def report(name, values, status='PASS'):
        results.append(dict(check=name, status=status, **values))
        print(f'{status} {name}: {values}', flush=True)
    for family in ('square', 'disk', 'lshape'):
        case = mesh(family, 8)
        area, _ = element_geometry(case['nodes'], case['triangles'])
        if family == 'lshape':
            centroid = case['nodes'][case['triangles']].mean(1)
            assert not np.any((centroid[:, 0] > .5) & (centroid[:, 1] > .5))
        report(f'geometry_{family}', dict(nodes=len(case['nodes']), triangles=len(case['triangles']),
               area=float(area.sum()), min_area=float(area.min()), boundary=len(case['boundary_nodes'])))
    errors = []
    for resolution in (8, 16):
        case = mesh('square', resolution); xy = case['nodes']
        exact = np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1]); zeros = np.zeros(len(xy))
        result, _ = solve(case, 'poisson', 2 * np.pi ** 2 * exact, zeros)
        error = relative_error(case, result['solution'][0], exact); errors.append(error)
        report(f'poisson_n{resolution}', dict(relative_l2=error), 'PASS' if error < .1 else 'WARN')
    assert errors[1] < errors[0], 'Poisson refinement did not improve accuracy'
    for pde in ('heat', 'wave'):
        case = mesh('square', 16); xy = case['nodes']
        spatial = np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1]); zeros = np.zeros(len(xy))
        a = .1 if pde == 'heat' else .7
        result, _ = solve(case, pde, zeros, spatial, zeros, a if pde == 'heat' else a ** 2, frames=21)
        errors, velocity_errors = {}, {}
        for i in (5, 10, 20):
            t = result['times'][i]
            exact = np.exp(-2 * a * np.pi ** 2 * t) * spatial if pde == 'heat' else np.cos(a * np.sqrt(2) * np.pi * t) * spatial
            errors[str(t)] = relative_error(case, result['solution'][i] - exact + spatial, spatial)
            if pde == 'wave':
                omega = a * np.sqrt(2) * np.pi
                velocity = -omega * np.sin(omega * t) * spatial
                velocity_errors[str(t)] = relative_error(case, result['velocity'][i] - velocity + omega * spatial, omega * spatial)
        values = dict(displacement_error_initial_scale=errors)
        if velocity_errors:
            values['velocity_error_initial_scale'] = velocity_errors
        report(pde, values, 'PASS' if max(list(errors.values()) + list(velocity_errors.values())) < .1 else 'WARN')
        if pde == 'wave':
            from .physics import assemble
            m, k = assemble(case, a ** 2)
            energy = np.array([.5 * (v @ (m @ v) + u @ (k @ u)) for u, v in zip(result['solution'], result['velocity'])])
            report('wave_energy', dict(relative_drift=float(np.ptp(energy) / energy[0])))
            nonzero, _ = solve(case, 'wave', zeros, spatial, .3 * spatial, a ** 2, frames=21)
            omega = a * np.sqrt(2) * np.pi
            exact = (np.cos(omega) + .3 / omega * np.sin(omega)) * spatial
            error = relative_error(case, nonzero['solution'][-1] - exact + spatial, spatial)
            report('wave_nonzero_initial_velocity', dict(displacement_error_initial_scale=error), 'PASS' if error < .1 else 'WARN')
    nonlinear = make_case(dict(tier='T4', resolution=8, frames=11, dt=.002),
                           dict(seed=39, pde='semilinear', geometry='lshape', split='train', id=0))
    assert nonlinear['meta']['solver']['nonlinear_residual'] < 1e-8 and np.isfinite(nonlinear['solution']).all()
    report('semilinear', nonlinear['meta']['solver'])
    case = mesh('lshape', 6); case['times'] = np.array([0., .5, 1.])
    case['solution'] = np.array([case['nodes'][:, 0] + 2 * case['nodes'][:, 1] + t for t in case['times']])
    q = np.array([[.2, .3, .15], [.1, .8, .5], [.8, .2, 1.]])
    err = np.max(abs(interpolate(case, q)[:, 0] - (q[:, 0] + 2 * q[:, 1] + q[:, 2])))
    assert err < 1e-12
    report('interpolation', dict(max_absolute=float(err)))
    cases = [make_case(dict(tier='G1', resolution=4, frames=5, dt=.002),
                      dict(seed=seed, pde='wave', geometry=geometry, split='train', id=seed))
             for seed, geometry in ((8, 'square'), (9, 'disk'))]
    scales = normalization(cases)
    for kind in ('A', 'C'):
        model = build_model(kind, 's'); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = next(model.parameters()).detach().clone(); optimizer.zero_grad(); total = 0.
        for case, count in zip(cases, (0, 7)):
            rng = np.random.default_rng(7); obs = observations(case, count, .01, rng)
            inputs = model_input(case, 'inverse', obs, scales, rng)
            queries = np.column_stack([case['nodes'][:8], np.full(8, .3)])
            target = torch.tensor(interpolate(case, queries), dtype=torch.float32)
            with torch.no_grad():
                output = model(inputs, query_input(case, queries))['field']
                assert output.shape == (8, 2) and torch.isfinite(output).all()
            loss, _ = case_loss(model, case, inputs, query_input(case, queries, requires_grad=True), target, scales,
                                parse_losses('field=1,kinematic=.001,boundary=.001'))
            assert torch.isfinite(loss)
            (loss / 2).backward(); total += float(loss.detach()) / 2
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step(); assert not torch.equal(before, next(model.parameters()))
        report(f'{kind}_forward_backward_update', dict(loss=total, observation_counts=[0, 7], output_shape=[8, 2], finite_gradients=True))
    counts = {}
    for kind in ('A', 'C'):
        counts[kind] = {}
        for size in PRESETS:
            with torch.device('meta'):
                model = build_model(kind, size, report=False)
            counts[kind][size] = sum(p.numel() for p in model.parameters())
    report('parameter_counts', counts)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(results, indent=2))
    print(f'{sum(r["status"] == "PASS" for r in results)} PASS, '
          f'{sum(r["status"] == "WARN" for r in results)} WARN, 0 FAIL; {path}')
    return results
