"""Focused regression checks for leakage, geometry, inference and weak forms."""
import unittest
import numpy as np
import torch
from pdeep_simple.geometry import mesh, element_geometry, sample_points, interpolate, locate
from pdeep_simple.data import make_case, normalization, observations, model_input, query_input
from pdeep_simple.baselines import gaussian_posterior
from pdeep_simple.models import build_model
from pdeep_simple.losses import weak_residual


def example(tier='G1', pde='heat', geometry='square', seed=11):
    return make_case(dict(tier=tier, resolution=6, frames=11, dt=.002),
                     dict(seed=seed, pde=pde, geometry=geometry, split='train', id=seed))


class ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_lshape_connectivity_and_outside_rejection(self):
        case = mesh('lshape', 8)
        area, _ = element_geometry(case['nodes'], case['triangles'])
        self.assertAlmostEqual(area.sum(), .75)
        points = sample_points(case, 500, np.random.default_rng(1))
        self.assertFalse(np.any((points[:, 0] > .5) & (points[:, 1] > .5)))
        with self.assertRaises(ValueError):
            locate(case, [[.75, .75]])

    def test_hidden_values_do_not_enter_model_inputs(self):
        for tier, pde in [('T2', 'heat'), ('T2', 'poisson'), ('G1', 'wave'), ('T4', 'semilinear')]:
            case = example(tier, pde); scales = normalization([case])
            obs = observations(case, 5, .01, np.random.default_rng(2))
            before = model_input(case, 'inverse', obs, scales, np.random.default_rng(3))
            changed = {k: v.copy() if isinstance(v, np.ndarray) else dict(v) for k, v in case.items()}
            if tier == 'T2':
                changed['meta']['coefficient'] = 123.
                changed['coefficient_fields'][:] = 123.
            elif pde == 'wave':
                changed['initial_u'][:] = 99.; changed['initial_v'][:] = -47.
            else:
                changed['source'][:] = 99.
            changed['solution'][:] = 987.
            after = model_input(changed, 'inverse', obs, scales, np.random.default_rng(3))
            for key in before:
                if torch.is_tensor(before[key]):
                    self.assertTrue(torch.equal(before[key], after[key]), (tier, key))

    def test_stationary_time_is_not_a_dynamic_coordinate(self):
        case = example('T1', 'poisson')
        with self.assertRaises(ValueError):
            interpolate(case, np.array([[.2, .2, .1]]))

    def test_gaussian_posterior_matches_numerical_integration(self):
        h, y, noise = np.array([[1.], [.5]]), np.array([.8, .2]), np.array([.2, .3])
        mean, covariance = gaussian_posterior(h, y, noise, 1.3)
        grid = np.linspace(-5, 5, 100001)
        logp = -.5 * (grid / 1.3) ** 2 - .5 * np.sum(((h * grid - y[:, None]) / noise[:, None]) ** 2, axis=0)
        probability = np.exp(logp - logp.max()); probability /= probability.sum()
        numerical_mean = probability @ grid
        numerical_variance = probability @ (grid - numerical_mean) ** 2
        self.assertAlmostEqual(mean[0], numerical_mean, places=9)
        self.assertAlmostEqual(covariance[0, 0], numerical_variance, places=9)

    def test_g1_bayes_conditioning_and_hidden_values(self):
        from unittest.mock import patch
        from pdeep_simple import baselines
        from pdeep_simple.data import known_inputs
        case = example('G1', 'poisson', 'disk')
        queries = np.column_stack([case['nodes'], np.zeros(len(case['nodes']))])
        obs = observations(case, 8, .01, np.random.default_rng(7))
        empty = {key: value[:0] for key, value in obs.items()}
        cache = {}
        with patch.object(baselines, 'solve', wraps=baselines.solve) as solver:
            prior = baselines.predict_baseline('prior', case, queries, empty, 'inverse', 5)[0]
            zero = baselines.prepare_baseline('bayes', case, empty, 'inverse', 5, cache)
            np.testing.assert_allclose(baselines.baseline_values(zero, case, queries)[0], prior)
            np.testing.assert_allclose(zero['amplitude_covariance'], 9 * np.eye(4))
            for mode in ('forward', 'hybrid'):
                result = baselines.prepare_baseline('bayes', case, obs, mode, 5, cache)
                field = baselines.baseline_values(result, case, queries)[0][:, 0]
                np.testing.assert_allclose(field, case['solution'][0], atol=1e-10)
                np.testing.assert_array_equal(result['amplitude_covariance'], np.zeros((4, 4)))
            # One prior solve and four basis solves, reused across modes/counts.
            self.assertEqual(solver.call_count, 5)
        before = baselines.predict_baseline('bayes', case, queries, obs, 'inverse', 5)
        changed_obs = dict(obs, values=obs['values'] + .1)
        after = baselines.predict_baseline('bayes', case, queries, changed_obs, 'inverse', 5)
        self.assertGreater(float(abs(before[0] - after[0]).max()), 1e-3)
        changed = dict(case, source=np.full_like(case['source'], 99.),
                       amplitudes=np.full_like(case['amplitudes'], -99.),
                       solution=np.full_like(case['solution'], 123.))
        hidden = baselines.predict_baseline('bayes', changed, queries, obs, 'inverse', 5)
        np.testing.assert_array_equal(before[0], hidden[0])
        # A single exact source row leaves three uncertain amplitude directions.
        values, masks, scalar, scalar_masks = known_inputs(case, 'inverse', np.random.default_rng(5))
        row = np.argmax(np.linalg.norm(case['basis'], axis=1))
        masks[row, 0] = 1; values[row, 0] = case['source'][row]
        with patch.object(baselines, 'known_inputs', return_value=(values, masks, scalar, scalar_masks)):
            partial0 = baselines.prepare_baseline('bayes', case, empty, 'hybrid', 5, cache)
            partial = baselines.prepare_baseline('bayes', case, obs, 'hybrid', 5, cache)
        constraint = case['basis'][row].astype(float)
        self.assertAlmostEqual(float(constraint @ partial['amplitude_mean']), case['source'][row], places=10)
        np.testing.assert_allclose(constraint @ partial['amplitude_covariance'], 0, atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(partial['amplitude_covariance'], tol=1e-10), 3)
        self.assertLess(np.trace(partial['amplitude_covariance']), np.trace(partial0['amplitude_covariance']))
        covariance = before[1]['amplitude_covariance']
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)

    def test_observation_permutation_and_query_chunking(self):
        case = example(); scales = normalization([case]); rng = np.random.default_rng(3)
        obs = observations(case, 9, .01, rng)
        permutation = np.arange(8, -1, -1)
        shuffled = {key: value[permutation] for key, value in obs.items()}
        inputs = model_input(case, 'inverse', obs, scales, np.random.default_rng(4))
        other = model_input(case, 'inverse', shuffled, scales, np.random.default_rng(4))
        coords = np.column_stack([case['nodes'][10:20], np.full(10, .3)])
        for kind in ('A', 'C'):
            model = build_model(kind, width=32, layers=2, heads=4, memory=4, report=False).eval()
            with torch.no_grad():
                context = model.encode(inputs)
                a = model.decode(context, query_input(case, coords))
                b = model(other, query_input(case, coords))['field']
                c = torch.cat([model.decode(context, query_input(case, coords[:4])),
                               model.decode(context, query_input(case, coords[4:]))])
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(a, c, atol=2e-6, rtol=2e-5)

    def test_variable_coefficient_nonlinear_weak_residual(self):
        case = example('T4', 'semilinear', 'disk'); scales = normalization([case])
        obs = observations(case, 0, .01, np.random.default_rng(0))
        inputs = model_input(case, 'forward', obs, scales, np.random.default_rng(0))
        # Insert a genuine converged FEM solution into the residual diagnostic.
        class Reference:
            def decode(self, context, queries):
                field = interpolate(case, queries['coords'].detach().numpy()) / scales['output']
                return torch.tensor(field, dtype=torch.float32)
        residual = weak_residual(Reference(), {}, case, inputs, scales)
        self.assertLess(float(residual.square().mean().sqrt()), 2e-4)

    def test_flow_gradients_and_whole_field_chunk_consistency(self):
        from unittest.mock import patch
        from pdeep_simple.flow import LatentFlow
        from pdeep_simple.losses import case_loss, parse_losses
        from pdeep_simple.evaluation import prediction_values
        case = example('G1', 'wave', 'lshape'); scales = normalization([case])
        rng = np.random.default_rng(91)
        obs = observations(case, 3, .01, rng)
        inputs = model_input(case, 'inverse', obs, scales, rng)
        coords = np.column_stack([case['nodes'][5:17], np.full(12, .4)])
        targets = torch.tensor(interpolate(case, coords), dtype=torch.float32)
        for kind in ('A', 'C'):
            torch.manual_seed(9)
            model = LatentFlow(build_model(kind, width=32, layers=2, heads=4, memory=4, report=False), 4, 24)
            model.set_stage('ae')
            context = model.with_latent(model.encode(inputs), model.reference_latent(case, scales))
            loss, _ = case_loss(model, case, inputs, query_input(case, coords, requires_grad=True), targets,
                                scales, parse_losses('field=1,kinematic=.001'), context)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(float(model.field_encoder.pool[-1].weight.grad.abs().sum()), 0.)
            model.calibrate([case], scales); model.set_stage('flow')
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            with torch.no_grad():
                context = model.encode(inputs); target = model.reference_latent(case, scales)
            fm = model.matching_loss(context, target, rng); fm.backward()
            self.assertTrue(torch.isfinite(fm))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.velocity.parameters()))
            with patch.object(model, 'reference_latent', side_effect=AssertionError('reference leaked into sampling')):
                latent = model.sample_latents(context, 3, 3, 49)
                mean, first = prediction_values(model, context, case, coords, scales, 5, latent)
                other, second = prediction_values(model, context, case, coords, scales, 11, latent)
            self.assertEqual(mean.shape, (12, 2))
            self.assertEqual(first['sample_fields'].shape, (3, 12, 2))
            np.testing.assert_allclose(mean, other, atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(first['sample_fields'], second['sample_fields'], atol=2e-6, rtol=2e-5)
            self.assertGreater(float(first['sample_std'].max()), 1e-7)

    def test_multi_resolution_domain_and_refined_inputs(self):
        from pdeep_simple.geometry import evaluation_grid, refine_mesh
        case = example('G1', 'heat', 'lshape')
        refined = refine_mesh(case, 1)
        self.assertAlmostEqual(element_geometry(case['nodes'], case['triangles'])[0].sum(),
                               element_geometry(refined['nodes'], refined['triangles'])[0].sum())
        for resolution in (12, 24):
            points, weights, arrays = evaluation_grid(case, resolution)
            self.assertEqual(arrays['grid_mask'].shape, (resolution, resolution))
            self.assertTrue(np.all(weights > 0))
            self.assertFalse(np.any((points[:, 0] > case['nodes'][:, 0].max() / 2 + 1e-12) &
                                    (points[:, 1] > case['nodes'][:, 1].max() / 2 + 1e-12)))
            q = np.column_stack([points, np.zeros(len(points))])
            source = np.stack([case['initial_u'], case['initial_u']], -1)[None]
            fine_source = np.stack([refined['initial_u'], refined['initial_u']], -1)[None]
            stored_times = case['times']; case['times'] = refined['times'] = np.array([0.])
            np.testing.assert_allclose(interpolate(case, q, source), interpolate(refined, q, fine_source), atol=1e-10)
            case['times'] = refined['times'] = stored_times

    def test_configured_fields_new_geometry_and_p_laplacian(self):
        from pdeep_simple.baselines import predict_baseline
        from pdeep_simple.physics import solve
        settings = dict(basis=dict(count=[5, 8], amplitude_mean=.2, amplitude_std=.7, scale_by_count=True),
                        pdes=dict(diffusion=dict(coefficient_kind='interface', conductivity=1.2),
                                  damped_wave=dict(velocity_basis=2, forcing_basis=3, damping=.5),
                                  p_laplacian=dict(p=2.)),
                        conditioning=dict(hybrid_known_source_fraction=0.))
        for tier, pde, geometry in [('G1', 'diffusion', 'two_holes'), ('G1', 'damped_wave', 'triangle'),
                                    ('T4', 'p_laplacian', 'star'), ('T4', 'eikonal', 'crescent')]:
            config = dict(tier=tier, resolution=6, frames=5, dt=.01, settings=settings)
            entry = dict(seed=17, pde=pde, geometry=geometry, split='test', id=0)
            case, repeated = make_case(config, entry), make_case(config, entry)
            np.testing.assert_array_equal(case['solution'], repeated['solution'])
            self.assertTrue(np.isfinite(case['solution']).all())
            np.testing.assert_allclose(case['solution'][:, case['boundary_nodes']], 0., atol=1e-12)
            for field, prior in case['meta']['field_priors'].items():
                self.assertEqual(case[f'{field}_basis'].shape[1], prior['count'])
                self.assertEqual(len(case[f'{field}_amplitudes']), prior['count'])
            if pde == 'damped_wave':
                self.assertEqual(case['initial_v_basis'].shape[1], 2)
                self.assertEqual(case['source_basis'].shape[1], 3)
                self.assertIn('velocity', case)
            obs = observations(case, 4, .01, np.random.default_rng(4))
            inputs = model_input(case, 'inverse', obs, normalization([case]), np.random.default_rng(5))
            q = query_input(case, np.column_stack([case['nodes'][:5], np.zeros(5)]))
            for kind in ('A', 'C'):
                model = build_model(kind, width=32, layers=2, heads=4, memory=4, report=False)
                prediction = model(inputs, q)['field']
                prediction.square().mean().backward()
                self.assertTrue(torch.isfinite(prediction).all())
            if pde == 'diffusion':
                empty = {k: v[:0] for k, v in obs.items()}
                queries = np.column_stack([case['nodes'], np.zeros(len(case['nodes']))])
                prior = predict_baseline('prior', case, queries, empty, 'inverse', 5)[0]
                bayes = predict_baseline('bayes', case, queries, empty, 'inverse', 5)
                np.testing.assert_allclose(bayes[0], prior, atol=1e-12)
                np.testing.assert_allclose(bayes[1]['amplitude_covariance'], np.eye(case['basis'].shape[1]) * case['meta']['amplitude_prior_std'] ** 2)
            if pde == 'p_laplacian':
                reference, _ = solve(case, 'poisson', case['source'], case['initial_u'])
                np.testing.assert_allclose(case['solution'], reference['solution'], atol=1e-12)

    def test_pod_gp_and_nonlinear_map(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from pdeep_simple.data import save_case, load_case
        from pdeep_simple.baselines import fit_pod, predict_baseline, prepare_baseline, baseline_values
        with TemporaryDirectory() as directory:
            root = Path(directory); entries, training = [], []
            for index in range(6):
                case = example('T1', 'poisson', seed=20 + index)
                split = 'train' if index < 4 else 'val' if index == 4 else 'test'
                name = f'{split}_{index}.npz'
                if split == 'train':
                    training.append(case['solution'][0])
                else:
                    case['solution'][:] = 999.  # never used to fit POD
                save_case(root / name, case); entries.append(dict(file=name, split=split))
            (root / 'manifest.json').write_text(json.dumps(dict(cases=entries)))
            pod = fit_pod(root, root / 'pod.npz', 4)
            settings = dict(pod_model=pod, pod_ridge=1e-10, gp_amplitude=.8)
            case = load_case(root / 'train_0.npz')
            queries = np.column_stack([case['nodes'], np.zeros(len(case['nodes']))])
            empty = observations(case, 0, .01, np.random.default_rng(4))
            mean, _ = predict_baseline('pod', case, queries, empty, 'inverse', 5, settings)
            np.testing.assert_allclose(mean[:, 0], np.mean(training, axis=0), atol=1e-8)
            obs = dict(queries=queries, quantity=np.zeros(len(queries), int), values=case['solution'][0],
                       noise=np.linspace(.005, .02, len(queries)))
            reconstructed = predict_baseline('pod', case, queries, obs, 'inverse', 5, settings)[0]
            np.testing.assert_allclose(reconstructed[:, 0], case['solution'][0], atol=1e-7)
            zero, uncertainty = predict_baseline('gp', case, queries, empty, 'inverse', 5, settings)
            np.testing.assert_array_equal(zero, 0.)
            np.testing.assert_allclose(uncertainty['solution_std'], .8)
            gp = prepare_baseline('gp', case, obs, 'inverse', 5, settings=settings)
            predicted, uncertainty = baseline_values(gp, case, queries)
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue(np.isfinite(uncertainty['solution_std']).all())
            self.assertTrue(np.all(uncertainty['solution_std'] <= .8))
            chunks = [baseline_values(gp, case, part)[0] for part in np.array_split(queries, 3)]
            np.testing.assert_allclose(predicted, np.concatenate(chunks), atol=1e-12)
            changed = dict(case, solution=np.full_like(case['solution'], 999.), source=np.full_like(case['source'], 99.),
                           amplitudes=np.full_like(case['amplitudes'], -99.))
            for method in ('pod', 'gp'):
                before = predict_baseline(method, case, queries, obs, 'inverse', 5, settings)[0]
                after = predict_baseline(method, changed, queries, obs, 'inverse', 5, settings)[0]
                np.testing.assert_array_equal(before, after)
        case = example('T4', 'semilinear')
        queries = np.column_stack([case['nodes'], np.zeros(len(case['nodes']))])
        obs = observations(case, 12, .01, np.random.default_rng(7))
        prediction, info = predict_baseline('map', case, queries, obs, 'inverse', 5)
        self.assertLess(info['map_objective'], info['map_initial_objective'])
        self.assertGreater(info['map_forward_solves'], 1)
        self.assertEqual(info['map_success'], 1)
        changed = dict(case, solution=np.full_like(case['solution'], 999.), source=np.full_like(case['source'], 99.),
                       amplitudes=np.full_like(case['amplitudes'], -99.))
        other = predict_baseline('map', changed, queries, obs, 'inverse', 5)[0]
        np.testing.assert_array_equal(prediction, other)


if __name__ == '__main__':
    unittest.main()
