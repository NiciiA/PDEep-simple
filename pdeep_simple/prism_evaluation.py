"""B-specific physical posterior summaries; ordinary field metrics stay shared."""
import numpy as np
import torch
from .prism import PHYSICAL_FIELDS, relevant_fields, physical_scales


def posterior_diagnostics(model, context, latents, case, interval):
    inputs = context['inputs']; weights = inputs['node_weights'].cpu().numpy(); weights /= weights.sum()
    with torch.no_grad():
        candidates = [model.with_latent(context, z) for z in latents]
        fields = np.stack([c['physical']['fields'].cpu().numpy() for c in candidates]).astype(np.float64)
        coefficients = np.array([float(c['physical']['coefficient']) for c in candidates])
        model.prepare_sensors(context)
        obs = inputs['observations'][~inputs['observation_mask']]
        predictions = [model.decode(c, context['sensor_queries'])[torch.arange(len(obs), device=obs.device), obs[:, 3].long()]
                       for c in candidates] if len(obs) else []
    mean = fields.mean(0); std = fields.std(0, ddof=1) if len(fields) > 1 else np.full_like(mean, np.nan)
    lower, upper = np.quantile(fields, [(1 - interval) / 2, (1 + interval) / 2], axis=0)
    reference = np.column_stack([case[name] for name in PHYSICAL_FIELDS])
    scales = physical_scales(inputs).cpu().numpy()
    masks = inputs['field_masks'].cpu().numpy()
    metrics = dict(information_seconds=context['information_seconds'], information_evaluations=len(context['information_records']),
                   observation_normalized_rmse=np.nan, observation_sample_normalized_rmse=np.nan,
                   observation_effective_noise_rmse=np.nan)
    for q, name in enumerate(('f', 'u0', 'v0', 'a')):
        active = bool(relevant_fields(inputs)[q]); unknown = (1 - masks[:, q]) * weights
        error = (mean[:, q] - reference[:, q]) ** 2
        metrics[f'physical_{name}_rmse'] = float(np.sqrt(error @ weights)) if active else np.nan
        mse = float(error @ unknown / unknown.sum()) if active and unknown.sum() else np.nan
        metrics[f'physical_{name}_unknown_rmse'] = float(np.sqrt(mse))
        metrics[f'physical_{name}_unknown_normalized_rmse'] = float(np.sqrt(mse) / scales[q])
        metrics[f'physical_{name}_coverage'] = (float(((reference[:, q] >= lower[:, q]) & (reference[:, q] <= upper[:, q])) @ unknown / unknown.sum())
                                                   if active and unknown.sum() else np.nan)
    metrics.update(physical_coefficient_std=float(coefficients.std(ddof=1)) if len(coefficients) > 1 else np.nan,
                   physical_coefficient_lower=float(np.quantile(coefficients, (1 - interval) / 2)),
                   physical_coefficient_upper=float(np.quantile(coefficients, (1 + interval) / 2)))
    if predictions:
        predicted = torch.stack(predictions); error = predicted - obs[:, 4]
        variance = obs[:, 5].square() + model.surrogate_variance[obs[:, 3].long()] + 1e-6
        metrics.update(observation_normalized_rmse=float(error.mean(0).square().mean().sqrt()),
                       observation_sample_normalized_rmse=float(error.square().mean().sqrt()),
                       observation_effective_noise_rmse=float((error.mean(0).square() / variance).mean().sqrt()))
    records = context['information_records']
    eigenvalues = np.stack([r['values'].cpu().numpy() for r in records]) if records else np.empty((0, 0))
    routing_weights = np.stack([r['weights'].cpu().numpy() for r in records]) if records else np.empty((0, 0))
    metrics.update(information_max_eigenvalue=float(eigenvalues.max()) if eigenvalues.size else 0.,
                   information_effective_directions=float(routing_weights.sum(1).mean()) if routing_weights.size else 0.,
                   information_well_informed=float((routing_weights >= .5).sum(1).mean()) if routing_weights.size else 0.,
                   information_weight_threshold=.5)
    arrays = dict(physical_mean=mean.astype(np.float32), physical_std=std.astype(np.float32),
                  physical_reference=reference.astype(np.float32), physical_lower=lower.astype(np.float32),
                  physical_upper=upper.astype(np.float32), physical_samples=fields[:4].astype(np.float32),
                  physical_scalar_samples=coefficients, physical_relevant=np.array(relevant_fields(inputs)),
                  physical_coefficient_means=fields[:, :, 3] @ weights,
                  physical_constant_coefficient=np.array(inputs['physical_layout']['constant_coefficient']),
                  information_eigenvalues=eigenvalues, information_weights=routing_weights,
                  information_flow_times=np.array([r['flow_time'] for r in records]),
                  information_sample_indices=np.array([r['sample_index'] for r in records], dtype=int),
                  information_states=np.stack([r['state'].cpu().numpy() for r in records]) if records else np.empty((0, model.latent_dim)),
                  surrogate_variance=model.surrogate_variance.cpu().numpy())
    return metrics, arrays


def physical_plots(path, out):
    import matplotlib.pyplot as plt
    from .plotting import field_panel
    with np.load(path, allow_pickle=False) as saved:
        if 'physical_mean' not in saved: return
        case = dict(nodes=saved['nodes'], triangles=saved['triangles'])
        names = ('forcing', 'initial u', 'initial v', 'coefficient')
        active = np.flatnonzero(saved['physical_relevant'])
        fig, axes = plt.subplots(len(active), 3, figsize=(11, 3 * len(active)), squeeze=False)
        for row, q in enumerate(active):
            status = 'known' if saved['known_field_mask'][:, q].all() else 'partly known' if saved['known_field_mask'][:, q].any() else 'hidden'
            for col, (values, title) in enumerate(((saved['physical_mean'][:, q], 'posterior mean'),
                    (saved['physical_std'][:, q], 'posterior std'),
                    (abs(saved['physical_mean'][:, q] - saved['physical_reference'][:, q]), 'absolute error'))):
                limits = (0., max(float(np.nanmax(values)), 1e-8)) if col and np.isfinite(values).any() else None
                field_panel(axes[row, col], case, values, names[q] + f' ({status}): ' + title, limits)
        fig.tight_layout(); fig.savefig(out / (path.stem + '_physical.png'), dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        coefficients = saved['physical_scalar_samples'] if bool(saved['physical_constant_coefficient']) else saved['physical_coefficient_means']
        axes[0].hist(coefficients, bins=min(20, max(1, len(coefficients))))
        import json
        meta = json.loads(str(saved['metadata']))
        constant = bool(saved['physical_constant_coefficient'])
        from .geometry import node_weights
        reference = (meta['coefficient'] if constant else
                     np.average(saved['physical_reference'][:, 3], weights=node_weights(case)))
        axes[0].axvline(reference, color='black', label='reference'); axes[0].legend()
        axes[0].set_title('Scalar coefficient' if constant else 'Domain-mean coefficient')
        for key, ax in zip(('information_eigenvalues', 'information_weights'), axes[1:]):
            for row in saved[key]: ax.plot(np.arange(1, len(row) + 1), row, alpha=.3)
            ax.set_title(key.replace('_', ' ')); ax.set_xlabel('retained direction')
        axes[2].set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(out / (path.stem + '_information.png'), dpi=150); plt.close(fig)
