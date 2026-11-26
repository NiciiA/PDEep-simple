"""F-Geo: E's decoder with aligned and refined initial-state inference."""
import torch
from .models import AGeo, EGeo
from .data import PDES
from .geometry import node_weights
from .physics import PARABOLIC, WAVES
from .wave import assimilate, observation_inputs, observed_values


class FGeo(EGeo):
    initial_pdes = PARABOLIC + WAVES

    def __init__(self, width, layers, heads, memory, wave_modes=32, initial_latent=32,
                 assimilation=True, infer_operator=False, f_variant='base', f_steps=3):
        super().__init__(width, layers, heads, memory, wave_modes, initial_latent, assimilation, infer_operator)
        if f_variant not in ('base', 'align', 'iterative', 'observable') or f_steps not in (2, 3, 4):
            raise ValueError('F needs base/align/iterative/observable and 2–4 correction steps')
        self.f_variant, self.f_steps = f_variant, f_steps
        self.flow_conditioned = False
        self.register_buffer('_f_version', torch.tensor(1))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + '_f_version' not in state_dict or int(state_dict[prefix + '_f_version']) != 1:
            raise ValueError('Old F checkpoints are incompatible with the new E-based F-Geo; use a new run')
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def encode(self, inputs, correct=True):
        pde = PDES[inputs['pde']]
        if pde not in self.initial_pdes:
            return super().encode(inputs, correct=False)
        raw = AGeo.encode(self, inputs)
        raw.update(wave=None, f_inputs=inputs, f_prior=self.initial_condition(raw['global_h']))
        observation_inputs(raw, inputs)
        context = self.state_context(raw, raw['f_prior'], raw['log_coefficient'])
        return self.finish(context) if correct else context

    def state_context(self, raw, latent, log_coefficient):
        """Decode only inferred initial values; E supplies all physical evolution."""
        inputs = raw['f_inputs']; context = dict(raw, log_coefficient=log_coefficient)
        pde = PDES[inputs['pde']]
        if pde == 'wave':
            context = EGeo.wave_context(self, context, inputs, latent, correct=False)
        else:
            weights = torch.as_tensor(node_weights(inputs['evolution_geometry']), dtype=latent.dtype, device=latent.device)
            geometry = self.initial_codec.geometry(inputs['nodes'], inputs['coordinate_bounds'], weights, inputs['features'][:, 2])
            initial = self.initial_codec.decode(geometry, latent) * inputs['output_scales']
            initial = torch.where(inputs['field_masks'][:, 1:3].bool(), inputs['field_values'][:, 1:3], initial)
            if pde in PARABOLIC:
                initial = torch.stack([initial[:, 0], torch.zeros_like(initial[:, 0])], -1)
            # Supply the decoded state internally, without changing the conditioning or its masks.
            fields = torch.cat([inputs['field_values'][:, :1], initial, inputs['field_values'][:, 3:]], -1)
            masks = inputs['field_masks'].clone(); masks[:, 1:3] = 1
            context = EGeo.evolution_context(self, context, dict(inputs, field_values=fields, field_masks=masks), pde)
        context.update(wave_inputs=inputs, f_raw=raw, f_latent=latent, refinement_steps=0,
                       f_refinement=self.f_variant in ('iterative', 'observable'))
        return context

    def finish(self, context):
        if self.assimilation:
            # A conditional flow has already seen these sensors: use only a small
            # consistency refinement, rather than another independent Bayes update.
            return self.refine(context, steps=1) if self.flow_conditioned else assimilate(self, context)
        if self.f_variant == 'observable':
            return self.refine(context, steps=0)
        context = dict(context, assimilation_applied=False)
        if len(context['assimilation_observations']):
            error = observed_values(self, context) - context['assimilation_observations'][:, 4]
            context.update(observation_error_before=error, observation_error_after=error)
        return context

    def with_initial_latent(self, context, latent):
        if 'f_raw' not in context:
            return context
        raw = context['f_raw']
        return self.finish(self.state_context(raw, latent, raw['log_coefficient']))

    def reference_latent(self, case, scales):
        if case['meta']['pde'] not in self.initial_pdes:
            return next(self.parameters()).new_zeros(self.initial_latent)
        geometry, values, weights = self.reference_initial(case, scales)
        return self.initial_codec(geometry[0], values, weights)

    def reference_initial(self, case, scales):
        geometry, values, weights = super().reference_initial(case, scales)
        if case['meta']['pde'] in PARABOLIC:
            values = torch.stack([values[:, 0], torch.zeros_like(values[:, 0])], -1)
        return geometry, values, weights

    def representation_loss(self, case, scales):
        geometry, target, weights = self.reference_initial(case, scales)
        latent = self.initial_codec(geometry[0], target, weights)
        channels = 2 if case['meta']['pde'] in WAVES else 1
        error = (self.initial_codec.decode(geometry, latent)[:, :channels] - target[:, :channels]).square().mean(-1)
        return (error * weights).sum() / weights.sum()

    def alignment_loss(self, context, case, scales):
        return (context['f_prior'] - self.reference_latent(case, scales).detach()).square().mean()

    def sensitivity(self, predicted, state):
        # A small observation-by-latent Jacobian. No second-order parameter graph.
        if state.device.type == 'mps':
            return torch.stack([torch.autograd.grad(y, state, retain_graph=True)[0] for y in predicted]).detach()
        identity = torch.eye(len(predicted), dtype=state.dtype, device=state.device)
        return torch.autograd.grad(predicted, state, identity, is_grads_batched=True, retain_graph=True)[0].detach()

    def precision(self, jacobian, variance):
        device = jacobian.device if jacobian.device.type != 'mps' else torch.device('cpu')
        j = jacobian.to(device=device, dtype=torch.float64)
        r = variance.to(device=device, dtype=torch.float64)
        return torch.eye(j.shape[1], device=device, dtype=j.dtype) + j.T @ (j / r[:, None])

    def refine(self, context, steps=None, return_stages=False):
        """First-order smoothing; optionally return (result, initial/accepted contexts)."""
        steps = self.f_steps if steps is None else steps
        stages = [context] if return_stages else None
        obs = context['assimilation_observations']; dimension = self.initial_latent
        observable = self.f_variant == 'observable'
        result = dict(context, assimilation_applied=False)
        if not len(obs):
            if observable:
                result['latent_covariance'] = torch.eye(dimension, dtype=context['f_latent'].dtype, device=context['f_latent'].device)
            return (result, stages) if return_stages else result
        gradients = torch.is_grad_enabled()
        raw = context['f_raw']; infer = bool(context.get('operator_inferred') and context['wave'] is not None)
        prior = torch.cat([context['f_latent'], context['log_coefficient'].reshape(1)]) if infer else context['f_latent']
        variance = obs[:, 5].square() + 1e-6
        with torch.enable_grad():
            state = prior if prior.requires_grad else prior.detach().requires_grad_(True)
            center = state.detach()
            def prediction(value):
                log_coefficient = value[-1] if infer else raw['log_coefficient']
                c = self.state_context(raw, value[:dimension], log_coefficient)
                return c, observed_values(self, c)
            def objective(value, predicted):
                return ((predicted - obs[:, 4]).square() / variance).sum() + (value - center).square().sum()
            current, predicted = prediction(state)
            before = predicted - obs[:, 4]; applied = 0
            for _ in range(steps):
                loss = objective(state, predicted)
                gradient = torch.autograd.grad(loss, state, retain_graph=True)[0].detach()
                if observable:
                    precision = self.precision(self.sensitivity(predicted, state), variance)
                    direction = .5 * torch.linalg.solve(precision, gradient.to(precision)).to(state)
                else:
                    direction = gradient / gradient.square().mean().sqrt().clamp_min(1.) * .25
                direction = direction.detach()
                if infer:
                    direction = torch.cat([direction[:dimension], direction[-1:].clamp(-.25, .25)])
                accepted = False
                for rate in (1., .5, .25):
                    candidate = state - rate * direction
                    with torch.no_grad():
                        _, trial = prediction(candidate)
                        better = bool(objective(candidate, trial) < loss.detach())
                    if better:
                        state = candidate; current, predicted = prediction(state)
                        applied += 1; accepted = True
                        if return_stages:
                            stages.append(dict(current, refinement_steps=applied, assimilation_applied=True))
                        break
                if not accepted:
                    break
            result = dict(current, assimilation_applied=applied > 0, refinement_steps=applied,
                          observation_error_before=before, observation_error_after=predicted - obs[:, 4])
            if observable:
                precision = self.precision(self.sensitivity(predicted, state), variance)
                covariance = torch.linalg.solve(precision, torch.eye(len(state), dtype=precision.dtype, device=precision.device))
                result['latent_covariance'] = covariance[:dimension, :dimension].to(state)
        if not gradients:
            # Inference keeps no optimizer/unrolled graph alive between query chunks.
            result = {k: v.detach() if torch.is_tensor(v) else v for k, v in result.items()}
            if result['wave'] is not None:
                result['wave'] = {k: v.detach() for k, v in result['wave'].items()}
        return (result, stages) if return_stages else result
