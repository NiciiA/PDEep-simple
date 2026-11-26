"""Small exact-posterior experiment for B's routed flow component.

Uses an exact linear forward map and identity physical codec to isolate flow
quality from surrogate/representation error. It is not a PDE calibration result.
No training runs unless --updates is explicitly positive.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pdeep_simple.models import mlp
from pdeep_simple.prism import RoutedFlow, information_matrix, information_subspace, project_information, route_velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, default=0)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--routing', choices=['ordinary', 'fixed', 'dynamic'], default='dynamic')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.updates < 0 or min(args.samples, args.steps) < 1: parser.error('invalid counts')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    h = torch.tensor([[1., .4, 0., 0.], [0., 1., .3, 0.]])
    variance = torch.full((2,), .16); y = torch.tensor([.3, -.2])
    gram = information_matrix(h, variance)
    covariance = torch.linalg.inv(torch.eye(4) + gram)
    mean = covariance @ h.T @ (y / variance)
    subspace = information_subspace(h, variance, rank=4, tau=1.)
    assert torch.linalg.eigvalsh(gram).min() > -1e-6
    torch.testing.assert_close(project_information(torch.eye(4), subspace), torch.eye(4) - covariance, atol=1e-6, rtol=1e-5)
    assert covariance[3, 3] == 1  # The unobserved direction retains prior variance.
    result = dict(exact_mean=mean.tolist(), exact_covariance=covariance.tolist(),
                  information_eigenvalues=subspace['values'].tolist(), updates=args.updates)
    if args.updates:
        network = RoutedFlow(32, 2, 4, 4)
        condition = mlp(2, 32, 32)
        optimizer = torch.optim.AdamW(list(network.parameters()) + list(condition.parameters()), lr=1e-3)
        empty = dict(vectors=torch.empty(4, 0), values=torch.empty(0), weights=torch.empty(0))
        info = empty if args.routing == 'ordinary' else subspace
        def context(value):
            token = condition(value)
            return dict(global_h=token, memory=token[None])
        for _ in range(args.updates):
            target = torch.randn(4); observed = h @ target + variance.sqrt() * torch.randn(2)
            base = torch.randn(4); clock = torch.rand(()); state = (1 - clock) * base + clock * target
            evidence, ambiguity = network(state, clock, context(observed), info)
            velocity = (evidence + ambiguity) / 2 if args.routing == 'ordinary' else route_velocity(evidence, ambiguity, info)
            objective = (velocity - (target - base)).square().mean()
            if args.routing != 'ordinary':
                first = project_information(evidence - target + base, info)
                second = ambiguity - target + base
                objective += .1 * (first.square().mean() + (second - project_information(second, info)).square().mean())
            optimizer.zero_grad(); objective.backward(); optimizer.step()
        draws = []
        with torch.no_grad():
            conditioning = context(y)
            for _ in range(args.samples):
                state = torch.randn(4)
                def velocity(z, s):
                    evidence, ambiguity = network(z, s, conditioning, info)
                    return (evidence + ambiguity) / 2 if args.routing == 'ordinary' else route_velocity(evidence, ambiguity, info)
                for index in range(args.steps):
                    clock = torch.tensor(index / args.steps)
                    first = velocity(state, clock)
                    second = velocity(state + first / args.steps, clock + 1 / args.steps)
                    state += (first + second) / (2 * args.steps)
                draws.append(state)
        draws = torch.stack(draws); sample_covariance = torch.cov(draws.T)
        result.update(sample_mean=draws.mean(0).tolist(), sample_covariance=sample_covariance.tolist(),
                      mean_error=float((draws.mean(0) - mean).norm()), covariance_error=float((sample_covariance - covariance).norm()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
