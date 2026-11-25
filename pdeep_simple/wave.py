"""Small cached FEM wave basis and exact undamped modal time evolution."""
import hashlib
import math
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from .physics import assemble


def modal_basis(inputs, count, cache):
    geometry = inputs['wave_geometry']; operator = inputs['wave_operator']
    reaction = inputs['wave_reaction']; device = inputs['nodes'].device
    digest = hashlib.blake2b(digest_size=16)
    for array in (*geometry.values(), operator, np.asarray(reaction)):
        digest.update(np.ascontiguousarray(array).tobytes())
    key = (digest.digest(), count, str(device))
    if key not in cache:
        free = np.setdiff1d(np.arange(len(geometry['nodes'])), geometry['boundary_nodes'])
        if not len(free):
            cache[key] = None
            return None
        mass, stiffness = assemble(geometry, operator, reaction=reaction)
        m, k = mass[free][:, free], stiffness[free][:, free]
        if len(free) == 1:
            values = np.array([k[0, 0] / m[0, 0]])
            vectors = np.array([[1. / np.sqrt(m[0, 0])]])
        else:
            # ARPACK needs k < n; retain at most n-1 modes on very small meshes.
            values, vectors = eigsh(k, k=min(count, len(free) - 1), M=m, sigma=0., which='LM',
                                    v0=np.linspace(1., 2., len(free)), tol=1e-9)
        order = np.argsort(values); values, vectors = values[order], vectors[:, order]
        if np.any(values <= 0):
            raise ValueError('E-Geo needs a positive definite known wave operator')
        vectors /= np.sqrt(np.sum(vectors * (m @ vectors), axis=0))
        vectors *= np.where(vectors[np.abs(vectors).argmax(0), np.arange(len(values))] < 0, -1., 1.)
        phi = np.zeros((len(geometry['nodes']), len(values))); phi[free] = vectors
        cache[key] = {name: torch.as_tensor(array, dtype=inputs['nodes'].dtype, device=device)
                      for name, array in dict(phi=phi, projection=(mass @ phi).T, omega=np.sqrt(values)).items()}
    return cache[key]


def sinc(x):
    # Smooth at zero, including the second derivatives needed to train velocity.
    small = x.abs() < .01
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x.square() / 6 + x.pow(4) / 120, safe.sin() / safe)


def modal_evolution(omega, a, b, forcing, time, frequency=0.):
    """q'' + omega^2 q = forcing*cos(2*pi*frequency*t), with q(0)=a, q'(0)=b.

    The sinc product also handles resonance and constant forcing without a
    division by omega^2 - forcing_frequency^2.
    """
    angle = time * omega
    cosine, sine = angle.cos(), angle.sin()
    plus = (omega + 2 * math.pi * frequency) * time / 2
    minus = (omega - 2 * math.pi * frequency) * time / 2
    response = .5 * time.square() * sinc(plus) * sinc(minus)
    response_dt = .5 * time * (sinc(plus) * minus.cos() + plus.cos() * sinc(minus))
    q = a * cosine + b / omega * sine + forcing * response
    velocity = -a * omega * sine + b * cosine + forcing * response_dt
    return q, velocity
