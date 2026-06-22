from __future__ import annotations
import numpy as np
from .scheme import Isotropic, Named, Axis
from .types import Expansion
from .array import Array

def variable(x0, order, name=None):
    if name is None:
        scheme = Isotropic(order, 1)
        coeffs = np.zeros(scheme.n_coeff, dtype=np.float64)
        coeffs[0] = float(x0)
        if order >= 1:
            coeffs[1] = 1.0
        return Expansion(coeffs, scheme)
    scheme = Named.of(order, [Axis(name, 1)])
    coeffs = np.zeros(scheme.n_coeff, dtype=np.float64)
    coeffs[0] = float(x0)
    if order >= 1:
        coeffs[1] = 1.0                       # the axis's single var -> flat index 1
    return Expansion(coeffs, scheme)

def variables(point, order, name=None):
    point = list(point)
    M = len(point)
    if M < 1:
        raise ValueError("variables(): point must have at least one element")
    if name is None:
        scheme = Isotropic(order, M)
    else:
        scheme = Named.of(order, [Axis(name, M)])
    data = np.zeros((M, scheme.n_coeff), dtype=np.float64)
    for i in range(M):
        data[i, 0] = float(point[i])
        if order >= 1:
            data[i, i + 1] = 1.0              # var i -> flat index i+1 (single axis block)
    return Array(data, scheme)
