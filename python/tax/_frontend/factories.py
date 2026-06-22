from __future__ import annotations
import numpy as np
from .scheme import Isotropic
from .types import Expansion

def variable(x0: float, order: int) -> Expansion:
    scheme = Isotropic(order, 1)
    coeffs = np.zeros(scheme.n_coeff, dtype=np.float64)
    coeffs[0] = float(x0)
    if order >= 1:
        coeffs[1] = 1.0
    return Expansion(coeffs, scheme)
