from __future__ import annotations
import math
import numpy as np

class Expansion:
    __slots__ = ("coeffs", "scheme")

    def __init__(self, coeffs, scheme):
        self.coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
        self.scheme = scheme
        if self.coeffs.shape != (scheme.n_coeff,):
            raise ValueError(
                f"coeffs length {self.coeffs.shape} != scheme.n_coeff {scheme.n_coeff}"
            )

    def value(self) -> float:
        return float(self.coeffs[0])

    def numpy(self) -> np.ndarray:
        return self.coeffs.copy()

    def coeff(self, k: int) -> float:
        if self.scheme.vars != 1:
            raise NotImplementedError("multivariate coeff() arrives in M2")
        if not (0 <= k < self.scheme.n_coeff):
            raise IndexError(f"coeff index {k} out of range [0, {self.scheme.n_coeff})")
        return float(self.coeffs[k])

    def derivative(self, k: int) -> float:
        return self.coeff(k) * math.factorial(k)

    # --- arithmetic: delegate to the eager engine (Task 11) ---
    def __add__(self, other):
        from .eager import binary
        return binary("add", self, other)

    def __radd__(self, other):
        from .eager import binary
        return binary("add", other, self)

    def __sub__(self, other):
        from .eager import binary
        return binary("sub", self, other)

    def __rsub__(self, other):
        from .eager import binary
        return binary("sub", other, self)

    def __mul__(self, other):
        from .eager import binary
        return binary("mul", self, other)

    def __rmul__(self, other):
        from .eager import binary
        return binary("mul", other, self)

    def __truediv__(self, other):
        from .eager import binary
        return binary("div", self, other)

    def __rtruediv__(self, other):
        from .eager import binary
        return binary("div", other, self)

    def __neg__(self):
        from .eager import unary
        return unary("neg", self)

    def __repr__(self):
        return f"Expansion(scheme={self.scheme}, value={self.value()})"
