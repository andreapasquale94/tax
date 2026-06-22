from __future__ import annotations
import math
import numpy as np
from .scheme import flat_index, unflat_index

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

    def coeff(self, *alpha, **axes) -> float:
        if axes:
            if alpha:
                raise ValueError("coeff: pass positional exponents OR axis keywords, not both")
            from .scheme import Named
            if not isinstance(self.scheme, Named):
                raise ValueError("coeff(**axes): keyword form requires a named expansion")
            a = [0] * self.scheme.vars
            for name, e in axes.items():
                if self.scheme.dim_of(name) != 1:
                    raise ValueError(
                        f"coeff keyword form supports 1-D axes only; {name!r} is multi-dim "
                        "— use positional coeff(*exponents)"
                    )
                a[self.scheme.var_offset(name)] = int(e)
            alpha = tuple(a)
        if len(alpha) != self.scheme.vars:
            raise ValueError(
                f"coeff expects {self.scheme.vars} exponents, got {len(alpha)}"
            )
        if any(x < 0 for x in alpha):
            raise ValueError("coeff: negative exponent")
        k = flat_index(alpha)
        return float(self.coeffs[k]) if k < self.scheme.n_coeff else 0.0

    def derivative(self, *alpha) -> float:
        fac = 1
        for a in alpha:
            fac *= math.factorial(a)
        return self.coeff(*alpha) * fac

    def gradient(self, name=None) -> np.ndarray:
        M = self.scheme.vars
        g = np.empty(M, dtype=np.float64)
        for i in range(M):
            e = tuple(1 if j == i else 0 for j in range(M))
            g[i] = self.coeff(*e)
        if name is None:
            return g
        from .scheme import Named
        if not isinstance(self.scheme, Named):
            raise ValueError("gradient(name): requires a named expansion")
        off = self.scheme.var_offset(name)
        return g[off: off + self.scheme.dim_of(name)]

    def hessian(self) -> np.ndarray:
        M = self.scheme.vars
        H = np.empty((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(M):
                a = [0] * M
                a[i] += 1
                a[j] += 1
                H[i, j] = self.derivative(*a)
        return H

    def eval(self, dx) -> float:
        M = self.scheme.vars
        if len(dx) != M:
            raise ValueError(f"eval expects {M} displacements, got {len(dx)}")
        total = 0.0
        for k in range(self.scheme.n_coeff):
            a = unflat_index(k, M)
            term = float(self.coeffs[k])
            for i in range(M):
                term *= dx[i] ** a[i]
            total += term
        return total

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
