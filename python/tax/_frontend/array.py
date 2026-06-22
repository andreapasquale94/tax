from __future__ import annotations
import numpy as np
from .types import Expansion


class Array:
    """A vector of K expansions over one shared scheme; contiguous (K, nCoeff) buffer."""
    __slots__ = ("coeffs", "scheme")

    def __init__(self, coeffs, scheme):
        self.coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
        self.scheme = scheme
        if self.coeffs.ndim != 2 or self.coeffs.shape[1] != scheme.n_coeff:
            raise ValueError(
                f"Array coeffs shape {self.coeffs.shape} != (K, {scheme.n_coeff})"
            )

    def __len__(self) -> int:
        return self.coeffs.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Array(self.coeffs[i], self.scheme)
        return Expansion(self.coeffs[i], self.scheme)

    def _rows(self) -> list:
        return [Expansion(self.coeffs[i], self.scheme) for i in range(len(self))]

    def value(self) -> np.ndarray:
        return self.coeffs[:, 0].copy()

    def numpy(self) -> np.ndarray:
        return self.coeffs.copy()

    def _map_unary(self, op):
        from . import eager
        results = [eager.unary(op, r) for r in self._rows()]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)

    def _map_binary(self, op, other):
        from . import eager
        rows = self._rows()
        if isinstance(other, Array):
            if len(other) != len(self):
                raise ValueError(f"Array length mismatch: {len(self)} vs {len(other)}")
            results = [eager.binary(op, a, b) for a, b in zip(rows, other._rows())]
        else:                                  # Expansion or Python scalar -> broadcast
            results = [eager.binary(op, a, other) for a in rows]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)

    def __add__(self, other): return self._map_binary("add", other)
    def __radd__(self, other): return self._map_binary("add", other)
    def __sub__(self, other): return self._map_binary("sub", other)
    def __rsub__(self, other):
        from . import eager
        rows = self._rows()
        results = [eager.binary("sub", other, a) for a in rows]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)
    def __mul__(self, other): return self._map_binary("mul", other)
    def __rmul__(self, other): return self._map_binary("mul", other)
    def __truediv__(self, other): return self._map_binary("div", other)
    def __rtruediv__(self, other):
        from . import eager
        rows = self._rows()
        results = [eager.binary("div", other, a) for a in rows]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)
    def __neg__(self): return self._map_unary("neg")

    def __repr__(self):
        return f"Array(K={len(self)}, scheme={self.scheme})"
