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

    def __repr__(self):
        return f"Array(K={len(self)}, scheme={self.scheme})"
