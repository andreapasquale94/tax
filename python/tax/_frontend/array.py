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
        return Expansion(self.coeffs[i].copy(), self.scheme)

    def _rows(self) -> list:
        return [Expansion(self.coeffs[i].copy(), self.scheme) for i in range(len(self))]

    def value(self) -> np.ndarray:
        return self.coeffs[:, 0].copy()

    def numpy(self) -> np.ndarray:
        return self.coeffs.copy()

    def eval(self, dx) -> np.ndarray:
        return np.array([r.eval(dx) for r in self._rows()], dtype=np.float64)

    def jacobian(self, name=None) -> np.ndarray:
        J = np.stack([r.gradient() for r in self._rows()])   # (K, vars)
        if name is None:
            return J
        from .scheme import Named
        if not isinstance(self.scheme, Named):
            raise ValueError("jacobian(name): requires a named Array")
        off = self.scheme.var_offset(name)
        return J[:, off: off + self.scheme.dim_of(name)]

    def hessian(self) -> np.ndarray:
        return np.stack([r.hessian() for r in self._rows()])

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
    def __pow__(self, p): return self._map_binary("pow", p)
    def __neg__(self): return self._map_unary("neg")

    def __repr__(self):
        return f"Array(K={len(self)}, scheme={self.scheme})"


def concatenate(items) -> Array:
    from .trace import Tracer, ArrayTracer
    if any(isinstance(it, (Tracer, ArrayTracer)) for it in items):
        rows = []
        for it in items:
            if isinstance(it, ArrayTracer):
                rows.extend(it._rows())
            else:
                rows.append(it)             # a scalar Tracer
        return ArrayTracer(rows, rows[0].scheme)
    from . import eager
    exps = []
    for it in items:
        if isinstance(it, Array):
            exps.extend(it._rows())
        elif isinstance(it, Expansion):
            exps.append(it)
        else:
            raise TypeError(f"concatenate: expected Expansion or Array, got {type(it).__name__}")
    if not exps:
        raise ValueError("concatenate(): empty input")
    target = exps[0].scheme
    for e in exps[1:]:
        target = target.union(e.scheme)
    rows = [eager._embed(e, target) for e in exps]
    return Array(np.stack(rows), target)


stack = concatenate


def dot(a, b):
    from . import eager
    ra, rb = a._rows(), b._rows()
    if len(ra) != len(rb):
        raise ValueError(f"dot: length mismatch {len(ra)} vs {len(rb)}")
    acc = eager.binary("mul", ra[0], rb[0])
    for i in range(1, len(ra)):
        acc = eager.binary("add", acc, eager.binary("mul", ra[i], rb[i]))
    return acc                              # Expansion


def norm(a):
    from . import eager
    return eager.unary("sqrt", dot(a, a))   # Expansion


def cross(a, b):
    from .trace import Tracer, ArrayTracer
    if isinstance(a, ArrayTracer) or isinstance(b, ArrayTracer):
        from . import eager
        ra, rb = a._rows(), b._rows()
        if len(ra) != len(rb) or len(ra) not in (2, 3):
            raise ValueError("cross requires two 2- or 3-vectors of equal length")
        mul = lambda x, y: eager.binary("mul", x, y)
        sub = lambda x, y: eager.binary("sub", x, y)
        if len(ra) == 2:
            return sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))     # scalar Tracer
        c0 = sub(mul(ra[1], rb[2]), mul(ra[2], rb[1]))
        c1 = sub(mul(ra[2], rb[0]), mul(ra[0], rb[2]))
        c2 = sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))
        return ArrayTracer([c0, c1, c2], c0.scheme)
    from . import eager
    ra, rb = a._rows(), b._rows()
    if len(ra) != len(rb) or len(ra) not in (2, 3):
        raise ValueError("cross requires two 2- or 3-vectors of equal length")
    mul = lambda x, y: eager.binary("mul", x, y)
    sub = lambda x, y: eager.binary("sub", x, y)
    if len(ra) == 2:
        return sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))     # scalar Expansion
    c0 = sub(mul(ra[1], rb[2]), mul(ra[2], rb[1]))
    c1 = sub(mul(ra[2], rb[0]), mul(ra[0], rb[2]))
    c2 = sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))
    return Array(np.stack([c0.coeffs, c1.coeffs, c2.coeffs]), c0.scheme)
