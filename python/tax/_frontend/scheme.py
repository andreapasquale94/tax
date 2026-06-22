from __future__ import annotations
from dataclasses import dataclass
from math import comb
import math

def num_monomials(order: int, vars: int) -> int:
    return comb(order + vars, vars)

def _binom(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)

def flat_index(alpha) -> int:
    """Graded-lex flat index of multi-index `alpha` (mirrors C++ tax::flatIndex)."""
    alpha = tuple(alpha)
    M = len(alpha)
    d = sum(alpha)
    idx = _binom(d + M - 1, M)
    rem = d
    for i in range(M - 1):
        idx += _binom(rem - alpha[i] + (M - 2 - i), M - 1 - i)
        rem -= alpha[i]
    return idx

def unflat_index(k: int, vars: int) -> tuple:
    """Inverse of flat_index for `vars` variables (mirrors C++ tax::unflatIndex)."""
    M = vars
    alpha = [0] * M
    d = 0
    while _binom(d + M, M) <= k:
        d += 1
    rank = k - _binom(d + M - 1, M)
    rem = d
    for i in range(M - 1):
        vars_left = M - i
        for ai in range(rem, -1, -1):
            block = _binom(rem - ai + vars_left - 2, vars_left - 2)
            if rank < block:
                alpha[i] = ai
                rem -= ai
                break
            rank -= block
    alpha[M - 1] = rem
    return tuple(alpha)

@dataclass(frozen=True)
class Isotropic:
    order: int
    vars: int

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("Isotropic.order must be >= 0")
        if self.vars < 1:
            raise ValueError("Isotropic.vars must be >= 1")

    @property
    def n_coeff(self) -> int:
        return num_monomials(self.order, self.vars)

    def cpp_type_string(self) -> str:
        return f"tax::IsotropicScheme<{self.order}, {self.vars}>"

    def descriptor_hash(self) -> str:
        return f"iso:{self.order}:{self.vars}"

    def union(self, other: "Isotropic") -> "Isotropic":
        if self.vars != other.vars:
            raise ValueError(
                f"isotropic union requires equal vars ({self.vars} != {other.vars})"
            )
        return Isotropic(max(self.order, other.order), self.vars)


@dataclass(frozen=True)
class Axis:
    name: str
    dim: int

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("Axis.dim must be >= 1")


@dataclass(frozen=True)
class Named:
    order: int
    axes: tuple  # tuple[Axis, ...], canonical: sorted by name, unique

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("Named.order must be >= 0")
        names = [a.name for a in self.axes]
        if names != sorted(names):
            raise ValueError("Named.axes must be sorted by name (use Named.of)")
        if len(set(names)) != len(names):
            raise ValueError("Named.axes has duplicate axis names")

    @classmethod
    def of(cls, order: int, axes) -> "Named":
        by_name: dict[str, Axis] = {}
        for a in axes:
            if a.name in by_name and by_name[a.name].dim != a.dim:
                raise ValueError(f"axis {a.name!r} used with conflicting dim")
            by_name[a.name] = a
        ordered = tuple(sorted(by_name.values(), key=lambda a: a.name))
        return cls(order, ordered)

    @property
    def vars(self) -> int:
        return sum(a.dim for a in self.axes)

    @property
    def n_coeff(self) -> int:
        return num_monomials(self.order, self.vars)

    def isotropic(self) -> Isotropic:
        return Isotropic(self.order, self.vars)

    def cpp_type_string(self) -> str:
        return self.isotropic().cpp_type_string()

    def descriptor_hash(self) -> str:
        # Identical emitted C++ to the isotropic twin -> share the cached .so.
        return self.isotropic().descriptor_hash()

    def axis_names(self) -> tuple:
        return tuple(a.name for a in self.axes)

    def dim_of(self, name: str) -> int:
        for a in self.axes:
            if a.name == name:
                return a.dim
        raise KeyError(name)

    def var_offset(self, name: str) -> int:
        off = 0
        for a in self.axes:
            if a.name == name:
                return off
            off += a.dim
        raise KeyError(name)

    def union(self, other: "Named") -> "Named":
        merged = Named.of(max(self.order, other.order), (*self.axes, *other.axes))
        return merged

    def axis_var_map(self, target: "Named") -> list:
        """Map each of this scheme's variable indices to the target's variable layout."""
        m = [0] * self.vars
        src = 0
        for a in self.axes:
            to = target.var_offset(a.name)
            for l in range(a.dim):
                m[src] = to + l
                src += 1
        return m
