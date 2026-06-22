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
