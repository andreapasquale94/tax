from __future__ import annotations
from dataclasses import dataclass
from math import comb

def num_monomials(order: int, vars: int) -> int:
    return comb(order + vars, vars)

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
