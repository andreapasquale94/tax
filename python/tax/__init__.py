from ._errors import (
    TaxError, CompilerNotFound, EigenNotFound, TaxIncludeNotFound,
    JitCompileError, DomainError,
)
from ._frontend.factories import variable, variables
from ._frontend.types import Expansion
from ._frontend.array import Array, concatenate, stack, dot, cross, norm

__all__ = [
    "TaxError", "CompilerNotFound", "EigenNotFound", "TaxIncludeNotFound",
    "JitCompileError", "DomainError",
]

__all__ += ["variable", "variables", "Expansion", "Array", "concatenate", "stack"]
__all__ += ["dot", "cross", "norm"]

from ._frontend import mathfns as _mathfns
for _n in _mathfns.__all__:
    globals()[_n] = getattr(_mathfns, _n)
__all__ += list(_mathfns.__all__)
