from ._errors import (
    TaxError, CompilerNotFound, EigenNotFound, TaxIncludeNotFound,
    JitCompileError, DomainError,
)
from ._frontend.factories import variable
from ._frontend.types import Expansion

__all__ = [
    "TaxError", "CompilerNotFound", "EigenNotFound", "TaxIncludeNotFound",
    "JitCompileError", "DomainError",
]

__all__ += ["variable", "Expansion"]

from ._frontend import mathfns as _mathfns
for _n in _mathfns.__all__:
    globals()[_n] = getattr(_mathfns, _n)
__all__ += list(_mathfns.__all__)
