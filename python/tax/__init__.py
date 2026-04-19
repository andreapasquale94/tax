"""
tax — runtime-dimensioned Truncated Algebraic eXpansions for Python.

The underlying C++ library is header-only and heavily templated on the
truncation order ``N`` and the number of variables ``M``. This package
exposes a runtime-dimensioned :class:`TE` class whose operations dispatch
to a precompiled grid of ``(N, M)`` instantiations at call time.

Example
-------
>>> import tax
>>> x, y = tax.variables(order=5, nvars=2, x0=[1.0, 2.0])
>>> f = tax.sin(x) * tax.exp(y)
>>> f.value()
6.211914291750007
>>> f.eval([0.1, -0.05])
5.924...

The largest compiled ``(N, M)`` combination is bounded at build time by
``TAX_PY_MAX_N``, ``TAX_PY_MAX_M``, and ``TAX_PY_MAX_COEFFS``. Requesting
an unsupported pair raises ``ValueError``.
"""

from ._tax import (  # noqa: F401
    TE,
    TAX_PY_MAX_COEFFS,
    TAX_PY_MAX_M,
    TAX_PY_MAX_N,
    __version__,
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    cbrt,
    constant,
    cos,
    cosh,
    cube,
    erf,
    exp,
    hypot,
    log,
    log10,
    pow,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    variable,
    variables,
)

__all__ = [
    "TE",
    "__version__",
    "TAX_PY_MAX_N",
    "TAX_PY_MAX_M",
    "TAX_PY_MAX_COEFFS",
    "constant",
    "variable",
    "variables",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "exp",
    "log",
    "log10",
    "sqrt",
    "cbrt",
    "square",
    "cube",
    "abs",
    "erf",
    "hypot",
    "pow",
]
