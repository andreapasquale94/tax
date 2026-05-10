# SPDX-License-Identifier: BSD-3-Clause
"""Truncated multivariate Taylor expansions.

Python users get a single storage type, ``DynTE``, with runtime-fixed
order and number of variables.  The static-extent C++ path
(``TruncatedTaylorExpansionT<T, N, M>``) is intentionally not exposed —
no ``std::variant`` over a (Order, Vars) grid, no JIT, and no per-shape
specialisation explosion.

Construction goes through module-level utility functions
(:func:`zero`, :func:`one`, :func:`constant`, :func:`variable`,
:func:`variables`); the ``DynTE`` class itself is not directly
constructible from Python.

Free math functions (``sin``, ``cos``, ``exp``, ``log``, ``sqrt``,
``tan``, ``sinh``, ``cosh``, ``tanh``, ``square``, ``cube``) and
arithmetic operators (``+``, ``-``, ``*``, ``/``) all return fresh
``DynTE`` instances — Python cannot meaningfully own lazy ET
temporaries across statements, so every call evaluates eagerly.
"""

from ._tax import (  # noqa: F401
    # type
    DynTE,
    # factories
    zero,
    one,
    constant,
    variable,
    variables,
    # trig + hyperbolic
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    sincos,
    sinhcosh,
    # inverse trig + hyperbolic
    asin,
    acos,
    atan,
    asinh,
    acosh,
    atanh,
    atan2,
    # exp / log
    exp,
    log,
    log10,
    # roots, powers
    sqrt,
    cbrt,
    square,
    cube,
    pow,
    hypot,
    # erf
    erf,
)

__all__ = [
    "DynTE",
    "zero",
    "one",
    "constant",
    "variable",
    "variables",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "sincos",
    "sinhcosh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "atan2",
    "exp",
    "log",
    "log10",
    "sqrt",
    "cbrt",
    "square",
    "cube",
    "pow",
    "hypot",
    "erf",
]
