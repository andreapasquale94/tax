# SPDX-License-Identifier: BSD-3-Clause
"""Truncated multivariate Taylor expansions.

Python users get a single storage type, ``DynTE``, with runtime-fixed
order and number of variables.  The static-extent C++ path
(``TruncatedTaylorExpansionT<T, N, M>``) is intentionally not exposed —
no ``std::variant`` over a (Order, Vars) grid, no JIT, and no per-shape
specialisation explosion.

Free math functions (``sin``, ``cos``, ``exp``, ``log``, ``sqrt``,
``tan``, ``sinh``, ``cosh``, ``tanh``, ``square``, ``cube``) are
re-exported here.
"""

from ._tax import (  # noqa: F401
    DynTE,
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    exp,
    log,
    sqrt,
    square,
    cube,
)

__all__ = [
    "DynTE",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "square",
    "cube",
]
