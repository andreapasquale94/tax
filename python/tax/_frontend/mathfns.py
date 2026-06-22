from __future__ import annotations
from .eager import unary, binary

_UNARY = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
          "asinh", "acosh", "atanh", "exp", "log", "sqrt", "cbrt", "square",
          "cube", "erf", "reciprocal"]

def _make_unary(opcode):
    def fn(x):
        return unary(opcode, x)
    fn.__name__ = opcode
    return fn

for _name in _UNARY:
    globals()[_name] = _make_unary(_name)

def pow(x, y):
    return binary("pow", x, y)

def atan2(y, x):
    return binary("atan2", y, x)

__all__ = _UNARY + ["pow", "atan2"]
