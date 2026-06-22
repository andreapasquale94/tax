from __future__ import annotations
from .array import Array
from .eager import unary, binary

_UNARY = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
          "asinh", "acosh", "atanh", "exp", "log", "sqrt", "cbrt", "square",
          "cube", "erf", "reciprocal"]

def _make_unary(opcode):
    def fn(x):
        if isinstance(x, Array):
            return x._map_unary(opcode)
        return unary(opcode, x)
    fn.__name__ = opcode
    return fn

for _name in _UNARY:
    globals()[_name] = _make_unary(_name)

def pow(x, y):
    if isinstance(x, Array):
        return x._map_binary("pow", y)
    return binary("pow", x, y)

def atan2(y, x):
    if isinstance(y, Array):
        return y._map_binary("atan2", x)
    return binary("atan2", y, x)

__all__ = _UNARY + ["pow", "atan2"]
