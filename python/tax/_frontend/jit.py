from __future__ import annotations
import numpy as np
from .types import Expansion
from .array import Array
from .trace import trace_function
from . import eager

def _classify_args(args):
    specs = []
    for a in args:
        if isinstance(a, Array):
            specs.append(("arr", a.scheme, len(a)))
        elif isinstance(a, Expansion):
            specs.append(("exp", a.scheme))
        elif isinstance(a, (int, float)):
            specs.append(("scalar", float(a)))
        else:
            raise TypeError(f"tax.jit: unsupported argument type {type(a).__name__}")
    return specs

def _sig_key(specs):
    # Hashable signature: schemes are frozen dataclasses; scalars by value.
    return tuple(specs)

def jit(fn):
    cache: dict = {}

    def wrapper(*args):
        specs = _classify_args(args)
        key = _sig_key(specs)
        tr = cache.get(key)
        if tr is None:
            tr = trace_function(fn, specs)
            cache[key] = tr
        g = tr.global_scheme
        in_buffers = []
        for a, spec in zip(args, specs):
            if spec[0] == "scalar":
                continue                            # baked
            if spec[0] == "exp":
                in_buffers.append(eager._embed(a, g))
            else:                                   # "arr"
                in_buffers.extend(eager._embed(row, g) for row in a._rows())
        out_sizes = [g.n_coeff] * tr.out_nrows
        outs = eager.run(tr.graph, in_buffers, out_sizes)
        if tr.out_kind == "exp":
            return Expansion(outs[0], g)
        return Array(np.stack(outs), g)

    wrapper.__name__ = getattr(fn, "__name__", "jitted")
    wrapper._tax_jit = True
    return wrapper
