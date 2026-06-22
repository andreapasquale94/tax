from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .types import Expansion
from .array import Array
from .trace import trace_function
from . import eager
from .scheme import Isotropic, Named, Axis
from .._codegen.emit_cpp import emit

@dataclass(frozen=True)
class _ScalarSpec:
    pass
f64 = _ScalarSpec()

@dataclass(frozen=True)
class ExpansionType:
    order: int
    name: object = None
    def _spec(self):
        sch = Named.of(self.order, [Axis(self.name, 1)]) if self.name else Isotropic(self.order, 1)
        return ("exp", sch)

@dataclass(frozen=True)
class ArrayType:
    order: int
    size: int
    name: object = None
    def _spec(self):
        sch = (Named.of(self.order, [Axis(self.name, self.size)]) if self.name
               else Isotropic(self.order, self.size))
        return ("arr", sch, self.size)

def _spec_of_type(t):
    if isinstance(t, (ExpansionType, ArrayType)):
        return t._spec()
    if isinstance(t, _ScalarSpec):
        return ("scalar", 0.0)            # placeholder; M4 bakes scalars, value supplied at call
    raise TypeError(f"tax.jit signature: unsupported type spec {t!r}")

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

def jit(fn=None, signature=None, dump=False):
    # Support @tax.jit, @tax.jit([...]), @tax.jit(signature=[...], dump=True)
    if isinstance(fn, list):               # @tax.jit([...])
        signature, fn = fn, None
    if fn is None:
        return lambda f: _build(f, signature, dump)
    return _build(fn, signature, dump)

def _build(fn, signature, dump):
    cache: dict = {}
    state = {"last_source": None}

    def _trace_and_record(specs):
        tr = trace_function(fn, specs)
        if dump:
            state["last_source"] = emit(tr.graph)
        return tr

    if signature is not None:
        sig_specs = [_spec_of_type(t) for t in signature]
        cache[_sig_key(sig_specs)] = _trace_and_record(sig_specs)   # eager trace at decoration

    def wrapper(*args):
        specs = _classify_args(args)
        if signature is not None:
            want = [_spec_of_type(t) for t in signature]
            if len(specs) != len(want):
                raise TypeError(
                    f"tax.jit signature expects {len(want)} arguments, got {len(specs)}"
                )
            # Validate schemes match the pinned signature (scalars compare loosely).
            for got, exp in zip(specs, want):
                if got[0] != exp[0] or (got[0] != "scalar" and got[1] != exp[1]):
                    raise TypeError(f"tax.jit signature mismatch: {got} vs declared {exp}")
        key = _sig_key(specs)
        tr = cache.get(key)
        if tr is None:
            tr = _trace_and_record(specs)
            cache[key] = tr
        g = tr.global_scheme
        in_buffers = []
        for a, spec in zip(args, specs):
            if spec[0] == "scalar":
                continue
            if spec[0] == "exp":
                in_buffers.append(eager._embed(a, g))
            else:
                in_buffers.extend(eager._embed(row, g) for row in a._rows())
        outs = eager.run(tr.graph, in_buffers, [g.n_coeff] * tr.out_nrows)
        return Expansion(outs[0], g) if tr.out_kind == "exp" else Array(np.stack(outs), g)

    wrapper.__name__ = getattr(fn, "__name__", "jitted")
    wrapper._tax_jit = True
    wrapper.dump_source = lambda: state["last_source"]
    return wrapper
