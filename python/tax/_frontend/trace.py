from __future__ import annotations
from dataclasses import dataclass
from .ir import Var, Const, Op, Graph

class TraceBuilder:
    def __init__(self):
        self.nodes: list = []

    def add(self, node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

class Tracer:
    """A symbolic stand-in for a scalar Expansion during tracing."""
    __slots__ = ("builder", "node", "scheme")

    def __init__(self, builder, node, scheme):
        self.builder = builder
        self.node = node
        self.scheme = scheme

    def __add__(self, o):
        from .eager import binary; return binary("add", self, o)
    def __radd__(self, o):
        from .eager import binary; return binary("add", o, self)
    def __sub__(self, o):
        from .eager import binary; return binary("sub", self, o)
    def __rsub__(self, o):
        from .eager import binary; return binary("sub", o, self)
    def __mul__(self, o):
        from .eager import binary; return binary("mul", self, o)
    def __rmul__(self, o):
        from .eager import binary; return binary("mul", o, self)
    def __truediv__(self, o):
        from .eager import binary; return binary("div", self, o)
    def __rtruediv__(self, o):
        from .eager import binary; return binary("div", o, self)
    def __neg__(self):
        from .eager import unary; return unary("neg", self)
    def __pow__(self, p):
        from .eager import binary; return binary("pow", self, p)


@dataclass(frozen=True)
class TraceResult:
    graph: object
    global_scheme: object
    out_kind: str          # "exp" or "arr"
    out_nrows: int


def _global_scheme(arg_specs):
    schemes = [spec[1] for spec in arg_specs if spec[0] in ("exp", "arr")]
    if not schemes:
        raise ValueError("tax.jit: at least one Expansion/Array argument is required")
    g = schemes[0]
    for s in schemes[1:]:
        g = g.union(s)
    return g


def trace_function(fn, arg_specs) -> TraceResult:
    g = _global_scheme(arg_specs)
    builder = TraceBuilder()
    slot = 0
    call_args = []
    for spec in arg_specs:
        kind = spec[0]
        if kind == "exp":
            call_args.append(Tracer(builder, builder.add(Var(slot, g)), g)); slot += 1
        elif kind == "arr":
            nrows = spec[2]
            rows = [Tracer(builder, builder.add(Var(slot + r, g)), g) for r in range(nrows)]
            slot += nrows
            call_args.append(ArrayTracer(rows, g))
        elif kind == "scalar":
            call_args.append(spec[1])            # raw scalar, baked when used
        else:
            raise ValueError(f"unknown arg spec kind {kind!r}")
    result = fn(*call_args)
    if isinstance(result, ArrayTracer):
        outputs = [t.node for t in result._rows()]
        out_kind, out_nrows = "arr", len(result)
    elif isinstance(result, Tracer):
        outputs = [result.node]
        out_kind, out_nrows = "exp", 1
    else:
        raise TypeError("tax.jit: function must return an Expansion or Array (got "
                        f"{type(result).__name__})")
    graph = Graph(builder.nodes, outputs, slot)
    return TraceResult(graph, g, out_kind, out_nrows)


def _trace_operand(builder, v, scheme) -> int:
    """A Tracer contributes its node; a Python scalar becomes a Const node."""
    if isinstance(v, Tracer):
        return v.node
    return builder.add(Const(float(v), scheme))

def trace_binary(opcode, a, b) -> Tracer:
    builder = a.builder if isinstance(a, Tracer) else b.builder
    scheme = a.scheme if isinstance(a, Tracer) else b.scheme
    na = _trace_operand(builder, a, scheme)
    nb = _trace_operand(builder, b, scheme)
    return Tracer(builder, builder.add(Op(opcode, (na, nb), scheme)), scheme)

def trace_unary(opcode, x) -> Tracer:
    return Tracer(x.builder, x.builder.add(Op(opcode, (x.node,), x.scheme)), x.scheme)


class ArrayTracer:
    """A symbolic stand-in for an Array during tracing: a list of scalar Tracers."""
    __slots__ = ("rows", "scheme")

    def __init__(self, rows, scheme):
        self.rows = list(rows)
        self.scheme = scheme

    def __len__(self):
        return len(self.rows)

    def _rows(self):
        return self.rows

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ArrayTracer(self.rows[i], self.scheme)
        return self.rows[i]

    def _map_unary(self, op):
        return ArrayTracer([trace_unary(op, r) for r in self.rows], self.scheme)

    def _map_binary(self, op, other):
        if isinstance(other, ArrayTracer):
            if len(other) != len(self):
                raise ValueError(f"ArrayTracer length mismatch: {len(self)} vs {len(other)}")
            return ArrayTracer([trace_binary(op, a, b) for a, b in zip(self.rows, other.rows)],
                               self.scheme)
        return ArrayTracer([trace_binary(op, a, other) for a in self.rows], self.scheme)

    def __add__(self, o): return self._map_binary("add", o)
    def __radd__(self, o): return self._map_binary("add", o)
    def __sub__(self, o): return self._map_binary("sub", o)
    def __rsub__(self, o): return ArrayTracer([trace_binary("sub", o, a) for a in self.rows], self.scheme)
    def __mul__(self, o): return self._map_binary("mul", o)
    def __rmul__(self, o): return self._map_binary("mul", o)
    def __truediv__(self, o): return self._map_binary("div", o)
    def __rtruediv__(self, o): return ArrayTracer([trace_binary("div", o, a) for a in self.rows], self.scheme)
    def __neg__(self): return self._map_unary("neg")
    def __pow__(self, p): return self._map_binary("pow", p)
