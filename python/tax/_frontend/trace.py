from __future__ import annotations
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
