from tax._frontend.trace import TraceBuilder, Tracer, ArrayTracer
from tax._frontend.ir import Var, Const, Op
from tax._frontend.scheme import Isotropic
import tax

def test_tracer_records_ops_not_eager():
    b = TraceBuilder()
    s = Isotropic(5, 1)
    x = Tracer(b, b.add(Var(0, s)), s)
    y = tax.sin(x) * tax.exp(x)                 # 2 unary ops + 1 mul
    assert isinstance(y, Tracer)
    kinds = [type(n).__name__ for n in b.nodes]
    assert kinds == ["Var", "Op", "Op", "Op"]
    assert b.nodes[1].opcode == "sin" and b.nodes[1].operands == (0,)
    assert b.nodes[2].opcode == "exp" and b.nodes[2].operands == (0,)
    assert b.nodes[3].opcode == "mul" and b.nodes[3].operands == (1, 2)
    assert y.node == 3

def test_tracer_scalar_becomes_const():
    b = TraceBuilder()
    s = Isotropic(3, 1)
    x = Tracer(b, b.add(Var(0, s)), s)
    y = 2.0 * x                                 # scalar -> Const node, then mul
    assert isinstance(b.nodes[1], Const) and b.nodes[1].value == 2.0
    assert b.nodes[2].opcode == "mul"

def test_array_tracer_concatenate_and_dot():
    b = TraceBuilder()
    s = Isotropic(2, 2)
    X = ArrayTracer([Tracer(b, b.add(Var(i, s)), s) for i in range(2)], s)
    assert isinstance(X[0], Tracer) and len(X) == 2
    Y = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    assert isinstance(Y, ArrayTracer) and len(Y) == 2
    d = tax.dot(X, X)                                  # x0*x0 + x1*x1 -> scalar Tracer
    assert isinstance(d, Tracer)

def test_array_tracer_elementwise_and_math():
    b = TraceBuilder()
    s = Isotropic(3, 2)
    X = ArrayTracer([Tracer(b, b.add(Var(i, s)), s) for i in range(2)], s)
    S = tax.sin(X)                                     # elementwise -> ArrayTracer
    assert isinstance(S, ArrayTracer) and len(S) == 2
    Z = 2.0 * X + X                                    # broadcast scalar + elementwise add
    assert isinstance(Z, ArrayTracer) and len(Z) == 2

from tax._frontend.trace import trace_function

def test_trace_function_global_scheme_and_outputs():
    s = Isotropic(4, 4)
    specs = [("scalar", 0.0), ("arr", s, 4), ("scalar", 398600.4418)]
    def rhs(t, x, mu):
        r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
        return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])
    tr = trace_function(rhs, specs)
    assert tr.global_scheme == s                  # only the Array contributes axes
    assert tr.out_kind == "arr" and tr.out_nrows == 4
    assert len(tr.graph.outputs) == 4
    assert tr.graph.n_inputs == 4                 # 4 Var slots (the Array rows); scalars baked

def test_trace_function_named_union():
    from tax._frontend.scheme import Named, Axis
    xs = Named.of(4, [Axis("x", 4)])
    mus = Named.of(4, [Axis("mu", 1)])
    specs = [("arr", xs, 4), ("exp", mus)]
    def f(x, mu):
        return mu * x[0]
    tr = trace_function(f, specs)
    assert tr.global_scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])   # union
    assert tr.out_kind == "exp"

def test_cross_in_trace_returns_arraytracer():
    from tax._frontend.trace import TraceBuilder, Tracer, ArrayTracer
    from tax._frontend.ir import Var
    from tax._frontend.scheme import Isotropic
    import tax
    b = TraceBuilder()
    s = Isotropic(1, 3)
    A = ArrayTracer([Tracer(b, b.add(Var(i, s)), s) for i in range(3)], s)
    B = ArrayTracer([Tracer(b, b.add(Var(3 + i, s)), s) for i in range(3)], s)
    C = tax.cross(A, B)
    assert isinstance(C, ArrayTracer) and len(C) == 3      # 3D cross -> 3-vector ArrayTracer
    z = tax.cross(A[:2], B[:2])
    assert isinstance(z, Tracer)                            # 2D cross -> scalar Tracer
