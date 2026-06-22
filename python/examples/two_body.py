"""Planar restricted two-body RHS as a tax.jit-compiled map — the north-star example.

The state is [rx, ry, vx, vy]; the RHS is [vx, vy, -mu rx / r^3, -mu ry / r^3]
with r = sqrt(rx^2 + ry^2). Two flavors:
  * unnamed_rhs(): integer-indexed coordinates, gravitational parameter mu as a
    plain float baked into the kernel.
  * named_rhs(): named axes "x" (the 4-D state) and "mu" (the parameter), so the
    result carries a named state-transition block jacobian("x") and a parameter
    sensitivity jacobian("mu").

Run:  python examples/two_body.py
"""
import os, sys, pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]
PYTHON = REPO / "python"
sys.path.insert(0, str(PYTHON))
os.environ.setdefault("TAX_INCLUDE", str(REPO / "include"))
os.environ.setdefault("TAX_CACHE_DIR", str(PYTHON / ".tax_cache"))

import tax

MU = 398600.4418
STATE = [1.0, 0.0, 0.0, 1.0]          # rx, ry, vx, vy (unit-radius circular-ish)

def _rhs(x, mu):
    r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

def unnamed_rhs():
    """Order-4 expansion in the 4 state coordinates; mu is a baked constant."""
    x = tax.variables(STATE, order=4)

    @tax.jit
    def rhs(t, x, mu):
        return _rhs(x, mu)

    return rhs(0.0, x, MU)

def named_rhs():
    """Named axes 'x' (state) and 'mu' (parameter), pinned with a jit signature."""
    x = tax.variables(STATE, order=4, name="x")
    mu = tax.variable(MU, order=4, name="mu")

    @tax.jit([tax.f64, tax.ArrayType(order=4, size=4, name="x"),
              tax.ExpansionType(order=4, name="mu")])
    def rhs(t, x, mu):
        return _rhs(x, mu)

    return rhs(0.0, x, mu)

def main():
    dx = unnamed_rhs()
    print("unnamed RHS value:", dx.value())
    print("state-transition jacobian:\n", dx.jacobian())
    dn = named_rhs()
    print("named RHS value:", dn.value())
    print("d(rhs)/d(x) block:\n", dn.jacobian("x"))
    print("d(rhs)/d(mu) sensitivity:\n", dn.jacobian("mu"))

if __name__ == "__main__":
    main()
