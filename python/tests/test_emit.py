from tax._frontend.ir import single_op_graph
from tax._frontend.scheme import Isotropic
from tax._codegen.emit_cpp import emit, CPP_EXPR

def test_emit_contains_scheme_and_signature():
    s = Isotropic(5, 1)
    src = emit(single_op_graph("sin", [s], s))
    assert "#include <tax/tax.hpp>" in src
    assert "tax::IsotropicScheme<5, 1>" in src
    assert 'extern "C" int tax_kernel' in src
    assert "sin(n0)" in src
    assert "std::copy_n(n1.coefficients().data()" in src

def test_emit_binary_mul():
    s = Isotropic(5, 1)
    src = emit(single_op_graph("mul", [s, s], s))
    assert "(n0 * n1)" in src

def test_cpp_expr_table_complete():
    for opc in ["add","sub","mul","div","neg","sin","cos","tan","asin","acos",
                "atan","sinh","cosh","tanh","asinh","acosh","atanh","exp","log",
                "sqrt","cbrt","square","cube","erf","reciprocal","pow","atan2"]:
        assert opc in CPP_EXPR
