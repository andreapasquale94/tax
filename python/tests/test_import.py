def test_package_imports_and_exposes_errors():
    import tax
    from tax import TaxError, CompilerNotFound, JitCompileError, DomainError
    assert issubclass(CompilerNotFound, TaxError)
    assert issubclass(JitCompileError, TaxError)
    assert issubclass(DomainError, TaxError)
