class TaxError(Exception):
    """Base class for all tax Python-layer errors."""

class CompilerNotFound(TaxError):
    """No C++ compiler could be discovered."""

class EigenNotFound(TaxError):
    """Eigen headers could not be located."""

class TaxIncludeNotFound(TaxError):
    """The tax header include directory could not be located."""

class JitCompileError(TaxError):
    """The generated translation unit failed to compile."""
    def __init__(self, cmd, stderr, source):
        self.cmd = cmd
        self.stderr = stderr
        self.source = source
        super().__init__(f"JIT compile failed:\n{' '.join(cmd)}\n\n{stderr}")

class DomainError(TaxError):
    """A kernel trapped a domain error at runtime (nonzero return code)."""
