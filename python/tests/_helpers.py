import glob
import os
import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Set at IMPORT time (before any skipif marker below is evaluated) so the
# codegen finds the repo headers and writes the JIT cache to a gitignored
# scratch dir. A session fixture would run too late — markers evaluate at
# collection/import.
os.environ.setdefault("TAX_INCLUDE", str(REPO_ROOT / "include"))
os.environ.setdefault("TAX_CACHE_DIR", str(REPO_ROOT / "python" / ".tax_cache"))


def _prefer_conda_toolchain() -> None:
    """Use the active conda env's C++ compiler + Eigen for reproducible builds.

    When run inside a conda/mamba env (the project's canonical `tax` env), point the
    JIT at the env's clang++ (the osx wrapper sets up the SDK) and the env's Eigen.
    Portable: keyed off CONDA_PREFIX, never a hard-coded path; respects an explicit
    TAX_CXX / TAX_EIGEN_INCLUDE if the caller already set one.
    """
    conda = os.environ.get("CONDA_PREFIX")
    if not conda:
        return
    if "TAX_CXX" not in os.environ:
        # The osx-arm64 wrapper (e.g. arm64-apple-darwin*-clang++) injects -isysroot;
        # prefer it over the bare clang++.
        wrappers = sorted(glob.glob(os.path.join(conda, "bin", "*-clang++")))
        plain = os.path.join(conda, "bin", "clang++")
        if wrappers:
            os.environ["TAX_CXX"] = wrappers[0]
        elif os.path.exists(plain):
            os.environ["TAX_CXX"] = plain
    if "TAX_EIGEN_INCLUDE" not in os.environ:
        eig = os.path.join(conda, "include", "eigen3")
        if os.path.isdir(eig):
            os.environ["TAX_EIGEN_INCLUDE"] = eig


_prefer_conda_toolchain()


def _have_toolchain() -> bool:
    from tax._codegen import build

    try:
        build.find_compiler()
        build.find_eigen_include()
        build.find_tax_include()
        return True
    except Exception:
        return False


needs_toolchain = pytest.mark.skipif(
    not _have_toolchain(), reason="C++ compiler / Eigen / tax headers not available"
)
