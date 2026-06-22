from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile

from .._errors import CompilerNotFound, EigenNotFound, TaxIncludeNotFound, JitCompileError


@functools.lru_cache(maxsize=None)
def find_compiler() -> str:
    for env in ("TAX_CXX", "CXX"):
        c = os.environ.get(env)
        if c:
            return c
    for name in ("c++", "clang++", "g++"):
        p = shutil.which(name)
        if p:
            return p
    raise CompilerNotFound("set TAX_CXX or put c++/clang++/g++ on PATH")


@functools.lru_cache(maxsize=None)
def compiler_id(cxx: str) -> str:
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True)
        first = out.stdout.splitlines()[0] if out.stdout else ""
    except OSError:
        first = ""
    return f"{cxx}|{first}"


@functools.lru_cache(maxsize=None)
def find_eigen_include() -> str:
    e = os.environ.get("TAX_EIGEN_INCLUDE")
    if e:
        return e
    pkg = shutil.which("pkg-config")
    if pkg:
        out = subprocess.run([pkg, "--cflags", "eigen3"], capture_output=True, text=True)
        if out.returncode == 0:
            for tok in out.stdout.split():
                if tok.startswith("-I"):
                    return tok[2:]
    for p in ("/usr/include/eigen3", "/usr/local/include/eigen3", "/opt/homebrew/include/eigen3"):
        if os.path.isdir(p):
            return p
    raise EigenNotFound("set TAX_EIGEN_INCLUDE or install eigen3")


@functools.lru_cache(maxsize=None)
def find_tax_include() -> str:
    t = os.environ.get("TAX_INCLUDE")
    if t:
        return t
    vendored = pathlib.Path(__file__).resolve().parents[1] / "_vendor" / "include"
    if vendored.is_dir():
        return str(vendored)
    raise TaxIncludeNotFound("set TAX_INCLUDE to the tax header directory")


def include_dirs() -> list[str]:
    return [find_tax_include(), find_eigen_include()]


ABI_VERSION = "0"
TAX_LIB_VERSION = "0.1.0"
STD_FLAG = "-std=c++23"


def flags_for_key(opt_flags: list[str]) -> str:
    """The canonical flag string for the cache key — mirrors what compile_kernel passes."""
    return " ".join([STD_FLAG, *opt_flags])


def cache_dir() -> pathlib.Path:
    d = os.environ.get("TAX_CACHE_DIR")
    base = pathlib.Path(d) if d else pathlib.Path.home() / ".cache" / "tax"
    return base


def cache_key(canonical: str, *, cid: str, flags: str, scalar: str = "float64") -> str:
    blob = "\x1f".join([ABI_VERSION, TAX_LIB_VERSION, cid, flags, scalar, canonical])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _pch_enabled() -> bool:
    return os.environ.get("TAX_USE_PCH", "1") != "0"


def pch_path(cxx: str, includes: list[str], opt_flags: list[str]):
    """Path to a cached <tax/tax.hpp> PCH built with `opt_flags`, or None.

    Built once per (compiler, flags) and reused; the PCH must use the SAME
    opt_flags as the kernel compile (clang bakes __OPTIMIZE__ into it). Any
    failure returns None so kernel compilation proceeds without a PCH.
    """
    if not _pch_enabled():
        return None
    key = cache_key("__pch__:tax/tax.hpp", cid=compiler_id(cxx),
                    flags=flags_for_key(opt_flags))
    out = cache_dir() / f"{key}.pch"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Write the wrapper header into the cache dir (permanent) so that
        # clang's embedded path reference remains valid after the tempdir is gone.
        hdr = out.with_suffix(".hpp")
        hdr.write_text("#include <tax/tax.hpp>\n")
        with tempfile.TemporaryDirectory(dir=cache_dir()) as td:
            tmp = pathlib.Path(td) / "tax.pch"
            cmd = [cxx, STD_FLAG, *opt_flags, "-x", "c++-header",
                   *[f"-I{i}" for i in includes], str(hdr), "-o", str(tmp)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return None
            os.replace(tmp, out)        # atomic publish
        return out
    except OSError:
        return None


def compile_kernel(source: str, key: str, *, cxx: str, includes: list[str],
                   opt_flags: list[str]) -> pathlib.Path:
    out_dir = cache_dir()
    so_path = out_dir / f"{key}.so"
    if so_path.exists():
        return so_path
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as td:
        cpp = pathlib.Path(td) / "kernel.cpp"
        cpp.write_text(source)
        tmp_so = pathlib.Path(td) / "kernel.so"
        pch = pch_path(cxx, includes, opt_flags)
        pch_flags = ["-include-pch", str(pch)] if pch is not None else []
        cmd = [cxx, STD_FLAG, *opt_flags, *pch_flags, "-shared", "-fPIC",
               *[f"-I{i}" for i in includes], str(cpp), "-o", str(tmp_so)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise JitCompileError(cmd, proc.stderr, source)
        os.replace(tmp_so, so_path)   # atomic publish into the cache
    return so_path
