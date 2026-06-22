from __future__ import annotations

import functools
import os
import pathlib
import shutil
import subprocess

from .._errors import CompilerNotFound, EigenNotFound, TaxIncludeNotFound


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
