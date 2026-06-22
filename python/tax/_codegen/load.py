from __future__ import annotations

import ctypes

import numpy as np

from .._errors import DomainError

_DBL_PP = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))


def load_kernel(so_path):
    lib = ctypes.CDLL(str(so_path))
    fn = lib.tax_kernel
    fn.argtypes = [_DBL_PP, _DBL_PP]
    fn.restype = ctypes.c_int
    return fn


def _as_pointer_array(buffers):
    arr = (ctypes.POINTER(ctypes.c_double) * len(buffers))()
    for i, b in enumerate(buffers):
        arr[i] = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return arr


def call_kernel(fn, in_buffers, out_sizes):
    ins = [np.ascontiguousarray(b, dtype=np.float64) for b in in_buffers]
    outs = [np.zeros(n, dtype=np.float64) for n in out_sizes]
    rc = fn(_as_pointer_array(ins), _as_pointer_array(outs))
    if rc != 0:
        raise DomainError(f"kernel returned {rc}")
    return outs
