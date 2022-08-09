import numpy as np
from numpy.typing import ArrayLike

from .utils._validations import _ensure_ndarray
from .lu.lu_band import lu_band_solve
from .sym_decomp.ldlt_band import sym_band_solve
from .sympos_decomp.cholesky_band import sympos_band_solve

def solve_band(
    a: ArrayLike,
    l: int,
    u: int,
    b: ArrayLike,
    overwrite_a: bool = False,
    overwrite_b: bool = False
):
    """
    Solve AX = B, where A is a banded matrix.
    Use LU decomposition approach

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        square input matrix A
    l : int
        lower bandwidth of A
    u : int
        upper bandwidth of A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    
    Returns
    -------
    b : ndarray of shape (n, m)
        overwriten array `b` with m solution vectors
    """
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy_a,
        dtype="float64"
    )
    copy_b = not overwrite_b
    b = _ensure_ndarray(
        b,
        copy=copy_b,
        dtype="float64"
    )
    
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "`a` and `b` must have equal number of columns,"
            f" got {a.shape[0]} and {b.shape[0]}."
        )

    b = lu_band_solve(a, l, u, b, overwrite_a=True, overwrite_b=True)
    return b

def solves_band(
    a: ArrayLike,
    d: int,
    b: ArrayLike,
    ensure_pos=False,
    overwrite_a: bool = False,
    overwrite_b: bool = False
):
    """
    Solve AX = B, where A is a symmetric banded matrix.
    Use band Cholesky if A is a SPD else use LDL^T.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        square input matrix A
    d : int
        bandwidth of matrix A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    ensure_pos : bool (default: False)
        - ``True`` use band Cholesky
        - ``False`` use band LDL^T

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    
    Returns
    -------
    b : ndarray of shape (n, m)
        overwriten array `b` with m solution vectors
    """
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy_a,
        dtype="float64"
    )
    copy_b = not overwrite_b
    b = _ensure_ndarray(
        b,
        copy=copy_b,
        dtype="float64"
    )

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "`a` and `b` must have equal number of columns,"
            f" got {a.shape[0]} and {b.shape[0]}."
        )

    if ensure_pos:
        b = sympos_band_solve(a, d, b, overwrite_a=True, overwrite_b=True)
    else:
        b = sym_band_solve(a, d, b, overwrite_a=True, overwrite_b=True)
    
    return b