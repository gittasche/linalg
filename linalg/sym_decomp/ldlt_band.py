import numpy as np
from numpy.typing import ArrayLike
from typing import Literal, Tuple

from ..utils._validations import _ensure_ndarray
from ..utils.solve_band import solve_lower_band
from ..utils.solve import solve_diag

def ldlt_band(
    a: ArrayLike,
    d: int,
    mode: Literal["full", "economic"] = "full",
    overwrite_a: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    LDL^T decomposition of symmetric banded matrix A.
    This implementation does not use symmetric pivoting due
    to difficulities with band structure. But this should work
    with SPD matrices as a Cholesky factorization, so it is a 
    "sqrt-free Cholesky".

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be symmetric.
    d : int
        bandwidth of matrix A
    mode : ["full", "economic"] (default: "full")
        return mode (see `Returns` section for details):
        - ``"full"`` is convinient form for further use
        - ``"economic"`` is a workspace economy mode

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    Returns
    -------
    if `mode == "full"` than return tuple (l, d):
        - `l` - ndarray of shape (n, n) unit lower triangle matrix
        - `d` - ndarray of shape (n,) diagonal elements of D
    
    if `mode == "economic"` than return a:
        - `a` - overwritten `a` with L[i, j] in i > j and
          D[i, i] on diagonal entries
    """
    copy = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy,
        dtype="float64"
    )

    if mode not in ["full", "economic"]:
        raise ValueError(
            "`mode` must be in ['full', 'economic'],"
            f" got {mode}."
        )
        
    n = a.shape[0]

    v = np.zeros(n)
    for i in range(n):
        l_b = np.maximum(i - d, 0)
        u_b = np.minimum(i + d + 1, n)

        v[l_b:i] = a[i, l_b:i] * np.diag(a[l_b:i, l_b:i])
        v[i] = a[i, i] - np.dot(a[i, l_b:i], v[l_b:i])
        a[i, i] = v[i]
        a[i + 1:u_b, i] = (a[i + 1:u_b, i] - np.dot(a[i + 1:u_b, l_b:i], v[l_b:i])) / v[i]
        
    if mode == "full":
        l = np.tril(a)
        np.fill_diagonal(l, 1.0)
        d = np.diag(a)
        return l, d
    elif mode == "economic":
        return a

def sym_band_solve(
    a: ArrayLike,
    d: int,
    b: ArrayLike,
    overwrite_a: bool = False,
    overwrite_b: bool = False
) -> np.ndarray:
    """
    Solve AX = B where A is a symmetric banded matrix
    using LDL^T decomposition.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be symmetric.
    d : int
        bandwidth of matrix A
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
    
    is_b1d = False
    if b.ndim == 1:
        b = b[:, np.newaxis]
        is_b1d = True

    a = ldlt_band(a, d, mode="economic", overwrite_a=True)

    b = solve_lower_band(a, d, b, overwrite_b=True, unit=True)
    b = solve_diag(a, b, overwrite_b=True)
    b = solve_lower_band(a, d, b, overwrite_b=True, transposed=True, unit=True)

    if is_b1d:
        b = b.ravel()

    return b