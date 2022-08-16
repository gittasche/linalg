import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal

from ..utils._validations import _ensure_ndarray
from ..utils.solve_band import solve_lower_band

def cholesky_band(
    a: ArrayLike,
    d: int,
    mode: Literal["full", "economic"] = "full",
    overwrite_a: bool = False
) -> NDArray:
    """
    Cholesky decomposition of symmetric positive definite (SPD) banded matrix A.
    A = LL^T, where L is a low triangle matrix.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix assumed to be SPD
    d : int
        bandwidth of matrix A
    mode : ["full", "economic"] (default: "full")
        return mode (see `Returns` section for details):
        - ``"full"`` is convinient form for further use
        - ``"economic"`` just does not take numpy.tril() (not too much economy)

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    Returns
    -------
    if `mode == "full"` than return l:
        - `l` - ndarray of shape (n, n) lower triangle matrix
    
    if `mode == "economic"` than return a:
        - `a` - overwritten `a` with L[i, j] in i >= j
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
    for i in range(n):
        # use gaxpy cholesky
        l_b = np.maximum(i - d, 0)
        u_b = np.minimum(i + d + 1, n)

        if i > 0:
            a[i:u_b, i] -= np.dot(a[i:u_b, l_b:i], a[i, l_b:i])
        
        if a[i, i] <= 0.0:
            raise RuntimeError(
                "Input matrix `a` is not symmetric positive definite."
                " Use solver `lu_band_solve` if `a` is banded."
            )
        a[i:u_b, i] /= np.sqrt(a[i, i])

    if mode == "full":
        l = np.tril(a)
        return l
    elif mode == "economic":
        return a

def sympos_band_solve(
    a: ArrayLike,
    d: int,
    b: ArrayLike,
    overwrite_a: bool = False,
    overwrite_b: bool = False
) -> NDArray:
    """
    Solve AX = B where A is a symmetric positive definite (SPD) banded matrix
    using cholesky LL^T decomposition.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be SPD.
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

    a = cholesky_band(a, d, mode="economic", overwrite_a=True)
    
    b = solve_lower_band(a, d, b, overwrite_b=True)
    b = solve_lower_band(a, d, b, overwrite_b=True, transposed=True)

    if is_b1d:
        b = b.ravel()

    return b