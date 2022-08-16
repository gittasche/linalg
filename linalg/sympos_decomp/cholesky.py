import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal

from ..utils._validations import _ensure_ndarray
from ..utils.solve import solve_lower

def cholesky(
    a: ArrayLike,
    mode: Literal["full", "economic"] = "full",
    overwrite_a: bool = False
) -> NDArray:
    """
    Cholesky decomposition of symmetric positive definite (SPD) matrix A.
    A = LL^T, where L is a low triangle matrix.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix assumed to be SPD
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
        if i > 0:
            a[i:, i] -= np.dot(a[i:, :i], a[i, :i])
            
        # check SPD
        if a[i, i] <= 0:
            raise RuntimeError(
                "Input matrix `a` is not symmetric positive definite."
                " Use symmetric solver `sym_solve` if `a` symmetric,"
                " general solver `lu_solve` if its not."
            )
        a[i:, i] /= np.sqrt(a[i, i])

    if mode == "full":
        l = np.tril(a)
        return l
    elif mode == "economic":
        return a

def sympos_solve(
    a: ArrayLike,
    b: ArrayLike,
    overwrite_a: bool = False,
    overwrite_b: bool = False
) -> NDArray:
    """
    Solve AX = B where A is a symmetric positive definite (SPD) matrix
    using cholesky LL^T decomposition.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be SPD.
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

    a = cholesky(a, mode="economic", overwrite_a=True)

    b = solve_lower(a, b, overwrite_b=True)
    b = solve_lower(a, b, overwrite_b=True, transposed=True)

    if is_b1d:
        b = b.ravel()

    return b