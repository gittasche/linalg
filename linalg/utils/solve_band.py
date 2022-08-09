import numpy as np
from numpy.typing import ArrayLike

from ._validations import _ensure_ndarray

def solve_lower_band(
    a: ArrayLike,
    l: int,
    b: ArrayLike,
    overwrite_b=False,
    transposed: bool = False,
    unit: bool = False
) -> np.ndarray:
    """
    Solve AX = B, where A is a lower triangular banded matrix
    with bandwidth `l`.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        square input matrix A
    l : int
        lower bandwidth of A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    transposed : bool (default: False)
        - ``True`` solve A^T x X = B
        - ``False`` solve A x X = B
    
    unit : bool (default: False)
        - ``True`` assume diagonal entries of A to 1.0
        - ``False`` no any assumptions

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    
    Returns
    -------
    b : ndarray of shape (n, m)
        overwriten array `b` with m solution vectors
    """
    a = _ensure_ndarray(
        a,
        ensure_square=True,
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

    n = a.shape[0]
    if not transposed:
        if not unit:
            for i in range(n):
                l_b = np.maximum(i - l, 0)
                b[i] = (b[i] - np.dot(a[i, l_b:i], b[l_b:i])) / a[i, i]
        else:
            for i in range(n):
                l_b = np.maximum(i - l, 0)
                b[i] = b[i] - np.dot(a[i, l_b:i], b[l_b:i])
    else:
        if not unit:
            for i in range(n-1, -1, -1):
                u_b = np.minimum(i + l + 1, n)
                b[i] = (b[i] - np.dot(a[i+1:u_b, i], b[i+1:u_b])) / a[i, i]
        else:
            for i in range(n-1, -1, -1):
                u_b = np.minimum(i + l + 1, n)
                b[i] = b[i] - np.dot(a[i+1:u_b, i], b[i+1:u_b])

    if is_b1d:
        b = b.ravel()

    return b

def solve_upper_band(
    a: ArrayLike,
    u: int,
    b: ArrayLike,
    overwrite_b=False,
    transposed: bool = False,
    unit: bool = False
) -> np.ndarray:
    """
    Solve AX = B, where A is an upper triangular banded matrix
    with bandwidth `u`.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        square input matrix A
    u : int
        upper bandwidth of A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    transposed : bool (default: False)
        - ``True`` solve A^T x X = B
        - ``False`` solve A x X = B
    
    unit : bool (default: False)
        - ``True`` assume diagonal entries of A to 1.0
        - ``False`` no any assumptions

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    
    Returns
    -------
    b : ndarray of shape (n, m)
        overwriten array `b` with m solution vectors
    """
    a = _ensure_ndarray(
        a,
        ensure_square=True,
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

    n = a.shape[0]
    if not transposed:
        if not unit:
            for i in range(n-1, -1, -1):
                u_b = np.minimum(i + u + 1, n)
                b[i] = (b[i] - np.dot(a[i, i+1:u_b], b[i+1:u_b])) / a[i, i]
        else:
            for i in range(n-1, -1, -1):
                u_b = np.minimum(i + u + 1, n)
                b[i] = b[i] - np.dot(a[i, i+1:u_b], b[i+1:u_b])
    else:
        if not unit:
            for i in range(n):
                l_b = np.maximum(i - u, 0)
                b[i] = (b[i] - np.dot(a[l_b:i, i], b[l_b:i])) / a[i, i]
        else:
            for i in range(n):
                l_b = np.maximum(i - u, 0)
                b[i] = b[i] - np.dot(a[l_b:i, i], b[l_b:i])

    if is_b1d:
        b = b.ravel()

    return b
