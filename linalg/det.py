import numpy as np
from numpy.typing import ArrayLike

from .lu.lu import lu
from .utils._validations import _ensure_ndarray

def det(
    a: ArrayLike,
    overwrite_a: bool = False
) -> float:
    """
    Get determinant of square matrix A
    using LU decomposition

    Use identity det(A) = det(L)*det(U) = det(U)

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A
    overwrite_a : bool (default: False)
        allow to overwrite `a`
    
    Returns
    -------
    det : float
        determinant of `a`
    """
    copy = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy,
        dtype="float64"
    )

    a, row_ids, _ = lu(a, mode="economic", overwrite_a=True)

    n = a.shape[0]
    ids = np.arange(n)
    n_transpos = np.sum(row_ids != ids) - 1
    n_transpos = 0 if n_transpos < 0 else n_transpos
    det_sign = (-1)**n_transpos

    return det_sign * np.prod(np.diag(a))