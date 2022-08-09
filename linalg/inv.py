import numpy as np
from numpy.typing import ArrayLike

from .lu.lu import lu
from .utils._validations import _ensure_ndarray
from .utils.solve import solve_lower, solve_upper

def inv(
    a: ArrayLike,
    overwrite_a: bool = False
) -> np.ndarray:
    """
    Inverse square matrix A using LU decomposition.
    It solves equation U x A^-1 = L^-1

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix
    overwrite_a : bool (default: False)
        allow to overwrite `a`
    
    Returns
    -------
    a_inv : ndarray of shape (n, n)
        inverse `a`
    """
    copy = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy,
        dtype="float64"
    )

    a, row_ids, _ = lu(a, overwrite_a=True, mode="economic")

    n = a.shape[0]
    # probably it can be solved without allocation of new identity
    b = np.identity(n)
    b[:] = b[row_ids]
    b = solve_lower(a, b, overwrite_b=True, unit=True)
    # U x A^-1 = L^-1
    b = solve_upper(a, b, overwrite_b=True)
    return b