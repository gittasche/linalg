import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils.solve import solve_lower, solve_upper
from .utils._validations import _ensure_ndarray

def solve_triangle(
    a: ArrayLike,
    b: ArrayLike,
    lower: bool = False,
    transposed: bool = False,
    unit: bool = False,
    overwrite_b: bool = False
) -> NDArray:
    """
    Solve AX = B, where A in general is a triangle matrix

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be triangle.
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    lower : bool (default: False)
        - ``True`` assume `a` is a lower triangle
        - ``False`` assume `a` is an upper triangle

    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    transposed : bool (default: False)
        - ``True`` solve A^T x X = B
        - ``False`` solve A x X = B
    
    unit : bool (default: False)
        - ``True`` assume diagonal entries of A to 1.0
        - ``False`` no any assumptions
    
    Returns
    -------
    b : ndarray of shape (n, m)
        overwriten array `b` with m solution vectors
    """
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=False,
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

    if lower:
        b = solve_lower(a, b, overwrite_b=True, transposed=transposed, unit=unit)
    else:
        b = solve_upper(a, b, overwrite_b=True, transposed=transposed, unit=unit)

    return b