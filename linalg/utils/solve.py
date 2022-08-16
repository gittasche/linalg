import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal

from ._validations import _ensure_ndarray
from .permutation import to_triangle

def solve_diag(
    a: ArrayLike,
    b: ArrayLike,
    overwrite_b=False
) -> NDArray:
    """
    Solve AX=B, where A is a diagonal matrix

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input matrix A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
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

    b[:] = b / np.diag(a)[:, np.newaxis]

    if is_b1d:
        b = b.ravel()
    
    return b

def solve_lower(
    a: ArrayLike,
    b: ArrayLike,
    overwrite_b=False,
    transposed: bool = False,
    unit: bool = False
) -> NDArray:
    """
    Solve AX=B, where A is a lower triangle matrix

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input matrix A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
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
                b[i] = (b[i] - np.dot(a[i, :i], b[:i])) / a[i, i]
        else:
            for i in range(n):
                b[i] = b[i] - np.dot(a[i, :i], b[:i])
    else:
        if not unit:
            for i in range(n-1, -1, -1):
                b[i] = (b[i] - np.dot(a[i+1:, i], b[i+1:])) / a[i, i]
        else:
            for i in range(n-1, -1, -1):
                b[i] = b[i] - np.dot(a[i+1:, i], b[i+1:])

    if is_b1d:
        b = b.ravel()

    return b

def solve_upper(
    a: ArrayLike,
    b: ArrayLike,
    overwrite_b=False,
    transposed: bool = False,
    unit: bool = False
) -> NDArray:
    """
    Solve AX=B, where A is an upper triangle matrix

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input matrix A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
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
                b[i] = (b[i] - np.dot(a[i, i+1:], b[i+1:])) / a[i, i]
        else:
            for i in range(n-1, -1, -1):
                b[i] = b[i] - np.dot(a[i, i+1:], b[i+1:])
    else:
        if not unit:
            for i in range(n):
                b[i] = (b[i] - np.dot(a[:i, i], b[:i])) / a[i, i]
        else:
            for i in range(n):
                b[i] = b[i] - np.dot(a[:i, i], b[:i])

    if is_b1d:
        b = b.ravel()

    return b

def solve_triangle(
    a: ArrayLike,
    b: ArrayLike,
    overwrite_a=False,
    overwrite_b=False,
    type: Literal["upper", "lower"] = "upper",
    transposed: bool = False,
    unit: bool = False
) -> NDArray:
    """
    Solve AX = B, where A in general is a permuted triangle matrix

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be permuted triangle.
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    type : ["upper", "lower"] (default: "upper")
        type of triangle form of matrix: if ``a`` closer to lower
        triangle form then use it.
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
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy_a
    )
    copy_b = not overwrite_b
    b = _ensure_ndarray(
        b,
        copy=copy_b,
        dtype="float64"
    )
    if type not in ["upper", "lower"]:
        raise ValueError(
            "Avilble `type` in ['upper', 'lower'],"
            f" got {type}."
        )

    a, row_ids, col_ids = to_triangle(a, type=type, overwrite_a=True)

    is_b1d = False
    if b.ndim == 1:
        b = b[:, np.newaxis]
        is_b1d = True

    n = a.shape[0]
    b[:] = b[row_ids]
    if type == "upper":
        b = solve_upper(a, b, overwrite_b=True, transposed=transposed, unit=unit)
    elif type == "lower":
        b = solve_lower(a, b, overwrite_b=True, transposed=transposed, unit=unit)
    ids = np.arange(n)
    b[col_ids] = b[ids]

    if is_b1d:
        b = b.ravel()
    
    return b
