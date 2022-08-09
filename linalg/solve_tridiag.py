import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Sequence
from typing import Union

from .utils._validations import _ensure_ndarray

def solve_tridiag(
    diags: Union[tuple[ArrayLike], list[ArrayLike]],
    d: ArrayLike,
    overwrite_diags: bool = False,
    overwrite_d: bool = False
) -> np.ndarray:
    """
    Solve AX = D, where A is a tridiagonal matrix (Thomas algorithm).
    This algorithm is not stable in general.

    Parameters
    ----------
    diags : Sequence of len 3
        tuple or list of diagonals ``(lower, main, upper)``:
        - ``lower`` - ArrayLike of shape (n-1,)
        - ``main`` - ArrayLike of shape (n,)
        - ``upper`` - ArrayLike of shape (n-1,)
    d : ArrayLike of shape (n, m)
        input matrix D, such that
        m - number of systems A x X[:,i] = D[:,i], i in [1, m]
    overwrite_diags : bool (default: False)
        allow to overwrite `diags`
    overwrite_d : bool (default: False)
        allow to overwrite `d`

    Returns
    -------
    d : ndarray of shape (n, m)
        overwriten array `d` with m solution vectors
    """
    message = (
            "`diags` must be list or tuple with 3 array-like nested,"
            f" got {diags} of type {type(diags).__name__}"
        )

    if not isinstance(diags, Sequence):
        raise TypeError(message)
    
    if len(diags) != 3:
        message += f" with length {len(diags)}"
        raise ValueError(message)

    a, b, c = diags
    copy_diags = not overwrite_diags
    a = _ensure_ndarray(
        a,
        ensure_1d=True,
        copy=copy_diags,
        dtype="float64"
    )
    b = _ensure_ndarray(
        b,
        ensure_1d=True,
        copy=copy_diags,
        dtype="float64"
    )
    c = _ensure_ndarray(
        c,
        ensure_1d=True,
        copy=copy_diags,
        dtype="float64"
    )
    copy_d = not overwrite_d
    d = _ensure_ndarray(
        d,
        copy=copy_d,
        dtype="float64"
    )

    is_d1d = False
    if d.ndim == 1:
        d = d[:, np.newaxis]
        is_d1d = True

    n = b.size
    for i in range(n-1):
        t = a[i] / b[i]
        b[i + 1] -= t * c[i]
        d[i + 1] -= t * d[i]
    d[-1] /= b[-1]
    for i in range(n-2, -1, -1):
        d[i] = (d[i] - c[i] * d[i + 1]) / b[i]

    if is_d1d:
        d = d.ravel()

    return d

def solve_sympos_tridiag(
    diags: Union[tuple[ArrayLike], list[ArrayLike]],
    b: ArrayLike,
    overwrite_diags: bool = False,
    overwrite_b: bool = False
) -> np.ndarray:
    """
    Solve AX = B, where A is a SPD tridiagonal matrix.
    This algorithm is well-conditioned due to SPD.

    Parameters
    ----------
    diags : Sequence of len 3
        tuple or list of diagonals ``(main, super)``:
        - ``main`` - ArrayLike of shape (n,)
        - ``super`` - ArrayLike of shape (n-1,)
    d : ArrayLike of shape (n, m)
        input matrix D, such that
        m - number of systems A x X[:,i] = D[:,i], i in [1, m]
    overwrite_diags : bool (default: False)
        allow to overwrite `diags`
    overwrite_d : bool (default: False)
        allow to overwrite `d`

    Returns
    -------
    d : ndarray of shape (n, m)
        overwriten array `d` with m solution vectors
    """
    message = (
            "`diags` must be list or tuple with 2 array-like nested,"
            f" got {diags} of type {type(diags).__name__}"
        )

    if not isinstance(diags, Sequence):
        raise TypeError(message)
    
    if len(diags) != 2:
        message += f" with length {len(diags)}"
        raise ValueError(message)

    d, e = diags
    copy_diags = not overwrite_diags
    d = _ensure_ndarray(
        d,
        ensure_1d=True,
        copy=copy_diags,
        dtype="float64"
    )
    e = _ensure_ndarray(
        e,
        ensure_1d=True,
        copy=copy_diags,
        dtype="float64"
    )
    copy_b = not overwrite_b
    b = _ensure_ndarray(
        b,
        ensure_1d=True,
        copy=copy_b,
        dtype="float64"
    )

    is_b1d = False
    if b.ndim == 1:
        b = b[:, np.newaxis]
        is_b1d = True

    n = d.size

    for i in range(n-1):
        t = e[i]
        e[i] /= d[i]
        d[i + 1] -= t * e[i]
        b[i + 1] -= e[i] * b[i]
    b[-1] /= d[-1]
    for i in range(n-2, -1, -1):
        b[i] = b[i] / d[i] - e[i] * b[i + 1]

    if is_b1d:
        b = b.ravel()

    return b