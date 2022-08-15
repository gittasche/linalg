import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

from ..utils._validations import _ensure_ndarray

def house(
    x: ArrayLike,
    i: int,
    overwrite_x: bool = False
) -> Tuple[float, np.ndarray]:
    """
    Get Householder reflection matrix ``P = 1 - beta * v @ v.T``
    such that ``P @ x = ||x||_2 * e_i``.

    Parameters
    ----------
    x : ArrayLike of shape (n,)
        input vector ``x``
    i : int
        nonzero entry in reduced vector
    overwrite_x : bool (default: False)
        allow to overwrite ``x`` with ``v``

    Returns
    -------
    beta : float
        coefficient in Householder reflection matrix ``P``
    x : ndarray of shape (n,)
        overwriten array ``x`` with vector ``v``
    """
    copy_x = not overwrite_x
    x = _ensure_ndarray(
        x,
        ensure_1d=True,
        copy=copy_x,
        dtype="float64"
    )

    x_wi = np.delete(x, i)
    x_i = x[i]
    sigma = np.dot(x_wi, x_wi)
    x[i] = 1.0
    if sigma == 0.0:
        beta = 0.0
    else:
        nu = np.sqrt(x_i**2 + sigma)
        if x_i <= 0.0:
            x[i] = x_i - nu
        else:
            x[i] = -sigma / (x_i + nu)
        beta = 2 * x[i]**2 / (sigma + x[i]**2)
        x /= x[i]
    return beta, x