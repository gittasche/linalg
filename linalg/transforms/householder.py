import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple

from ..utils._validations import _ensure_ndarray
from .cy_householder import cy_house

def house(
    x: ArrayLike,
    i: int,
    overwrite_x: bool = False
) -> Tuple[float, NDArray]:
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

    if not (0 <= i < x.size):
        raise ValueError(
            f"`i` must be in [0, {x.size}),"
            f" got {i}."
        )

    # get `beta` and change `x` in-place
    beta = cy_house(x, i)

    return beta, x