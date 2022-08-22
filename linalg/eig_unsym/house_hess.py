import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..utils._validations import _ensure_ndarray
from ..transforms.householder import house

def house_hess(
    a: ArrayLike,
    mode: Literal["full", "hess"] = "full",
    overwrite_a: bool = False
) -> Union[Tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy_a,
        dtype="float64"
    )

    n = a.shape[0]
    if mode == "full":
        betas = np.zeros(n - 2)
    for i in range(n - 2):
        beta, v = house(a[i + 1:, i], 0)
        if mode == "full":
            betas[i] = beta
        a[i + 1:, i:] -= beta * np.dot(np.outer(v, v), a[i + 1:, i:])
        a[:, i + 1:] -= beta * np.dot(a[:, i + 1:], np.outer(v, v))
        if mode == "full":
            a[i + 2:, i] = v[1:n - i - 1]
    
    if mode == "full":
        u = np.identity(n)
        v = np.zeros(n - 1)
        for i in range(n - 3, -1, -1):
            v[i] = 1.0
            v[i + 1:] = a[i + 2:, i]
            a[i + 2:, i] = 0.0
            u[i + 1:, i + 1:] -= betas[i] * np.dot(np.outer(v[i:], v[i:]), u[i + 1:, i + 1:])

        return u, a
    elif mode == "hess":
        return a
