import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..utils._validations import _ensure_ndarray
from ..transforms.householder import house

def bidiag(
    a: ArrayLike,
    mode: Literal["full", "economic"] = "full",
    overwrite_a: bool = False
) -> Union[Tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """
    Upper bidiagonalization of matrix A using Householder reflections.
    A = UBV^T, where U, V - orthogonal, B - upper bidiagonal

    Parameters
    ----------
    a : ArrayLike of shape (m, n)
        input matrix A
    mode : ["full", "economic"] (default: "full")
        return options:
        - ``"full"`` - return ``u``, ``b``, ``v``, such that ``a = u @ b @ v.T``
        - ``"economic"`` - return ``a`` with B on diagonal and superdiagonal and
        u and v in triangles in factored forms.
    
    overwrite_a : bool (default: False)
        allow to overwrite ``a``

    Returns
    -------
    - ``"full"`` - return ``u``, ``b``, ``v``, such that ``a = u @ b @ v.T``
    - ``"economic"`` - return ``a`` with B on diagonal and superdiagonal and
    u and v in triangles in factored forms.
    """
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_2d=True,
        copy=copy_a,
        dtype="float64"
    )

    m, n = a.shape
    k = np.minimum(m, n)
    if mode == "full":
        u_betas = np.zeros(k)
        v_betas = np.zeros(k)
    for i in range(k):
        beta, x = house(a[i:, i], 0)
        if mode == "full":
            u_betas[i] = beta
        a[i:, i:] -= beta * np.dot(np.outer(x, x), a[i:, i:])
        a[i + 1:, i] = x[1:m - i]
        if i < n - 2:
            beta, x = house(a[i, i + 1:], 0)
            if mode == "full":
                v_betas[i] = beta
            a[i:, i + 1:] -= beta * np.dot(a[i:, i + 1:], np.outer(x, x))
            a[i, i + 2:] = x[1:n - i - 1]
    
    if mode == "full":
        u = np.identity(m)
        x = np.zeros(m)
        for i in range(k-1, -1, -1):
            x[i] = 1.0
            x[i + 1:] = a[i + 1:, i]
            u[i:, i:] -= u_betas[i] * np.dot(np.outer(x[i:], x[i:]), u[i:, i:])

        v = np.identity(n)
        x = np.zeros(n - 1)
        for i in range(k-1, -1, -1):
            if i < n - 2:
                x[i] = 1.0
                x[i + 1:] = a[i, i + 2:]
                v[i + 1:, i + 1:] -= v_betas[i] * np.dot(np.outer(x[i:], x[i:]), v[i + 1:, i + 1:])

        b = np.zeros_like(a)
        ids = np.arange(k)
        b[ids, ids] = np.diag(a)
        if m >= n:
            b[ids[:-1], ids[1:]] = np.diag(a, k=1)
        else:
            b[ids, ids + 1] = np.diag(a, k=1)
    
        return u, b, v
    elif mode == "economic":
        return a
    