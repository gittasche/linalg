import numpy as np
from numpy.typing import ArrayLike
from typing import Literal

from ..transforms.householder import house
from ..utils._validations import _ensure_ndarray

def qr_house(
    a: ArrayLike,
    mode: Literal["full", "economic", "r"] = "full",
    overwrite_a=False
):
    """
    Get QR decomposition of rectangular matrix A (``A = QR``),
    where Q is an orthogonal and R is an upper triangular.
    Use householder reflections approach.

    Parameters
    ----------
    a : ArrayLike of shape (m, n)
        input matrix A
    mode : ["full", "economic", "r"] (default: "full")
        return options (see ``Returns`` section for details)
    overwrite_a : bool (default: False)
        allow to overwrite ``a``

    Returns
    -------
    q : ndarray
        Q matrix of shape (m, m) if ``mode == "full"``,
        of shape (m, k) if ``mode == "economic"``, where k = min(m, n).
        Not returned if ``mode == "r"``
    r : ndarray
        R matrix of shape (m, n) if ``mode == "full"`` or ``mode == "r"``,
        of shape (k, n) if ``mode == "economic"``, where k = min(m, n)
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
    if mode in ["full", "economic"]:
        betas = np.zeros(n)
    
    # compute R matrix and Householder vectors
    for i in range(k):
        beta, v = house(a[i:, i], i=0)
        if mode in ["full", "economic"]:
            betas[i] = beta
        p = np.identity(m - i) - beta * np.outer(v, v)
        a[i:, i:] = np.dot(p, a[i:, i:])
        a[i + 1:, i] = v[1:m - i]

    # compute Q matrix if needed
    if mode in ["full", "economic"]:
        if mode == "full":
            q = np.identity(m)
        elif mode == "economic":
            q = np.eye(m, k)
        v = np.zeros(m)
        for i in range(k-1, -1, -1):
            v[i] = 1.0
            v[i+1:] = a[i+1:, i]
            p = np.identity(m - i) - betas[i] * np.outer(v[i:], v[i:])
            q[i:, i:] = np.dot(p, q[i:, i:])

    if mode == "full":
        return q, np.triu(a)
    elif mode == "economic":
        return q, np.triu(a[:k])
    elif mode == "r":
        return np.triu(a)