import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..transforms.householder import house
from ..utils._validations import _ensure_ndarray
from ..utils.permutation import decode_permutation

def qr_house(
    a: ArrayLike,
    mode: Literal["full", "economic", "r"] = "full",
    overwrite_a=False
) -> Union[Tuple[NDArray, ...], NDArray]:
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

    if mode not in ["full", "economic", "r"]:
        raise ValueError(
            "Availible `mode` only in ['full', 'economic', 'r'],"
            f" got {mode}."
        )

    m, n = a.shape
    k = np.minimum(m, n)
    if mode in ["full", "economic"]:
        betas = np.zeros(k)
    
    # compute R matrix and Householder vectors
    for i in range(k):
        beta, v = house(a[i:, i], i=0)
        if mode in ["full", "economic"]:
            betas[i] = beta
        a[i:, i:] -= beta * np.dot(np.outer(v, v), a[i:, i:])
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
            v[i + 1:] = a[i + 1:, i]
            q[i:, i:] -= betas[i] * np.dot(np.outer(v[i:], v[i:]), q[i:, i:])

    if mode == "full":
        return q, np.triu(a)
    elif mode == "economic":
        return q, np.triu(a[:k])
    elif mode == "r":
        return np.triu(a)

def qr_house_piv(
    a: ArrayLike,
    mode: Literal["full", "economic", "r"] = "full",
    decode_p: bool = True,
    overwrite_a=False
) -> Tuple[NDArray, ...]:
    """
    Get QR decomposition of rectangular matrix A (``A = QR``),
    where Q is an orthogonal and R is an upper triangular.
    Use householder reflections with column pivoting approach.

    Parameters
    ----------
    a : ArrayLike of shape (m, n)
        input matrix A
    mode : ["full", "economic", "r"] (default: "full")
        return options (see ``Returns`` section for details)
    decode_p: bool (default: True)
        return permutation matrix:
        - ``True`` in full (n, n) form
        - ``False`` in encoded (n,) form

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
    pivs : ndarray
        permutation matrix of shape (n, n) if ``decode_p == True`` such that A = QRP,
        encoded permutation matrix of shape (n,) if ``decode_p == False`` such that
        A[:, pivs] = QR
    """
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_2d=True,
        copy=copy_a,
        dtype="float64"
    )

    if mode not in ["full", "economic", "r"]:
        raise ValueError(
            "Availible `mode` only in ['full', 'economic', 'r'],"
            f" got {mode}."
        )

    m, n = a.shape
    k = np.minimum(m, n)
    # squared cols euclidean norms
    c = np.einsum("ij,ij->j", a, a)
    piv_idx = np.argmax(c)
    pivs = np.arange(n)
    if mode in ["full", "economic"]:
        betas = np.zeros(k)
    for i in range(k):
        if c[piv_idx] == 0.0:
            break
        a[:, [i, piv_idx]] = a[:, [piv_idx, i]]
        c[[i, piv_idx]] = c[[piv_idx, i]]
        pivs[[i, piv_idx]] = pivs[[piv_idx, i]]
        beta, v = house(a[i:, i], 0)
        if mode in ["full", "economic"]:
            betas[i] = beta
        a[i:, i:] -= beta * np.dot(np.outer(v, v), a[i:, i:])
        a[i + 1:, i] = v[1:m - i]
        c[i + 1:] -= a[i, i + 1:]**2
        if i < n - 1:
            piv_idx = np.argmax(c[i + 1:]) + i + 1
    
    if mode in ["full", "economic"]:
        if mode == "full":
            q = np.identity(m)
        elif mode == "economic":
            q = np.eye(m, k)
        v = np.zeros(m)
        for i in range(k-1, -1, -1):
            v[i] = 1.0
            v[i + 1:] = a[i + 1:, i]
            q[i:, i:] -= betas[i] * np.dot(np.outer(v[i:], v[i:]), q[i:, i:])

    if decode_p:
        pivs = decode_permutation(pivs).T

    if mode == "full":
        return q, np.triu(a), pivs
    elif mode == "economic":
        return q, np.triu(a[:k]), pivs
    elif mode == "r":
        return np.triu(a), pivs
        