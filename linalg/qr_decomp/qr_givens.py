import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..transforms.givens import givens
from ..utils._validations import _ensure_ndarray

def qr_givens(
    a: ArrayLike,
    mode: Literal["full", "economic", "r"] = "full",
    overwrite_a: bool = False
) -> Union[Tuple[NDArray, ...], NDArray]:
    """
    Get QR decomposition of rectangular matrix A (``A = QR``),
    where Q is an orthogonal and R is an upper triangular.
    Use givens rotations method.

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
        q = np.identity(m)

    for i in range(k):
        for j in range(m-1, i, -1):
            g = givens(a[j - 1, i], a[j, i], mode="ndarray")
            a[j - 1:j + 1, i:] = np.dot(g, a[j - 1:j + 1, i:])
            if mode in ["full", "economic"]:
                q[:, j - 1:j + 1] = np.dot(q[:, j - 1:j + 1], g.T)

    if mode == "full":
        return q, np.triu(a)
    elif mode == "economic":
        return q[:, :k], np.triu(a[:k])
    elif mode == "r":
        return np.triu(a)