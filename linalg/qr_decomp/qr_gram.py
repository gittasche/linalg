import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple

from ..utils._validations import _ensure_ndarray

def qr_gram(
    a: ArrayLike,
    overwrite_a: bool = False
) -> Tuple[NDArray, ...]:
    """
    Get QR decomposition of rectangular matrix A (``A = QR``),
    where Q is an orthonormal and R is an upper triangular.
    Use Gram-Schmidt method.

    Parameters
    ----------
    a : ArrayLike of shape (m, n)
        input matrix A
    overwrite_a : bool (default: False)
        allow to overwrite ``a``

    Returns
    -------
    a : ndarray of shape (m, n)
        Q matrix in economic form
    r : ndarray of shape (n, n)
        R matrix in economic form
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
    r = np.zeros((n, n))
    for i in range(k):
        r[i, i] = np.linalg.norm(a[:, i])
        a[:, i] /= r[i, i]
        r[i, i + 1:] = np.dot(a[:, i], a[:, i + 1:])
        a[:, i + 1:] -= np.dot(a[:, i, np.newaxis], r[np.newaxis, i, i + 1:])
    return a, r