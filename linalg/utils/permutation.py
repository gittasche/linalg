import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple

from ._validations import _ensure_ndarray

def decode_permutation(
    p_ids: ArrayLike
) -> NDArray:
    """
    Decode permutation vector

    Parameters
    ----------
    p_ids : ArrayLike of shape (n,)
        encoded permutation vector
    
    Returns
    -------
    p : ndarray of shape (n, n)
        permutation matrix such that
        - p @ A permutes rows of A
        - A @ p permutes cols of A
    """
    p_ids = _ensure_ndarray(
        p_ids,
        ensure_1d=True,
        dtype="int32"
    )
    n = p_ids.size
    ids = np.arange(n, dtype="int32")

    p = np.identity(n, dtype="int32")
    p[p_ids] = p[ids]
    return p

def encode_permutation(
    p: ArrayLike
) -> NDArray:
    """
    Encode permutation matrix

    Parameters
    ----------
    p : ndarray of shape (n, n)
        permutation matrix such that
        - p @ A permutes rows of A
        - A @ p permutes cols of A
    
    Returns
    -------
    p_ids : ArrayLike of shape (n,)
        encoded permutation vector
    """
    p = _ensure_ndarray(
        p,
        ensure_square=True,
        dtype="int32"
    )
    n = p.shape[0]
    p_ids = np.arange(n, dtype="int32")
    return np.dot(p_ids, p)

def to_triangle(
    a: ArrayLike,
    type: Literal["upper", "lower"] = "upper",
    overwrite_a: bool = False
) -> Tuple[NDArray, ...]:
    """
    Transform matrix `a` into triangle form.
    This function very unstable (does not work in many cases).

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix
    type : ["upper", "lower"] (default: "upper")
        type of output triangle matrix:
        - ``"upper"`` - upper triangular matrix
        - ``"lower"`` - lower triangular matrix

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    Returns
    -------
    a : ndarray of shape (n, n)
        overwriten `a` if `overwrite_a`, else new
        ndarray instance with triangular matrix
    row_ids, col_ids : ndarrays of shape (n,)
        encoded permutation matrices
    """
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy_a
    )
    if type not in ["upper", "lower"]:
        raise ValueError(
            "Avilble `type` in ['upper', 'lower'],"
            f" got {type}."
        )

    row_zero_count = np.sum(a == 0, axis=1)
    col_zero_count = np.sum(a == 0, axis=0)

    if type == "upper":
        row_ids = np.argsort(row_zero_count)
        col_ids = np.argsort(-col_zero_count)
    elif type == "lower":
        row_ids = np.argsort(-row_zero_count)
        col_ids = np.argsort(col_zero_count)
    
    a = a[row_ids]
    a = a[:, col_ids]

    return a, row_ids, col_ids

def extract_band(
    a: ArrayLike,
    diags: ArrayLike
) -> NDArray:
    """
    Extract band from matrix A

    Parameters
    ----------
    a : ArrayLike
        input matrix A
    diags : ArrayLike
        indeces of A diagonals

    Returns
    -------
    a_band : ndarray of shape `a.shape`
        banded part of `a`
    """
    diags = _ensure_ndarray(
        diags,
        ensure_1d=True,
        copy=False,
        dtype="int32"
    )

    if diags.size <= 0:
        raise ValueError(
            "`diags` must have attribute `size > 0`,"
            f" got {diags.size}"
        )

    a_band = np.zeros_like(a)
    for diag in diags:
        a_band += np.diag(np.diag(a, k=diag), k=diag)

    return a_band