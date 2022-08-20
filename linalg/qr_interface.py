import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union, Optional

from .qr_decomp.qr_givens import qr_givens
from .qr_decomp.qr_house import qr_house, qr_house_piv

def qr(
    a: ArrayLike,
    mode: Literal["full", "economic", "r"] = "full",
    pivoting: bool = False,
    decode_p: Optional[bool] = None,
    overwrite_a=False
) -> Union[Tuple[NDArray, ...], NDArray]:
    """
    Get QR decomposition of matrix A (``A = QR``).
    Interface for QR decomposition routines.

    Parameters
    ----------
    a : ArrayLike of shape (m, n)
        input matrix A
    mode : ["full", "economic", "r"] (default: "full")
        return options (see ``Returns`` section for details)
    pivoting : bool (default: False)
        enable column pivoting
    decode_p: bool or None (default: None)
        if not ``None``return permutation matrix:
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
        A[:, pivs] = QR. Not returned if ``pivoting == False``.
    """
    if mode not in ["full", "economic", "r"]:
        raise ValueError(
            "Availible `mode` only in ['full', 'economic', 'r'],"
            f" got {mode}."
        )

    if pivoting:
        if decode_p is None:
            raise ValueError(
                "`decode_p` must be of type bool if `pivoting == True`",
                f"got {decode_p} of type {type(decode_p).__name__}."
            )
        return qr_house_piv(a, mode=mode, decode_p=decode_p, overwrite_a=overwrite_a)
    else:
        if mode == "full":
            return qr_givens(a, mode=mode, overwrite_a=overwrite_a)
        elif mode == "economic":
            return qr_house(a, mode=mode, overwrite_a=overwrite_a)
        elif mode == "r":
            return qr_givens(a, mode=mode, overwrite_a=overwrite_a)