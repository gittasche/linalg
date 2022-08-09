import numpy as np
from typing import Literal
from numpy.typing import ArrayLike

from ..utils._validations import _ensure_ndarray

def determinant(
    a: ArrayLike,
    piv_option: Literal["row", "col", "full"] = "row"
) -> float:
    """
    Get determinant of a matrix using
    Gauss-Jordan diagonalization.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A
    piv_option : ["row", "col", "full"] (default: "row")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"row"`` to interchange rows only
        - ``"col"`` to interchange cols only
        - ``"full"`` to interchange both rows and cols
    
    Returns
    -------
    det : float
        det(a)
    """
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        dtype="float64"
    )

    if piv_option not in ["row", "col", "full"]:
        raise ValueError(
            "`piv_option` must be in ['row', 'col', 'full'],"
            f" got {piv_option}."
        )
    
    n = a.shape[0]
    
    # any row or col swap changes det sign
    det_sign = 1
    for i in range(n-1):
        if piv_option == "row":
            # get pivoting elemnt
            piv_row = np.argmax(np.abs(a[i:, i]))
            piv_row += i
            
            if piv_row != i:
                det_sign *= -1
            
            a[[piv_row, i]] = a[[i, piv_row]]
        elif piv_option == "col":
            # get pivoting elemnt
            piv_col = np.argmax(np.abs(a[i, i:]))
            piv_col += i

            if piv_col != i:
                det_sign *= -1

            a[:, [piv_col, i]] = a[:, [i, piv_col]]
        elif piv_option == "full":
            # get pivoting elemnt
            piv_idx = np.argmax(np.abs(a[i:, i:]))
            piv_row, piv_col = np.unravel_index(piv_idx, a[i:, i:].shape)
            piv_row += i
            piv_col += i
            
            if piv_row != i:
                det_sign *= -1
            if piv_col != i:
                det_sign *= -1
            
            a[[piv_row, i]] = a[[i, piv_row]]
            a[:, [piv_col, i]] = a[:, [i, piv_col]]
        
        if a[i, i] == 0.0:
            raise RuntimeError("`a` is a singular matrix.")
        
        a[:i, i:] -= a[:i, i, np.newaxis] * a[i, i:] / a[i, i]
        a[i+1:, i:] -= a[i+1:, i, np.newaxis] * a[i, i:] / a[i, i]
        
    if a[-1, -1] == 0.0:
        raise RuntimeError("`a` is a singular matrix.")
    
    return np.prod(np.diag(a)) * det_sign
    