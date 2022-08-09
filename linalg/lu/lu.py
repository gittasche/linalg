import numpy as np
import warnings
from numpy.typing import ArrayLike
from typing import Literal, Tuple, Union

from ..utils._validations import _ensure_ndarray
from ..utils._validations import PivotingWarning
from ..utils.solve import solve_lower, solve_upper
from ..utils.permutation import decode_permutation

def lu(
    a: ArrayLike,
    piv_option: Union[None, Literal["row", "col", "full"]] = "row",
    mode: Literal["full", "economic"] = "full",
    overwrite_a: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Get LU (PLU, LUQ, PLUQ) decomposition of square matrix.
    If no pivoting: A = LU
    If row pivoting: A = PLU, P - permutation matrix
    If col pivoting: A = LUQ, Q - permutation matrix
    If full pivoting: A = PLUQ

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix to decompose
    piv_option : ["row", "col", "full"] (default: "row")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"row"`` to interchange rows only
        - ``"col"`` to interchange cols only
        - ``"full"`` to interchange both rows and cols
        
    mode : ["full", "economic"] (default: "full")
        return mode (see `Returns` section for details):
        - ``"full"`` is convinient form for further use
        - ``"economic"`` is a workspace economy mode
    
    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    Returns
    -------
    if `mode == "full"` than return tuple(L, U, P, Q):
        `l`, `u`, `p`, `q` - ndarrays of shape (n, n), such that A = PLUQ
    if `mode == "economic"` than return tuple(a, row_ids, col_ids):
        - `a` - ndarray of shape (n, n) contains L in low triangle, U in up triangle
        - `row_ids` - ndarray of shape (n,) encoded P
        - `col_ids` - ndarray of shape (n,) encoded Q
    """
    copy = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy,
        dtype="float64"
    )

    if piv_option not in [None, "row", "col", "full"]:
        raise ValueError(
            "`piv_option` must be in ['row', 'col', 'full'],"
            f" got {piv_option}."
        )
    
    if mode not in ["full", "economic"]:
        raise ValueError(
            "`mode` must be in ['full', 'economic'],"
            f" got {mode}."
        )
    
    n = a.shape[0]

    row_ids = np.arange(n)
    col_ids = np.arange(n)
    for i in range(n-1):
        if piv_option is None:
            warnings.warn(
                "Disable pivoting is a bad practice, ensure "
                "there are no zeros on diagonal of `a` matrix.",
                PivotingWarning
            )
        elif piv_option == "row":
            # get pivoting elemnt
            piv_row = np.argmax(np.abs(a[i:, i]))
            piv_row += i
            
            # interchange rows of `a`
            a[[piv_row, i]] = a[[i, piv_row]]
            row_ids[[piv_row, i]] = row_ids[[i, piv_row]]
        elif piv_option == "col":
            # get pivoting elemnt
            piv_col = np.argmax(np.abs(a[i, i:]))
            piv_col += i
            
            # interchange cols of `a` and note this
            # permutation in `col_ids`
            a[:, [piv_col, i]] = a[:, [i, piv_col]]
            col_ids[[piv_col, i]] = col_ids[[i, piv_col]]
        elif piv_option == "full":
            # get pivoting elemnt
            piv_idx = np.argmax(np.abs(a[i:, i:]))
            piv_row, piv_col = np.unravel_index(piv_idx, a[i:, i:].shape)
            piv_row += i
            piv_col += i
            
            # interchange rows of `a`
            a[[piv_row, i]] = a[[i, piv_row]]
            row_ids[[piv_row, i]] = row_ids[[i, piv_row]]
            
            # interchange cols of `a` and note this
            # permutation in `col_ids`
            a[:, [piv_col, i]] = a[:, [i, piv_col]]
            col_ids[[piv_col, i]] = col_ids[[i, piv_col]]

        if a[i, i] == 0.0:
            raise RuntimeError("`a` is a singular matrix.")

        # perform outer product algorithm
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i, np.newaxis] * a[i, i+1:]

    if mode == "full":
        l = np.tril(a)
        np.fill_diagonal(l, 1.0)
        u = np.triu(a)
        p = decode_permutation(row_ids)
        q = decode_permutation(col_ids).T
        return l, u, p, q
    elif mode == "economic":
        return a, row_ids, col_ids

def lu_solve(
    a: ArrayLike,
    b: ArrayLike,
    piv_option: Union[None, Literal["row", "col", "full"]] = "row",
    overwrite_a: bool = False,
    overwrite_b: bool = False
) -> np.ndarray:
    """
    Solve AX=B using LU decomposition
    
    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        square input matrix A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    piv_option : ["row", "col", "full"] (default: "row")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"row"`` to interchange rows only
        - ``"col"`` to interchange cols only
        - ``"full"`` to interchange both rows and cols

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    
    Returns
    -------
    b : ndarray of shape (n, m)
        overwriten array `b` with m solution vectors
    """
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy_a,
        dtype="float64"
    )
    copy_b = not overwrite_b
    b = _ensure_ndarray(
        b,
        copy=copy_b,
        dtype="float64"
    )
    
    is_b1d = False
    if b.ndim == 1:
        b = b[:, np.newaxis]
        is_b1d = True

    # use workspace effective LU
    a, row_ids, col_ids = lu(a, piv_option=piv_option, mode="economic", overwrite_a=True)

    n = a.shape[0]
    b[:] = b[row_ids]
    b = solve_lower(a, b, overwrite_b=True, unit=True)
    b = solve_upper(a, b, overwrite_b=True)
    ids = np.arange(n)
    b[col_ids] = b[ids]

    if is_b1d:
        b = b.ravel()

    return b
    