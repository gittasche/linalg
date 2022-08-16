import numpy as np
import warnings
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..utils._validations import _ensure_ndarray
from ..utils._validations import PivotingWarning
from ..utils.solve_band import solve_upper_band
from ..utils.solve import solve_lower
from ..utils.permutation import decode_permutation

def lu_band(
    a: ArrayLike,
    l: int,
    u: int,
    piv_option: Union[None, Literal["row"]] = "row",
    mode: Literal["full", "economic"] = "full",
    overwrite_a: bool = False
) -> Tuple[NDArray, ...]:
    """
    LU decomposition of square banded matrix A.
    Only partial pivoting implemented, because full
    pivoting totally breaks band structure.
    
    A = PLU, where
    P is a permutation matrix for row pivoting,
    L is a unit lower triangle banded matrix,
    U is an upper triangle banded matrix. 

    Paramters
    ---------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be banded
    l : int
        lower bandwidth of A
    u : int
        upper bandwidth of A
    piv_option : None or "row" (default: "row")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"row"`` to interchange rows only
        
    mode : ["full", "economic"] (default: "full")
        return mode (see `Returns` section for details):
        - ``"full"`` is convinient form for further use
        - ``"economic"`` is a workspace economy mode
        
    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    Returns
    -------
    if `mode == "full"` than return tuple(L, U, P, Q):
        `l`, `u`, `p`- ndarrays of shape (n, n), such that A = PLU
    if `mode == "economic"` than return tuple(a, row_ids, col_ids):
        - `a` - ndarray of shape (n, n) contains L in low triangle, U in up triangle
        - `row_ids` - ndarray of shape (n,) encoded P
    """
    copy = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy,
        dtype="float64"
    )

    if piv_option not in [None, "row"]:
        raise ValueError(
            "`piv_option` must be None or 'row',"
            f" got {piv_option}."
        )

    if mode not in ["full", "economic"]:
        raise ValueError(
            "`mode` must be in ['full', 'economic'],"
            f" got {mode}."
        )

    n = a.shape[0]

    if l + u + 1 > n:
        raise ValueError(
            "`l + u + 1` must be <= a.shape[0],"
            f" got {l + u + 1} > {a.shape[0]}."
        )

    row_ids = np.arange(n)
    for i in range(n-1):
        if piv_option is None:
            warnings.warn(
                "Disable pivoting is a bad practice, ensure "
                "there are no zeros on diagonal of `a` matrix.",
                PivotingWarning
            )
        elif piv_option == "row":
            # get pivoting elemnt
            piv_row = np.argmax(np.abs(a[i:i + l + 1, i]))
            piv_row += i
            
            # interchange rows of `a`
            a[[piv_row, i]] = a[[i, piv_row]]
            row_ids[[piv_row, i]] = row_ids[[i, piv_row]]
        
        if a[i, i] == 0.0:
            raise RuntimeError("`a` is a singular matrix.")
        
        l_b = np.minimum(i + l + 1, n)
        u_b = np.minimum(i + u + l + 1, n)

        a[i + 1:l_b, i] /= a[i, i]
        a[i + 1:l_b, i + 1:u_b] -= a[i + 1:l_b, i, np.newaxis] * a[i, i + 1:u_b]

    if mode == "full":
        l = np.tril(a)
        np.fill_diagonal(l, 1.0)
        u = np.triu(a)
        p = decode_permutation(row_ids)
        return l, u, p
    elif mode == "economic":
        return a, row_ids

def lu_band_solve(
    a: ArrayLike,
    l: int,
    u: int,
    b: ArrayLike,
    piv_option: Union[None, Literal["row"]] = "row",
    overwrite_a: bool = False,
    overwrite_b: bool = False
) -> NDArray:
    """
    Solve AX = B, where A is a square nonsingular matrix
    using band LU decomposition A = PLU.

    
    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        square input matrix A
    l : int
        lower bandwidth of A
    u : int
        upper bandwidth of A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    piv_option : None or "row" (default: "row")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"row"`` to interchange rows only

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

    a, row_ids = lu_band(a, l, u, piv_option=piv_option, mode="economic", overwrite_a=True)

    b[:] = b[row_ids]
    # L have unknown bandwidth, U have upper bandwidth l + u
    b = solve_lower(a, b, overwrite_b=True, unit=True)
    b = solve_upper_band(a, u + l, b, overwrite_b=True)

    if is_b1d:
        b = b.ravel()

    return b
