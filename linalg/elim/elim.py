import numpy as np
import warnings
from typing import Literal, Union
from numpy.typing import ArrayLike

from ..utils._validations import _ensure_ndarray
from ..utils._validations import PivotingWarning
from ..utils.solve import solve_diag, solve_upper
    
def solve_elim(
    a: ArrayLike,
    b: ArrayLike,
    piv_option: Union[None, Literal["row", "col", "full"]] = "row",
    strategy: Literal["gauss", "gaussj"] = "gauss",
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    return_a: bool = False
) -> np.ndarray:
    """
    Solve AX = B using row elimination strategies.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        square input matrix A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    piv_option : None or ["row", "col", "full"] (default: "row")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"row"`` to interchange rows only
        - ``"col"`` to interchange cols only
        - ``"full"`` to interchange both rows and cols

    strategy : ["gauss", "gaussj"] (default: "gauss")
        elimination strategy:
        - ``"gauss"`` elimintates only lower rows to get upper
          triangular matrix from `a`
        - ``"gaussj"`` eliminates both lower and upper rows to
          get diagonal matrix from `a`
          
    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    overwrite_b : bool (default: False)
        allow to overwrite `b` matrix
    
    return_a : bool (default: False)
        return permuted `a` if True

    Returns
    -------
    b : ndarray of shape (n, m)
        overwriten array `b` with m solution vectors
    a : ndarray of shape (n, n)
        permuted `a` if `return_a`
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
    
    if piv_option not in [None, "row", "col", "full"]:
        raise ValueError(
            "`piv_option` must be None or in ['row', 'col', 'full'],"
            f" got {piv_option}."
        )

    if strategy not in ["gauss", "gaussj"]:
        raise ValueError(
            "`strategy` must be in ['gauss', 'gaussj'],"
            f" got {strategy}."
        )
    n = a.shape[0]
    # indeces to reconstruct original solution
    # after columns permutation
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
            
            # interchange rows of `a` and `b`
            a[[piv_row, i]] = a[[i, piv_row]]
            b[[piv_row, i]] = b[[i, piv_row]]
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
            
            # interchange rows of `a` and `b`
            a[[piv_row, i]] = a[[i, piv_row]]
            b[[piv_row, i]] = b[[i, piv_row]]
            
            # interchange cols of `a` and note this
            # permutation in `col_ids`
            a[:, [piv_col, i]] = a[:, [i, piv_col]]
            col_ids[[piv_col, i]] = col_ids[[i, piv_col]]
        
        if a[i, i] == 0.0:
            raise RuntimeError("`a` is a singular matrix.")
        
        # perform outer product algorithm
        if strategy == "gauss":
            b[i+1:] -= a[i+1:, i, np.newaxis] * b[i] / a[i, i]
            a[i+1:, i:] -= a[i+1:, i, np.newaxis] * a[i, i:] / a[i, i]
        elif strategy == "gaussj":
            b[:i] -= a[:i, i, np.newaxis] * b[i] / a[i, i]
            b[i+1:] -= a[i+1:, i, np.newaxis] * b[i] / a[i, i]
            a[:i, i:] -= a[:i, i, np.newaxis] * a[i, i:] / a[i, i]
            a[i+1:, i:] -= a[i+1:, i, np.newaxis] * a[i, i:] / a[i, i]
        
    if a[-1, -1] == 0.0:
        raise RuntimeError("`a` is a singular matrix.")

    if strategy == "gauss":
        b = solve_upper(a, b, overwrite_b=True)
    elif strategy == "gaussj":
        b[:-1] -= a[:-1, -1, np.newaxis] * b[-1] / a[-1, -1]
        a[:-1, -1] = 0.0
        b = solve_diag(a, b, overwrite_b=True)

    # get output `b` in the same form as input
    if is_b1d:
        b = b.ravel()

    # interchange rows back
    ids = np.arange(n)
    b[col_ids] = b[ids]

    if return_a:
        return b, a
    else:
        return b
