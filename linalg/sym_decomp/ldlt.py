import numpy as np
import warnings
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..utils._validations import _ensure_ndarray
from ..utils._validations import PivotingWarning
from ..utils.solve import solve_lower, solve_diag
from ..utils.permutation import decode_permutation

def ldlt(
    a: ArrayLike,
    piv_option: Union[None, Literal["sym"]] = "sym",
    mode: Literal["full", "economic"] = "full",
    overwrite_a: bool = False
) -> Tuple[NDArray, ...]:
    """
    LDL^T (PLDL^TP^T) decomposition of a symmetric matrix A.
    Availible only symmetric pivoting to preserve symmetry of A,
    such that A = PLDL^TP^T, where P is a permutation matrix.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be symmetric.
    piv_option : None or "sym" (default: "sym")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"sym"`` symmetric pivoting searchs max in diagonal
    
    mode : ["full", "economic"] (default: "full")
        return mode (see `Returns` section for details):
        - ``"full"`` is convinient form for further use
        - ``"economic"`` is a workspace economy mode

    overwrite_a : bool (default: False)
        allow to overwrite `a` matrix
    
    Returns
    -------
    if `mode == "full"` than return tuple (l, d, p):
        - `l` - ndarray of shape (n, n) unit lower triangle matrix
        - `d` - ndarray of shape (n,) diagonal elements of D
        - `p` - ndarray of shape (n, n) permutation matrix
    
    if `mode == "economic"` than return tuple (a, diag_ids):
        - `a` - overwritten `a` with L[i, j] in i > j and
          D[i, i] on diagonal entries
        - `diag_ids` - ndarray of shape (n,) encoded permutation matrix
    """
    copy = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy,
        dtype="float64"
    )

    if piv_option not in [None, "sym"]:
        raise ValueError(
            "`piv_option` must be in [None, 'sym'],"
            f" got {piv_option}."
        )
    
    if mode not in ["full", "economic"]:
        raise ValueError(
            "`mode` must be in ['full', 'economic'],"
            f" got {mode}."
        )
        
    n = a.shape[0]

    v = np.zeros(n)
    diag_ids = np.arange(n)
    for i in range(n):
        if piv_option is None:
            warnings.warn(
                "Disable pivoting is a bad practice, ensure "
                "there are no zeros on diagonal of `a` matrix.",
                PivotingWarning
            )
        elif piv_option == "sym":
            piv_diag = np.argmax(np.diag(a[i:, i:]))
            if a[piv_diag + i, piv_diag + i] == 0.0:
                # take L1 norm if pivoting element is zero
                l1_norms = np.sum(np.abs(a[i:, :i+1]), axis=1)
                piv_diag = np.argmax(l1_norms)
            piv_diag += i
            a[[piv_diag, i]] = a[[i, piv_diag]]
            a[:, [piv_diag, i]] = a[:, [i, piv_diag]]
            diag_ids[[piv_diag, i]] = diag_ids[[i, piv_diag]]

        v[:i] = a[i, :i] * np.diag(a[:i, :i])
        v[i] = a[i, i] - np.dot(a[i, :i], v[:i])
        a[i, i] = v[i]
        a[i + 1:, i] = (a[i + 1:, i] - np.dot(a[i + 1:, :i], v[:i])) / v[i]

    if mode == "full":
        l = np.tril(a)
        np.fill_diagonal(l, 1.0)
        d = np.diag(a)
        p = decode_permutation(diag_ids)
        return l, d, p
    elif mode == "economic":
        return a, diag_ids

def sym_solve(
    a: ArrayLike,
    b: ArrayLike,
    piv_option: Union[None, Literal["sym"]] = "sym",
    overwrite_a: bool = False,
    overwrite_b: bool = False
) -> NDArray:
    """
    Solve AX = B where A is a symmetric indefinite matrix
    using LDL^T decomposition.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square matrix A assumed to be symmetric.
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    piv_option : None or "sym" (default: "sym")
        pivoting strategy:
        - ``None`` no pivoting (bad option)
        - ``"sym"`` symmetric pivoting searchs max in diagonal
        
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

    a, diag_ids = ldlt(a, piv_option=piv_option, mode="economic", overwrite_a=True)

    n = a.shape[0]
    b[:] = b[diag_ids]
    b = solve_lower(a, b, overwrite_b=True, unit=True)
    b = solve_diag(a, b, overwrite_b=True)
    b = solve_lower(a, b, overwrite_b=True, transposed=True, unit=True)
    ids = np.arange(n)
    b[diag_ids] = b[ids]

    if is_b1d:
        b = b.ravel()

    return b
    