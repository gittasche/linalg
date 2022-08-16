import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal

from .utils._validations import _ensure_ndarray
from .lu.lu import lu_solve
from .sym_decomp.ldlt import sym_solve
from .sympos_decomp.cholesky import sympos_solve

def solve(
    a: ArrayLike,
    b: ArrayLike,
    assume_a: Literal["gen", "sym", "pos"] = "gen",
    overwrite_a: bool = False,
    overwrite_b: bool = False
) -> NDArray:
    """
    Interface to solve AX = B, where A is square full-rank
    N x N matrix, B is a N x NRHS matrix.

    Parameters
    ----------
    a : ArrayLike of shape (n, n)
        input square full-rank matrix A
    b : ArrayLike of shape (n, m)
        input matrix B, such that
        m - number of systems A x X[:,i] = B[:,i], i in [1, m]
    assume_a : ["gen", "sym", "pos"] (default: "gen")
        Assume type of input matrix A:
        - ``"gen"`` - general square full-rank matrix,
          use LU decomposition to solve
        - ``"sym"`` - symmetric full-rank matrix,
          use LDL^T decomposition to solve
        - ``"pos"`` - symmetric positive definite full-rank matrix,
          use Cholesky decomposition to solve
    
    overwrite_a : bool (default: False)
        allow to overwrite `a`
    overwrite_b : bool (default: False)
        allow to overwrite `b`
    
    Returns
    -------
    b : ndarray of shape (n, m)
        matrix of solutions
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
    
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "`a` and `b` must have equal number of columns,"
            f" got {a.shape[0]} and {b.shape[0]}."
        )

    if assume_a not in ["gen", "sym", "pos"]:
        raise ValueError(
            "Unknown `assume_a` option: availible ['gen', 'sym', 'pos'],"
            f" got {assume_a}."
        )

    if assume_a == "gen":
        b = lu_solve(a, b, overwrite_a=True, overwrite_b=True)
    elif assume_a == "sym":
        b = sym_solve(a, b, overwrite_a=True, overwrite_b=True)
    elif assume_a == "pos":
        b = sympos_solve(a, b, overwrite_a=True, overwrite_b=True)
    
    return b