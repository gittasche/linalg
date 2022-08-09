import numpy as np
from typing import Literal
from numpy.typing import ArrayLike

from .elim import solve_elim
from ..utils._validations import _ensure_ndarray

def inverse(
    a: ArrayLike,
    piv_option: Literal["row", "col", "full"] = "row",
    strategy: Literal["gauss", "gaussj"] = "gauss"
) -> np.ndarray:
    """
    Get inverse matrix using gauss elimination

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
    
    strategy : ["gauss", "gaussj"] (default: "gauss")
        elimination strategy:
        - ``"gauss"`` elimintates only lower rows to get upper
          triangular matrix from `a`
        - ``"gaussj"`` eliminates both lower and upper rows to
          get diagonal matrix from `a`
    
    Returns
    -------
    a : ndarray of shape (n, n)
        output permuted matrix A, such that A^-1 x A = A x A^-1 = I
    """
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        dtype="float64"
    )
    n = a.shape[0]

    b = np.identity(n, dtype=np.float64)
    
    return solve_elim(a, b, piv_option=piv_option, strategy=strategy)
    