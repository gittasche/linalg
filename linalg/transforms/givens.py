import numpy as np
import numbers
from numpy.typing import ArrayLike
from typing import Union, Tuple, Literal

from ..utils._validations import _ensure_ndarray

def givens(
    a: Union[int, float],
    b: Union[int, float],
    mode: Literal["tuple", "ndarray"] = "tuple",
) -> Union[Tuple[float, float], np.ndarray]:
    """
    Compute Givens matrix or its elements. Givens matrix ``G`` is a ``2 x 2`` matrix
    such that ``G @ [a, b]^T = [*, 0]^T`` for given [a, b], where ``*`` is 
    an unknown number.

    Paramters
    ---------
    a : int or float
        first entry in [a, b]
    b : int or float
        second entry in [a, b]
    mode : {"tuple", "ndarray"} (default: "tuple")
        return option (see ``Returns`` section)
    
    Returns
    -------
    if ``mode == "tuple"``
        - ``c`` (float) - cosine entry in Givens matrix
        - ``s`` (float) - sine entry in Givens matrix
    if ``mode == "ndarray"``
        - ``g`` (ndarray of shape (2, 2)) - Givens matrix
    """
    if not isinstance(a, (numbers.Integral, numbers.Real)):
        raise TypeError(
            "`a` must be real or integer scalar",
            f" got {a} of type {type(a).__name__}."
        )
    if not isinstance(b, (numbers.Integral, numbers.Real)):
        raise TypeError(
            "`b` must be real or integer scalar",
            f" got {b} of type {type(b).__name__}."
        )
    
    if b == 0.0:
        c = 1.0
        s = 0.0
    else:
        if np.abs(b) > np.abs(a):
            tau = -a / float(b)
            s = 1 / np.sqrt(1.0 + tau**2)
            c = s * tau
        else:
            tau = -b / float(a)
            c = 1 / np.sqrt(1.0 + tau**2)
            s = c * tau
    
    if mode == "tuple":
        return c, s
    elif mode == "ndarray":
        g = np.array([
            [c, -s],
            [s, c]
        ])
        return g