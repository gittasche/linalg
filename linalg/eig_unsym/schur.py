import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..utils._validations import _ensure_ndarray
from ..transforms.givens import givens
from .house_hess import house_hess

def schur(a, overwrite_a=False):
    copy_a = not overwrite_a
    a = _ensure_ndarray(
        a,
        ensure_square=True,
        copy=copy_a,
        dtype="float64"
    )

    u, a = house_hess(a, mode="full", overwrite_a=True)
    
