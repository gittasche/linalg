import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union

from ..utils._validations import _ensure_ndarray
from ..transforms.givens import givens
from .house_hess import house_hess

def schur(a, compute_u = False, overwrite_a=False):
    """
    Not implemented
    """
    raise NotImplementedError()