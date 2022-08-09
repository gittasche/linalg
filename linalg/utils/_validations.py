import numpy as np
import warnings
from collections.abc import Sequence

AVAILIBLE_DTYPES = ["float64", "int64", "float32", "int32"]

def _is_arraylike(x):
    if isinstance(x, Sequence) or isinstance(x, np.ndarray):
        return True

def _ensure_ndarray(
    x,
    ensure_1d=False,
    ensure_2d=False,
    ensure_square=False,
    copy=True,
    dtype=None
):
    if not isinstance(x, np.ndarray):
        type_name = type(x).__name__
        if not _is_arraylike(x):
            raise ValueError(f"Array-like expected, got {type_name}.")

        # raise error if `x` is a nested sequences
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            x = np.asarray(x)
    
    if ensure_1d and ensure_2d:
        raise ValueError(
            "Array can not be both 1d and 2d."
        )
    if ensure_1d:
        if x.ndim != 1:
            raise ValueError(
                "`x` must be 1d matrix"
                f", got {x.ndim}d."
            )
    if ensure_2d:
        if x.ndim != 2:
            raise ValueError(
                "`x` must be 2d matrix"
                f", got {x.ndim}d."
            )

    if ensure_square:
        if x.ndim != 2:
            raise ValueError(
                "non 2d `x` can not be square matrix"
                f", got {x.ndim}d."
            )
        if x.shape[0] != x.shape[1]:
            raise ValueError(
                "`x` must be a square matrix,"
                f" got array of shape {x.shape}."
            )

    if dtype is not None:
        if dtype not in AVAILIBLE_DTYPES:
            raise ValueError(
                f"`dtype` must be in {AVAILIBLE_DTYPES},"
                f" got {dtype}."
            )
        if x.dtype != dtype:
            # prevent copy for effective workspace use
            x = x.astype(dtype=dtype, copy=False)

    if copy:
        return np.copy(x)
    else:
        return x

def is_symmetric(a):
    a = _ensure_ndarray(
        a,
        ensure_square=True
    )
    return np.allclose(a, a.T)

class PivotingWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
        