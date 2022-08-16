import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple, Union, Optional

from .qr_decomp.qr_givens import qr_givens
from .qr_decomp.qr_house import qr_house, qr_house_piv

def qr(
    a: ArrayLike,
    mode: Literal["full", "economic", "r"] = "full",
    pivoting: bool = False,
    decode_p: Optional[bool] = None,
    overwrite_a=False
) -> Union[Tuple[NDArray, ...], NDArray]:
    if mode not in ["full", "economic", "r"]:
        raise ValueError(
            "Availible `mode` only in ['full', 'economic', 'r'],"
            f" got {mode}."
        )

    if pivoting:
        if decode_p is None:
            raise ValueError(
                "`decode_p` must be of type bool if `pivoting == True`",
                f"got {decode_p} of type {type(decode_p).__name__}."
            )
        return qr_house_piv(a, mode=mode, decode_p=decode_p, overwrite_a=overwrite_a)
    else:
        if mode == "full":
            return qr_givens(a, mode=mode, overwrite_a=overwrite_a)
        elif mode == "economic":
            return qr_house(a, mode=mode, overwrite_a=overwrite_a)
        elif mode == "r":
            return qr_givens(a, mode=mode, overwrite_a=overwrite_a)