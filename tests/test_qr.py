import numpy as np
import pytest
from numpy.testing import assert_allclose

from linalg.qr_decomp import qr_house, qr_givens, qr_gram, qr_house_piv

@pytest.mark.parametrize("a",
    [
        np.array([
            [1, 2, 3],
            [1, 5, 6],
            [1, 8, 9],
            [1, 11, 12]
        ]),
        np.array([
            [12, -51, 4],
            [6, 167, -68],
            [-4, 24, -41]
        ]),
        np.random.standard_normal((9, 6)),
        np.random.standard_normal((6, 9))
    ]
)
@pytest.mark.parametrize("qr_func", [qr_house, qr_house_piv, qr_givens, qr_gram])
@pytest.mark.parametrize("mode", ["full", "economic"])
def test_qr(a, qr_func, mode):
    qr_func_args = qr_func.__code__.co_varnames[:qr_func.__code__.co_argcount]
    if "mode" in qr_func_args:
        if "decode_p" in qr_func_args:
            q, r, p = qr_func(a, mode=mode, decode_p=True)
            assert_allclose(a, q @ r @ p, atol=1e-12)
        else:
            q, r = qr_func(a, mode=mode)
            assert_allclose(a, q @ r, atol=1e-12)
    else:
        q, r = qr_func(a)
        assert_allclose(a, q @ r, atol=1e-12)
